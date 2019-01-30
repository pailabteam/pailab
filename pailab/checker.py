import logging
from pailab.repo import MLRepo, MLObjectType, NamingConventions
from pailab.repo_objects import RepoInfoKey  # pylint: disable=E0401
from pailab.repo_store import RepoStore  # pylint: disable=E0401
logger = logging.getLogger(__name__)




def get_initial_config(repo):
    model_names = repo.get_names(MLObjectType.MODEL)
    model_checks ={}
    for m in model_names:
        model_checks[m] = {'correct': False, 'check_for_latest': True, 'model_version': RepoStore.LAST_VERSION}
    
    labels = repo.get_names(MLObjectType.LABEL)
    labeled_model_checks = {}
    for l in labels:
            labeled_model_checks[l] = {'correct': False, 'check_for_latest': True}
    model_checks = {'models': model_checks, 'labels': labeled_model_checks}

    data_checks = {'overlapping': True, 'usage': False}
    repo_checks = {'integrity': True}
    return {'model_checks': model_checks, 'data_checks': data_checks, 'repo_checks': repo_checks}

class Model:

    @staticmethod
    def __check_model(repo, model_name, correct, model_version, check_for_latest=False):
        """Check if the model is calibrated and evaluated on the latest versions

        Arguments:
            repo {MLRepo} -- the ml repo
            model_name {str} -- model name
            model_version {version} -- model version to check

        Keyword Arguments:
            correct {bool} -- determine whether training, evaluations, measures and tests will be qutomatically triggered if check fails to correct (default: {False})
            check_for_latest {bool} -- if true, some additional checks are performed to see whether the latest model is calibrated on th latest data

        Returns:
            dict -- dictionary containing the modifier versions and the latest version of the objects, if dictionary is empty noc model inconsistencies could be found
        """
        logging.info('Checking model ' + model_name + ', version: ' + str(model_version) +
                        ', correct: ' + str(correct) + ', check_for_latest: ' + str(check_for_latest))
        model_main_name = model_name.split('/')[0]
        result = {}
        repo_store = repo.get_ml_repo_store()

        # first check if all versions of the models modifiers are still the latest version
        if check_for_latest:
            tmp = {}
            m = repo.get(MLRepo.get_calibrated_model_name(
                model_name), version=model_version)
            for k, v in m.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
                latest_version = repo_store.get_latest_version(k)
                if not str(v) == str(latest_version):
                    tmp[k] = {'modifier version': v,
                                'latest version': latest_version}
            if len(tmp) > 0:
                result['latest model version not on latest inputs'] = tmp
                if correct == True:
                    job_id = repo.run_training(model_name)
                    job_ids = repo.run_evaluation(
                        model_name, message='running evaluations triggered by new training of ' + model_name, predecessors=[job_id])
                    job_ids = repo.run_measures(
                        model_name, message='running measures triggered by new training of ' + model_name, predecessors=job_ids)
                    job_ids = repo.run_tests(
                        model_name, message='running tests triggered by new training of ' + model_name, predecessors=job_ids)
                    return result  # we can return here since everything was corrected
        # simply check if model has been evaluated on the latest data sets
        data_names = repo.get_names(MLObjectType.TRAINING_DATA)
        data_names.extend(repo.get_names(MLObjectType.TEST_DATA))
        eval_check_result = {}
        data_corrected = set()
        for data in data_names:
            latest_data_version = repo_store.get_latest_version(data)
            eval_name = str(NamingConventions.EvalData(
                data=data, model=model_main_name))
            tmp = repo.get(eval_name, version=None, modifier_versions={
                model_name: model_version, data: latest_data_version}, 
                throw_error_not_exist=False, throw_error_not_unique=False)
            if tmp == []:
                eval_check_result[data] = latest_data_version
                if correct:
                    repo.run_evaluation(model_name, message='automatically corrected from _check_model',
                                        model_version=model_version, run_descendants=True)
                    data_corrected.add(data)

        if len(eval_check_result) > 0:
            result['evaluations missing'] = eval_check_result
        # simply check if all measures have been computed on latest data for the respective model
        measures = repo.get_names(MLObjectType.MEASURE_CONFIGURATION)
        measure_check_result = set()
        if len(measures) > 0:  # only if measures were defined
            for m in measures:
                measure_config = repo.get(m)
                for measure_name in measure_config.measures.keys():
                    for data in data_names:
                        tmp = str(NamingConventions.Measure(
                            str(NamingConventions.EvalData(data=data, model=model_main_name)), measure_type=measure_name))
                        obj = repo.get(tmp, version=None, modifier_versions={
                            model_name: model_version}, throw_error_not_exist=False, throw_error_not_unique=False)
                        if obj == []:
                            measure_check_result.add(tmp)
                            # we do only have to correct if the underlying evaluation data has not been corrected (then predecessors are included)
                            if correct and not data in data_corrected:
                                repo.run_measures(model_name, message='automatically corrected from _check_model',
                                                    model_version=model_version, run_descendants=True)

        if len(measure_check_result) > 0:
            result['measures not calculated'] = measure_check_result
        return result

    @staticmethod
    def run(repo: MLRepo, model_name=None, correct=False, model_version=RepoStore.LAST_VERSION, model_label=None, check_for_latest = True):
        """Perform consistency checks for specified model versions

        Args:
            :param repo (MLRepo): ml repository
            :model_name (str, optional): Defaults to None. If specified, the model version(s) specified by the name and the model_version are cheked.
            :param correct (bool, optional): Defaults to False. If True, th metho starts the corresponding jobs to fix the found isues.
            :param model_version (str or list of str, optional): Defaults to RepoStore.LAST_VERSION. The model version(s) of th models to check
            :param model_label ([type], optional): Defaults to None. 

        Raises:
            Exception: Raises if a model version but no model name is specified 

        Returns:
            [dict]: dictionary with model+version to found issues. May be empty if no issues exist.
        """

        logger.info('Start checking model.')
        result = {}
        if model_name is None and model_label is None:
            raise Exception('Please specify a model or a label.')
        if model_label is not None:  # check the model defined by the label
            label = repo.get(model_label)
            tmp = Model.__check_model(repo, label.name, correct,
                                                model_version=label.version)
            if len(tmp) > 0:
                result[model_label] = tmp

        if model_name is not None:  # check the model defined by name and versions
            if len(model_name.split('/')) == 1:
                model_name = model_name + '/model'
            latest_version = repo.get(model_name).repo_info.version
            if model_version is None:
                model_version = RepoStore.LAST_VERSION
            if isinstance(model_version, str):
                model_versions = [model_version]
            else:
                model_versions = model_version
            for version in model_versions:
                if str(latest_version) == str(version) or str(version) == RepoStore.LAST_VERSION:
                    if str(version) == RepoStore.LAST_VERSION:
                        logger.debug(
                            'Latest version found, check if latest version ran on latest data.')
                        tmp = Model.__check_model(repo, model_name, correct,
                                                            latest_version, check_for_latest=check_for_latest)
                    else:
                        tmp = Model.__check_model(repo, model_name, correct,
                                                            version, check_for_latest=check_for_latest)
                else:
                    tmp = Model.__check_model(repo, model_name, correct,
                                        version, check_for_latest=check_for_latest)
                if len(tmp) > 0:
                    result[model_name+':' + str(version)] = tmp

        # if no versions or a fixed model label is specified, check for all labels
        if model_version is None and model_label is None:
            labels = repo.get_names(MLObjectType.LABEL)
            for l in labels:
                label = repo.get(l)
                tmp = Model.__check_model(
                    repo, label.name, correct, label.version, check_for_latest=check_for_latest)
                if len(tmp) > 0:
                    result[label.repo_info.name] = tmp
        logger.info('Finished checking model.')
        return result

    
class Data:
    @staticmethod
    def __check_no_overlapping_training_test(repo):
        """Checks if current training data does overlap with test data

        Args:
            repo (MLRepo): ml repository
        """
        def compute_start_end_index(dataset, rawdata):
            start = dataset.start_index
            if start < 0:
                start = start + rawdata.n_data
            end = dataset.end_index
            if end is None:
                end = rawdata.n_data
            elif end < 0:
                end = rawdata.n_data + end
            return (start, end)

        def overlap(interval_1, interval_2):
            if interval_1[0] < interval_2[0] and interval_1[1] < interval_2[0]:
                return False
            if interval_2[0] < interval_1[0] and interval_2[1] < interval_1[0]:
                return False
            return True

        tmp = repo.get_names(MLObjectType.TRAINING_DATA)
        if len(tmp) == 0:
            raise Exception('Repo contains no training data.')
        training_data = repo.get(tmp[0])
        training_raw = repo.get(training_data.raw_data,
                                version=training_data.raw_data_version)
        training_indices = compute_start_end_index(
            training_data, training_raw)

        test_data_names = repo.get_names(MLObjectType.TEST_DATA)
        result = {}
        for data in test_data_names:
            test_data = repo.get(data)
            if test_data.raw_data == training_data.raw_data:
                test_data_raw = repo.get(
                    test_data.raw_data, version=test_data.raw_data_version)
                test_indices = compute_start_end_index(
                    test_data, test_data_raw)
                if overlap(training_indices, test_indices):
                    result[data] = {'training and test data overlap': {
                        data: test_data.repo_info.version, training_data.repo_info.name: training_data.repo_info.version}}
        return result

    @staticmethod
    def __check_usage(repo):
        """Check if all RawData is used

        Args:
            repo ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """

        def compute_start_end_index(dataset, rawdata):
            start = dataset.start_index
            if start < 0:
                start = start + rawdata.n_data
            end = dataset.end_index
            if end is None:
                end = rawdata.n_data
            elif end < 0:
                end = rawdata.n_data + end
            return (start, end)
        raise NotImplementedError()

    @staticmethod
    def run(repo, overlapping = True, usage = True):
        result = {}
        if overlapping:
            result = Data.__check_no_overlapping_training_test(repo)
        if usage:
            #todo must be implemented
            result.update(Data.__check_usage(repo))
        return result

class Repo:
    @staticmethod
    def __integrity(repo):
        ml_repo = repo.get_ml_repo_store()
        if hasattr(ml_repo, 'check_integrity'):
            return ml_repo.check_integrity()
        return {}

    @staticmethod
    def run(repo, integrity = True):
        if integrity:
            return Repo.__integrity(repo)

class Tests:
    @staticmethod
    def __check_test(repo: MLRepo, model, model_version, test_definition):
        # first create from definition the test for the given model to gt the test name
        test_definition.labels = None
        test_definition.models = {model: model_version}
        tests = test_definition.create(repo) #create tests to get the names to search for them
        results = {}
        for t in tests:
            test = repo.get(t.repo_info.name, version = None, modifier_versions={model: model_version},
                            throw_error_not_exist=False, throw_error_not_unique=False)
            if test == []:
                results[t.repo_info.name] = 'Test missing'
                continue             
            if isinstance(test, list): # search test 
                t = test[0]
                for k in range(1, len(test)):
                    if test[k].repo_info.commit_date > t.repo_info.commit_date:
                        t = test[k] 
                test = t
            result = test._check(repo)
            if not result is None:
                results[t.repo_info.name] = result
        return results

    @staticmethod
    def run(repo: MLRepo):
        test_definitions = repo.get_names(MLObjectType.TEST_DEFINITION)
        labels = repo.get_names(MLObjectType.LABEL)
        results = {}
        for l in labels:
            label = repo.get(l)
            for t in test_definitions:
                test_definition = repo.get(t)
                model = str(NamingConventions.CalibratedModel(label.name))
                result = Tests.__check_test(repo, model, label.version, test_definition)
                if len(result) > 0:
                    results[label.repo_info.name] = result
        models = repo.get_names(MLObjectType.CALIBRATED_MODEL)
        for model in models: #check latest models
            for t in test_definitions:
                test_definition = repo.get(t)
                result = Tests.__check_test(repo, model, RepoStore.LAST_VERSION, test_definition)
                if len(result) > 0:
                    results[model + ':latest'] = result
        return results

def run(repo: MLRepo, config : dict = None):
    
    if config is None:
        if 'check' not in repo._config.keys():
            config = get_initial_config(repo)
            repo._config['check'] = config
            repo._save_config()    
        config = repo._config['check']
    else:
        config = get_initial_config(repo)
        repo._config['check'] = config
        repo._save_config()

    result = []
    for k,v in config['model_checks']['models'].items():
        tmp = Model.run(repo, model_name=k, correct=v['correct'], model_version = v['model_version'], check_for_latest=v['check_for_latest'])
        if len(tmp) > 0:
            result.append(tmp)
    for k,v in config['model_checks']['labels'].items():
        tmp = Model.run(repo, correct=v['correct'], model_label = k, check_for_latest=v['check_for_latest'])
        if len(tmp) > 0:
            result.append(tmp)
    tmp = Data.run(repo, **config['data_checks'])        
    if len(tmp) > 0:
        result.append(tmp)
    tmp = Repo.run(repo, **config['repo_checks'])
    return result