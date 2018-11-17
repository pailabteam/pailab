import logging
from pailab.repo import MLObjectType, MLRepo  # pylint: disable=E0611, E0401
from pailab.repo_objects import RepoInfoKey  # pylint: disable=E0401
from pailab.repo_store import RepoStore  # pylint: disable=E0401
logger = logging.getLogger(__name__)


class _ObjNames:
    def __init__(self, names=None, sep='/'):
        if isinstance(names, list):
            for x in names:
                self.__add(x)
        else:
            if isinstance(names, dict):
                for k, v in names.items():
                    self.__add(v, k.split(sep))

    def __add(self, name, name_separated=None, sep='/'):
        if name_separated is None:
            name_separated = name.split(sep)
        if len(name_separated) == 1:
            leaf = _ObjNames()
            setattr(leaf, '__name', name)
            setattr(self, name_separated[0], leaf)
        else:
            if hasattr(self, name_separated[0]):
                d = getattr(self, name_separated[0])
                d.__add(name, name_separated[1:])
            else:
                sub_obj = _ObjNames()
                sub_obj.__add(name, name_separated[1:])
                setattr(self, name_separated[0], sub_obj)

    def __call__(self, containing_str=None):
        if len(self.__dict__) == 1:
            for k, v in self.__dict__.items():
                if isinstance(v, str):
                    return v
                else:
                    return v()
        else:
            result = []
            for k, v in self.__dict__.items():
                if isinstance(v, str):
                    result.append(v)
                else:
                    d = v()
                    if isinstance(d, str):
                        result.append(d)
                    else:
                        result.extend(d)
            if containing_str is not None:
                return [x for x in result if containing_str in x]
            return result


def path_to_names(repo, include_object_types=[MLObjectType.MEASURE, MLObjectType.CALIBRATED_MODEL,
                                              MLObjectType.EVAL_DATA, MLObjectType.JOB, MLObjectType.LABEL,
                                              MLObjectType.MODEL_PARAM, MLObjectType.TEST_DATA, MLObjectType.TRAINING_DATA,
                                              MLObjectType.TRAINING_PARAM]):
    """Returns an object which allows easy access to the names of all repo objects

    Args:
        repo ([type]): [description]
    """
    obj_names = []
    for k in include_object_types:
        names = repo.get_names(k.value)
        obj_names.extend(names)
    repo_obj = _ObjNames(obj_names)
    return repo_obj


def check_model(repo, model_name, correct=False, model_version=RepoStore.LAST_VERSION):
    """Check if the model is calibrated and evaluated on the latest versions

    Arguments:
        repo {MLRepo} -- the ml repo
        model_name {str} -- model name

    Keyword Arguments:
        correct {bool} -- determine whether training, evaluations, measures and tests will be qutomatically triggered if check fails to correct (default: {False})
        model_version {repo_version} --  (default: {RepoStore.LAST_VERSION})

    Returns:
        dict -- dictionary containing the modifier versions and the latest version of the objects, if dictionary is empty noc model inconsistencies could be found
    """

    result = {}

    m = repo.get(MLRepo.get_calibrated_model_name(
        model_name), version=model_version)
    # first check if all versions of the models modifiers are still the latest version
    repo_store = repo.get_ml_repo_store()
    for k, v in m.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
        latest_version = repo_store.get_latest_version(k)
        if not v == latest_version:
            result[k] = {'modifier version': v,
                         'latest version': latest_version}
    if len(result) > 0 and correct == True:
        job_id = repo.run_training(model_name)
        job_ids = repo.run_evaluation(
            model_name, message='running evaluations triggered by new training of ' + model_name, predecessors=[job_id])
        job_ids = repo.run_measures(
            model_name, message='running measures triggered by new training of ' + model_name, predecessors=job_ids)
        job_ids = repo.run_tests(
            model_name, message='running tests triggered by new training of ' + model_name, predecessors=job_ids)
    return result


def _check_evaluation(repo, model, data):
    pass


def check_evaluation(repo):
    """Checks if all defined evaluations in the repo are on latest model and datasets


    Arguments:
        repo {MLRepo} -- the repository the checks are applied to
    """
    logger.info('Start checking evaluations.')
