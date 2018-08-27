"""
Machine learning repository
"""
import importlib
from enum import Enum
from copy import deepcopy
import logging
import repo.repo_objects as repo_objects
from repo.repo_objects import repo_object_init  # pylint: disable=E0401
LOGGER = logging.getLogger('repo')


class MLObjectType(Enum):
    """Enums describing all ml object types.
    """
    EVAL_DATA = 'eval_data'
    RAW_DATA = 'raw_data'
    TRAINING_DATA = 'training_data'
    TEST_DATA = 'test_data'
    TEST_RESULT = 'test_result'
    MODEL_PARAM = 'model_param'
    TRAINING_PARAM = 'training_param'
    TRAINING_FUNCTION = 'training_function'
    MODEL_EVAL_FUNCTION = 'model_eval_function'
    PREP_PARAM = 'prep_param'
    PREPROCESSOR = 'preprocessor'
    MODEL_INFO = 'model_info'
    LABEL = 'label'
    MODEL = 'model'
    COMMIT_INFO = 'commit_info'
    MAPPING = 'mapping'

    def _get_key(category):  # pylint: disable=E0213
        """Returns a standardized ky for the given category

        Arguments:
            category {MLObjectType or string (name or value of MLObjectType)} -- MLObjectType or string defining the enum

        Raises:
            Exception -- If no MLObjectType exists matchin the given category or if category is of wrong type

        Returns:
            string -- string which can be used e.g. in a dictionary
        """

        category_name = None
        if isinstance(category, MLObjectType):
            category_name = category.value
        if isinstance(category, str):
            for k in MLObjectType:
                if k.value == category or k.name == category:
                    category_name = k.value
                    break
        if category_name is None:
            raise Exception('No category ' + str(category) + ' exists.')
        return category_name


class ModelInfo:
    """ This class summarizes information of a special model version.

        This class summarizes information of a special model version,
        such as version numbers of all objects used such as trainin data, training parametere, etc..
    """
    @repo_object_init()
    def __init__(self):
        self.model_version = None
        self.training_data_version = {}
        self.test_version = {}
        self.test_result_version = {}
        self.model_param_version = None
        self.training_param_version = None
        self.prep_param_version = None
        self.model_training_function_version = None


class Mapping:
    """Provides a mapping from MLObjectType to all objects in the repo belonging to this type
    """
    @repo_object_init()
    def __init__(self, **kwargs):
        for key in MLObjectType:
            setattr(self, key.value, [])
        self.set_fields(kwargs)

    def set_fields(self, kwargs):
        """Set  fields from a dictionary

        Args:
            :param kwargs: dictionary
        """
        if not kwargs is None:
            for key in MLObjectType:
                if key.name in kwargs.keys():
                    setattr(self, key.value, kwargs[key.name])
                else:
                    if key.value in kwargs.keys():
                        setattr(self, key.value, kwargs[key.value])
                    else:
                        if key in kwargs.keys():
                            setattr(self, key.value, kwargs[key])

    def add(self, category, name):
        """Add a object to a category if it does not already exists

        Args:
            :param category: Either a string (MLObjectType.value or MLObjectType.name) or directly a MLObjectType enum
            :param name: name of object

            :return true, if mapping has changed, false otherwise
        """
        category_name = MLObjectType._get_key(category)
        mapping = getattr(self, category_name)
        # if name in mapping:
        #    raise Exception('Cannot add object: A ' + category_name + ' object with name ' + name + ' already exists in repo, please use update to modify it.')
        if category_name == MLObjectType.TRAINING_DATA.value and len(mapping) > 0 and mapping[0] != name:
            raise Exception('Only one set of training data allowed.')
        if not name in mapping:
            mapping.append(name)
            return True
        return False

    def __getitem__(self, category):
        """Get an item.

        Args:
            :param category: Either a string (MLObjectType.value or MLObjectType.name) or directly a MLObjectType enum

        Returns:
            Key of specified category.
        """
        category_name = MLObjectType._get_key(category)
        return getattr(self, category_name)

class EvalJob:
    """definition of a model evaluation job
    """
    @repo_object_init()
    def __init__(self, model, data, user, eval_function_version=-1, model_version=-1, data_version=-1):
        self.model = model
        self.data = data
        self.user = user
        self.eval_function_version = eval_function_version
        self.model_version = model_version
        self.data_version = data_version

    def run(self, repo, jobid):
        """Run the job with data from the given repo

        Arguments:
            repo {MLrepository} -- repository used to get and store the data
            jobid {string} -- id of job which will be run
        """
        model = repo.get(self.model, self.model_version)
        data = repo._get(self.data, self.data_version, full_object=True)
        eval_func = repo._get(model.eval_function, self.eval_function_version)
        tmp = importlib.import_module(eval_func.module_name)
        module_version = None
        if hasattr(tmp, '__version__'):
            module_version = tmp.__version__
        f = getattr(tmp, eval_func.function_name)
        y = f(model.model(), data.x_data)
        result = repo_objects.RawData(y, data.y_coord_names, repo_info={
                            repo_objects.RepoInfoKey.NAME.value: MLRepo.get_default_eval_name(model, data),
                            repo_objects.RepoInfoKey.CATEGORY.value: MLObjectType.EVAL_DATA.value,
                            repo_objects.RepoInfoKey.MODIFICATION_INFO.value: {'model_version': model.repo_info[repo_objects.RepoInfoKey.VERSION],
                                                                'data_version': data.repo_info[repo_objects.RepoInfoKey.VERSION],
                                                                'jobid': jobid}
                            }
                            )
        repo.add(result)

class MLRepo:
    """ Repository for doing machine learning

        The repository and his extensions provide a solid fundament to do machine learning science supporting features such as:
            - auditing/versioning of the data, models and tests
            - best practice standardized plotting for investigating model performance and model behaviour
            - automated quality checks

        The repository needs three different handlers/repositories 

    """
    def __init__(self, user, script_repo, numpy_repo, ml_repo, job_runner):
        """ Constructor of MLRepo

            :param script_repo: repository for the user's modules providing the customized model evaluation, raining, calibration and preprocessing methods
            :param numpy_repo: repository where the numpy data is stored in versions
            :param ml_repo: repository where the repo_objects are stored
            :param job_runner: the jobrunner to execute calibration, evaluations etc.
        """
        self._script_repo = script_repo
        self._numpy_repo = numpy_repo
        self._ml_repo = ml_repo
        self._user = user
        self._job_runner = job_runner
        # check if the ml mapping is already contained in the repo, otherwise add it
        try:
            self._mapping = self._ml_repo.get('repo_mapping', -1)
        except Exception:
            self._mapping = Mapping(  # pylint: disable=E1123
                repo_info={repo_objects.RepoInfoKey.NAME.value: 'repo_mapping', 
                repo_objects.RepoInfoKey.CATEGORY.value: MLObjectType.MAPPING.value})
           
    def _adjust_version(self, version, name):
        """Checks if version is negative and then adjust it according to the typical python 
        way for list where -1 is the last element of the list, -2 the second last etc.

        :param version: version number
        :param name: name of object for which version is adjusted
        :return adjusted version
        """
        if version >= 0:
            return version
        return self._ml_repo.get_latest_version(name) + 1 + version

    def _add(self, repo_object, message='', category = None):
            """ Add a repo_object to the repository.

                :param repo_object: repo_object to be added, will be modified so that it contains the version number
                :param message: commit message
                :param category: Category of repo_object which is used as fallback if th object does not define a category.

                Raises an exception if the category of the object is not defined in the object and if it is not defined with the category argument.
                It raises an exception if an object with this id does already exist.

                :return version number of object added and boolean if mapping has changed
            """        
            if repo_object.repo_info[repo_objects.RepoInfoKey.CATEGORY.value] is None:
                if category is None:
                    raise Exception('Category of repo_object not set and no fallback category defined.')
                else:
                    repo_object.repo_info[repo_objects.RepoInfoKey.CATEGORY.value] = category
            
            mapping_changed = self._mapping.add(repo_object.repo_info[repo_objects.RepoInfoKey.CATEGORY], repo_object.repo_info[repo_objects.RepoInfoKey.NAME])

            repo_object.repo_info[repo_objects.RepoInfoKey.COMMIT_MESSAGE.value] = message
            obj_dict = repo_objects.create_repo_obj_dict(repo_object)
            version = self._ml_repo.add(obj_dict)
            repo_object.repo_info[repo_objects.RepoInfoKey.VERSION.value] = version
            if len(repo_object.repo_info[repo_objects.RepoInfoKey.BIG_OBJECTS]) > 0:
                np_dict = repo_object.numpy_to_dict()
                self._numpy_repo.add(repo_object.repo_info[repo_objects.RepoInfoKey.NAME],
                                    repo_object.repo_info[repo_objects.RepoInfoKey.VERSION],
                                    np_dict)
            return version, mapping_changed

    def add(self, repo_object, message='', category = None):
        """ Add a repo_object or list of repo objects to the repository.

            :param repo_object: repo_object or list of repo_objects to be added, will be modified so that it contains the version number
            :param message: commit message
            :param category: Category of repo_object which is used as fallback if the object does not define a category.

            Raises an exception if the category of the object is not defined in the object and if it is not defined with the category argument.
            It raises an exception if an object with this id does already exist.

            :return version number of object added or dictionary of names and versions of objects added
        """
        result = {}
        repo_list = repo_object
        mapping_changed = False
        if not isinstance(repo_list,list):
            repo_list = [repo_object]
        if isinstance(repo_list, list):
            for obj in repo_list:
                result[obj.repo_info[repo_objects.RepoInfoKey.NAME]], mapping_changed_tmp = self._add(obj, message, category)
                mapping_changed = mapping_changed or mapping_changed_tmp
        if mapping_changed:
            mapping_version, dummy = self._add(self._mapping)
            result['repo_mapping'] = mapping_version
            
        commit_message = repo_objects.CommitInfo(message, self._user, result, repo_info = {repo_objects.RepoInfoKey.CATEGORY.value: MLObjectType.COMMIT_INFO.value,
                repo_objects.RepoInfoKey.NAME.value: 'CommitInfo'} )
        self._add(commit_message)
        if len(result) == 1 or (mapping_changed and len(result) == 2):
            return result[repo_object.repo_info[repo_objects.RepoInfoKey.NAME]]
        return result

    def get_training_data(self, version=-1, full_object=True):
        """Returns training data 

        Keyword Arguments:
            version {integer} -- version of data object
            full_object {bool} -- if True, the complete data is returned including numpy data (default: {True})
        """
        if self._mapping[MLObjectType.TRAINING_DATA] is None:
            raise Exception("No training_data in repository.")
        return self.get_data(self._mapping[MLObjectType.TRAINING_DATA][0], version, full_object)

    def get_data(self, name, version=-1, full_object=True):
        """Return a data object

        Arguments:
            name {string} -- data object

        Keyword Arguments:
            version {integer} -- version of data object to be returned, default is latest object
            full_object {bool} -- if true, the full object including numpy objects is returned (default: {True})
        """
        result = self._get(name, version, full_object)
        return result

    def add_eval_function(self, module_name, function_name, repo_name = None):
        """Add the function to evaluate the model

        Arguments:
            module_name {string} -- module where function is located
            function_name {string} -- function name
            repo_name {tring} -- identifier of the repo object used to store the information (default: None), if None, the name is set to module_name.function_name
        """
        name = repo_name
        if name is None:
            name = module_name + "." + function_name
        func = repo_objects.Function(module_name, function_name, repo_info={
                                     repo_objects.RepoInfoKey.NAME.value: name,
                                     repo_objects.RepoInfoKey.CATEGORY.value: MLObjectType.MODEL_EVAL_FUNCTION.value})
        self.add(func, 'add model evaluation function ' + name)
    
    def add_training_function(self, module_name, function_name, repo_name = None):
        """Add function to train a model

        Arguments:
            module_name {string} -- module where function is located
            function_name {string} -- function name
            repo_name {tring} -- identifier of the repo object used to store the information (default: None), if None, the name is set to module_name.function_name
        """
        name = repo_name
        if name is None:
            name = module_name + "." + function_name
        func = repo_objects.Function(module_name, function_name, repo_info={
                                     repo_objects.RepoInfoKey.NAME.value: name,
                                     repo_objects.RepoInfoKey.CATEGORY.value: MLObjectType.TRAINING_FUNCTION.value})
        self.add(func, 'add model training function ' + name)

    def add_model(self, model_name, model_eval = None, model_training = None, model_param = None, training_param = None):
        """Add a new model to the repo
        
        Arguments:
            model_name {string} -- identifier of the model
        
        Keyword Arguments:
            model_eval {string} -- identifier of the evaluation function in the repo to evaluate the model, 
                                    if None and there is only one evaluation function in the repo, this function will be used
            model_training {string} -- identifier of the training function in the repo to train the model, 
                                    if None and there is only one evaluation function in the repo, this function will be used
            model_param {string} -- identifier of the model parameter in the repo (default: {None}), if None and there is exactly one ModelParameter in teh repo, this will be used,
                                    otherwise it is assumed that no model_params are needed
            training_param {string} -- identifier of the training parameter (default: {None}), if None and there is only one training_parameter object in the repo, 
                                        this will be used
        """
        model = repo_objects.Model(repo_info={repo_objects.RepoInfoKey.CATEGORY.value: MLObjectType.MODEL, 
                                               repo_objects.RepoInfoKey.NAME.value: model_name })
        model.eval_function = model_eval
        if model.eval_function is None:
            mapping = self._mapping[MLObjectType.MODEL_EVAL_FUNCTION]
            if len(mapping) == 1:
                model.eval_function = mapping[0]
            else:
                raise Exception('More than one or no eval function in repo, therefore you must explicitely specify an eval function.')
       
        model.training_function = model_training
        if  model.training_function is None:
            mapping = self._mapping[MLObjectType.TRAINING_FUNCTION]
            if len(mapping) == 1:
                model.training_function = mapping[0]
            else:
                raise Exception('More than one or no training function in repo, therefore you must explicitely specify a training function.')
        
        model.training_param = training_param
        if model.training_param is None:
            mapping = self._mapping[MLObjectType.TRAINING_PARAM]
            if len(mapping) == 1:
                model.training_param = mapping[0]
            else:
                raise Exception('More than one or no training parameter in repo, therefore you must explicitely specify a training parameter.')

        model.model_param = model_param
        if model.model_param is None:
            mapping = self._mapping[MLObjectType.MODEL_PARAM]
            if len(mapping) == 1:
                model.model_param = mapping[0]
            else:
                if len(mapping) > 1
                raise Exception('More than one model parameter in repo, therefore you must explicitely specify a model parameter.')
        self.add(model)
        
    def _get(self, name, version=-1, full_object=False):
        """ Get repo objects. It throws an exception, if an object with the name does not exist.

            :param name: Object name
            :param version: object version, default is latest (-1)
            :param full_object: flag to determine whether the numpy objects are loaded (True->load)
        """
        repo_dict = self._ml_repo.get(name, version)
        result = repo_objects.create_repo_obj(repo_dict)
        if isinstance(result, repo_objects.DataSet):
            raw_data = self._get(
                result.raw_data, result.raw_data_version, full_object)
            setattr(result, 'x_data', raw_data.x_data)
            setattr(result, 'x_coord_names', raw_data.x_coord_names)
            setattr(result, 'y_data', raw_data.y_data)
            setattr(result, 'y_coord_names', raw_data.y_coord_names)

        numpy_dict = {}
        if len(result.repo_info[repo_objects.RepoInfoKey.BIG_OBJECTS]) > 0 and full_object:
            numpy_dict = self._numpy_repo.get(
                result.repo_info[repo_objects.RepoInfoKey.NAME], result.repo_info[repo_objects.RepoInfoKey.VERSION])
        for x in result.repo_info[repo_objects.RepoInfoKey.BIG_OBJECTS]:
            if not x in numpy_dict:
                numpy_dict[x] = None
        result.numpy_from_dict(numpy_dict)
        return result

    def get_default_eval_name( model_name, data_name ):
        """Return name of the object containing evaluation results
        
        Arguments:
            data_name {[type]} -- [description]
        
        Returns:
            string -- name of valuation results
        """
        return  model_name + '/eval/' + data_name

    def get_names(self, ml_obj_type):
        """ Get the list of names of all repo_objects from a given repo_object_type in the repository.

            :param ml_obj_type: MLObjectType specifying the types of objects names are returned for.

            :return list of object names for the gigven category.
        """
        return self._ml_repo.get_names(ml_obj_type)

    def get_history(self, name, repo_info_fields=[], obj_member_fields=[], version_start=0, version_end=-1, label=None):
        """ Return a list of histories of object member variables.

        :param repo_info_field: List of fields from repo_info which will be returned in the dictionary. 
                                If List contains flag 'ALL', all fields will be returned.
        :param obj_member_fields: List of member atributes from repo_object which will be returned in the dictionary. 
                                If List contains flag 'ALL', all attributes will be returned.
        """
        version_list = []
        if not label is None:
            raise Exception('Not yet implemented.')
        version_list = range(self._adjust_version(
            version_start, name), self._adjust_version(version_end, name)+1)
        fields = [x for x in repo_info_fields]
        if len(repo_info_fields) == 0:
            fields = [x.value for x in repo_objects.RepoInfoKey]
        fields.extend(obj_member_fields)
        return self._ml_repo.get_history(name, fields, version_list)

    def get_commits(self,  version_start=0, version_end=-1):
        version_list = []
        version_list = range(self._adjust_version(
            version_start, 'CommitInfo'), self._adjust_version(version_end, 'CommitInfo')+1)
        result = []
        for i in version_list:
            result.append(self._get('CommitInfo', i))
        return result

    def run_training(self, message='', job_runner=None):
        """ Run the training algorithm. 

        :param message: commit message
        :param job_runner: job runner executing the raining script. Default is single threaded local jobrunner.

        :return ticket number and new model version
        """
        pass

    def run_evaluation(self, model, message=None, model_version=-1, datasets={}):
        """ Evaluate the model on all datasets. 

            :param model: name of model to evaluate, if None and only one model exists
            :message: message inserted into commit (default None), if Noe, an autmated message is created
            :param model_version: Version of model to be evaluated.
            :datasets: Dictionary of datasets (names and version numbers) on which the model is evaluated. 
                Default is all datasets from testdata on latest version.
            Raises:
                Exception if model_name is None and more then one model exists
        """
        datasets_ = deepcopy(datasets)
        if len(datasets_) == 0:
            names = self.get_names(MLObjectType.TEST_DATA)
            for n in names:
                v = self._ml_repo.get_latest_version(n)
                datasets_[n] = v
        m = model
        if m is None:
            models = self.get_names(MLObjectType.MODEL)
            if len(models) == 1:
                m = models[0]
            else:
                raise Exception('No model is specified but repository contains more than one model.')
        for n, v in datasets:
            eval_job = EvalJob(m, n, self._user, model_version=model_version, data_version=v)
            self._job_runner.add(eval_job)

    def run_tests(self, message='', model_version=-1, tests={}, job_runner=None):
        """ Run tests for a specific model version.

            :param message: Commit message for this operation.
            :param model_version: Version or label of model for which the tests are executed.
            :param tests: Dictionary of tests (names and version numbers) run. Default is all tests on latest version. 
            :param job_runner: job runner executing the tests. Default is single threaded local jobrunner.

            :return ticket number of job
        """
        pass

    def get_model_info(self, model_version=-1):
        """ Return model info for the given model version.

        :param model_version: Version or label of model, default is latest.
        """
        pass

    def set_label(self, model_version, message='', force=True):
        """ Label a certain model version.

            This method labels a certain model version. If force == False it checks if the label 
            already exists an raises in this case an exception.

            :param message: Commit message
            :param model_version: model version for which the label is set.
            :param force: lag determining if ann exception is raise if the label is already used (force->False, exception is raised)
        """
        pass
