"""Machine Learning Repository

This module contains pailab's machine learning repository, i.e. the repository 
Machine learning repository
"""
import importlib
import abc

from numpy import linalg
from numpy import inf
from enum import Enum
from copy import deepcopy
import logging
import pailab.repo_objects as repo_objects
from pailab.repo_objects import RepoInfoKey
from pailab.repo_objects import repo_object_init  # pylint: disable=E0401
import pailab.repo_store as repo_store
logger = logging.getLogger(__name__)


class MLObjectType(Enum):
    """Enums describing all ml object types.
    """
    EVAL_DATA = 'EVAL_DATA'
    RAW_DATA = 'RAW_DATA'
    TRAINING_DATA = 'TRAINING_DATA'
    TEST_DATA = 'TEST_DATA'
    TEST_RESULT = 'TEST_RESULT'
    MODEL_PARAM = 'MODEL_PARAM'
    TRAINING_PARAM = 'TRAINING_PARAM'
    TRAINING_FUNCTION = 'TRAINING_FUNCTION'
    MODEL_EVAL_FUNCTION = 'MODEL_EVAL_FUNCTION'
    PREP_PARAM = 'PREP_PARAM'
    PREPROCESSOR = 'PREPROCESSOR'
    MODEL_INFO = 'MODEL_INFO'
    LABEL = 'LABEL'
    MODEL = 'MODEL'
    CALIBRATED_MODEL = 'CALIBRATED_MODEL'
    COMMIT_INFO = 'COMMIT_INFO'
    MAPPING = 'MAPPING'
    MEASURE = 'MEASURE'
    MEASURE_CONFIGURATION = 'MEASURE_CONFIGURATION'
    JOB = 'JOB'

    @staticmethod
    def _get_key(category): 
        """Returns a standardized key for the given category

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


class Mapping:
    """Provides a mapping from MLObjectType to all objects in the repo belonging to this type
    """
    @repo_object_init()
    def __init__(self, **kwargs):
        logging.debug('Initializing map with kwargs: ' + str(kwargs))
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

    def __str__(self):
        result = ''
        for k in MLObjectType:
            result += k.value +': '
            if hasattr(self, k.value):
                result +=str(getattr(self, k.value)) + ','
            else:
                result += '[],'
        return result

def _add_modification_info(repo_obj, *args):
    """Add modificaion info to a repo object from a list if repo objects which were used to create it.

    Arguments:
        repo_obj {repoobject} -- the object the modification info is added to
    """
    result = {}
    for v in args:
        if v is not None:
            result[v.repo_info[RepoInfoKey.NAME]
                ] = v.repo_info[RepoInfoKey.VERSION]
            for k, w in v.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
                # check if there is a bad inconsistency: if an object modifying the current object has already been used with a different version
                # number, this must be annotated
                if k in result.keys() and result[k] != w:
                    result['inconsistency'] = 'object ' + k + \
                        ' has already been used with a different version number'
                result[k] = w
    repo_obj.repo_info[RepoInfoKey.MODIFICATION_INFO] = result

# region Jobs

class Job(abc.ABC):
    """Abstract class defining the interfaces needed for a job to be used in the JobRunner

    """

    def set_predecessor_jobs(self, predecessors):
        """Add list of jobids which must have been run successfull before the job can be started

        Arguments:
            predecessors {list o jobids} -- predecessors
        """
        self._predecessors = predecessors

    def get_predecessor_jobs(self):
        """Return list of jobids which must have been run sucessfully before the job will be executed
        """
        if hasattr(self, '_predecessors'):
            return self._predecessors
        return []

    @abc.abstractmethod
    def run(self, ml_repo, jobid):
        pass

 
class EvalJob(Job):
    """definition of a model evaluation job
    """
    @repo_object_init()
    def __init__(self, model, data, user, eval_function_version=repo_store.RepoStore.LAST_VERSION,
                model_version=repo_store.RepoStore.LAST_VERSION, data_version=repo_store.RepoStore.LAST_VERSION):
        self.model = model
        self.data = data
        self.user = user
        self.eval_function_version = eval_function_version
        self.model_version = model_version
        self.data_version = data_version
        # list of jobids which must have been run before this job should be excuted
        self.predecessors = []

    def run(self, repo, jobid):
        """Run the job with data from the given repo

        Arguments:
            repo {MLrepository} -- repository used to get and store the data
            jobid {string} -- id of job which will be run
        """
        logging.info('Start evaluation job ' + str(jobid) +' on model ' + self.model + ' ' + str(self.model_version) + ' ')
        model = repo._get(self.model, self.model_version)
        model_definition_name = self.model.split('/')[0]
        model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO.value][model_definition_name]
        model_definition = repo._get(model_definition_name, model_def_version)
        
        data = repo._get(self.data, self.data_version, full_object=True)
        eval_func = repo._get(model_definition.eval_function, self.eval_function_version)
        tmp = importlib.import_module(eval_func.module_name)
        module_version = None
        if hasattr(tmp, '__version__'):
            module_version = tmp.__version__
        f = getattr(tmp, eval_func.function_name)
        y = f(model, data.x_data)
        result = repo_objects.RawData(y, data.y_coord_names, repo_info={
                            RepoInfoKey.NAME: MLRepo.get_eval_name(model_definition, data),
                            RepoInfoKey.CATEGORY: MLObjectType.EVAL_DATA.value
                            }
                            )
        _add_modification_info(result, model, data)
        repo.add(result, 'evaluate data ' +
                 self.data + ' with model ' + self.model)
        logging.info('Finished evaluation job ' + str(jobid))
        

class TrainingJob(Job):
    """definition of a model training job
    """
    @repo_object_init()
    def __init__(self, model, user, training_function_version=repo_store.RepoStore.LAST_VERSION, model_version=repo_store.RepoStore.LAST_VERSION,
                training_data_version=repo_store.RepoStore.LAST_VERSION, training_param_version=repo_store.RepoStore.LAST_VERSION,
                 model_param_version=repo_store.RepoStore.LAST_VERSION):
        self.model = model
        self.user = user
        self.training_function_version = training_function_version
        self.model_version = model_version
        self.training_param_version = training_param_version
        self.model_param_version = model_param_version
        self.training_data_version = training_data_version
        # list of jobids which must have been run before this job should be excuted
        self.predecessors = []

    def run(self, repo, jobid):
        """Run the job with data from the given repo

        Arguments:
            repo {MLrepository} -- repository used to get and store the data
            jobid {string} -- id of job which will be run
        """
        model = repo._get(self.model, self.model_version)
        train_data = repo.get_training_data(
            self.training_data_version, full_object=True)
        train_func = repo._get(model.training_function,
                               self.training_function_version)
        train_param = None
        if not model.training_param == '':
            train_param = repo._get(model.training_param, self.training_param_version)
        model_param = None
        if not model.model_param is None:
            model_param = repo._get(
                model.model_param, self.model_param_version)
        tmp = importlib.import_module(train_func.module_name)
        module_version = None
        if hasattr(tmp, '__version__'):
            module_version = tmp.__version__
        f = getattr(tmp, train_func.function_name)
        m = None
        if model_param is None:
            m = f(train_param, train_data.x_data, train_data.y_data)
        else:
            if train_param is None:
                m = f(model_param, train_data.x_data, train_data.y_data)
            else:
               m = f(model_param, train_param, train_data.x_data, train_data.y_data)
        m.repo_info[RepoInfoKey.NAME] = self.model + '/model'
        m.repo_info[RepoInfoKey.CATEGORY] = MLObjectType.CALIBRATED_MODEL.value
        _add_modification_info(m, model_param, train_param, train_data, model)
        repo.add(m, 'training of model ' + self.model)


class MeasureJob(Job):
    @repo_object_init()
    def __init__(self, result_name, measure_type, coordinates, data_name, model_name, data_version=repo_store.RepoStore.LAST_VERSION,
                model_version=repo_store.RepoStore.LAST_VERSION):
        """Constructor

        Arguments:
            measure_type {string} -- string describing the measure type
            coordinates {string or list} -- either a string describing the coordinate (or simply all coordinates if the string equals MeasureConfiguration._ALL_COORDINATES) or a list of strings
            data_name {string} -- name of data for which the measure shall be calculated
            model_name {string} -- name of model for which the measure shall be calculated

        Keyword Arguments:
            data_version {versionnumber} -- version of data to be used (default: {repo_store.RepoStore.LAST_VERSION})
            model_version {versionnumber} -- version of model to be used (default: {repo_store.RepoStore.LAST_VERSION})
        """

        self.measure_type = measure_type
        self.coordinates = coordinates
        self.model_name = model_name
        self.model_version = model_version
        self.data_name = data_name
        self.data_version = data_version

    def run(self, repo, jobid):
        target = repo._get(self.data_name, version=self.data_version, full_object = True)
        m_name = self.model_name.split('/')[0] #if given model name is a name of calibated model, split to find the evaluation
        eval_data_name = NamingConventions.EvalData(data = self.data_name, model = m_name)
        eval_data = repo._get(str(eval_data_name), modifier_versions={self.model_name: self.model_version, self.data_name: self.data_version}, full_object = True )
        logger.info('run MeasureJob on data ' + self.data_name + ':' + str(self.data_version) 
                        + ', ' + str(eval_data_name) + ':' + str(eval_data.repo_info[RepoInfoKey.VERSION])
                )
        #if len(self.coordinates) == 0 or repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
        #    return target.y_data, eval_data.x_data
        columns = []
        if not repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
            for x in self.coordinates:
                columns.append(target.y_coord_names.index(x))
        if len(columns) == 0:
            v = self._compute(target.y_data, eval_data.x_data)
        else:
            v = self._compute(target.y_data[:,columns], eval_data.x_data[:columns])
        result_name = str(NamingConventions.Measure(eval_data_name, measure_type = self.measure_type))
        if not repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
            for x in self.coordinates:
                result_name = result_name + ':' + x
        result = repo_objects.Measure( v, 
                                repo_info = {RepoInfoKey.NAME : result_name, RepoInfoKey.CATEGORY: MLObjectType.MEASURE.value})
        _add_modification_info(result, eval_data, target)
        repo.add(result, 'computing  measure ' + self.measure_type + ' on data ' + self.data_name)

    def _compute(self, target_data, eval_data):
        if self.measure_type == repo_objects.MeasureConfiguration.MAX:
            return self._compute_max(target_data, eval_data)
        if self.measure_type == repo_objects.MeasureConfiguration.R2:
           return self._compute_r2(target_data, eval_data)
        else:
            raise NotImplementedError
        
    def _compute_max(self, target_data, eval_data):
        logger.debug('computing maximum error')
        return linalg.norm(target_data-eval_data, inf)

    def _compute_r2(self, target_data, eval_data):
        from sklearn.metrics import r2_score
        return r2_score(target_data, eval_data)        
# endregion
class DataSet:
    """Class used to access training or test data.

    This class refers to some RawData object and a start- and endindex 

    """
    @repo_object_init()
    def __init__(self, raw_data, start_index=0, 
        end_index=None, raw_data_version=repo_store.RepoStore.LAST_VERSION):
        """Constructor

        Arguments:
            :argument raw_data: {string} -- id of raw_data the dataset refers to
            :argument start_index: {integer} -- index of first entry of the raw data used in the dataset
            :argument end_index: {integer} -- end_index of last entry of the raw data used in the dataset (if None, all including last element are used)
            :argument raw_data_version: {integer} -- version of RawData object the DataSet refers to (default is latest)
        """
        self.raw_data = raw_data
        self.start_index = start_index
        self.end_index = end_index
        self.raw_data_version = raw_data_version
        
        if self.end_index is not None:
            if self.start_index > 0 and self.end_index > 0 and self.start_index >= self.end_index:
                raise Exception('Startindex must be less then endindex.')

    def set_data(self, raw_data):
        """Set the data from the given raw_data.
        
        Arguments:
            raw_data {RawData} -- the raw data used to set the data from
        
        Raises:
            Exception -- if end_index id less than start_index
        
        """

        if raw_data.x_data is not None:
            setattr(self, 'x_data',raw_data.x_data) # todo more efficient implementation over numpy_repo to avoid loading all and then cutting off
        if raw_data.y_data is not None:
            setattr(self, 'y_data', raw_data.y_data)
        setattr(self, 'x_coord_names', raw_data.x_coord_names)
        setattr(self, 'y_coord_names', raw_data.y_coord_names)
        setattr(self, 'n_data', raw_data.n_data)
    
    def __str__(self):
        return str(self.to_dict())

class Name:
    def __init__(self, name_order, tag):
        self.name_order = name_order.split('/')
        self.tag = tag
        self.model = None

        
    def __str__(self):
        result = self.values[self.name_order[0]]
        for i in range(1,len(self.name_order)):
            result = result + '/' + self.values[self.name_order[i]]
        return result

    def _set_from_string(self, v):
        tmp = v.split('/')
        if len(tmp) > len(self.name_order):
            raise Exception('Incorrect name depth: ' + str(len(tmp)) + ' != ' + str(len(self.name_order))) 
        for i in range(len(tmp)):
            if  self.name_order[i] != '*':
                self.values[self.name_order[i]] = tmp[i]
        
    def _set_from_dictionary(self, values):
        for k,v in values.items():
            if k in self.name_order and k != '*':
                self.values[k] = v

    def _set_from(self, v):
        if isinstance(v, Name):
            self._set_from_dictionary(v.values)
        if isinstance(v,list):
            for x in v:
                self._set_from(x)
        if isinstance(v, str):
            self._set_from_string(v)
        if isinstance(v, dict):
            self._set_from_dictionary(v)
        
    def __call__(self, name=None, **args):
        self.values = {'*': self.tag}
        if args is not None:
            self._set_from([name, args])
        else:
            self._set_from(name)
        return self
    
class NamingConventions:
    
    evaluation_tag = 'eval'
    measure_tag = 'measure'

    def _get_object_name(name):
        if not isinstance(name, str):
            return eval_data.repo_info[RepoInfoKey.NAME]
        return name

    def get_measure_result_name(data_name, measure_type, model = None):
        d_name = RepoNamingConventions._get_object_name(data_name)
        model_name = RepoNamingConventions._get_object_name(model)
        tmp = d_name.split('/')
        #region validate
        if len(tmp) < 3 and model_name is None:
            raise Exception('Please specify a model name.')
        if len(tmp) == 3 and not model_name is None:
            if tmp[0] != model_name:
                raise Exception('Model name does not equal model which was used to compute the veal results')
        #endregion
        if model_name is None:
            model_name = tmp[0]
        return model_name + '/' + RepoNamingConventions.measure_tag + '/' + tmp[-1] + '/' + measure_type

    def get_model_from_name(name):
        return name.split('/')[0]

    def get_model_param_name(model_name):
        return NamingConventions.get_model_from_name(model_name) +'/model_param'

    def get_calibrated_model_name(model):
        return model.split('/')[0]+'/model'

    def get_eval_name(model_name, data_name):
        return model_name + '/eval/' + data_name

    def get_eval_name_from_measure_name(measure_name):
        model = NamingConventions.get_model_from_name(measure_name)
        data_name = measure_name.split('/')[2] 
        return NamingConventions.get_eval_name(model, data_name)
    
    Measure = Name('model/*/data/measure_type', 'measure')
    EvalData = Name('model/*/data', 'eval')
    Data = Name('data', '')
    CalibratedModel = Name('model/*', 'model')
    Model = Name('model', '')
    ModelParam = Name('model/*', 'model_param')
    
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
        logging.info('Get mapping.')
        repo_dict = []
        try:
            repo_dict = self._ml_repo.get('repo_mapping', versions=repo_store.RepoStore.LAST_VERSION)
        except:
            logging.info('No mapping found, creating new mapping.')
        if len(repo_dict) > 1:
            raise Exception('More than on mapping found.')
        if len(repo_dict) == 1:
           self._mapping = repo_objects.create_repo_obj(repo_dict[0])
        else:
            self._mapping = Mapping(  # pylint: disable=E1123
                repo_info={RepoInfoKey.NAME: 'repo_mapping', 
                RepoInfoKey.CATEGORY: MLObjectType.MAPPING.value})
           
    def _add(self, repo_object, message='', category = None):
            """ Add a repo_object to the repository.

                :param repo_object: repo_object to be added, will be modified so that it contains the version number
                :param message: commit message
                :param category: Category of repo_object which is used as fallback if th object does not define a category.

                Raises an exception if the category of the object is not defined in the object and if it is not defined with the category argument.
                It raises an exception if an object with this id does already exist.

                :return version number of object added and boolean if mapping has changed
            """        
            if repo_object.repo_info[RepoInfoKey.CATEGORY] is None:
                if category is None:
                    raise Exception('Category of repo_object not set and no fallback category defined.')
                else:
                    repo_object.repo_info[RepoInfoKey.CATEGORY] = category
            
            mapping_changed = self._mapping.add(repo_object.repo_info[RepoInfoKey.CATEGORY], repo_object.repo_info[RepoInfoKey.NAME])

            repo_object.repo_info[RepoInfoKey.COMMIT_MESSAGE.value] = message
            obj_dict = repo_objects.create_repo_obj_dict(repo_object)
            version = self._ml_repo.add(obj_dict)
            repo_object.repo_info[RepoInfoKey.VERSION.value] = version
            if len(repo_object.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0:
                np_dict = repo_object.numpy_to_dict()
                self._numpy_repo.add(repo_object.repo_info[RepoInfoKey.NAME],
                                    repo_object.repo_info[RepoInfoKey.VERSION],
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
                result[obj.repo_info[RepoInfoKey.NAME]], mapping_changed_tmp = self._add(obj, message, category)
                mapping_changed = mapping_changed or mapping_changed_tmp
        if mapping_changed:
            mapping_version, dummy = self._add(self._mapping)
            result['repo_mapping'] = mapping_version
            
        commit_message = repo_objects.CommitInfo(message, self._user, result, repo_info = {RepoInfoKey.CATEGORY: MLObjectType.COMMIT_INFO.value,
                RepoInfoKey.NAME: 'CommitInfo'} )
        self._add(commit_message)
        if not isinstance(repo_object, list):
            if len(result) == 1 or (mapping_changed and len(result) == 2):
                return result[repo_object.repo_info[RepoInfoKey.NAME]]
        return result

    def get_training_data(self, version=repo_store.RepoStore.LAST_VERSION, full_object=True):
        """Returns training data 

        Keyword Arguments:
            version {integer} -- version of data object
            full_object {bool} -- if True, the complete data is returned including numpy data (default: {True})
        """
        if self._mapping[MLObjectType.TRAINING_DATA] is None:
            raise Exception("No training_data in repository.")
        return self._get(self._mapping[MLObjectType.TRAINING_DATA][0], version, full_object)

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
                                     RepoInfoKey.NAME: name,
                                     RepoInfoKey.CATEGORY: MLObjectType.MODEL_EVAL_FUNCTION.value})
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
                                     RepoInfoKey.NAME: name,
                                     RepoInfoKey.CATEGORY: MLObjectType.TRAINING_FUNCTION.value})
        self.add(func, 'add model training function ' + name)

    def add_data(self, data_name, data, input_variables = None, target_variables = None):
        """Add raw data to the repository

        Arguments:
            data_name {name of data} -- the name of the data added
            data {pandas datatable} -- the data as pndas datatable
        
        Keyword Arguments:
            input_variables {list of strings} -- list of column names defining the input variables for the machine learning (default: {None}). If None, all variables are used as input
            target_variables {list of strings} -- list of column names defining the target variables for the machine learning (default: {None}). If None, no target data is added from the table.
        """
        if target_variables is not None:
            raw_data = repo_objects.RawData(data.as_matrix(columns=input_variables), input_variables, data.as_matrix(columns=target_variables), 
                target_variables, repo_info = {RepoInfoKey.NAME: data_name})
        else:
            raw_data = repo_objects.RawData(data.as_matrix(), list(data), repo_info = {RepoInfoKey.NAME: data_name})
        self.add(raw_data, 'data ' + data_name + ' added to repository' , category = MLObjectType.RAW_DATA)

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
                                        this will be used. If an empty string is given as training parameter, we assume that the algorithm does not need a training pram.
        """
        model = repo_objects.Model(repo_info={RepoInfoKey.CATEGORY: MLObjectType.MODEL.value, 
                                               RepoInfoKey.NAME: model_name })
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
                if len(mapping) > 1:
                    raise Exception('More than one model parameter in repo, therefore you must explicitely specify a model parameter.')
        self.add(model, 'add model ' + model_name)

    def add_measure(self, measure, coordinates = None):
        """Add a measure to the repository
        
        If the measure already exists, it returns the message
        Arguments:
            measure {str} -- string defining the measure, i.e MAX,...
        
        Keyword Arguments:
            coordinates {list of str} -- list ofstrings defining the coordinates (by name) used for the measure (default: {None}), if None, all coordinates will be used
        """
        #if a measure configuration already exists, use this, otherwise creat a new one
        measure_config = None
        tmp = self.get_names(MLObjectType.MEASURE_CONFIGURATION)
        if len(tmp) > 0:
            measure_config = self._get(tmp[0])
        else:
            measure_config = repo_objects.MeasureConfiguration([], repo_info={RepoInfoKey.NAME: 'measure_config', 
                    RepoInfoKey.CATEGORY: MLObjectType.MEASURE_CONFIGURATION.value})
        measure_config.add_measure(measure, coordinates)
        coord = 'all'
        if not coordinates is None:
            coord = str(coordinates)
        self.add(measure_config, message='added measure ' + measure +' for coordinates '+ coord)

    def get_ml_repo_store(self):
        """Return the storage for the ml repo
        
        Returns:
            RepoStore -- the storage for the RepoObjects
        """
        return self._ml_repo

    def _get(self, name, version=repo_store.RepoStore.LAST_VERSION, full_object=False,
             modifier_versions=None, obj_fields=None,  repo_info_fields=None):
        """ Get repo objects. It throws an exception, if an object with the name does not exist.

            :param name: Object name
            :param version: object version, default is latest (-1). If the fields are nested (an element of a dictionary which is an element of a 
                    dictionary, use path notation to the element, i.e. p/elem1/elem2 to get p[elem1][elem2])
            :param full_object: flag to determine whether the numpy objects are loaded (True->load)
        """
        logging.debug('Getting ' + name + ', version ' + str(version))
        repo_dict = self._ml_repo.get(name, version, modifier_versions, obj_fields, repo_info_fields)
        if len(repo_dict) == 0:
            raise Exception('No object found with name ' +  name + ' and version ' + str(version))
        
        tmp = []
        for x in repo_dict:
            result = repo_objects.create_repo_obj(x)
            if isinstance(result, DataSet):
                raw_data = self._get(result.raw_data, result.raw_data_version, False)
                if full_object:
                    numpy_data = self._numpy_repo.get(result.raw_data, raw_data.repo_info[RepoInfoKey.VERSION], 
                                                        result.start_index, result.end_index)
                    repo_objects.repo_object_init.numpy_from_dict(raw_data, numpy_data)
                result.set_data(raw_data)

            numpy_dict = {}
            if len(result.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0 and full_object:
                numpy_dict = self._numpy_repo.get(
                    result.repo_info[RepoInfoKey.NAME], result.repo_info[RepoInfoKey.VERSION])
            for x in result.repo_info[RepoInfoKey.BIG_OBJECTS]:
                if not x in numpy_dict:
                    numpy_dict[x] = None
            result.numpy_from_dict(numpy_dict)
            tmp.append(result)
        if len(tmp) == 1:
            return tmp[0]
        return tmp

    def get_measure_result_name(eval_data, measure_type, model_name = None):    
        data_name = eval_data
        if not isinstance(eval_data, str):
            data_name = eval_data.repo_info[RepoInfoKey.NAME]
        tmp = data_name.split('/')
        if len(tmp) == 1:
            if model_name is None:
                raise Exception('Please specify a model name.')
            return model_name + '/measure/' + tmp[0] + '/' + measure_type
        if len(tmp) == 3:
            return tmp[0] + '/measure/' + tmp[-1] + '/' + measure_type
        raise Exception('Given data name is not valid.')

    def get_calibrated_model_name(model_name):
        tmp = model_name.split('/')
        return tmp[0] + '/model'

    def get_eval_name( model, data ):
        """Return name of the object containing evaluation results
        
        Arguments:
            model {ModelDefinition object or str} -- 
            data {RawData or DataSet object or str} --
        
        Returns:
            string -- name of valuation results
        """
        model_name = model
        if not isinstance(model,str):
            model_name =model.repo_info[RepoInfoKey.NAME]
        data_name = data
        if not isinstance(data, str):
            data_name = data.repo_info[RepoInfoKey.NAME]
        return  model_name + '/eval/' + data_name
        
    def get_names(self, ml_obj_type):
        """ Get the list of names of all repo_objects from a given repo_object_type in the repository.

            :param ml_obj_type: MLObjectType specifying the types of objects names are returned for.

            :return list of object names for the gigven category.
        """
        if isinstance(ml_obj_type, MLObjectType):
            return self._ml_repo.get_names(ml_obj_type.value)
        else:
            return self._ml_repo.get_names(ml_obj_type)

    def append_raw_data(self, name, x_data, y_data = None):
        """Append data to a RawData object

           It appends data to the given RawData object and updates all training and test DataSets which implicitely changed by this update.
        Args:
            name (string): name of RawData object
            x_data (numpy matrix): the x_data to append
            y_data (numpy matrix, optional): Defaults to None. The y_data to append
        
        Raises:
            Exception: If the data is not consistent to the RawData (e.g. different number of x-coordinates) it throws an exception.
        """
        logger.info('Start appending ' + str(x_data.shape[0]) + ' datapoints to RawData' + name)
        raw_data = self._get(name)
        if len(raw_data.x_coord_names) != x_data.shape[1]:
            raise Exception('Number of columns of x_data of RawData object is not equal to number of columns of additional x_data.')
        if raw_data.y_coord_names is None and y_data is not None:
            raise Exception('RawData object does not contain y_data but y_data is given')
        if raw_data.y_coord_names is not None:
            if y_data is None:
                raise Exception('RawData object has y_data but no y_data is given')
            if y_data.shape[1] != len(raw_data.y_coord_names ):
                raise Exception('Number of columns of y_data of RawData object is not equal to number of columns of additional y_data.')
        numpy_dict = {'x_data' : x_data}
        if raw_data.y_coord_names is not None:
            numpy_dict['y_data'] =  y_data
        raw_data.n_data += x_data.shape[0]
        old_version = raw_data.repo_info[RepoInfoKey.VERSION]
        new_version = self.add(raw_data)
        self._numpy_repo.append(name, old_version, new_version, numpy_dict)
        # now find all datasets which are affected by the updated data
        changed_data_sets = []
        training_data = self.get_training_data(full_object = False)
        if isinstance(training_data, DataSet):
            if training_data.raw_data == name and training_data.raw_data_version == repo_store.RepoStore.LAST_VERSION:
                if training_data.end_index is None or training_data.end_index < 0:
                    training_data.raw_data_version = new_version
                    changed_data_sets.append(training_data)
        test_data = self.get_names(MLObjectType.TEST_DATA)
        for d in test_data:
            data = self._get(d)
            if isinstance(data, DataSet):
                if data.raw_data == name and data.raw_data_version == repo_store.RepoStore.LAST_VERSION:
                    if data.end_index is None or data.end_index < 0:
                        data.raw_data_version = new_version
                        changed_data_sets.append(data)
        self.add(changed_data_sets, 'RawData ' + name + ' updated, add DataSets depending om the updated RawData.')
        logger.info('Finished appending data to RawData' + name)
                
                    

    def get_history(self, name, repo_info_fields=None, obj_member_fields=None, version_start=repo_store.RepoStore.FIRST_VERSION, 
                    version_end=repo_store.RepoStore.LAST_VERSION):
        """ Return a list of histories of object member variables without bigobjects

        :param repo_info_field: List of fields from repo_info which will be returned in the dictionary. 
                                If List contains flag 'ALL', all fields will be returned.
        :param obj_member_fields: List of member atributes from repo_object which will be returned in the dictionary. 
                                If List contains flag 'ALL', all attributes will be returned.
        """
        tmp = self._ml_repo.get(name, versions=(version_start, version_end), obj_fields=obj_member_fields,  repo_info_fields=repo_info_fields)
        if isinstance(tmp, list):
            return tmp
        return [tmp]

    def get_commits(self,  version_start=repo_store.RepoStore.FIRST_VERSION, version_end=repo_store.RepoStore.LAST_VERSION):
        tmp = self._get('CommitInfo', (version_start, version_end))
        if isinstance(tmp, list):
            return tmp
        return [tmp]
        
    def run_training(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, 
                    training_function_version=repo_store.RepoStore.LAST_VERSION,
                    training_data_version=repo_store.RepoStore.LAST_VERSION, training_param_version = repo_store.RepoStore.LAST_VERSION, 
                    model_param_version = repo_store.RepoStore.LAST_VERSION):
        """ Run the training algorithm. 

        :param message: commit message
        """
        if model is None:
            m_names = self.get_names(MLObjectType.MODEL.value)
            if len(m_names) == 0:
                Exception('No model definition exists, please define a model first.')
            if len(m_names) > 1:
                Exception('More than one model in repository, please specify a model to evaluate.')
            model = m_names[0]
        train_job = TrainingJob(model, self._user, training_function_version=training_function_version, model_version=model_version,
            training_data_version=training_data_version, training_param_version= training_param_version, 
            model_param_version=model_param_version, repo_info = {RepoInfoKey.NAME: model + '/jobs/training',
                RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
        self.add(train_job)
        self._job_runner.add(train_job.repo_info[RepoInfoKey.NAME], train_job.repo_info[RepoInfoKey.VERSION], self._user)
        logging.info('Training job ' + train_job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                + str(train_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
        return train_job.repo_info[RepoInfoKey.NAME], str(train_job.repo_info[RepoInfoKey.VERSION])
        
    def run_evaluation(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, predecessors = []):
        """ Evaluate the model on all datasets. 

            :param model: name of model to evaluate, if None and only one model exists
            :message: message inserted into commit (default None), if Noe, an autmated message is created
            :param model_version: Version of model to be evaluated.
            :datasets: Dictionary of datasets (names and version numbers) on which the model is evaluated. 
            :predecessors: list of jobs which shall have been completed successfull before the evaluation is started
                Default is all datasets from testdata on latest version.
            Raises:
                Exception if model_name is None and more then one model exists
        """
        if model is None:
            m_names = self.get_names(MLObjectType.CALIBRATED_MODEL)
            if len(m_names) == 0:
                raise Exception('No model exists, please train a model first.')
            if len(m_names) > 1:
                raise Exception('More than one model in repository, please specify a model to evaluate.')
            model = m_names[0]
        datasets_ = deepcopy(datasets)
        if len(datasets_) == 0: #if nothing is specified, add evaluation jobs on all training and test datasets
            names = self.get_names(MLObjectType.TEST_DATA.value)
            for n in names:
                v = self._ml_repo.get_version_number(n, -1)
                datasets_[n] = v
            training_data = self.get_training_data(full_object = False)
            datasets_[training_data.repo_info[RepoInfoKey.NAME]] = training_data.repo_info[RepoInfoKey.VERSION] 
        job_ids = []
        for n, v in datasets_.items():
            eval_job = EvalJob(model, n, self._user, model_version=model_version, data_version=v,
                        repo_info = {RepoInfoKey.NAME: model + '/jobs/eval_job/' + n,
                                    RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
            eval_job.set_predecessor_jobs(predecessors)
            self._add(eval_job)
            self._job_runner.add(eval_job.repo_info[RepoInfoKey.NAME], eval_job.repo_info[RepoInfoKey.VERSION], self._user)
            logging.info('Eval job ' + eval_job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                + str(eval_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
            job_ids.append((eval_job.repo_info[RepoInfoKey.NAME], str(eval_job.repo_info[RepoInfoKey.VERSION])))
        return job_ids

    def run_measures(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, measures = {}, predecessors = []):
        if model is None:
            m_names = self.get_names(MLObjectType.CALIBRATED_MODEL)
            if len(m_names) == 0:
                raise Exception('No model exists, please train a model first.')
            if len(m_names) > 1:
                raise Exception('More than one model in repository, please specify a model to evaluate.')
            model = m_names[0]
        datasets_ = deepcopy(datasets)
        if len(datasets_) == 0:
            names = self.get_names(MLObjectType.TEST_DATA)
            names.extend(self.get_names(MLObjectType.TRAINING_DATA))
            for n in names:
                v = self._ml_repo.get_version_number(n, -1)
                datasets_[n] = v #todo include training data into measures
            
        measure_config = self.get_names(MLObjectType.MEASURE_CONFIGURATION)[0]
        measure_config = self._get(measure_config)
        measures_to_run = {}
        if len(measures) == 0: # if no measures are specified, use all from configuration
            measures_to_run = measure_config.measures
        else:
            for k, v in measure.items():
                measures_to_run[k] = v
        job_ids = []
        for n, v in datasets_.items():
            for m_name, m in measures_to_run.items():
                measure_job = MeasureJob(m_name, m[0], m[1], n, model, v, model_version, 
                        repo_info = {RepoInfoKey.NAME: model + '/jobs/measure/' + n + '/' + m[0],
                        RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
                measure_job.set_predecessor_jobs(predecessors)
                self._add(measure_job)
                job_id = self._job_runner.add(measure_job.repo_info[RepoInfoKey.NAME], measure_job.repo_info[RepoInfoKey.VERSION], self._user)
                job_ids.append(job_id)
                logging.info('Measure job ' + measure_job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                    + str(measure_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
        return job_ids

    def run_tests(self, message='', model_version=repo_store.RepoStore.LAST_VERSION, tests={}, job_runner=None, predecessors = []):
        """ Run tests for a specific model version.

            :param message: Commit message for this operation.
            :param model_version: Version or label of model for which the tests are executed.
            :param tests: Dictionary of tests (names and version numbers) run. Default is all tests on latest version. 
            :param job_runner: job runner executing the tests. Default is single threaded local jobrunner.

            :return ticket number of job
        """
        pass

    def set_label(self, label_name, model_name = None, model_version = repo_store.RepoStore.LAST_VERSION, message=''):
        """ Label a certain model version.

            It checks if a model with this version really exists and throws an exception if such a model does not exist.

            This method labels a certain model version. 
            :param message: Commit message
            :param model_name: name of model
            :param model_version: model version for which the label is set.
        """
        # check if a model with this version exists
        if not self._ml_repo.object_exists(model_name, model_version):
            raise Exception('Cannot set label, model ' + model_name + ' with version ' + str(model_version) + ' does not exist.')
        label = repo_objects.Label(model_name, model_version, repo_info={RepoInfoKey.NAME: label_name, 
            RepoInfoKey.CATEGORY: MLObjectType.LABEL.value})
        self._ml_repo.add(label)        
        

