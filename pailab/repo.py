"""Machine Learning Repository

This module contains pailab's machine learning repository, i.e. the repository 
Machine learning repository
"""
import abc
import json
from datetime import datetime
from numpy import linalg
from numpy import inf, load
from enum import Enum
from copy import deepcopy
import logging
import pailab.repo_objects as repo_objects
from pailab.repo_objects import RepoInfoKey, DataSet, MeasureConfiguration
from pailab.repo_objects import repo_object_init, RepoInfo, RepoObject  # pylint: disable=E0401
import pailab.repo_store as repo_store
from pailab.repo_store_factory import RepoStoreFactory
logger = logging.getLogger(__name__)



class MLObjectType(Enum):
    """Enums describing all ml object types.
    """
    EVAL_DATA = 'EVAL_DATA'
    RAW_DATA = 'RAW_DATA'
    TRAINING_DATA = 'TRAINING_DATA'
    TEST_DATA = 'TEST_DATA'
    TEST = 'TEST'
    TEST_DEFINITION = 'TEST_DEFINITION'
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
    TRAINING_STATISTIC = 'TRAINING_STATISTIC'

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


class Mapping(RepoObject):
    """Provides a mapping from MLObjectType to all objects in the repo belonging to this type
    """
    def __init__(self, repo_info = RepoInfo(), **kwargs):
        super(Mapping, self).__init__(repo_info)
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
    """Add/update modificaion info to a repo object from a list if repo objects which were used to create it.

    Arguments:
        repo_obj {repoobject} -- the object the modification info is added to
    """
    mod_info =  repo_obj.repo_info[RepoInfoKey.MODIFICATION_INFO]
    for v in args:
        if v is not None:
            mod_info[v.repo_info[RepoInfoKey.NAME]
                ] = v.repo_info[RepoInfoKey.VERSION]
            #for k, w in v.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
                # check if there is a bad inconsistency: if an object modifying the current object has already been used with a different version
                # number, this must be annotated
                #if k in mod_info.keys() and mod_info[k] != w:
                #    mod_info['inconsistency'] = 'object ' + k + \
                #        ' has already been used with a different version number'
                #mod_info[k] = w
    
# region Jobs

class Job(RepoObject, abc.ABC):

    def __init__(self, repo_info):
        super(Job, self).__init__(repo_info)
        self.state = 'created'
        self.started = 'not yet started'
        self.finished = 'not yet finished'

    class RepoWrapper:
        
        def __init__(self, job, ml_repo):
            self.versions = {}
            self.ml_repo = ml_repo
            for k in job.get_predecessor_jobs():
                job = ml_repo.get(k[0], version=k[1])
                for k,v in job.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
                    self.versions[k] = v
            self.modification_info = {}

        def _get_version(self, name, orig_version):
            if orig_version is None:
                return orig_version
            if name in self.versions.keys():
                return self.versions[name]
            return orig_version

        def get(self, name, version=None, full_object = False,
                modifier_versions=None, obj_fields=None,  repo_info_fields=None,
                throw_error_not_exist=True, throw_error_not_unique=True, adjust_modification_info = True):
            new_v = self._get_version(name, version)
            m_v = None
            if modifier_versions is not None:
                m_v = deepcopy(modifier_versions)
                for k,v in m_v.items():
                    m_v[k] = self._get_version(k,v)
            obj = self.ml_repo.get(name, version=new_v, full_object=full_object,
                modifier_versions=m_v, obj_fields=obj_fields,  repo_info_fields=repo_info_fields,
                throw_error_not_exist=throw_error_not_exist, throw_error_not_unique=throw_error_not_unique)
            if isinstance(obj, list):
                if len(obj) == 0:
                    if throw_error_not_exist:
                        raise Exception('More than one object with name ' + name + ' found meeting the conditions.')
                    else:
                        return []
                else:
                    if throw_error_not_unique:
                        raise Exception('More than one object with name ' + name + ' found meeting the conditions.')
                    else:
                        return []
            if adjust_modification_info:
                self.modification_info[name] = obj.repo_info[RepoInfoKey.VERSION]
                for k,v in obj.repo_info.modification_info.items():
                    self.modification_info[k] = v
            return obj

        def add(self, obj, message , category = None):
            self.ml_repo.add(obj, message, category = category)
            if isinstance(obj, list):
                for o in obj:
                    self.modification_info[o.repo_info[RepoInfoKey.NAME]] = o.repo_info[RepoInfoKey.VERSION]   
            else:     
                self.modification_info[obj.repo_info[RepoInfoKey.NAME]] = obj.repo_info[RepoInfoKey.VERSION]

        def get_training_data(self, obj_version, full_object):
            tmp = self.ml_repo.get_training_data(obj_version, full_object = False) # TODO: replace get_trainign data by get using the training data name
            return self.get(tmp.repo_info[RepoInfoKey.NAME], obj_version, full_object=full_object)    
        
        def get_names(self, ml_obj_type):
            return self.ml_repo.get_names(ml_obj_type)
            
    """Abstract class defining the interfaces needed for a job to be used in the JobRunner

    """
    def add_predecessor_job(self, job):
        """Add job as predecessor which must have been run successfull before the job can be started

        Arguments:
            predecessors {list o jobids} -- predecessors
        """
        if isinstance(job, tuple):
            self._predecessors.append(job)    
        else:
            self._predecessors.append((job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION]))

    def set_predecessor_jobs(self, predecessors):
        self._predecessors=[]
        for obj in predecessors:
            self.add_predecessor_job(obj)

    def get_predecessor_jobs(self):
        """Return list of jobids which must have been run sucessfully before the job will be executed
        """
        if hasattr(self, '_predecessors'):
            return self._predecessors
        return []

    def run(self, ml_repo, jobid):
        wrapper = Job.RepoWrapper(self, ml_repo)
        self.state = 'running'
        self.started = str(datetime.now())
        ml_repo._update_job(self)
        try:
            self._run(wrapper,  jobid)
            self.finished = str(datetime.now())
            self.state ='finished'
            self.repo_info[RepoInfoKey.MODIFICATION_INFO] = wrapper.modification_info
            ml_repo._update_job(self)
        except Exception as e:
            self.finished = str(datetime.now())
            self.state = 'error'
            self.repo_info[RepoInfoKey.MODIFICATION_INFO] = wrapper.modification_info
            self.error_message = str(e)
            ml_repo._update_job(self)
            raise e from None

    def check_rerun(self, ml_repo):
        name, modifier_versions = self.get_modifier_versions(ml_repo)
        job = ml_repo.get(self.repo_info.name, version = None, modifier_versions=modifier_versions,throw_error_not_exist=False)
        if job == []:
            return True
        if job.state == 'error':
            return True
        return False

    @abc.abstractmethod
    def _run(self, ml_repo, jobid):
        pass

    @abc.abstractmethod
    def get_modifier_versions(self, ml_repo):
        """Returns name of outpu variable as well as all relevant modifiers with their defined versions.

        This method is used to check if a job needs to be rerun.
        
        Args:
            ml_repo (MLRepo): the repository
        """

        pass
 
class EvalJob(Job):
    """definition of a model evaluation job
    """
    def __init__(self, model, data, user, eval_function_version=repo_store.RepoStore.LAST_VERSION,
                model_version=repo_store.RepoStore.LAST_VERSION, data_version=repo_store.RepoStore.LAST_VERSION,
                repo_info = RepoInfo()):
        super(EvalJob, self).__init__(repo_info)
        self.model = model
        self.data = data
        self.user = user
        self.eval_function_version = eval_function_version
        self.model_version = model_version
        self.data_version = data_version
        # list of jobids which must have been run before this job should be excuted
        self.predecessors = []

    def _run(self, repo, jobid):
        """Run the job with data from the given repo

        Arguments:
            repo {MLrepository} -- repository used to get and store the data
            jobid {string} -- id of job which will be run
        """
        logging.info('Start evaluation job ' + str(jobid) +' on model ' + self.model + ' ' + str(self.model_version) + ' ')
        model = repo.get(self.model, self.model_version, full_object = True)
        model_definition_name = self.model.split('/')[0]
        model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO][model_definition_name]
        model_definition = repo.get(model_definition_name, model_def_version)
        
        data = repo.get(self.data, self.data_version, full_object=True)
        eval_func = repo.get(model_definition.eval_function, self.eval_function_version)

        y = eval_func.create()(model, data.x_data)
        
        result = repo_objects.RawData(y, data.y_coord_names, repo_info={
                            RepoInfoKey.NAME: MLRepo.get_eval_name(model_definition, data),
                            RepoInfoKey.CATEGORY: MLObjectType.EVAL_DATA.value
                            }
                            )
         # create modification info
        _add_modification_info(result, model, model_definition, data, eval_func)
        model_param_name = str(NamingConventions.ModelParam(model = model_definition_name))
        if model_param_name in model.repo_info.modification_info.keys():
            result.repo_info.modification_info[model_param_name] = model.repo_info.modification_info[model_param_name]
        training_param_name = str(NamingConventions.TrainingParam(model = model_definition_name))
        if training_param_name in model.repo_info.modification_info.keys():
            result.repo_info.modification_info[training_param_name] = model.repo_info.modification_info[training_param_name]
        repo.add(result, 'evaluate data ' +
                 self.data + ' with model ' + self.model)
        logging.info('Finished evaluation job ' + str(jobid))
    
    def get_modifier_versions(self, repo):
        # get if evaluation with given inputs has already been computed
        model_definition_name = self.model.split('/')[0]
        model = repo.get(self.model, self.model_version, full_object = False, throw_error_not_exist = False)
        if model == []: #no model has been calibrated so far
            model_def_version = repo_store.RepoStore.LAST_VERSION
        else:
            model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO][model_definition_name]
        model_definition = repo.get(model_definition_name, model_def_version)
        result_name = str(NamingConventions.EvalData(model = self.model, data = self.data))
        return result_name, {self.model: self.model_version, self.data: self.data_version, model_definition.eval_function: self.eval_function_version}
        
class TrainingJob(Job):
    """definition of a model training job
    """
    @repo_object_init()
    def __init__(self, model, user, training_function_version=repo_store.RepoStore.LAST_VERSION, model_version=repo_store.RepoStore.LAST_VERSION,
                training_data_version=repo_store.RepoStore.LAST_VERSION, training_param_version=repo_store.RepoStore.LAST_VERSION,
                 model_param_version=repo_store.RepoStore.LAST_VERSION, repo_info = RepoInfo()):
        super(TrainingJob, self).__init__(repo_info)
        self.model = model
        self.user = user
        self.training_function_version = training_function_version
        self.model_version = model_version
        self.training_param_version = training_param_version
        self.model_param_version = model_param_version
        self.training_data_version = training_data_version
        # list of jobids which must have been run before this job should be excuted
        self.predecessors = []

    def _run(self, repo, jobid):
        """Run the job with data from the given repo

        Arguments:
            repo {MLrepository} -- repository used to get and store the data
            jobid {string} -- id of job which will be run
        """
        model = repo.get(self.model, self.model_version)
        train_data = repo.get_training_data(self.training_data_version, full_object = True)
        train_func = repo.get(model.training_function,
                               self.training_function_version)
        train_param = None
        if not model.training_param == '':
            train_param = repo.get(model.training_param, self.training_param_version)
        model_param = None
        if not model.model_param is None:
            model_param = repo.get(
                model.model_param, self.model_param_version)
        if model_param is None:
            result = train_func.create()(train_param, train_data.x_data, train_data.y_data)
        else:
            if train_param is None:
                result = train_func.create()(model_param, train_data.x_data, train_data.y_data)
            else:
               result = train_func.create()(model_param, train_param, train_data.x_data, train_data.y_data)
        if isinstance(result, tuple):
            calibrated_model = result[0]
            training_stat = result[1]
            training_stat.repo_info[RepoInfoKey.NAME] = self.model + '/training_stat'
            training_stat.repo_info[RepoInfoKey.CATEGORY] = MLObjectType.TRAINING_STATISTIC.value
            _add_modification_info(training_stat, model_param, train_param, train_data, model, train_func)
        else:
            calibrated_model = result    
            training_stat=None
        calibrated_model.repo_info[RepoInfoKey.NAME] = self.model + '/model'
        calibrated_model.repo_info[RepoInfoKey.CATEGORY] = MLObjectType.CALIBRATED_MODEL.value
        # create modification info
        _add_modification_info(calibrated_model, model_param, train_param, train_data, model, train_func)
        
        if training_stat is not None:
            repo.add([training_stat, calibrated_model], 'training of model ' + self.model)
        else: 
            repo.add(calibrated_model, 'training of model ' + self.model)

    
    def get_modifier_versions(self, repo):
        modifiers = {}
        modifiers[self.model] = self.model_version
        model = repo.get(self.model, self.model_version)
        train_data = repo.get_training_data(self.training_data_version, full_object = False)
        modifiers[train_data.repo_info.name] = self.training_data_version
        modifiers[model.training_function] = self.training_function_version
        if not model.training_param == '':
            modifiers[model.training_param] = self.training_param_version
        if not model.model_param is None:
            modifiers[model.model_param] = self.model_param_version
        return self.model + '/model', modifiers

class MeasureJob(Job):
    @repo_object_init()
    def __init__(self, result_name, measure_type, coordinates, data_name, model_name, data_version=repo_store.RepoStore.LAST_VERSION,
                model_version=repo_store.RepoStore.LAST_VERSION, repo_info = RepoInfo()):
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
        super(MeasureJob, self).__init__(repo_info)
        self.measure_type = measure_type
        self.coordinates = coordinates
        self.model_name = model_name
        self.model_version = model_version
        self.data_name = data_name
        self.data_version = data_version

    def _run(self, repo, jobid):
        logging.info('Start measure job ' + self.repo_info.name)
        target = repo.get(self.data_name, self.data_version, full_object = True)
        m_name = self.model_name.split('/')[0] #if given model name is a name of calibated model, split to find the evaluation
        eval_data_name = NamingConventions.EvalData(data = self.data_name, model = m_name)
        eval_data = repo.get(str(eval_data_name), modifier_versions={self.model_name: self.model_version, self.data_name: self.data_version}, full_object = True )
        logger.info('run MeasureJob on data ' + self.data_name + ':' + str(self.data_version) 
                        + ', ' + str(eval_data_name) + ':' + str(eval_data.repo_info[RepoInfoKey.VERSION])
                )
        columns = []
        measure_name = MeasureConfiguration.get_name(self.measure_type)
        if not repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
            measure_name = MeasureConfiguration.get_name((self.measure_type, self.coordinates))
            for x in self.coordinates:
                columns.append(target.y_coord_names.index(x))
        if len(columns) == 0:
            v = self._compute(target.y_data, eval_data.x_data)
        else:
            v = self._compute(target.y_data[:,columns], eval_data.x_data[:,columns])
        result_name = str(NamingConventions.Measure(eval_data_name, measure_type = measure_name))
        result = repo_objects.Measure( v, 
                                repo_info = {RepoInfoKey.NAME : result_name, RepoInfoKey.CATEGORY: MLObjectType.MEASURE.value})
        
        # create modification info
        result.repo_info.modification_info[self.model_name] = eval_data.repo_info.modification_info[self.model_name]
        result.repo_info.modification_info[self.data_name] = eval_data.repo_info.modification_info[self.data_name]
        result.repo_info.modification_info[str(eval_data_name)] = eval_data.repo_info.version
        result.repo_info.modification_info[m_name] = eval_data.repo_info.modification_info[m_name]
        model_param_name = str(NamingConventions.ModelParam(model = m_name))
        if model_param_name in eval_data.repo_info.modification_info.keys():
            result.repo_info.modification_info[model_param_name] = eval_data.repo_info.modification_info[model_param_name]
        training_param_name = str(NamingConventions.TrainingParam(model = m_name))
        if training_param_name in eval_data.repo_info.modification_info.keys():
            result.repo_info.modification_info[training_param_name] = eval_data.repo_info.modification_info[training_param_name]
        ################

        logging.debug('Add result ' + result_name)

        repo.add(result, 'computing  measure ' + self.measure_type + ' on data ' + self.data_name)
        logging.info('Finished measure job ' + self.repo_info.name)
        
    def get_modifier_versions(self, repo):
        modifiers = {}
        modifiers[self.data_name] =  self.data_version
        modifiers[self.model_name] = self.model_version
        measure_name = MeasureConfiguration.get_name(self.measure_type)
        if not repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
            measure_name = MeasureConfiguration.get_name((self.measure_type, self.coordinates))
        return measure_name, modifiers

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
    @staticmethod
    def _get_object_name(name):
        if not isinstance(name, str):
            return name.repo_info[RepoInfoKey.NAME]
        return name

    @staticmethod
    def get_model_from_name(name):
        return name.split('/')[0]

    @staticmethod
    def get_model_param_name(model_name):
        return NamingConventions.get_model_from_name(model_name) +'/model_param'

    @staticmethod
    def get_calibrated_model_name(model):
        return model.split('/')[0]+'/model'

    @staticmethod
    def get_eval_name(model_name, data_name):
        return model_name + '/eval/' + data_name

    @staticmethod
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
    TrainingParam = Name('model/*', 'training_param')
    Test = Name('model/*/test_name/data', 'tests')


   



class MLRepo:   

    """ Repository for doing machine learning

        The repository and his extensions provide a solid fundament to do machine learning science supporting features such as:
            - auditing/versioning of the data, models and tests
            - best practice standardized plotting for investigating model performance and model behaviour
            - automated quality checks

        The repository needs three different handlers/repositories 

    """
    
    @staticmethod
    def __create_default_config(user, workspace):
        if user is None:
            raise Exception('Please specify a user.')
        return {'user': user, 'workspace': workspace, 'repo_store': 
                    {'type': 'memory_handler', 
                    'config': {} }}

    def _save_config(self):
        if 'workspace' in self._config.keys():
            if self._config['workspace']  is not None:
                with open(self._config['workspace']  + '/.config.json', 'w') as f:
                    json.dump(self._config, f, indent=4, separators=(',', ': '))

    def __init__(self,  workspace = None, user=None, config = None, numpy_repo =None, job_runner=None, save_config = False):
        """ Constructor of MLRepo

            :param numpy_repo: repository where the numpy data is stored in versions. If None, a NumpyHDFHandler will be used with directory equal to repo_dir.
            :param ml_repo: repository where the repo_objects are stored. If None, a RepoDiskHandler with directory repo_dir will be used.
            :param job_runner: the jobrunner to execute calibration, evaluations etc. If None, a SimpleJobRunner is used.
        """
        self._config = config
        if config is None:
            if workspace is not None:
                with open(workspace + '/.config.json', 'r') as f:
                    self._config = json.load(f)
            else:
                self._config = MLRepo.__create_default_config(user, workspace)
        
        self._numpy_repo = numpy_repo
        self._ml_repo = RepoStoreFactory.get(self._config['repo_store']['type'], **self._config['repo_store']['config'])
        
        if numpy_repo is None:
            from pailab.numpy_handler_hdf import NumpyHDFStorage
            self._numpy_repo = NumpyHDFStorage(self._config['workspace'] + '/hdf') 
        self._user = self._config['user']
        self._job_runner = job_runner
        # check if the ml mapping is already contained in the repo, otherwise add it
        logging.info('Get mapping.')
        repo_dict = []
        if self._ml_repo.object_exists('repo_mapping', version=repo_store.RepoStore.LAST_VERSION):
            repo_dict = self._ml_repo.get('repo_mapping', versions=repo_store.RepoStore.LAST_VERSION)
        else:
            logging.info('No mapping found, creating new mapping.')
        if len(repo_dict) > 1:
            raise Exception('More than on mapping found.')
        if len(repo_dict) == 1:
           self._mapping = repo_objects.create_repo_obj(repo_dict[0])
        else:
            self._mapping = Mapping(  # pylint: disable=E1123
                repo_info={RepoInfoKey.NAME: 'repo_mapping', 
                RepoInfoKey.CATEGORY: MLObjectType.MAPPING.value})
        
        self._add_triggers = []

        if job_runner is None:
            from pailab.job_runner.job_runner import SimpleJobRunner
            self._job_runner = SimpleJobRunner(self)
            #self._job_runner.set_repo(self)
        if save_config:
            self._save_config()
      
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

            repo_object.repo_info[RepoInfoKey.COMMIT_MESSAGE] = message
            repo_object.repo_info[RepoInfoKey.COMMIT_DATE] = str(datetime.now())
            repo_object.repo_info[RepoInfoKey.AUTHOR] = self._user
            obj_dict = repo_objects.create_repo_obj_dict(repo_object)
            version = self._ml_repo.add(obj_dict)
            if len(repo_object.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0:
                np_dict = repo_object.numpy_to_dict()
                self._numpy_repo.add(repo_object.repo_info[RepoInfoKey.NAME],
                                    repo_object.repo_info[RepoInfoKey.VERSION],
                                    np_dict)
            for trigger in self._add_triggers:
                trigger()
            return version, mapping_changed

    def _update_job(self, job_object):
        """Update a job object without incrementing version number
        
            Updates a repo object without incrementing version number or adding commit message. This should be only used internally by the jobs
            to update the job objects according to their state, modification info and runtime.
        Args:
            job_object (Job-object): the job object to be updated

        Raises:
        exception if given object is not a job object
        """
        obj_dict = repo_objects.create_repo_obj_dict(job_object)
        self._ml_repo.replace(obj_dict)
        if len(job_object.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0:
            raise Exception('Jobs with big objects cannot be updated.')

    def add(self, repo_object, message='', category = None):
        """ Add a repo_object or list of repo objects to the repository.

            :param repo_object: repo_object or list of repo_objects to be added, will be modified so that it contains the version number
            :param message: commit message
            :param category: Category of repo_object which is used as fallback if the object does not define a category.

            Raises an exception if the category of the object is not defined in the object and if it is not defined with the category argument.
            It raises an exception if an object with this id does already exist.

            :return version number of object added or dictionary of names and versions of objects added
        """
        version = repo_store._version_str()
        result = {}
        repo_list = repo_object
        mapping_changed = False
        if not isinstance(repo_list,list):
            repo_list = [repo_object]
        if isinstance(repo_list, list):
            for obj in repo_list:
                obj.repo_info.version = version
                result[obj.repo_info[RepoInfoKey.NAME]], mapping_changed_tmp = self._add(obj, message, category)
                mapping_changed = mapping_changed or mapping_changed_tmp
        if mapping_changed:
            self._mapping.repo_info.version = version
            mapping_version, dummy = self._add(self._mapping)
            result['repo_mapping'] = mapping_version
            
        commit_message = repo_objects.CommitInfo(message, self._user, result, repo_info = {RepoInfoKey.CATEGORY: MLObjectType.COMMIT_INFO.value,
                RepoInfoKey.NAME: 'CommitInfo', RepoInfoKey.VERSION : version} )
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
        return self.get(self._mapping[MLObjectType.TRAINING_DATA][0], version, full_object)

    def add_eval_function(self, f, repo_name = None):
        """Add the function to evaluate the model

        Arguments:
            module_name {string} -- module where function is located
            function_name {string} -- function name
            repo_name {tring} -- identifier of the repo object used to store the information (default: None), if None, the name is set to module_name.function_name
        """
        name = repo_name
        if name is None:
            name = f.__module__ + "." + f.__name__
        func = repo_objects.Function(f, repo_info={
                                     RepoInfoKey.NAME: name,
                                     RepoInfoKey.CATEGORY: MLObjectType.MODEL_EVAL_FUNCTION.value})
        self.add(func, 'add model evaluation function ' + name)
    
    def add_training_function(self, f, repo_name = None):
        """Add function to train a model

        Arguments:
            module_name {string} -- module where function is located
            function_name {string} -- function name
            repo_name {tring} -- identifier of the repo object used to store the information (default: None), if None, the name is set to module_name.function_name
        """
        name = repo_name
        if name is None:
            name = f.__module__ + "." + f.__name__
        func = repo_objects.Function(f, repo_info={
                                     RepoInfoKey.NAME: name,
                                     RepoInfoKey.CATEGORY: MLObjectType.TRAINING_FUNCTION.value})
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
            measure_config = self.get(tmp[0])
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

    def get_numpy_data_store(self):
        return self._numpy_repo
        
    def get(self, name, version=repo_store.RepoStore.LAST_VERSION, full_object=False,
             modifier_versions=None, obj_fields=None,  repo_info_fields=None,
             throw_error_not_exist=True, throw_error_not_unique=True):
        """ Get repo objects. It throws an exception, if an object with the name does not exist.

            :param name: Object name
            :param version: object version, default is latest (-1). If the fields are nested (an element of a dictionary which is an element of a 
                    dictionary, use path notation to the element, i.e. p/elem1/elem2 to get p[elem1][elem2])
            :param full_object: flag to determine whether the numpy objects are loaded (True->load)
        """
        logging.debug('Getting ' + name + ', version ' + str(version))
        repo_dict = self._ml_repo.get(name, version, modifier_versions, obj_fields, repo_info_fields, 
                                      throw_error_not_exist, throw_error_not_unique)
        if len(repo_dict) == 0:
            if throw_error_not_exist:
                logger.error('No object found with name ' +  name + ' and version ' + str(version) + 'modifier_versions: ' + str(modifier_versions))
                raise Exception('No object found with name ' +  name + ' and version ' + str(version))
            else:
                return []
        
        tmp = []
        for x in repo_dict:
            result = repo_objects.create_repo_obj(x)
            if isinstance(result, DataSet):
                raw_data = self.get(result.raw_data, result.raw_data_version, False)
                if full_object:
                    numpy_data = self._numpy_repo.get(result.raw_data, raw_data.repo_info[RepoInfoKey.VERSION], 
                                                        result.start_index, result.end_index)
                    repo_objects.repo_object_init.numpy_from_dict(raw_data, numpy_data)
                result.set_data(raw_data)

            numpy_dict = {}
            if len(result.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0 and full_object:
                numpy_dict = self._numpy_repo.get(
                    result.repo_info[RepoInfoKey.NAME], result.repo_info[RepoInfoKey.VERSION])
            #for x in result.repo_info[RepoInfoKey.BIG_OBJECTS]:
            #    if not x in numpy_dict:
            #        numpy_dict[x] = None
            result.numpy_from_dict(numpy_dict)
            tmp.append(result)
        if len(tmp) == 1:
            return tmp[0]
        return tmp

    @staticmethod
    def get_calibrated_model_name(model_name):
        tmp = model_name.split('/')
        return tmp[0] + '/model'
    
    @staticmethod
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
        tmp = self.get('CommitInfo', (version_start, version_end))
        if isinstance(tmp, list):
            return tmp
        return [tmp]
    
    def __obj_latest(self, name, modification_info = {}):
        """Returns version of object created on latest data (modulo specified versions)
            If the object does not exist, it returns None.
        
        Args:
            name (str): name of object checked
            modification_info (dict): dictionay containing objects and their version used within the check (objects not contained are assumed to be LAST_VERSION)
        """
        tmp = None
        try:
            tmp = self.get(name, version=repo_store.RepoStore.LAST_VERSION)
        except:
            return None
        # now loop over all modified objects     
        modifiers = {}
        for k,v in tmp.repo_info.modification_info.items():
            if k in modification_info.keys():
                modifiers[k] = v
            else:
                modifiers[k] = repo_store.RepoStore.LAST_VERSION
        try:
            tmp = self.get(name, version = None, modifier_versions = modifiers)
        except:
            return None
        return tmp

    def run_training(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, 
                    training_function_version=repo_store.RepoStore.LAST_VERSION,
                    training_data_version=repo_store.RepoStore.LAST_VERSION, training_param_version = repo_store.RepoStore.LAST_VERSION, 
                    model_param_version = repo_store.RepoStore.LAST_VERSION, run_descendants = False):
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
        # Only start training if input data has changed since last run 
        if train_job.check_rerun(self):
            self.add(train_job)
            self._job_runner.add(train_job.repo_info[RepoInfoKey.NAME], train_job.repo_info[RepoInfoKey.VERSION], self._user)
            logging.info('Training job ' + train_job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                    + str(train_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
            if run_descendants:
                self.run_evaluation(model + '/model', 'evaluation triggered as descendant of run_training', 
                    predecessors=[(train_job.repo_info[RepoInfoKey.NAME], train_job.repo_info[RepoInfoKey.VERSION])], run_descendants=True)
            return train_job.repo_info[RepoInfoKey.NAME], str(train_job.repo_info[RepoInfoKey.VERSION])
        else:
            return 'No new training started: A model has already been trained on the latest data.'

    def _get_default_object_name(self, obj_name, obj_category):
        """Returns unique object name of given category if given object_name is None
        
        Args:
            obj_name (str or None): if not None, the given string will simply be returned
            obj_category (MLObjectType): category of object
        
        Raises:
            Exception: If not exactly one object of desired type exists in repo
        
        Returns:
            [str]: Resulting object name
        """

        if obj_name is not None:
            return obj_name
        names = self.get_names(obj_category)
        if len(names) == 0:
            raise Exception('No object of type ' + obj_category.value + ' exists.')
        if len(names) > 1:
            raise Exception('More than one object of type ' + obj_category.value + ' exist, please specify a specific object name.')
        return names[0]
            
    def _create_evaluation_jobs(self, model=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, predecessors = [], labels = None):
        models = [(self._get_default_object_name(model, MLObjectType.CALIBRATED_MODEL), model_version)]
        if model is None and labels is None:
            labels = self.get_names(MLObjectType.LABEL)
        if labels is not None:
            if isinstance(labels, str):
                labels=[labels]
            for l in labels:
                tmp = self.get(l)
                models.append((tmp.name, tmp.version) )
        datasets_ = deepcopy(datasets)
        if len(datasets_) == 0: #if nothing is specified, add evaluation jobs on all training and test datasets
            names = self.get_names(MLObjectType.TEST_DATA.value)
            for n in names:
                v = self._ml_repo.get_version(n, -1)
                datasets_[n] = v
            training_data = self.get_training_data(full_object = False)
            datasets_[training_data.repo_info[RepoInfoKey.NAME]] = training_data.repo_info[RepoInfoKey.VERSION] 
        jobs = []
        for m in models:
            for n, v in datasets_.items():
                eval_job = EvalJob(m[0], n, self._user, model_version=m[1], data_version=v,
                            repo_info = {RepoInfoKey.NAME: m[0] + '/jobs/eval_job/' + n,
                                        RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
                if eval_job.check_rerun(self):
                    eval_job.set_predecessor_jobs(predecessors)
                    jobs.append(eval_job)
                #else:
                #    logger.info('Skip running ' + eval_job.repo_info.name + ' has already been run on the specified data, job version: ' + tmp.repo_info.version)
        return jobs

    # def run_jobs(self, job, modifier_versions={}):
    #     """Runs jobs matching a regular expression.

    #     It checks if the job has already been run on the same input data.
        
    #     Args:
    #         job (str): Regular expression defining used to define the jobs to run
    #         modifier_versions (dict, optional): Defaults to {}. Only jobs which have not already been run on the input data are started.
    #             The dictionary defines the object version which shall be used within the job run, if an object is not specified in this dict, the latest version is used.
    #     """
    #     tmp = self.get_names(MLObjectType.JOB)
    #     import re
    #     jobs = [x for x in tmp and re.match(job, x)]
    #     jobs_to_run = []
    #     for j in jobs:
    #         ref_job = self.get(j) #get job to get all modification infos
    #         mod_info = deepcopy(ref_job.repo_info.modification_info)
    #         for k,v in mod_info.items():
    #             if k in modifier_versions.keys():
    #                 mod_info[k] = v
    #         try:
    #             job = self.get(j, modifier_versions=mod_info)
    #         except:
    #             jobs_to_run.append(job)

    def run_evaluation(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, 
        datasets={}, predecessors = [], run_descendants=False, labels = None):
        """ Evaluate the model on all datasets. 

        Args:
            model: name of model to evaluate, if None and only one model exists
            message: message inserted into commit (default None), if Noe, an autmated message is created
            model_version: Version of model to be evaluated.
            datasets: Dictionary of datasets (names and version numbers) on which the model is evaluated. 
            predecessors: list of jobs which shall have been completed successfull before the evaluation is started. Default is all datasets from testdata on latest version.

            Raises:
                Exception if model_name is None and more then one model exists
        """
        jobs = self._create_evaluation_jobs(model, model_version, datasets, predecessors, labels=labels)
        job_ids = []
        for job in jobs:
            if job.check_rerun(self):            
                self.add(job)
                self._job_runner.add(job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION], self._user)
                logging.info('Eval job ' + job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                    + str(job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
                job_ids.append((job.repo_info[RepoInfoKey.NAME], str(job.repo_info[RepoInfoKey.VERSION])))
                if run_descendants:
                    self.run_measures(model, 'run_mesaures started as predecessor of run_evaluation', datasets={job.data: repo_store.RepoStore.LAST_VERSION}, 
                        predecessors=[(job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION])])
        return job_ids

    def run_measures(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, measures = {}, predecessors = [], labels=None):
        models = [(self._get_default_object_name(model, MLObjectType.CALIBRATED_MODEL), model_version)]
        if model is None and labels is None:
            labels = self.get_names(MLObjectType.LABEL)
        if labels is not None:
            if isinstance(labels, str):
                labels=[labels]
            for l in labels:
                tmp = self.get(l)
                models.append((tmp.name, tmp.version) )

        datasets_ = deepcopy(datasets)
        if len(datasets_) == 0:
            names = self.get_names(MLObjectType.TEST_DATA)
            names.extend(self.get_names(MLObjectType.TRAINING_DATA))
            for n in names:
                datasets_[n] = repo_store.RepoStore.LAST_VERSION

        measure_names =   self.get_names(MLObjectType.MEASURE_CONFIGURATION)
        if len(measure_names) == 0:
            logger.warning('No measures defined.')
            return      
        measure_config = measure_names[0]
        measure_config = self.get(measure_config)
        measures_to_run = {}
        if len(measures) == 0: # if no measures are specified, use all from configuration
            measures_to_run = measure_config.measures
        else:
            for k, v in measures.items():
                measures_to_run[k] = v
        job_ids = []
        for mod in models:
            for n, v in datasets_.items():
                for m_name, m in measures_to_run.items():
                    measure_job = MeasureJob(m_name, m[0], m[1], n, mod[0], v, mod[1],
                        repo_info = {RepoInfoKey.NAME: mod[0] + '/jobs/measure/' + n + '/' + m[0],
                            RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
                    if measure_job.check_rerun(self) :
                        measure_job.set_predecessor_jobs(predecessors)
                        self.add(measure_job)
                        self._job_runner.add(measure_job.repo_info[RepoInfoKey.NAME], measure_job.repo_info[RepoInfoKey.VERSION], self._user)
                        job_ids.append((measure_job.repo_info[RepoInfoKey.NAME], measure_job.repo_info[RepoInfoKey.VERSION]))
                        logging.info('Measure job ' + measure_job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                        + str(measure_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
        return job_ids

    def run_tests(self, test_definitions = None, predecessors = []):
        """ Run tests for a specific model version.

            :param test_definition (list/set): List or set of names of the test definitions which shall b executed. If None, all test definitions are executed.
        
            :return ticket number of job
        """
        test_defs = test_definitions
        if test_defs is None:
            test_defs = self._ml_repo.get_names(MLObjectType.TEST_DEFINITION.value)
        job_ids = []
        for t in test_defs:
            tmp = self.get(t)
            tests = tmp.create(self)
            for tt in tests:
                if tt.check_rerun(self):
                    self.add(tt, category = MLObjectType.TEST)
                    self._job_runner.add(tt.repo_info.name, tt.repo_info.version, self._user)
                    job_ids.append((tt.repo_info.name, tt.repo_info.version))
        return job_ids

    def set_label(self, label_name, model = None, model_version = repo_store.RepoStore.LAST_VERSION, message=''):
        """ Label a certain model version.

            It checks if a model with this version really exists and throws an exception if such a model does not exist.

            This method labels a certain model version. 
            :param message: Commit message
            :param model_name: name of model
            :param model_version: model version for which the label is set.
        """
        model = self._get_default_object_name(model, MLObjectType.CALIBRATED_MODEL)
        # check if a model with this version exists
        m = self.get(model)
        if model_version == repo_store.RepoStore.LAST_VERSION:
            model_version = m.repo_info[RepoInfoKey.VERSION]
        label = repo_objects.Label(model, model_version, repo_info={RepoInfoKey.NAME: label_name, 
            RepoInfoKey.CATEGORY: MLObjectType.LABEL.value})
        self.add(label)        
        
    def _object_exists(self, name):
        """returns True if an object with this name exists in repo
        
        Args:
            name (str): name of object
        
        Returns:
            [type]: [description]
        """

        return self._ml_repo.object_exists(name)

