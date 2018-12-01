"""Machine Learning Repository

This module contains pailab's machine learning repository, i.e. the repository 
Machine learning repository
"""
import abc
from numpy import linalg
from numpy import inf, load
from enum import Enum
from copy import deepcopy
from deepdiff import DeepDiff
import logging
import pailab.repo_objects as repo_objects
from pailab.repo_objects import RepoInfoKey, DataSet
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
    """Add/update modificaion info to a repo object from a list if repo objects which were used to create it.

    Arguments:
        repo_obj {repoobject} -- the object the modification info is added to
    """
    mod_info =  repo_obj.repo_info[RepoInfoKey.MODIFICATION_INFO]
    for v in args:
        if v is not None:
            mod_info[v.repo_info[RepoInfoKey.NAME]
                ] = v.repo_info[RepoInfoKey.VERSION]
            for k, w in v.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
                # check if there is a bad inconsistency: if an object modifying the current object has already been used with a different version
                # number, this must be annotated
                if k in mod_info.keys() and mod_info[k] != w:
                    mod_info['inconsistency'] = 'object ' + k + \
                        ' has already been used with a different version number'
                mod_info[k] = w
    
# region Jobs

class Job(abc.ABC):

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

        def get(self, obj_name, obj_version=None, full_object = False,
                modifier_versions=None, obj_fields=None,  repo_info_fields=None):
            new_v = self._get_version(obj_name, obj_version)
            m_v = None
            if modifier_versions is not None:
                m_v = deepcopy(modifier_versions)
                for k,v in m_v.items():
                    m_v[k] = self._get_version(k,v)
            obj = self.ml_repo.get(obj_name, version=new_v, full_object=full_object,
                modifier_versions=m_v, obj_fields=obj_fields,  repo_info_fields=repo_info_fields)
            if isinstance(obj, list):
                raise Exception("More than one object found meeting the conditions.")
            self.modification_info[obj_name] = obj.repo_info[RepoInfoKey.VERSION]
            return obj

        def add(self, obj, message , category = None):
            self.ml_repo.add(obj, message, category = category)
            self.modification_info[obj.repo_info[RepoInfoKey.NAME]] = obj.repo_info[RepoInfoKey.VERSION]

        def get_training_data(self, obj_version, full_object):
            tmp = self.ml_repo.get_training_data(obj_version, full_object = False) # TODO: replace get_trainign data by get using the training data name
            return self.get(tmp.repo_info[RepoInfoKey.NAME], obj_version, full_object=full_object)    
            
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
        self._run(wrapper,  jobid)
        self.repo_info[RepoInfoKey.MODIFICATION_INFO] = wrapper.modification_info
        ml_repo._update_job(self)

    @abc.abstractmethod
    def _run(self, ml_repo, jobid):
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

    def _run(self, repo, jobid):
        """Run the job with data from the given repo

        Arguments:
            repo {MLrepository} -- repository used to get and store the data
            jobid {string} -- id of job which will be run
        """
        logging.info('Start evaluation job ' + str(jobid) +' on model ' + self.model + ' ' + str(self.model_version) + ' ')
        model = repo.get(self.model, self.model_version, full_object = True)
        #if model.repo_info[RepoInfoKey.CATEGORY] == MLObjectType.Model:
        #    model = repo.get(self.model + '/model', self.model_version)
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
        _add_modification_info(result, model, data, eval_func)
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
        m = None
        if model_param is None:
            m = train_func.create()(train_param, train_data.x_data, train_data.y_data)
        else:
            if train_param is None:
                m = train_func.create()(model_param, train_data.x_data, train_data.y_data)
            else:
               m = train_func.create()(model_param, train_param, train_data.x_data, train_data.y_data)
        m.repo_info[RepoInfoKey.NAME] = self.model + '/model'
        m.repo_info[RepoInfoKey.CATEGORY] = MLObjectType.CALIBRATED_MODEL.value
        _add_modification_info(m, model_param, train_param, train_data, model, train_func)
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

    def _run(self, repo, jobid):
        logging.info('Start measure job ' + self.repo_info.name)
        target = repo.get(self.data_name, self.data_version, full_object = True)
        m_name = self.model_name.split('/')[0] #if given model name is a name of calibated model, split to find the evaluation
        eval_data_name = NamingConventions.EvalData(data = self.data_name, model = m_name)
        eval_data = repo.get(str(eval_data_name), modifier_versions={self.model_name: self.model_version, self.data_name: self.data_version}, full_object = True )
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
            v = self._compute(target.y_data[:,columns], eval_data.x_data[:,columns])
        result_name = str(NamingConventions.Measure(eval_data_name, measure_type = self.measure_type))
        if not repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
            for x in self.coordinates:
                result_name = result_name + '_' + x
        result = repo_objects.Measure( v, 
                                repo_info = {RepoInfoKey.NAME : result_name, RepoInfoKey.CATEGORY: MLObjectType.MEASURE.value})
        _add_modification_info(result, eval_data, target)
        logging.debug('Add result ' + result_name)
        repo.add(result, 'computing  measure ' + self.measure_type + ' on data ' + self.data_name)
        logging.info('Finished measure job ' + self.repo_info.name)
        
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
    
    def _get_object_name(name):
        if not isinstance(name, str):
            return eval_data.repo_info[RepoInfoKey.NAME]
        return name

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

#region collections and items
class RepoObjectItem:

    def __init__(self, name, ml_repo, repo_obj = None):
        self._name = name
        self._repo = ml_repo
        if repo_obj is not None:
            self.obj = repo_obj
     
    def _set(self, path, items):
        if len(path) > 0:
            if len(path) == 1:
                setattr(self, path[0], items[0])
                return
            if hasattr(self, path[0]):
                getattr(self, path[0])._set(path[1:], items[1:])
            else:
                setattr(self, path[0], items[0])
                items[0]._set(path[1:], items[1:])

    def load(self, version=repo_store.LAST_VERSION, full_object=False,
            modifier_versions=None, containing_str=None):
            if containing_str is None or containing_str in self._name:
                try:
                    self.obj = self._repo.get(self._name, version, full_object, modifier_versions)
                except:
                    pass
            for k, v in self.__dict__.items():
                if hasattr(v,'load'):
                    v.load(version, full_object, modifier_versions, containing_str)

    def modifications(self, commit=False, commit_message=''):
        if self._name is not None:
            try:
                obj_orig = self._repo.get(
                    self.obj.repo_info[RepoInfoKey.NAME], version=self.obj.repo_info[RepoInfoKey.VERSION])
                diff = DeepDiff(obj_orig, self.obj,
                                ignore_order=True)
            except AttributeError:
                return None
            if len(diff) == 0:
                return None
            else:
                if commit:
                    version = self._repo.add(
                        self.obj, message=commit_message)
                    self.obj = self._repo.get(self._name, version=version)
                return {self._name: diff}
        else:
            result = {}
            for k, v in self.__dict__.items():
                if hasattr(v, 'modifications'):
                    tmp = v.modifications(commit, commit_message)
                    if tmp is not None:
                        result.update(tmp)
            return result

    def __call__(self, containing_str=None):
        # if len(self.__dict__) == 1:
        if containing_str is not None:
            result = []
            if containing_str in self._name:
                result.append(self._name)
            for k, v in self.__dict__.items():
                if isinstance(v, RepoObjectItem):
                    d = v(containing_str)
                    if isinstance(d, str):
                        result.append(d)
                    else:
                        result.extend(d)
            return [x for x in result if containing_str in x]
        else:
            return self._name
        return result

class RawDataItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(RawDataItem,self).__init__(name, ml_repo, repo_obj)

    def append(self, x_data, y_data = None):
        """Append data to a RawData object

        It appends data to the given RawData object and updates all training and test DataSets which implicitely changed by this update.

        Args:
            name (string): name of RawData object
            x_data (numpy matrix): the x_data to append
            y_data (numpy matrix, optional): Defaults to None. The y_data to append
        
        Raises:
            Exception: If the data is not consistent to the RawData (e.g. different number of x-coordinates) it throws an exception.
        """
        logger.info('Start appending ' + str(x_data.shape[0]) + ' datapoints to RawData' + self._name)
        raw_data = self._repo.get(self._name)
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
        new_version = self._repo.add(raw_data)
        self._repo._numpy_repo.append(self._name, old_version, new_version, numpy_dict)
        # now find all datasets which are affected by the updated data
        changed_data_sets = []
        training_data = self._repo.get_training_data(full_object = False)
        if isinstance(training_data, DataSet):
            if training_data.raw_data == self._name and training_data.raw_data_version == repo_store.RepoStore.LAST_VERSION:
                if training_data.end_index is None or training_data.end_index < 0:
                    training_data.raw_data_version = new_version
                    changed_data_sets.append(training_data)
        test_data = self._repo.get_names(MLObjectType.TEST_DATA)
        for d in test_data:
            data = self._repo.get(d)
            if isinstance(data, DataSet):
                if data.raw_data == self._name and data.raw_data_version == repo_store.RepoStore.LAST_VERSION:
                    if data.end_index is None or data.end_index < 0:
                        data.raw_data_version = new_version
                        changed_data_sets.append(data)
        self._repo.add(changed_data_sets, 'RawData ' + self._name + ' updated, add DataSets depending om the updated RawData.')
        if hasattr(self, 'obj'):#update current object
            self.obj = self._repo.get(self._name, version=new_version)
        logger.info('Finished appending data to RawData' + self._name)

class RawDataCollection(RepoObjectItem):
    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]

    def __init__(self, repo):
        super(RawDataCollection, self).__init__('raw_data', repo)
        names = repo.get_names(MLObjectType.RAW_DATA)
        for n in names:
            setattr(self, RawDataCollection.__get_name_from_path(n), RawDataItem(n, repo))
        self._repo = repo

    def add(self, name, data, input_variables, target_variables):
        """Add raw data to the repository

        Arguments:
            data_name {name of data} -- the name of the data added
            data {pandas datatable} -- the data as pndas datatable
        
        Keyword Arguments:
            input_variables {list of strings} -- list of column names defining the input variables for the machine learning (default: {None}). If None, all variables are used as input
            target_variables {list of strings} -- list of column names defining the target variables for the machine learning (default: {None}). If None, no target data is added from the table.
        """
        path = 'raw_data/' + name
      
        if target_variables is not None:
            raw_data = repo_objects.RawData(data.as_matrix(columns=input_variables), input_variables, data.as_matrix(columns=target_variables), 
                target_variables, repo_info = {RepoInfoKey.NAME: path})
        else:
            raw_data = repo_objects.RawData(data.as_matrix(), list(data), repo_info = {RepoInfoKey.NAME: path})
        v = self._repo.add(raw_data, 'data ' + path + ' added to repository' , category = MLObjectType.RAW_DATA)
        obj = self._repo.get(path, version=v, full_object = False)
        setattr(self, name, RawDataItem(path, self._repo, obj))

    def add_from_numpy_file(self, name, filename_X, x_names, filename_Y=None, y_names = None):
        path = name
        X = load(filename_X)
        Y = None
        if filename_Y is not None:
            Y = load(filename_Y)
        raw_data =  repo_objects.RawData(X, x_names, Y, y_names, repo_info = {RepoInfoKey.NAME: path})
        v = self._repo.add(raw_data, 'data ' + path + ' added to repository' , category = MLObjectType.RAW_DATA)
        obj = self._repo.get(path, version=v, full_object = False)
        setattr(self, name, RawDataItem(path, self._repo, obj))

class TrainingDataCollection(RepoObjectItem):
    def __get_name_from_path(path):
        return path.split('/')[-1]
    
    def __init__(self, repo):
        super(TrainingDataCollection, self).__init__('training_data', repo)
        
        names = repo.get_names(MLObjectType.TRAINING_DATA)
        for n in names:
            setattr(self, TrainingDataCollection.__get_name_from_path(n), RepoObjectItem(n, repo))
        self._repo = repo

    def add(self, name, raw_data, start_index=0, 
        end_index=None, raw_data_version='last'):
        #path = 'training_data/' + name
        data_set = repo_objects.DataSet(raw_data, start_index, end_index, 
                raw_data_version, repo_info = {RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        v = self._repo.add(data_set)
        tmp = self._repo.get(name, version=v)
        item = RepoObjectItem(name, self._repo, tmp)
        setattr(self, name, item)

class TestDataCollection(RepoObjectItem):
    def __get_name_from_path(path):
        return path.split('/')[-1]
    
    def __init__(self, repo):
        super(TestDataCollection, self).__init__('test_data', repo)
        names = repo.get_names(MLObjectType.TEST_DATA)
        for n in names:
            setattr(self, TestDataCollection.__get_name_from_path(n), RepoObjectItem(n,repo))
        self._repo = repo

    def add(self, name, raw_data, start_index=0, 
        end_index=None, raw_data_version='last'):
        data_set = repo_objects.DataSet(raw_data, start_index, end_index, 
                raw_data_version, repo_info = {RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        v = self._repo.add(data_set)
        tmp = self._repo.get(name, version=v)
        item = RepoObjectItem(name, self._repo, tmp)
        setattr(self, name, item)

class MeasureItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(MeasureItem, self).__init__(name, ml_repo, repo_obj) 

class JobItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(JobItem, self).__init__(name, ml_repo, repo_obj) 

class MeasureCollection(RepoObjectItem):
    def __init__(self, name, ml_repo, model_name):
        super(MeasureCollection, self).__init__('measures', ml_repo)
        names = ml_repo.get_names(MLObjectType.MEASURE)
        for n in names:
            path = n.split('/')[2:]
            items = [None] * len(path)
            for i in range(len(items)-1):
                items[i] = RepoObjectItem(path[i], None)
            items[-1] = MeasureItem(n, ml_repo)
            self._set(path, items)
            #items[-2] = MeasuresOnDataItem
            
class JobCollection(RepoObjectItem):
    def __init__(self, name, ml_repo, model_name):
        super(JobCollection, self).__init__('jobs', ml_repo)
        names = ml_repo.get_names(MLObjectType.JOB)
        for n in names:
            if model_name in n:
                path = n.split('/')
                path = path[path.index('jobs')+1:]
                items = [None] * len(path)
                for i in range(len(items)-1):
                    items[i] = RepoObjectItem(path[i], None)
                items[-1] = JobItem(n, ml_repo)
                self._set(path, items)

class ModelItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(ModelItem,self).__init__(name, ml_repo, repo_obj)
        self.model = RepoObjectItem(name + '/model', ml_repo)
        self.eval = RepoObjectItem(name + '/eval', ml_repo)
        self.model_param = RepoObjectItem(name + '/model_param', ml_repo)
        self.training_param = RepoObjectItem(name + '/training_param', ml_repo)
        self.measures = MeasureCollection(name+ '/measure', ml_repo, name)
        self.jobs = JobCollection(name+'/jobs', ml_repo, name)
        #self.param = RepoObjectItem(name)

    def set_label(self, label_name, version = repo_store.RepoStore.LAST_VERSION, message=''):
        self._repo.set_label(label_name, self._name+ '/model', version, message)

class ModelCollection:
    @staticmethod
    def __get_name_from_path(name):
        return name

    def __init__(self, repo):
        names = repo.get_names(MLObjectType.MODEL)
        for n in names:
            setattr(self, ModelCollection.__get_name_from_path(n), ModelItem(n,repo))
        self._repo = repo

    def add(self, name):
        setattr(self, name, ModelItem(name,self._repo))
#endregion

class MLRepo:   
    """ Repository for doing machine learning

        The repository and his extensions provide a solid fundament to do machine learning science supporting features such as:
            - auditing/versioning of the data, models and tests
            - best practice standardized plotting for investigating model performance and model behaviour
            - automated quality checks

        The repository needs three different handlers/repositories 

    """
    def reload(self):
        self.raw_data = RawDataCollection(self)
        self.training_data = TrainingDataCollection(self)
        self.test_data = TestDataCollection(self)
        self.models = ModelCollection(self)

    def __init__(self, user, numpy_repo =None, ml_repo=None, job_runner=None, repo_dir = None):
        """ Constructor of MLRepo

            :param numpy_repo: repository where the numpy data is stored in versions. If None, a NumpyHDFHandler will be used with directory equal to repo_dir.
            :param ml_repo: repository where the repo_objects are stored. If None, a RepoDiskHandler with directory repo_dir will be used.
            :param job_runner: the jobrunner to execute calibration, evaluations etc. If None, a SimpleJobRunner is used.
        """
        self._numpy_repo = numpy_repo
        self._ml_repo = ml_repo
        if ml_repo is None:
            if repo_dir is None:
                raise Exception('You must either specify a repository directory or the ml_repo directly.')
            from pailab.disk_handler import RepoObjectDiskStorage
            self._ml_repo = RepoObjectDiskStorage(repo_dir + '/objects')
        if numpy_repo is None:
            if repo_dir is None:
                raise Exception('You must either specify a repository directory or the numpy repo directly.')
            from pailab.numpy_handler_hdf import NumpyHDFStorage
            self._numpy_repo = NumpyHDFStorage(repo_dir + '/hdf') 
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
        self.reload()
        if job_runner is None:
            from pailab.job_runner.job_runner import SimpleJobRunner
            self._job_runner = SimpleJobRunner(self)
            #self._job_runner.set_repo(self)
        
    
        
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
            repo_object.repo_info[RepoInfoKey.AUTHOR] = self._user
            obj_dict = repo_objects.create_repo_obj_dict(repo_object)
            version = self._ml_repo.add(obj_dict)
            if len(repo_object.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0:
                np_dict = repo_object.numpy_to_dict()
                self._numpy_repo.add(repo_object.repo_info[RepoInfoKey.NAME],
                                    repo_object.repo_info[RepoInfoKey.VERSION],
                                    np_dict)
            self.reload() # todo make this more efficient by just updating collections and items which are affected by this
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

    def get(self, name, version=repo_store.RepoStore.LAST_VERSION, full_object=False,
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
            logger.error('No object found with name ' +  name + ' and version ' + str(version) + 'modifier_versions: ' + str(modifier_versions))
            raise Exception('No object found with name ' +  name + ' and version ' + str(version))
        
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
        self.add(train_job)
        self._job_runner.add(train_job.repo_info[RepoInfoKey.NAME], train_job.repo_info[RepoInfoKey.VERSION], self._user)
        logging.info('Training job ' + train_job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                + str(train_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
        if run_descendants:
            eval_jobs = self.run_evaluation(model + '/model', 'evaluation triggered as descendant of run_training', 
                predecessors=[(train_job.repo_info[RepoInfoKey.NAME], train_job.repo_info[RepoInfoKey.VERSION])], run_descendants=True)
        
            
        return train_job.repo_info[RepoInfoKey.NAME], str(train_job.repo_info[RepoInfoKey.VERSION])

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
            
        
    def _create_evaluation_jobs(self, model=None,model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, predecessors = []):
        model = self._get_default_object_name(model, MLObjectType.CALIBRATED_MODEL)
        datasets_ = deepcopy(datasets)
        if len(datasets_) == 0: #if nothing is specified, add evaluation jobs on all training and test datasets
            names = self.get_names(MLObjectType.TEST_DATA.value)
            for n in names:
                v = self._ml_repo.get_version(n, -1)
                datasets_[n] = v
            training_data = self.get_training_data(full_object = False)
            datasets_[training_data.repo_info[RepoInfoKey.NAME]] = training_data.repo_info[RepoInfoKey.VERSION] 
        jobs = []
        for n, v in datasets_.items():
            eval_job = EvalJob(model, n, self._user, model_version=model_version, data_version=v,
                        repo_info = {RepoInfoKey.NAME: model + '/jobs/eval_job/' + n,
                                    RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
            eval_job.set_predecessor_jobs(predecessors)
            jobs.append(eval_job)
        return jobs

    def run_evaluation(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, predecessors = [], run_descendants=False):
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
        jobs = self._create_evaluation_jobs(model, model_version, datasets, predecessors)
        job_ids = []
        for job in jobs:
            self.add(job)
            self._job_runner.add(job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION], self._user)
            logging.info('Eval job ' + job.repo_info[RepoInfoKey.NAME]+ ', version: ' 
                + str(job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
            job_ids.append((job.repo_info[RepoInfoKey.NAME], str(job.repo_info[RepoInfoKey.VERSION])))
            if run_descendants:
                self.run_measures(model, 'run_mesaures started as predecessor of run_evaluation', datasets={job.data: repo_store.RepoStore.LAST_VERSION}, 
                        predecessors=[(job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION])])
        return job_ids

    def run_measures(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, measures = {}, predecessors = []):
        model = self._get_default_object_name(model, MLObjectType.CALIBRATED_MODEL)
        datasets_ = deepcopy(datasets)
        if len(datasets_) == 0:
            names = self.get_names(MLObjectType.TEST_DATA)
            names.extend(self.get_names(MLObjectType.TRAINING_DATA))
            for n in names:
                v = self._ml_repo.get_version(n, -1)
                datasets_[n] = v #todo include training data into measures

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
            for k, v in measure.items():
                measures_to_run[k] = v
        job_ids = []
        for n, v in datasets_.items():
            for m_name, m in measures_to_run.items():
                measure_job = MeasureJob(m_name, m[0], m[1], n, model, v, model_version, 
                        repo_info = {RepoInfoKey.NAME: model + '/jobs/measure/' + n + '/' + m[0],
                        RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
                measure_job.set_predecessor_jobs(predecessors)
                self.add(measure_job)
                self._job_runner.add(measure_job.repo_info[RepoInfoKey.NAME], measure_job.repo_info[RepoInfoKey.VERSION], self._user)
                job_ids.append((measure_job.repo_info[RepoInfoKey.NAME], measure_job.repo_info[RepoInfoKey.VERSION]))
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
        

