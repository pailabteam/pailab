import abc
import datetime
import importlib

import numpy as np
from enum import Enum
from types import MethodType

import logging

logger = logging.getLogger(__name__)

def _get_attribute_dict(clazz, excluded=set()):
    """Return dictionary of non-static members and their values for an instance of  class

    Args:
        :param clazz: object instance of a crtain class
        :param excluded: names of member attributes which are excluded from dictionary
    """
    return {name: attr for name, attr in clazz.__dict__.items()
            if not name.startswith("__")
            and not callable(attr)
            and not type(attr) is staticmethod
            and not name in excluded
            }


class RepoInfoKey(Enum):
    """ Enums to describe all possible repository informations.
    """
    VERSION = 'version'
    NAME = 'name'
    CLASSNAME = 'classname'
    MODIFICATION_INFO = 'modification_info'
    DESCRIPTION = 'description'
    CATEGORY = 'category'
    BIG_OBJECTS = 'big_objects'
    COMMIT_MESSAGE = 'commit_message'
    AUTHOR = 'author'
    COMMIT_DATE = 'commit_date'

class RepoInfo:
    """Contains all repo relevent information

    This class contains all repo relevant information such as version, name, descriptions.
    It must be a member of all objects which are handled by the repo.
    """

    def __init__(self, kwargs = {}, name = None, version = None, category = None, modification_info = None):
        self.version = version
        self.name = name
        self.classname = None
        self.modification_info = {}
        if modification_info is not None:
            self.modification_info = modification_info
        self.description = None
        self.category = category
        self.big_objects = []
        self.commit_message = None
        self.author = None
        self.commit_date = None
        self.set_fields(kwargs)
            
    def set_fields(self, kwargs):
        """Set repo info fields from a dictionary

        Args:
            :param kwargs: dictionary
        """
        for k,v in kwargs.items():
            self[k] = v
    
    def __setitem__(self, field, value):
        """Set an item.

        Args:
            :param field: Either a string (RepoInfoKey.value or RepoInfoKey.name) or directly a RepoInfoKey enum
            :param value: value for the field
        """
        if isinstance(field, RepoInfoKey):
            setattr(self, field.value, value)
        if isinstance(field, str):
            for k in RepoInfoKey:
                if k.value == field:
                    setattr(self, field, value)
                    break
                else:
                    if k.name == field:
                        setattr(self, k.value, value)
                        break

    def __getitem__(self, field):
        """Get an item.

        Args:
            :param field: Either a string (RepoInfoKey.value or RepoInfoKey.name) or directly a RepoInfoKey enum

        Returns:
            Value of specified field.
        """
        if isinstance(field, RepoInfoKey):
            return getattr(self, field.value)
        if isinstance(field, str):
            for k in RepoInfoKey:
                if k.value == field:
                    return getattr(self, field)
                else:
                    if k.name == field:
                        return getattr(self, k.value)
        return None

    def get_dictionary(self):
        """Return repo info as dictionary
        """
        return _get_attribute_dict(self)

    def __str__(self):
        return str(self.get_dictionary())

def create_repo_obj_dict(obj):
    """ Create from a repo_object a dictionary with all values to handle the object within the repo

    Args:
        :param obj: repository object
    """
    result = _get_attribute_dict(obj, obj.repo_info[RepoInfoKey.BIG_OBJECTS])
    result['repo_info'] = obj.repo_info.get_dictionary()
    return result


class RepoObject:
   
    def __init__(self, repo_info):
        self.repo_info = RepoInfo()
        # if 'repo_info' in kwargs.keys():
        #     repo_info = kwargs['repo_info']
        #     del kwargs['repo_info']
        if isinstance(repo_info, dict):
            self.repo_info = RepoInfo(repo_info)
        else:
            self.repo_info = repo_info
    
        self.repo_info.classname = self.__class__.__module__ + \
        '.' + self.__class__.__name__

       #RepoObject._init_repo_info(**kwargs)
        #if 'repo_info' in kwargs.keys():
        #    del kwargs['repo_info']
        #self.from_dict(kwargs)

    @classmethod
    def _create_from_dict(cls, **kwargs):
        repo_info = RepoInfo()
        if 'repo_info' in kwargs.keys():
            repo_info = kwargs['repo_info']
            del kwargs['repo_info']
            if isinstance(repo_info, dict):
                repo_info = RepoInfo(repo_info)
            else:
                repo_info = repo_info
        if '_init_from_dict' in kwargs.keys():
            del kwargs['_init_from_dict']
        repo_info.classname = cls.__module__ + '.' + cls.__name__
        result = cls.__new__(cls)
        result.from_dict(kwargs)
        setattr(result, 'repo_info', repo_info)
        return result

    def to_dict(self):
        """ Return a data dictionary for a given repo_object without the big data objects

        Args:
         
            :returns: dictionary of data
        """
        excluded = [
            x for x in self.repo_info[RepoInfoKey.BIG_OBJECTS]]  # pylint: disable=E1101
        excluded.append('repo_info')
        return _get_attribute_dict(self, excluded)
    
    def from_dict(self, repo_obj_dict):  # pylint: disable=E0213
        """ set object from a dictionary

        Args:
            :param repo_object_dict: dictionary with the object data
        """
        for key, value in repo_obj_dict.items():
            self.__dict__[key] = value

    
    def numpy_to_dict(self):  # pylint: disable=E0213
        result = {}
        for x in self.repo_info[RepoInfoKey.BIG_OBJECTS]:  # pylint: disable=E1101
            result[x] = getattr(self, x)
        return result

    def numpy_from_dict(self, repo_numpy_dict):  # pylint: disable=E0213
        for x in self.repo_info[RepoInfoKey.BIG_OBJECTS]:  # pylint: disable=E1101
            if x in repo_numpy_dict.keys():
                setattr(self, x, repo_numpy_dict[x])
            else:
                setattr(self, x, None)

class repo_object_init:  # pylint: disable=too-few-public-methods
    """ Decorator class to modify a constructor so that the class can be used within the ml repository as repo_object.

    """
    def to_dict(repo_obj):  # pylint: disable=E0213
        """ Return a data dictionary for a given repo_object

        Args:
            :param repo_obj: A repo_object, i.e. object which provides the repo_object interface

            :returns: dictionary of data
        """
        excluded = [
            x for x in repo_obj.repo_info[RepoInfoKey.BIG_OBJECTS]]  # pylint: disable=E1101
        excluded.append('repo_info')
        return _get_attribute_dict(repo_obj, excluded)

    def from_dict(repo_obj, repo_obj_dict):  # pylint: disable=E0213
        """ set object from a dictionary

        Args:
            :param repo_object: repo_object which will be set from the dictionary
            :param repo_object_dict: dictionary with the object data
        """
        for key, value in repo_obj_dict.items():
            repo_obj.__dict__[key] = value

    def numpy_to_dict(repo_obj):  # pylint: disable=E0213
        result = {}
        for x in repo_obj.repo_info[RepoInfoKey.BIG_OBJECTS]:  # pylint: disable=E1101
            result[x] = getattr(repo_obj, x)
        return result

    def numpy_from_dict(repo_obj, repo_numpy_dict):  # pylint: disable=E0213
        for x in repo_obj.repo_info[RepoInfoKey.BIG_OBJECTS]:  # pylint: disable=E1101
            if x in repo_numpy_dict.keys():
                setattr(repo_obj, x, repo_numpy_dict[x])
            else:
                setattr(repo_obj, x, None)
    def __init__(self, big_objects=[]):
        """
        """
        self._big_objects = big_objects

    def init_repo_object(self, init_self, repo_info):  # pylint: disable=E0213
        repo_info[RepoInfoKey.CLASSNAME] = init_self.__class__.__module__ + \
            '.' + init_self.__class__.__name__
        if not self._big_objects is None:
            repo_info[RepoInfoKey.BIG_OBJECTS] = self._big_objects
        setattr(init_self, 'repo_info', repo_info)
        if not hasattr(init_self, 'to_dict'):
            setattr(init_self, 'to_dict', MethodType(
                repo_object_init.to_dict, init_self))
        if not hasattr(init_self, 'from_dict'):
            setattr(init_self, 'from_dict', MethodType(
                repo_object_init.from_dict, init_self))
        if not hasattr(init_self, 'numpy_from_dict'):
            setattr(init_self, 'numpy_from_dict',  MethodType(
                repo_object_init.numpy_from_dict, init_self))
        if not hasattr(init_self, 'numpy_to_dict'):
            setattr(init_self, 'numpy_to_dict',  MethodType(
                repo_object_init.numpy_to_dict, init_self))
        
                
    def __call__(self, f):
        def wrap(init_self, *args, **kwargs):
            repo_info = None
            if 'repo_info' in kwargs.keys():  # check if arguments contain a repo_info object
                repo_info_dict = kwargs['repo_info']
                del kwargs['repo_info']
                if isinstance(repo_info_dict, dict):
                    repo_info = RepoInfo(repo_info_dict)   
                else:
                    repo_info = RepoInfo()
                if not '_init_from_dict' in kwargs.keys():
                    f(init_self, *args, **kwargs)
                self.init_repo_object(init_self, repo_info)
                if '_init_from_dict' in kwargs.keys() and kwargs['_init_from_dict'] == True:
                    del kwargs['_init_from_dict']
                    init_self.from_dict(kwargs)
            else:
                f(init_self, *args, **kwargs)
        return wrap


def get_object_from_classname(classname, data):
    """ Returns an object instance for given classname and data dictionary.

    :param classname: Full classname as string including the modules, e.g. repo.Y if class Y is defined in module repo.
    :param data: Dictionary of data used to initialize the object instance.

    :returns:
        Instance object of class.
    """
    parts = classname.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    try:
        result = m._create_from_dict(**data)
        return result
    except:
        return m(_init_from_dict=True, **data)


def create_repo_obj(obj):
    """Create a repo_object from a dictionary.

    This function creates a repo object from a dictionary in a factory-like fashion.
    It uses the obj['repo_info']['classname'] within the dictionary and constructs the class
    using get_object_from_classname. It throws an exception if the dictionary does not contain an 'repo_info' key.

    Args:
        :param obj: dictionary containing all informations for a repo_object.

    """
    if not 'repo_info' in obj.keys():
        logger.error('Given dictionary is not a repo dictionary, '
                        'since repo_info key is missing.')
        raise Exception('Given dictionary is not a repo dictionary, '
                        'since repo_info key is missing.')
    repo_info = obj['repo_info']
    result = get_object_from_classname(repo_info['classname'], obj)
    return result

class RawData(RepoObject):
    """Class to store numpy data.
    
    """

    def _cast_data_to_numpy(data):  # pylint: disable=E0213
        if data is None:
            return data
        if isinstance(data, list):
            data = np.reshape(data, (len(data), 1))
        if len(data.shape) == 1:  # pylint: disable=E1101
            data = np.reshape(
                data, (data.shape[0], 1))  # pylint: disable=E1101
        return data

    """Class encapsulating raw data
    """
    def __init__(self, x_data, x_coord_names, y_data=None, y_coord_names=None, repo_info = RepoInfo()):
        """Constructor

        Arguments:
            :param x_data: numpy object (array or matrix) containing x_data
            :param x_coord_names: list of x_data coordinate names
            :param y_data: numpy object (array or matrix) containing x_data
            :param y_coord_names: list of y_data coordinate names
        """
        super(RawData, self).__init__(repo_info)
        self.repo_info.big_objects = ['x_data', 'y_data']
        if x_data is None:
            raise Exception("No x_data specified.")
        x_data = RawData._cast_data_to_numpy(x_data)
        self.n_data = x_data.shape[0]  # pylint: disable=E1101
        if x_data.shape[1] != len(x_coord_names):  # pylint: disable=E1101
            logger.error('Number of x-coordinates does not equal number of names for x-coordinates.')
            raise Exception(
                'Number of x-coordinates does not equal number of names for x-coordinates.')
        y_data = RawData._cast_data_to_numpy(y_data)
        if not y_data is None:
            if y_data.shape[1] != len(y_coord_names):  # pylint: disable=E1101
                logger.error('Nmber of y-coordinates does not equal number of names for y-coordinates.')
                raise Exception(
                    'Nmber of y-coordinates does not equal number of names for y-coordinates.')
            if y_data.shape[0] != x_data.shape[0]:  # pylint: disable=E1101
                raise Exception(
                    'Number of x-datapoints does not equal number of y-datapoints.')
        self.x_data = x_data
        self.x_coord_names = x_coord_names
        
        self.y_data = None
        self.y_coord_names = None
        if not y_data is None:
            self.y_data = y_data
            self.y_coord_names = y_coord_names
        

    def __str__(self):
        return str(self.to_dict()) # pylint: disable=E1101
   
class Model(RepoObject):
    def __init__(self, preprocessors = None, 
                eval_function = None, train_function = None, train_param = None, 
                model_param = None, repo_info = RepoInfo()):
        """Defines all model relevant information
        
        Keyword Arguments:
            preprocessors {string} -- list of preprocessor names (default: {None})
            eval_function {string} -- name of function object for model evaluation (default: {None})
            train_function {string} -- name of function object for model training (default: {None})
            train_param {string} -- name of training parameer object used for model training (default: {None})
            model_param {string} -- name of model parameter object used for creating the model, i.e. network architecture (default: {None})
            model {object} -- the object defining the model
        """
        super(Model, self).__init__(repo_info)
        self.preprocessors = preprocessors
        self.eval_function = eval_function
        self.training_function = train_function
        self.training_param = train_param
        self.model_param = model_param

class Preprocessor(RepoObject):
    def __init__(self, transforming_function, fitting_function = None,
                preprocessing_param = None, repo_info = RepoInfo()):
        """Defines all relevant information for the preprocessor
        
        Keyword Arguments:
            preprocessing {string} -- name of preprocessing used (default: {None})
            preprocessing_param {string} -- name of preprocessing used (default: {None})
            eval_function {string} -- name of function object for model evaluation (default: {None})
            train_function {string} -- name of function object for model training (default: {None})
            train_param {string} -- name of training parameer object used for model training (default: {None})
            model_param {string} -- name of model parameter object used for creating the model, i.e. network architecture (default: {None})
            model {object} -- the object defining the model
        """
        super(Preprocessor, self).__init__(repo_info)
        self.fitting_function = fitting_function
        self.transforming_function = transforming_function
        self.preprocessing_param = preprocessing_param
    
class Function(RepoObject):
    """Function
    """
    def __init__(self, f, repo_info = RepoInfo()):
        super(Function, self).__init__(repo_info)
        self._module_name = f.__module__
        self._function_name = f.__name__
        self._module_version = None

        tmp = importlib.import_module(self._module_name)
        self._module_version = 'None'
        if hasattr(tmp, '__version__'):
            self._module_version = str(tmp.__version__)
        else:
            logger.warning('Used module does not define a version which may lead to irreproducable results.')
        
    def create(self):
        """Returns the function object
        
        Returns:
            function object: the function object
        """
        tmp = importlib.import_module(self._module_name)
        if hasattr(tmp, '__version__'):
            if self._module_version != str(tmp.__version__):
                raise Exception('Module has version different to last version: ' + str(tmp.__version__) + ', orig version: ' 
                    + self._module_version +'. Either version add the function of newer module again or change module to original version.')
        return getattr(tmp, self._function_name)

    def get_version(self):
       return self._module_version

class Result(RepoObject):
    def __init__(self, data, big_data = None, repo_info = RepoInfo()):
        super(Result, self).__init__(repo_info)
        self.repo_info.category = 'RESULT'
        self.result = data
        self.big_data = big_data
        if self.big_data is not None:
            self.repo_info.big_objects = 'big_data'
        
    def numpy_to_dict(self):  # pylint: disable=E0213
        return self.big_data

    def numpy_from_dict(self, repo_numpy_dict):  # pylint: disable=E0213
        self.big_data = repo_numpy_dict

class CommitInfo(RepoObject):
    def __init__(self, message, author, objects, repo_info = RepoInfo()):
        """Constructor
        
        Arguments:
            message {string} -- commit message
            author {string} -- author
            objects {dictionary} --  dictionary of names of committed objects and version numbers
        """
        super(CommitInfo, self).__init__(repo_info)
        self.message = message
        self.author = author
        self.objects = objects
        self.time = datetime.datetime.now()

    def __str__(self):
        """Get meaningfull string of object
        """
        result =  'time: ' + str(self.time) + ', author: ' + self.author + ', message: '+ self.message + ', objects: ' + str(self.objects)
        return result

class Label(RepoObject):
    """RepoObject to label a certain model version
    """
    def __init__(self, model_name, model_version, repo_info = RepoInfo()):
        super(Label, self).__init__(repo_info)
        self.name = model_name
        self.version = model_version
    
    
class MeasureConfiguration(RepoObject):
    """RepoObject defining a configuration for all measures which shall be computed.
    """    
    L2 = 'l2'
    MSE = 'mse'
    MAX = 'max'
    R2 = 'r2'
    
    _ALL_COORDINATES = '_all'
    
    def __init__(self, measures, repo_info = RepoInfo()):
        """Constructor
        
        Arguments:
            measures {list} -- list containing defintions for measures. The list contains either tuples of string (describing measure type) and a list of strings 
            (containing names of coordinates used computing the measure) or only strings (just the measure types where we assume that then all coordinates are used o compute the measure)

        """
        super(MeasureConfiguration, self).__init__(repo_info)
        self.measures = {}
        for x in measures:
            if isinstance(x,tuple):
                self.measures[MeasureConfiguration._create_name(x)] = x
            else:
                if isinstance(x,str):
                    self.measures[MeasureConfiguration._create_name(x)] = (x, MeasureConfiguration._ALL_COORDINATES)
                else:
                    raise Exception('Given list of measures contains invalid element.')

    def add_measure(self, measure, coords=None):
        if not coords is None:
            self.measures[MeasureConfiguration._create_name((measure,coords))] = (measure,coords)
        else:
            self.measures[MeasureConfiguration._create_name(measure)] = (measure, MeasureConfiguration._ALL_COORDINATES)
    
    @staticmethod
    def get_name(measure_def):
        return MeasureConfiguration._create_name(measure_def)

    #region private
    @staticmethod
    def _create_name(measure_def):
        if isinstance(measure_def, tuple):
            name = measure_def[0]
            if isinstance(measure_def[1], list):
                separator = '_'
                name = name +'_' + separator.join(measure_def[1])
            if isinstance(measure_def[1], str):
                if not measure_def[1] == MeasureConfiguration._ALL_COORDINATES:
                    name = name + '_' + measure_def[1]
            return name
        if isinstance(measure_def, str):
            return measure_def
        raise Exception('Measure definition has wrong type.')
    
    #endregion
    
    def __str__(self):
        return str(self.to_dict()) # pylint: disable=E1101

class Measure(RepoObject):
    def __init__(self, value, repo_info = RepoInfo()):
        super(Measure, self).__init__(repo_info)
        self.value = value
    
    def __str__(self):
        return str(self.to_dict()) # pylint: disable=E1101

class DataSet(RepoObject):
    """Class used to access training or test data.

    This class refers to some RawData object and a start- and endindex 

    """
    def __init__(self, raw_data, start_index=0, 
        end_index=None, raw_data_version='last', repo_info = RepoInfo()):
        """Constructor

        Arguments:
            :argument raw_data: {string} -- id of raw_data the dataset refers to
            :argument start_index: {integer} -- index of first entry of the raw data used in the dataset
            :argument end_index: {integer} -- end_index of last entry of the raw data used in the dataset (if None, all including last element are used)
            :argument raw_data_version: {integer} -- version of RawData object the DataSet refers to (default is latest)
        """
        super(DataSet, self).__init__(repo_info)
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

    def get_pandas_data(self):
        """ Returns a pandas data frame
        
        Arguments:
        """

        if self.y_data is None:
            return pd.DataFrame(data = self.x_data, columns = self.x_coord_names)
        else:
            return pd.DataFrame(data = np.concatenate((self.x_data, self.y_data), axis=1), \
                columns = self.x_coord_names + self.y_coord_names)
    
    def __str__(self):
        return str(self.to_dict())

