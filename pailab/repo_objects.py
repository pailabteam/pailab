import datetime

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
    HISTORY = 'history'
    CLASSNAME = 'classname'
    MODIFICATION_INFO = 'modification_info'
    DESCRIPTION = 'description'
    CATEGORY = 'category'
    BIG_OBJECTS = 'big_objects'
    COMMIT_MESSAGE = 'commit_message'
    AUTHOR = 'author'

class RepoInfo:
    """Contains all repo relevent information

    This class contains all repo relevant information such as version, name, descriptions.
    It must be a member of all objects which are handled by the repo.
    """

    def __init__(self, kwargs):
        for key in RepoInfoKey:
            setattr(self, key.value, None)
        self.set_fields(kwargs)
        if self[RepoInfoKey.VERSION] is None:
            self[RepoInfoKey.VERSION] = 0
        if self[RepoInfoKey.BIG_OBJECTS] is None:
            self[RepoInfoKey.BIG_OBJECTS] = set()
        if self[RepoInfoKey.MODIFICATION_INFO] is None:
            self[RepoInfoKey.MODIFICATION_INFO] = {}
            
    def set_fields(self, kwargs):
        """Set repo info fields from a dictionary

        Args:
            :param kwargs: dictionary
        """
        for k,v in kwargs.items():
            self[k] = v
        return 
        if not kwargs is None:
            for key in RepoInfoKey:
                if key.name in kwargs.keys():
                    setattr(self, key.value, kwargs[key.name])
                else:
                    if key.value in kwargs.keys():
                        setattr(self, key.value, kwargs[key.value])
                    else:
                        if key in kwargs.keys():
                            setattr(self, key.value, kwargs[key])

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
            setattr(repo_obj, x, repo_numpy_dict[x])

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
                repo_info = kwargs['repo_info']
                del kwargs['repo_info']
                if isinstance(repo_info, dict):
                    repo_info = RepoInfo(repo_info)
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


class RawData:
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
    @repo_object_init(['x_data', 'y_data'])
    def __init__(self, x_data, x_coord_names, y_data=None, y_coord_names=None):
        """Constructor

        Arguments:
            :param x_data: numpy object (array or matrix) containing x_data
            :param x_coord_names: list of x_data coordinate names
            :param y_data: numpy object (array or matrix) containing x_data
            :param y_coord_names: list of y_data coordinate names
        """
        x_data = RawData._cast_data_to_numpy(x_data)
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
        self.n_data = x_data.shape[0]  # pylint: disable=E1101
        self.y_data = None
        self.y_coord_names = None
        if not y_data is None:
            self.y_data = y_data
            self.y_coord_names = y_coord_names

    def __str__(self):
        return str(self.to_dict()) # pylint: disable=E1101
   
class Model:
    @repo_object_init()
    def __init__(self, preprocessing = None, preprocessing_param = None, eval_function = None, train_function = None, train_param = None, model_param = None):
        """Defines all model relevant information
        
        Keyword Arguments:
            preprocessing {string} -- name of preprocessing used (default: {None})
            preprocessing_param {string} -- name of preprocessing used (default: {None})
            eval_function {string} -- name of function object for model evaluation (default: {None})
            train_function {string} -- name of function object for model training (default: {None})
            train_param {string} -- name of training parameer object used for model training (default: {None})
            model_param {string} -- name of model parameter object used for creating the model, i.e. network architecture (default: {None})
            model {object} -- the object defining the model
        """
        self.preprocessing_function = preprocessing
        self.preprocessing_param = preprocessing_param
        self.eval_function = eval_function
        self.training_function = train_function
        self.training_param = train_param
        self.model_param = model_param
    
class Function:
    """Function
    """
    @repo_object_init()
    def __init__(self, module_name, function_name):
        self.module_name = module_name
        self.function_name = function_name

class CommitInfo:
    @repo_object_init()
    def __init__(self, message, author, objects):
        """Constructor
        
        Arguments:
            message {string} -- commit message
            author {string} -- author
            objects {dictionary} --  dictionary of names of committed objects and version numbers
        """
        self.message = message
        self.author = author
        self.objects = objects
        self.time = datetime.datetime.now()

    def __str__(self):
        """Get meaningfull string of object
        """
        result =  'time: ' + str(self.time) + ', author: ' + self.author + ', message: '+ self.message + ', objects: ' + str(self.objects)
        return result
class Label:
    """RepoObject to label a certain model
    """
    @repo_object_init()
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
    
    
class MeasureConfiguration:
    """RepoObject defining a configuration for all measures which shall be computed.
    """    
    L2 = 'l2'
    MSE = 'mse'
    MAX = 'max'
    R2 = 'r2'
    
    _ALL_COORDINATES = '_all'
    
    @repo_object_init()
    def __init__(self, measures):
        """Constructor
        
        Arguments:
            measures {list} -- list containing defintions for measures. The list contains either tuples of string (describing measure type) and a list of strings 
            (containing names of coordinates used computing the measure) or only strings (just the measure types where we assume that then all coordinates are used o compute the measure)

        """
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
            self.measures[MeasureConfiguration._create_name(measure)] = (measure,[MeasureConfiguration._ALL_COORDINATES])
    #region private
    @staticmethod
    def _create_name(measure_def):
        if isinstance(measure_def, tuple):
            name = measure_def[0]
            if isinstance(measure_def[1], list):
                name = name +'(' + str(list) + ')'
            if isinstance(measure_def[1], str):
                name = name + '(' + measure_def[1] + ')'
            return name
        if isinstance(measure_def, str):
            return measure_def
        raise Exception('Measure definition has wrong type.')
    
    #endregion
    
    def __str__(self):
        return str(self.to_dict()) # pylint: disable=E1101

class Measure:
    @repo_object_init()
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return str(self.to_dict()) # pylint: disable=E1101