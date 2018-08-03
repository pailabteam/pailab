from enum import Enum, auto
from functools import partial
import copy


def _get_attribute_dict(clazz):
    """Return dictionary of non-static members and their values for an instance of  class

    Args:
        :param clazz: object instance of a crtain class
    """
    return { name: attr for name, attr in clazz.__dict__.items()
            if not name.startswith("__") 
            and not callable(attr)
            and not type(attr) is staticmethod }


class RepoInfoKey(Enum):
    VERSION = 'version'
    NAME = 'name'
    HISTORY = 'history'
    CLASSNAME = 'classname'
    MODIFICATION_INFO = 'modification_info'
    DESCRIPTION = 'description'
    CATEGORY = 'category'
    ML_TYPE = 'ml_type'
    BIG_OBJECTS = 'big_objects'



class RepoInfo:
    
    def __init__(self, **kwargs):
        for key in RepoInfoKey:
            setattr(self, key.value, None)
        self.set_fields(kwargs)

    def set_fields(self, kwargs):
        """Set repo info fields from a dictionary

        Args:
            :param kwargs: dictionary 
        """
        if not kwargs is None:
            for key in RepoInfoKey:
                if key.name in kwargs.keys():
                    setattr(self, key.value, kwargs[key.name])
                else:
                    if  key.value in kwargs.keys():
                        setattr(self, key.value, kwargs[key.value])
                    else:
                        if key in kwargs.keys():
                            setattr(self, key.value, kwargs[key])
                    
              
    def __setitem__(self, field, value):
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

class repo_object_init: # pylint: disable=too-few-public-methods
    """ Decorator class to modify a constructor so that the class can be used within the ml repository.

    """
    def init_from_class_data(init_self, **kwargs):# pylint: disable=E0213
        repo_info = RepoInfo()
        repo_info.classname =  init_self.__class__.__module__ + '.' + init_self.__class__.__name__
        applied = False
        if kwargs is not None:
            if 'repo_info' in kwargs.keys():
                repo_info = kwargs['repo_info']
                del  kwargs['repo_info']

            if 'repo_id' in kwargs.keys():
                repo_info['id'] = kwargs['repo_id']
                del kwargs['repo_id']
    
            if 'obj' in kwargs.keys():
                data_dict = kwargs['obj']
                for k, v in data_dict.items():
                    setattr(init_self, k, v)
                applied = True

        setattr(init_self, 'repo_info', repo_info)  
        return applied

    def __call__(self, f):
        def wrap(init_self, *args, **kwargs):
            if not repo_object_init.init_from_class_data(init_self, **kwargs):
                if 'repo_id' in kwargs.keys():
                    del kwargs['repo_id']
                f(init_self,*args,**kwargs)
        return wrap

def get_object_from_classname(classname, data):
    """ Returns an object instance for for given classname and data dictionary.
    
    :param classname: Full classname as string including the modules, e.g. repo.Y if class Y is defined in module repo.
    :param data: Dictionary of data used to initialize the object instance.

    :returns: 
        Instance object of class.
    """
    parts = classname.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m(**data)

def _create_repo_dictionary(obj):
    """
    """
    def get_members(repo_object):
        if isinstance(repo_object, dict):
            return repo_object
        else:
            return _get_attribute_dict(repo_object)

    obj_dict = get_members(obj)
    result = {'obj':{} }
    members = result['obj']
    for k, v in obj_dict.items():
        if not k == 'repo_info':
            members[k] = v
        else:
            result[k] = v
    return result

def _create_object_from_repo_object(obj):
    if not 'repo_info' in obj.keys():
        raise Exception('Given dictionary is not a repo dictionary, '\
            'since repo_info key is missing.')
    repo_info = obj['repo_info']
    result = get_object_from_classname(repo_info.classname, obj)
    return result




class RepoObject:

    def __init__(self,  obj_data, repo_info = RepoInfo()):
        self._repo_info =  RepoInfo()
        self._set_repo_info(repo_info)
        self._obj = obj_data
        #self._obj_dict =

    def _set_repo_info(self, repo_info):
        if isinstance(repo_info, RepoInfo):
            self._repo_info = repo_info
        else:
             if isinstance(repo_info, dict):
                self._repo_info.set_fields(repo_info)

    def _get_repo_info(self):
        return self._repo_info

    repo_info = property(_get_repo_info, _set_repo_info)
