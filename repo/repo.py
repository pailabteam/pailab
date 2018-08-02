"""
Machine learning repository
"""
import logging

LOGGER = logging.getLogger('repo')



def _create_repo_object_info(version = 0, id = None, message = '', modifier = '', repo_type = '', description = ''):
    """ Create dictionary with all informations of a repo object
    """
    result = {'version': 0, 'id': id, 'class': '', 'message': message,
                'modifier': '', 'repo_type': repo_type }
    return result

class repo_object_init: # pylint: disable=too-few-public-methods
    """ Decorator class to modify a constructor so that the class ban be used within the ml repository.


    """
    def init_from_class_data(init_self, **kwargs):# pylint: disable=E0213
        repo_info = _create_repo_object_info()
        repo_info['class'] = init_self.__class__.__module__ + '.' + init_self.__class__.__name__
        applied = False
        if kwargs is not None:
            if 'repo_info' in kwargs.keys():
                repo_info = kwargs['repo_info']
                del  kwargs['repo_info']

            if 'repo_id' in kwargs.keys():
                repo_info['id'] = kwargs['repo_id']
                del kwargs['repo_id']
    
            if 'members' in kwargs.keys():
                data_dict = kwargs['members']
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


def _create_repo_object(obj):
    """
    """
    def get_dictionary(repo_object):
        if isinstance(repo_object, dict):
            return repo_object
        else:
            return repo_object.__dict__

    obj_dict = get_dictionary(obj)
    result = {'members':{} }
    members = result['members']
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
    if 'class' in obj['repo_info'].keys():
        result = get_object_from_classname(obj['repo_info']['class'], obj)
    return result



class RepoObject:
    def __init__(self, obj, repo_id = None, version = 0):
        if repo_id is None:
            raise Exception('The repo_id is missing in the argument list.')
        if isinstance(obj, RepoObject):
            self._data = obj._data 
        else:
            self._data = _create_repo_object(obj)
        self.id = repo_id
        self.version = 0

    def get_class(self):
        """Return the underlying used defined object. 
        """
        return _create_object_from_repo_object(self._data)

    def __get_version(self):
        return self._data['repo_info']['version']

    def __set_version(self, v):
        self._data['repo_info']['version'] = v

    version = property (__get_version, __set_version)

    def __set_id(self, id):
        self._data['repo_info']['id'] = id

    def __get_id(self):
        return self._data['repo_info']['id']

    id = property(__get_id, __set_id)


class ObjectRepository:
    pass

class DataStorage:
    pass

class CodeRepository:
    pass

class MLRepository:
    def __init__(repo_object_storage, data_storage, code_repo):
        pass

    def add_raw_data(self, data):
        pass
    def get_raw_data(self, data):
        pass
    def add_training_data(self, data_set):
        pass
    def get_training_data(self, full):
        pass
    def add_preprocessing(self, prep_function, prep_param):
        pass
    

