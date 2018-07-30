import logging

logger = logging.getLogger('repo')

class repo_object_init:
    """ Decorator class to modify a constructor such that the respective class ban be used within the ml repository.


    """
    def init_from_class_data(init_self, **kwargs):
        repo_info = {'id' : '', 'version': ''}
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
                for k,v in data_dict.items():
                    setattr(init_self, k, v)
                applied = True
        
        setattr(init_self, 'repo_info', repo_info)  
        return applied
        
    def __call__(self,f):
        def wrap(init_self,*args,**kwargs):
            if not repo_object_init.init_from_class_data(init_self, **kwargs):
                if 'repo_id' in kwargs.keys():
                    del kwargs['repo_id']
                f(init_self,*args,**kwargs)
        return wrap

def get_object_from_classname(classname, data):
    """ Returns for a given classname and a data dictionary an instance o the respective class initialized from the given dictionary.
    
    :param classname: Full classname as string including the modules, e.g. repo.Y if class Y is defined in module repo.
    :param data: Dictionary of data used to initialize the object instance.

    :returns: 
        Instance object of class.
    """
    parts = class_name.split('.')
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
    for k,v in obj_dict.items():
        if not k=='repo_info':
            members[k] = v
        else:
            result[k] = v
    return result

def _create_object_from_repo_object(obj):
    if not 'repo_info' in obj.keys():
        raise Exception('Given dictionary is not a repo dictionary, since repo_info key is missing.')
    if 'class' in obj['repo_info'].keys():
        result = get_object_from_classname(obj['repo_info']['class'], obj)
    return result

class RepoObject:
    def __init__(self, obj):
        if isinstance(obj, RepoObject):
            self._data = obj._data 
        else:
            self._data = _create_repo_object(obj)

    def get_class(self):
        return _create_object_from_repo_object(self._data)

    def __get_version(self):
        return self._data['repo_info']['version']

    def __set_version(self, v):
        self._data['repo_info']['version'] = v

    version = property (__get_version, __set_version)

    def __set_id(self, id):
        self._data['repo_info']['id'] = id
