import repo.repo_objects as repo_objects
import repo.repo as repo

class RepoObjectMemoryStorage:
    def __init__(self):
        self._store = {}
        self._name_to_category = {}
        self._categories = {}

    def add(self, obj):
        """ Add an object of given category to the storage.

        The objects version will be set to the latest version+1.
    
        :param obj: The object added.

        :return: version number of added object
        """
        category = obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value]
        name = obj['repo_info'][repo_objects.RepoInfoKey.NAME.value]
        if not category in self._store.keys():
            self._store[category] = {}
        tmp = self._store[category]
        if not name in tmp.keys():
            tmp[name] = [obj]
        else:
            tmp[name].append(obj)
        obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value] = len(tmp[name])-1
        self._name_to_category[name] = category
        if not category in self._categories.keys():
            self._categories[category] = set()
        self._categories[category].add(name)
        return obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def get(self, name, version=-1):
        """
        """
        category = self._name_to_category[name]
        if not category in self._store.keys():
            raise Exception('No object ' + name + ' in category ' + category)
        if not name in self._store[category].keys():
            raise Exception('No object ' + name + ' in category ' + category)
        tmp = self._store[category][name]
        if version >= len(tmp):
            raise Exception('No valid versionnumber (' + str(version) + ') for object ' + name + ' in category ' + category)
        return self._store[category][name][version]

class NumpyMemoryStorage:
    def __init__(self):
        self._store = {}

    def add(self, name, version, numpy_dict):
        """ Add numpy data from an object to the storage.

        :param name: Name (as string) of object
        :param version: object version
        :param numpy_dict: numpy dictionary

        """
        if not name in self._store.keys():
            self._store[name] = {version : numpy_dict}
        else:
            self._store[name][version] = numpy_dict

    def get(self, name, version):
        """
        """
        if not name in self._store.keys():
            raise Exception('No numpy data for object ' + name + ' with version ' + str(version))
        if not version in self._store[name].keys():
            raise Exception('No numpy data for object ' + name + ' with version ' + str(version))
        return self._store[name][version]
