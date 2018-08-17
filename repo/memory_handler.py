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
        obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value] = len(
            tmp[name])-1
        self._name_to_category[name] = category
        if not category in self._categories.keys():
            self._categories[category] = set()
        self._categories[category].add(name)
        return obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def _get_object_list(self, name):
        """Return list of all versions of an object.

        :param name: name of object

        :return list of versions of object
        """
        if not name in self._name_to_category.keys():
            raise Exception('No object with name ' + name + ' in store.')
        category = self._name_to_category[name]
        if not category in self._store.keys():
            raise Exception('No object ' + name + ' in category ' + category)
        if not name in self._store[category].keys():
            raise Exception('No object ' + name + ' in category ' + category)
        return self._store[category][name]

    def get(self, name, version):
        """
        """
        tmp = self._get_object_list(name)
        if version >= len(tmp) or abs(version) > len(tmp):
            raise Exception('No valid versionnumber (' + str(version) +
                            ') for object ' + name)
        return tmp[version]

    def get_latest_version(self, name):
        """Return latest version number of an object.

        :param name: name of object
        """
        return len(self._get_object_list(name))-1

    def get_names(self, category):
        """Return object names of all object in a category.

        :param category: category name

        :return: list of object names belonging to the category
        """
        if not category in self._store.keys():
            raise Exception('Category ' + category + ' not in storage.')
        return [x for x in self._store[category].keys()]

    def get_history(self, name, fields=[], version_list=[]):
        """Return history of an object.

        This is an interface which has no effect in the case of the Memory-Handler but may enhance performance using other
        handlers. Therefore the ML Repo uses this interface and we have to implement it.

        :param name: name of object
        :param fields: the fields which will be taken from the repo. Since this is only for performance isues
        which have no effect in the MemoryStorage case, we ignore this argument.
        :param version_list: list of version whose history will be used. 
        """
        tmp = self._get_object_list(name)
        return [tmp[x] for x in version_list]


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
            self._store[name] = {version: numpy_dict}
        else:
            self._store[name][version] = numpy_dict

    def get(self, name, version):
        """
        """
        if not name in self._store.keys():
            raise Exception('No numpy data for object ' +
                            name + ' with version ' + str(version))
        if not version in self._store[name].keys():
            raise Exception('No numpy data for object ' +
                            name + ' with version ' + str(version))
        return self._store[name][version]
