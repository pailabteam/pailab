from copy import deepcopy
import repo.repo_objects as repo_objects
import repo.repo as repo
from repo.repo_store import RepoStore


class RepoObjectMemoryStorage(RepoStore):
    # region private
    def _replace_version_keyword(self, name, versionset):
        def replace_keyword_by_version(n):
            if n is None:
                return None
            if isinstance(n, str):
                if n == RepoStore.FIRST_VERSION:
                    return 0
                if n == RepoStore.LAST_VERSION:
                    return self.get_latest_version(name)
            return n
        if isinstance(versionset, tuple):
            return (replace_keyword_by_version(versionset[0]), replace_keyword_by_version(versionset[1]))
        v = deepcopy(versionset)
        v = replace_keyword_by_version(v)
        if isinstance(v, list):
            for i in range(len(v)):
                v[i] = replace_keyword_by_version(v[i])
        return v

    def _is_in_versions(self, name, version, versions):
        if versions is None:
            return True
        v = self._replace_version_keyword(name, versions)
        if isinstance(v, list):
            return version in v
        if isinstance(v, tuple):
            return (version >= v[0]) and (version <= v[1])
        return version == v

    def _is_in_modifications(self, obj, modifications):
        """Check if dictionary contains all modifications as given and if their version is in the respective version spec

        Arguments:
            obj {dict} -- object dictionary
            modifications {dict} -- [description]

        Returns:
            [bool] -- either True or False depending if given modification criteria is met
        """
        if modifications is None:
            return True
        modification_info = obj['repo_info'][repo_objects.RepoInfoKey.MODIFICATION_INFO.value]
        result = True
        for k, v in modifications.items():
            if not k in modification_info.keys():
                return False
            result = result and self._is_in_versions(
                k, modification_info[k], v)
            if result == False:
                return result
        return result

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
# endregion

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
        if not isinstance(category, str):
            category = category.value
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

    def get(self, name, versions=None, modifier_versions=None, obj_fields=None,  repo_info_fields=None):
        tmp = self._get_object_list(name)
        result = []
        for x in tmp:
            if self._is_in_versions(x['repo_info'][repo_objects.RepoInfoKey.NAME.value], x['repo_info'][repo_objects.RepoInfoKey.VERSION.value], versions):
                if self._is_in_modifications(x, modifier_versions):
                    result.append(x)
        return result

    def get_version_number(self, name, offset):
        if offset < 0:
            return len(self._get_object_list(name)) + offset
        return offset

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
            return []
            #raise Exception('Category ' + category + ' not in storage.')
        return [x for x in self._store[category].keys()]


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
