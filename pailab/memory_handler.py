from copy import deepcopy
from numpy import concatenate
import pailab.repo_objects as repo_objects
import pailab.repo as repo
from pailab.repo_store import RepoStore, NumpyStore
import logging
logger = logging.getLogger(__name__)


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
            logger.error('No object with name ' + name + ' in store.')
            raise Exception('No object with name ' + name + ' in store.')
        category = self._name_to_category[name]
        if not category in self._store.keys():
            logger.error('No object ' + name + ' in category ' + category)
            raise Exception('No object ' + name + ' in category ' + category)
        if not name in self._store[category].keys():
            logger.error('No object ' + name + ' in category ' + category)
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
        logger.debug(obj['repo_info'][repo_objects.RepoInfoKey.NAME.value] +
                     ' added with version ' + str(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value]) + ', category: ' + category)

        return obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def _get(self, name, versions=None, modifier_versions=None, obj_fields=None,  repo_info_fields=None):
        tmp = self._get_object_list(name)
        result = []
        for x in tmp:
            if self._is_in_versions(x['repo_info'][repo_objects.RepoInfoKey.NAME.value], x['repo_info'][repo_objects.RepoInfoKey.VERSION.value], versions):
                if self._is_in_modifications(x, modifier_versions):
                    result.append(deepcopy(x))
        return result

    def get_version(self, name, offset):
        tmp = self._get_object_list(name)
        if (offset < 0 and abs(offset)>len(tmp)) or offset >= len(tmp):
            raise Exception('Offset larger then number of versions.')
        if len(tmp) == 0:
            raise Exception('No object with name ' +
                            name + ' exists in storage')
        return self._get_object_list(name)[offset]['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def get_latest_version(self, name):
        """Return latest version number of an object.

        :param name: name of object
        """
        tmp = self._get_object_list(name)
        if len(tmp) == 0:
            raise Exception('No object with name ' +
                            name + ' exists in storage')
        return tmp[-1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def get_first_version(self, name):
        """Return latest version number of an object.

        :param name: name of object
        """
        tmp = self._get_object_list(name)
        if len(tmp) == 0:
            raise Exception('No object with name ' +
                            name + ' exists in storage')
        return self._get_object_list(name)[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def get_names(self, category):
        """Return object names of all object in a category.

        :param category: category name

        :return: list of object names belonging to the category
        """
        if not category in self._store.keys():
            return []
            # raise Exception('Category ' + category + ' not in storage.')
        return [x for x in self._store[category].keys()]

    def replace(self, obj):
        """Overwrite existing object without incrementing version

        Args:
            obj (RepoObject): repo object to be overwritten
        """
        category = obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value]
        if not isinstance(category, str):
            category = category.value
        name = obj['repo_info'][repo_objects.RepoInfoKey.NAME.value]
        if not category in self._store.keys():
            self._store[category] = {}
        tmp = self._store[category]
        if not name in tmp.keys():
            logger.error('Cannot replace object: No object with name ' +
                         name + ' and category ' + category + ' exists.')
            raise Exception('Cannot replace object: No object with name ' +
                            name + ' and category ' + category + ' exists.')
        version = int(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value])
        if version >= len(tmp[name]):
            logger.error('Cannot replace objct: The version ' + str(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value])
                         + ' does not exist in storage.')
            raise Exception('Cannot replace objct: The version ' + str(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value])
                            + ' does not exist in storage.')
        tmp[name][version] = obj


class NumpyMemoryStorage(NumpyStore):
    def __init__(self):
        self._store = {}

    def add(self, name, version, numpy_dict):
        """ Add numpy data from an object to the storage.

        :param name: Name (as string) of object
        :param version: object version
        :param numpy_dict: numpy dictionary

        """
        logger.debug('Adding data for ' + name + ' an version ' + str(version))
        if not name in self._store.keys():
            self._store[name] = {version: numpy_dict}
        else:
            self._store[name][version] = numpy_dict

    def append(self, name, version_old, version_new, numpy_dict):
        if not name in self._store.keys():
            logger.error("Cannot append data because " +
                         name + " does not exist.")
            raise Exception("Cannot append data because " +
                            name + " does not exist.")
        self._store[name][version_new] = {
            'previous': version_old,  'numpy_dict': numpy_dict}

    def get(self, name, version, from_index=0, to_index=None):
        """
        """
        logger.debug('Get data for ' + name + ' and version ' + str(version))
        if not name in self._store.keys():
            raise Exception('No numpy data for object ' +
                            name + ' with version ' + str(version))
        if not version in self._store[name].keys():
            raise Exception('No numpy data for object ' +
                            name + ' with version ' + str(version))
        result = self._store[name][version]
        new_result = result
        if 'previous' in result.keys():
            new_result = {}
            prev = self.get(name, result['previous'])
            for k, v in result['numpy_dict'].items():
                new_result[k] = concatenate((v, prev[k]), axis=0)

        # adjust for given start and end indices
        if from_index != 0 or (to_index is not None):
            tmp = {}
            logger.debug('Slice data from_index: ' +
                         str(from_index) + ', to_index: ' + str(to_index))
            if from_index != 0 and (to_index is not None):
                for k, v in new_result.items():
                    tmp[k] = v[from_index:to_index, :]
                    logger.debug(k + ' added with new shape ' + str(v.shape))
            else:
                if from_index != 0:
                    for k, v in new_result.items():
                        tmp[k] = v[from_index:-1, :]
                if to_index is not None:
                    for k, v in new_result.items():
                        tmp[k] = v[0:to_index, :]
            return tmp
        return result
