import datetime
from copy import deepcopy
from numpy import concatenate
import pailab.repo_objects as repo_objects
import pailab.repo as repo
from pailab.repo_store import RepoStore, NumpyStore, _time_from_version
import logging
logger = logging.getLogger(__name__)


class RepoObjectMemoryStorage(RepoStore):
    # region private

    def _is_in_versions(self, version, versions):
        if version is None:
            return True
        if versions is None:
            return True
        v = versions
        if isinstance(v, list):
            return version in v
        if isinstance(v, tuple):
            if v[0] is None:
                start_time = datetime.datetime(1980, 1, 1, 0, 0)
            else:
                start_time = _time_from_version(v[0])
            if v[1] is None:
                end_time = datetime.datetime(2300, 1, 1, 0, 0)
            else:
                end_time = _time_from_version(v[1])
            time = _time_from_version(version)
            return (time >= start_time) and (time <= end_time)
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
            result = result and self._is_in_versions(modification_info[k], v)
            if result == False:
                return result
        return result

    def _get_object_list(self, name, throw_error_not_exist=True, throw_error_not_unique=True):
        """Return list of all versions of an object.

        :param name: name of object

        :return list of versions of object
        """
        if not name in self._name_to_category.keys():
            if throw_error_not_exist:
                logger.error('No object with name ' + name + ' in store.')
                raise Exception('No object with name ' + name + ' in store.')
            else:
                return []
        category=self._name_to_category[name]
        if not category in self._store.keys():
            if throw_error_not_exist:
                logger.error('No object ' + name + ' in category ' + category)
                raise Exception('No object ' + name + ' in category ' + category)
            else:
                return []
        if not name in self._store[category].keys():
            if throw_error_not_exist:
                logger.error('No object ' + name + ' in category ' + category)
                raise Exception('No object ' + name + ' in category ' + category)
            else:
                return []
        return self._store[category][name]
# endregion

    def __init__(self):
        self._store={}
        self._name_to_category={}
        self._categories={}

    def _add(self, obj):
        """ Add an object of given category to the storage.

        The objects version will be set to the latest version+1.

        :param obj: The object added.

        :return: version number of added object
        """
        category=obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value]
        if not isinstance(category, str):
            category=category.value
        name=obj['repo_info'][repo_objects.RepoInfoKey.NAME.value]

        if not category in self._store.keys():
            self._store[category]={}
        tmp=self._store[category]
        if not name in tmp.keys():
            tmp[name]=[obj]
        else:
            tmp[name].append(obj)
        self._name_to_category[name]=category
        if not category in self._categories.keys():
            self._categories[category]=set()
        self._categories[category].add(name)
        logger.debug(obj['repo_info'][repo_objects.RepoInfoKey.NAME.value] +
                     ' added with version ' + str(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value]) + ', category: ' + category)

    def _get(self, name, versions = None, modifier_versions = None, obj_fields = None,  repo_info_fields = None,
             throw_error_not_exist=True, throw_error_not_unique=True):
        tmp=self._get_object_list(name, throw_error_not_exist, throw_error_not_unique)
        result=[]
        for x in tmp:
            if self._is_in_versions(x['repo_info'][repo_objects.RepoInfoKey.VERSION.value], versions):
                if self._is_in_modifications(x, modifier_versions):
                    result.append(deepcopy(x))
        return result

    def get_version(self, name, offset, throw_error_not_exist=True):
        tmp=self._get_object_list(name, throw_error_not_exist)
        if (offset < 0 and abs(offset) > len(tmp)) or offset >= len(tmp):
            if throw_error_not_exist:
                raise Exception('Offset larger then number of versions.')
            else:
                return []
        if len(tmp) == 0:
            if throw_error_not_exist:
                raise Exception('No object with name ' +
                                name + ' exists in storage')
            else:
                return []
        return self._get_object_list(name)[offset]['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def get_latest_version(self, name, throw_error_not_exist=True):
        """Return latest version number of an object.

        :param name: name of object
        """
        tmp=self._get_object_list(name, throw_error_not_exist)
        if len(tmp) == 0:
            if throw_error_not_exist:
                raise Exception('No object with name ' +
                                name + ' exists in storage')
            else:
                return []
        return tmp[-1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

    def get_first_version(self, name, throw_error_not_exist=True):
        """Return latest version number of an object.

        :param name: name of object
        """
        tmp=self._get_object_list(name, throw_error_not_exist)
        if len(tmp) == 0:
            if throw_error_not_exist:
                raise Exception('No object with name ' +
                                name + ' exists in storage')
            else:
                return []
        return tmp[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value]

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
        category=obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value]
        if not isinstance(category, str):
            category=category.value
        name=obj['repo_info'][repo_objects.RepoInfoKey.NAME.value]
        if not category in self._store.keys():
            self._store[category]={}
        tmp=self._store[category]
        if not name in tmp.keys():
            logger.error('Cannot replace object: No object with name ' +
                         name + ' and category ' + category + ' exists.')
            raise Exception('Cannot replace object: No object with name ' +
                            name + ' and category ' + category + ' exists.')
        all_obj = tmp[name]
        version = obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value]
        for i, x in enumerate(all_obj):
            if version == x['repo_info'][repo_objects.RepoInfoKey.VERSION.value]:
                all_obj[i] = obj
                return

        logger.error('Cannot replace object: The version ' + str(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value])
                     + ' does not exist in storage.')
        raise Exception('Cannot replace object: The version ' + str(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value])
                        + ' does not exist in storage.')


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
