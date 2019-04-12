import datetime
from copy import deepcopy
from numpy import concatenate
import pailab.ml_repo.repo_objects as repo_objects
import pailab.ml_repo.repo as repo
from pailab.ml_repo.repo_store import RepoStore, NumpyStore, _time_from_version
import logging
logger = logging.getLogger(__name__)


class RepoObjectMemoryStorage(RepoStore):
    """ The repo object memory storage. 
    This class is used to store repo object (excluding large objects) in the memory.
    The importance of the handler is mostly for testing purposes.
    """

    # region private

    def _is_in_versions(self, version, versions):
        """ Check whether the version exists between the dates of the two versions
        
        Arguments:
            version {str} -- the version to check whether it is here
            versions {list of str} -- a list of possible two versions, 
                                takes the date of the first version as a start date and the second as the end date
        
        Returns:
            bool -- returns true if the version is in between the two dates
        """

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
        """ Check if dictionary contains all modifications as given and if their version is in the respective version spec

        Arguments:
            obj {dict} -- object dictionary
            modifications {dict} -- [description]

        Returns:
            bool -- either True or False depending if given modification criteria is met
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
        """ Return list of all versions of an object.
        
        Arguments:
            name {str} -- name of the object
        
        Keyword Arguments:
            throw_error_not_exist {bool} -- true - throw error if not exists, else return [] (default: {True})
            throw_error_not_unique {bool} -- true - throw error if item is not unique, else return [] (default: {True})
        
        Raises:
            Exception -- raises an exception if no object with the name is found
            Exception -- raises an exception if no object in the category with the name is found
        
        Returns:
            list of str -- list of versions of object
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
        """ Initializes the handler
        """

        self._store={}
        self._name_to_category={}
        self._categories={}

    def _delete(self, name, version):
        """ Delete an object from the repo

        Arguments:
            name {str} -- the identifier of the object
            version {str} -- the version of the object to delete
        """

        category = self._name_to_category[name]
        objs = self._store[category][name]
        counter = -1
        for i in range(len(objs)):
            if objs[i]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] == version:
                counter = i
                break
        if counter >-1:
            del objs[counter]
            if len(objs) == 0:
                del self._store[category][name]
                del self._name_to_category[name]
        
    def _add(self, obj):
        """ Adds an object to the storage

        The objects version will be set to the latest version+1.

        Arguments:
            obj {RepoObject} -- the repo object to add to git
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
        """ Get a dictionary/list of dictionaries fulffilling the conditions.

            Returns a list of objects matching the name and whose
                -version is in the given list of versions
                -modifiers match the version number/are in the list of version numbers of the given modifiers
        Arguments:
            name {str} -- object id

        Keyword Arguments:
            versions {list, version_number, tuple} -- either a list of versions or a single version of the objects to be returned (default: {None}),
                    if None, the condition on version is ignored. If a tuple is given, this tuple defines a version intervall, i.e.
                    all versions between the first and last entry (both including) are returned. In addition FIRST_VERSION and LAST_VERSION can be used for versions to access
                    the last/first version.
            modifier_versions {dictionary} -- modifier ids together with version specs which are matched by the returned object (default: {None}).
            obj_fields {list of str or str} -- list of strings identifying the fields which will be returned in the dictionary,
                                                        if None, no fields are returned, if set to 'all', all fields will be returned  (default: {None})
            repo_info_fields {list of str or str} -- list of strings identifying the fields of the repo_info dict which will be returned in the dictionary,
                                                        if None, no fields are returned, if set to 'all', all fields will be returned (default: {None})
            throw_error_not_exist {bool} -- true - throw error if not exists, else return [] (default: {True})
            throw_error_not_unique {bool} -- true - throw error if item is not unique, else return [] (default: {True})

        Returns:
            RepoObject or list thereof -- The repo object
        """

        tmp=self._get_object_list(name, throw_error_not_exist, throw_error_not_unique)
        result=[]
        for x in tmp:
            if self._is_in_versions(x['repo_info'][repo_objects.RepoInfoKey.VERSION.value], versions):
                if self._is_in_modifications(x, modifier_versions):
                    result.append(deepcopy(x))
        return result

    def get_version(self, name, offset, throw_error_not_exist=True):
        """ Return the newest version up to offset versions
        
        Arguments:
            name {str} -- the identifier of the object
            offset {int} -- the offset
        
        Keyword Arguments:
            throw_error_not_exist {bool} -- true - throw error if not exists, else return [] (default: {True})
        
        Raises:
            Exception -- raises an error if the offset is higher than the number of versions available
            Exception -- raises an exception if the object does not exists and throw_error_not_exist == True
        
        Returns:
            str -- the version
        """

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
        """ Determine the latest version of the object
        
        Arguments:
            name {str} -- identifier of the object
        
        Keyword Arguments:
            throw_error_not_exist {bool} -- true - throw error if not exists, else return [] (default: {True})
        
        Raises:
            Exception -- Raises an exception if the object does not exists
        
        Returns:
            str -- the latest version string of the object
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
        """ Determine the first version of the object
        
        Arguments:
            name {str} -- identifier of the object
        
        Keyword Arguments:
            throw_error_not_exist {bool} -- true - throw error if not exists, else return [] (default: {True})
        
        Raises:
            Exception -- Raises an exception if the object does not exists
        
        Returns:
            str -- the first version string of the object
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
        """ Return the names of all objects belonging to the given category.
        
        Arguments:
            ml_obj_type {str} -- Value of MLObjectType-Enum specifying the category for which all names will be returned
        
        Returns:
            list of str -- a list of all objects in the category
        """

        if not category in self._store.keys():
            return []
            # raise Exception('Category ' + category + ' not in storage.')
        return [x for x in self._store[category].keys()]

    def replace(self, obj):
        """ Overwrite existing object without incrementing version
        
        Arguments:
            obj {RepoObject} --  repo object to be overwritten
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

    def _delete(self, name, version):
        """ Delete an object from the repo

        Arguments:
            name {str} -- the identifier of the object
            version {str} -- the version of the object to delete
        """

        if name in self._store.keys():
            if version in self._store[name].keys():
                del self._store[name][version]
                if len(self._store[name]) == 0:
                    del self._store[name]

    def add(self, name, version, numpy_dict):
        """ Add numpy data from an object to the storage.
        
        Arguments:
            name {str} -- identifier (as string) of object
            version {str} -- object version
            numpy_dict {numpy dict} -- numpy dictionary
        """

        logger.debug('Adding data for ' + name + ' an version ' + str(version))
        if not name in self._store.keys():
            self._store[name] = {version: numpy_dict}
        else:
            self._store[name][version] = numpy_dict

    def append(self, name, version_old, version_new, numpy_dict):
        """ appends an numpy dictionary to an existing object
        
        Arguments:
            name {str} -- identifier of the object
            version_old {str} -- the old version of the object
            version_new {str} -- the new version of the object
            numpy_dict {numpy dict} -- the numpy dictionary to append
        
        Raises:
            Exception -- raises an exception if the object does not exist
        """

        if not name in self._store.keys():
            logger.error("Cannot append data because " +
                         name + " does not exist.")
            raise Exception("Cannot append data because " +
                            name + " does not exist.")
        self._store[name][version_new] = {
            'previous': version_old,  'numpy_dict': numpy_dict}

    def get(self, name, version, from_index=0, to_index=None):
        """ get the numpy object for a name and a version, rows can be used
        
        Arguments:
            name {str} -- identifier of the object
            version {str} -- version of the object
        
        Keyword Arguments:
            from_index {int} -- the index from which the data should be taken (default: {0})
            to_index {int or None} -- the index to which the data is returned (None means till the end) (default: {None})
        
        Raises:
            Exception -- raises an exception if no object with the name exists
            Exception -- raises an exception if no object and with the version exists 
        
        Returns:
            numpy array -- the numpy object to return
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
                new_result[k] = concatenate((prev[k], v), axis=0)

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
        return new_result
