import uuid
import datetime

import abc
from pailab.ml_repo.repo_objects import RepoInfoKey  # pylint: disable=E0401


def _version_str():
    return str(uuid.uuid1())


def _time_from_version(v):
    """Return the time included in uuid

    Args:
        :param v (str): string representin the uuid

    Returns:
        datetime: the datetime
    """
    if isinstance(v, str):
        tmp = uuid.UUID(v)
        return datetime.datetime(1582, 10, 15) + datetime.timedelta(microseconds=tmp.time / 10)
    return datetime.datetime(1582, 10, 15) + datetime.timedelta(microseconds=v.time / 10)


FIRST_VERSION = 'first'
LAST_VERSION = 'last'


class RepoScriptStore(abc.ABC):
    @abc.abstractmethod
    def add(self, script_file):
        """Add a script to the storage.


        Arguments:
            script_file {string} -- file (incluing path) of script

        """
        pass

    @abc.abstractmethod
    def get(self, name, versions=None):
        """

        """
        pass


class RepoStore(abc.ABC):

    FIRST_VERSION = FIRST_VERSION
    LAST_VERSION = LAST_VERSION

    """Abstract base class for all storages which can be used in the ML repository

    """

    def _replace_version_placeholder(self, name, versions, throw_error_not_exist=True):
        def replace_version(name, version, throw_error_not_exist):
            if version == FIRST_VERSION:
                return self.get_first_version(name, throw_error_not_exist)
            if version == LAST_VERSION:
                return self.get_latest_version(name, throw_error_not_exist)
            if isinstance(version, int):
                return self.get_version(name, version, throw_error_not_exist)
            return version
        if isinstance(versions, str):
            versions = replace_version(name, versions, throw_error_not_exist)
        if isinstance(versions, tuple):
            versions = (replace_version(name, versions[0], throw_error_not_exist),
                        replace_version(name, versions[1], throw_error_not_exist))
        if isinstance(versions, list):
            versions = [replace_version(
                name, v, throw_error_not_exist) for v in versions]
        return versions

    @abc.abstractmethod
    def _add(self, obj):
        """Add an object to the storage.

        This method is called internally by the method add that is setting a version if there has not version already been set.

        Arguments:
            obj {RepoObject} -- repository object

        Raises:
            Exception if an object with same name already exists.
        """
        pass

    def add(self, obj):
        """Add an object to the storage.


        Arguments:
            obj {RepoObject|list(RepoObject)} -- repository object or list o repository objects

        Raises:
            Exception if an object with same name already exists.
        """
        if obj['repo_info'][RepoInfoKey.VERSION.value] is None:
            obj['repo_info'][RepoInfoKey.VERSION.value] = _version_str()
        self._add(obj)
        return obj['repo_info'][RepoInfoKey.VERSION.value]

    @abc.abstractmethod
    def replace(self, obj):
        """Overwrite existing object without incrementing version

        Args:
            obj (RepoObject): repo object to be overwritten
        """
        pass

    def _get_by_modification_info(self, modifier_name, modifier_version, object_types = []):
        """Return list of all objects which were modified by a given object.

        This method may be overwritten by subclasses to enhance performance.
        
        Args:
            modifier_name (str): name of object which modified th searched objects
            modifier_version (str): version of object which modified th searched objects
            object_types (list(str)): lit of strings defining the object types
        Returns:
            [list]: list of objects, empty if no such objects exist
        """

        result = []
        modifier = {modifier_name: modifier_version}
        for category in object_types:
            names = self.get_names(category)
            for n in names:
                objs = self.get(n, modifier_versions=modifier, throw_error_not_exist=False, throw_error_not_unique=False)
                if isinstance(objs, list):
                    result.extend(objs)
                else:
                    result.append(objs)
        return result

    def get(self, name, versions=None, modifier_versions=None, obj_fields=None,  repo_info_fields=None,
            throw_error_not_exist=True, throw_error_not_unique=True):
        versions = self._replace_version_placeholder(
            name, versions, throw_error_not_exist)
        if modifier_versions is not None:
            for k, v in modifier_versions.items():
                modifier_versions[k] = self._replace_version_placeholder(
                    k, v, throw_error_not_exist)
        return self._get(name, versions, modifier_versions,
                         obj_fields, repo_info_fields,
                         throw_error_not_exist, throw_error_not_unique)

    def push(self):
        """Push changes to an eternal repo.
        """

        pass

    def pull(self):
        """Pull changes from an external repo
        """

        pass

    @abc.abstractmethod
    def _delete(self, name, version):
        """Delete an object with a predefined version

        Args:
            name (str): name of object
            version (str): object version
        """
        
        pass

    @abc.abstractmethod
    def _get(self, name, versions=None, modifier_versions=None, obj_fields=None,  repo_info_fields=None,
             throw_error_not_exist=True, throw_error_not_unique=True):
        """Get a dictionary/list of dictionaries fulffilling the conditions.

            Returns a list of objects matching the name and whose
                -version is in the given list of versions
                -modifiers match the version number/are in the list of version numbers of the given modifiers
        Arguments:
            name {string} -- object id

        Keyword Arguments:
            versions {list, version_number, string, tuple} -- either a list of versions or a single version of the objects to be returned (default: {None}),
                    if None, the condition on version is ignored. If a tuple is given, this tuple defines a version intervall, i.e.
                    all versions between the first and last entry (both including) are returned. In addition FIRST_VERSION and LAST_VERSION can be used for versions to access
                    the last/first version.
            modifier_versions {dictionary} -- modifier ids together with version specs which are matched by the returned object (default: {None}).
            obj_fields {list of strings or string} -- list of strings identifying the fields which will be returned in the dictionary,
                                                        if None, no fields are returned, if set to 'all', all fields will be returned  (default: {None})
            repo_info_fields {list of strings or string} -- list of strings identifying the fields of the repo_info dict which will be returned in the dictionary,
                                                        if None, no fields are returned, if set to 'all', all fields will be returned (default: {None})
        Returns:
            List of all objects fullfilling the conditions.
        """
        pass

    @abc.abstractmethod
    def get_names(self, ml_obj_type):
        """Return the names of all objects belonging to the given category.

        Arguments:
            ml_obj_type {string} -- Value of MLObjectType-Enum specifying the category for which all names will be returned
        """
        pass

    @abc.abstractmethod
    def get_version(self, name, offset, throw_error_not_exist=True):
        """Return versionnumber for the given offset

        If offset >= 0 it returns the version number of the offset version, if <0 it returns according to the
        python list logic the version number of the (offset-1)-last version

        Arguments:
            name {string} -- name of object
            offset {int} -- offset
        """
        pass

    @abc.abstractmethod
    def get_latest_version(self, name, throw_error_not_exist=True):
        """Return latest version number of object in the storage

        Arguments:
            :param name {string} -- object name

        Returns:
            version number -- latest version number
        """
        return self.get(name, versions=RepoStore.LAST_VERSION, repo_info_fields=[RepoInfoKey.VERSION])[0]['repo_info'][RepoInfoKey.VERSION.value]

    @abc.abstractmethod
    def get_first_version(self, name, throw_error_not_exist=True):
        """Return version number of first (in a temporal sense) object in storage

        Args:
            name (str): object name for which the version is returned
        """
        raise NotImplementedError()

    def object_exists(self, name, version=LAST_VERSION):
        """Returns True if an object with the given name and version exists.

        Arguments:
            name {string} -- object name

        Keyword Arguments:
            version {version number} -- version number (default: {LAST_VERSION})
        """
        obj = []
        try:
            obj = self.get(name, versions=[version],
                           throw_error_not_exist=False, throw_error_not_unique=False)
        except:
            pass
        return len(obj) != 0


class NumpyStore(abc.ABC):
    @abc.abstractmethod
    def _delete(self, name, version):
        """Delete an object with a predefined version.
        
        This method is only internally used by the MLRepo.

        Args:
            name (str): name of object
            version (str): object version
        """

        pass

    @abc.abstractmethod
    def add(self, name, version, numpy_dict):
        """ Add numpy data from an object to the storage.

        :param name: Name (as string) of object
        :param version: object version
        :param numpy_dict: numpy dictionary

        """
        pass

    @abc.abstractmethod
    def append(self, name, version_old, version_new, numpy_dict):
        """Append data

        Args:
            name (string): name of data object to be returned
            version_old (version): version of the object where the data will be appended
            version_new (version): version of the new objct after appending the data
            numpy_dict (dict): dictionary containign the values

        """
        pass

    @abc.abstractmethod
    def get(self, name, version, from_index=0, to_index=None):
        """Get specific data

        Args:
            name ([type]): [description]
            version ([type]): [description]
        """

    def push(self):
        """Push changes to an eternal repo.
        """

        pass

    def pull(self):
        """Pull changes from an external repo
        """

        pass
