import abc
from pailab.repo_objects import RepoInfoKey  # pylint: disable=E0401


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

    FIRST_VERSION = 'first'
    LAST_VERSION = 'last'

    """Abstract base class for all storages which can be used in the ML repository

    """
    @abc.abstractmethod
    def add(self, obj):
        """Add an object to the storage.


        Arguments:
            obj {RepoObject} -- repository object

        Raises:
            Exception if an object with same name already exists.
        """
        pass

    @abc.abstractmethod
    def get(self, name, versions=None, modifier_versions=None, obj_fields=None,  repo_info_fields=None):
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

    def get_version_number(self, name, offset):
        """Return versionnumber for the given offset

        If offset >= 0 it returns the version number of the offest version, if <0 it returns according to the
        python list logic the version number of the (offset-1)-last version

        Arguments:
            name {string} -- name of object
            offset {int} -- offset
        """
        pass

    def get_latest_version(self, name):
        """Return latest version number of object in the storage

        Arguments:
            name {string} -- object name

        Returns:
            version number -- latest version number
        """
        return self.get(name, versions=RepoStore.LAST_VERSION, repo_info_fields=[RepoInfoKey.VERSION])[0]['repo_info'][RepoInfoKey.VERSION.value]

    def object_exists(self, name, version=LAST_VERSION):
        """Returns True if an object with the given name and version exists.

        Arguments:
            name {string} -- object name

        Keyword Arguments:
            version {version number} -- version number (default: {LAST_VERSION})
        """
        obj = self.get(name, versions=[version])
        return len(obj) != 0


class NumpyStore(abc.ABC):
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
