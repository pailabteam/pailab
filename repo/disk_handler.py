import os
import sqlite3
import uuid
from datetime import datetime, timedelta
import json
import repo.repo_objects as repo_objects
import repo.repo as repo
from repo.repo_store import RepoStore
import logging
logger = logging.getLogger(__name__)


class RepoObjectDiskStorage(RepoStore):

    # region private

    # region json encoding for enums
    PUBLIC_ENUMS = {
        'MLObjectType': repo.MLObjectType,
        # ...
    }

    class EnumEncoder(json.JSONEncoder):
        def default(self, obj):
            if type(obj) in RepoObjectDiskStorage.PUBLIC_ENUMS.values():
                return {"__enum__": str(obj)}
            return json.JSONEncoder.default(self, obj)

    @staticmethod
    def as_enum(d):
        if "__enum__" in d:
            name, member = d["__enum__"].split(".")
            return getattr(RepoObjectDiskStorage.PUBLIC_ENUMS[name], member)
        return d
    # endregion

    def _sqlite_db_name(self):
        return self._main_dir + '/.version.sqlite'

    def _execute(self, command):
        logging.info('Executing: ' + command)
        return self._conn.execute(command)

    def _create_new_db(self):
        self._conn = sqlite3.connect(self._sqlite_db_name())
        # three tables: one with category->name mapping, one with category, version, and one with modification info
        # mapping
        self._execute(
            '''CREATE TABLE mapping (name text PRIMARY KEY, category text)''')
        # versions
        self._execute(
            '''CREATE TABLE versions (name TEXT NOT NULL, version TEXT NOT NULL, file TEXT NOT NULL, uuid_time TIMESTAMP,
                                    insert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, PRIMARY KEY(name, version) ) ''')

        # modification_info
        self._execute(
            '''CREATE TABLE modification_info (name TEXT NOT NULL, version TEXT NOT NULL, modifier TEXT NOT NULL, modifier_version TEXT NOT NULL, modifier_uuid_time TIMESTAMP, PRIMARY KEY(name, version, modifier) ) ''')
        self._conn.commit()

    def _setup_new(self):
        if not os.path.exists(self._main_dir):
            os.makedirs(self._main_dir)
        if not os.path.exists(self._sqlite_db_name()):
            self._create_new_db()
        else:
            self._conn = sqlite3.connect(self._sqlite_db_name())

    @staticmethod
    def _get_time_from_uuid(uid):
        return datetime(1582, 10, 15) + timedelta(microseconds=uid.time//10)

    def _get_last_version(self, name):
        for row in self._execute("select version from versions where name = '" + name + "' order by uuid_time DESC LIMIT 1"):
            return row[0]
        raise Exception('No object with name ' + name + ' exists.')

    def _get_first_version(self, name):
        for row in self._execute("select version from versions where name = '" + name + "' order by uuid_time ASC LIMIT 1"):
            return row[0]
        raise Exception('No object with name ' + name + ' exists.')

    def _replace_version_placeholder(self, name, version):
        if version == RepoStore.FIRST_VERSION:
            return self._get_first_version(name)
        else:
            if version == RepoStore.LAST_VERSION:
                return self._get_last_version(name)
        return version

    # endregion

    def __init__(self, folder):
        self._main_dir = folder
        self._setup_new()

    def get_names(self, ml_obj_type):
        """Return the names of all objects belonging to the given category.

        Arguments:
            ml_obj_type {string} -- Value of MLObjectType-Enum specifying the category for which all names will be returned
        """
        result = []
        for row in self._conn.execute('select name, category from  mapping where category = ' + ml_obj_type):
            result.append(row['name'])
        return result

    def add(self, obj):
        """Add an object to the storage.


        Arguments:
            obj {RepoObject} -- repository object

        Raises:
            Exception if an object with same name already exists.
        """
        uid = uuid.uuid1()
        uid_time = RepoObjectDiskStorage._get_time_from_uuid(uid)
        name = obj['repo_info'][repo_objects.RepoInfoKey.NAME.value]
        category = obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value].name
        obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value] = str(uid)
        exists = False
        # region write mapping
        for row in self._execute('select * from mapping where name = ' + "'" + name + "'"):
            exists = True
            break
        if not exists:
            self._execute(
                "insert into mapping (name, category) VALUES ('" + name + "', '" + category + "')")
        # endregion
        # region write file info
        version = str(uid)
        file_sub_dir = category + '/' + name + '/'
        os.makedirs(self._main_dir + '/' + file_sub_dir, exist_ok=True)
        filename = file_sub_dir + '/' + str(uid) + '.json'
        self._conn.execute("insert into versions (name, version, file, uuid_time) VALUES('" +
                           name + "', '" + version + "','" + filename + "','" + str(uid_time) + "')")
        # endregion
        # region write modification info
        if repo_objects.RepoInfoKey.MODIFICATION_INFO.value in obj['repo_info']:
            for k, v in obj['repo_info'][repo_objects.RepoInfoKey.MODIFICATION_INFO.value].items():
                tmp = RepoObjectDiskStorage._get_time_from_uuid(uuid.UUID(v))
                self._conn.execute("insert into modification_info (name, version, modifier, modifier_version, modifier_uuid_time) VALUES ('"
                                   + name + "','" + version + "','" + k + "','" + str(v) + "','" + str(tmp) + "')")
        # endregion
        self._conn.commit()
        # region write file
        logging.debug(
            'Write object as json file with filename ' + filename)
        with open(self._main_dir + '/' + filename, 'w') as f:
            json.dump(obj, f, cls=RepoObjectDiskStorage.EnumEncoder)
        # endregion

    def get_version_condition(self, name, versions, version_column, time_column):
        version_condition = ''
        if isinstance(versions, str):
            version_condition = "and " + version_column + " = '" + \
                self._replace_version_placeholder(name, versions) + "'"
        else:
            if isinstance(versions, tuple):
                uid = uuid.UUID(
                    self._replace_version_placeholder(name, versions[0]))
                start_time = RepoObjectDiskStorage._get_time_from_uuid(
                    uid)
                uid = uuid.UUID(
                    self._replace_version_placeholder(name, versions[1]))
                end_time = RepoObjectDiskStorage._get_time_from_uuid(
                    uid)
                version_condition = "'" + str(
                    start_time) + "' <= " + time_column + " and " + time_column + "<= '" + str(end_time) + "'"
            else:
                if isinstance(versions, list):
                    version_condition = version_column + ' in ('
                    tmp = "','"
                    tmp = tmp.join([self._replace_version_placeholder(name, v)
                                    for v in versions])
                    version_condition += "'" + tmp + "')"
        return version_condition

    def get(self, name, versions=None, modifier_versions=None, obj_fields=None,  repo_info_fields=None):
        """Get a dictionary/list of dictionaries fulffilling the conditions.

            Returns a list of objects matching the name and whose
                -version is in the given list of versions
                -modifiers match the version number/are in the list of version numbers of the given modifiers
        Arguments:
            name {string} -- object id

        Keyword Arguments:
            versions {list, version_number, tuple} -- either a list of versions or a single version of the objects to be returned (default: {None}),
                    if None, the condition on version is ignored. If a tuple is given, this tuple defines a version intervall, i.e.
                    all versions between the first and last entry (both including) are returned. In addition FIRST_VERSION and LAST_VERSION can be used for versions to access
                    the last/first version.
            modifier_versions {dictionary} -- modifier ids together with version specs which are matched by the returned object (default: {None}).
            obj_fields {list of strings or string} -- list of strings identifying the fields which will be returned in the dictionary,
                                                        if None, no fields are returned, if set to 'all', all fields will be returned  (default: {None})
            repo_info_fields {list of strings or string} -- list of strings identifying the fields of the repo_info dict which will be returned in the dictionary,
                                                        if None, no fields are returned, if set to 'all', all fields will be returned (default: {None})
        """
        category = None
        for row in self._execute('select category from mapping where name = ' + "'" + name + "'"):
            category = repo.MLObjectType[row[0]]
        if category is None:
            raise Exception('no object ' + name + ' in storage.')

        version_condition = self.get_version_condition(
            name, versions, 'version', 'uuid_time')

        # region modifier condition
        # as v, as m
        modifier_conditions = []
        if modifier_versions is not None:
            for k, v in modifier_versions.items():
                tmp = self.get_version_condition()
        # endregion
        if len(modifier_conditions) == 0:
            select_statement = "select file from versions where name = '" + name + "'"
            if version_condition != '':
                select_statement += " and " + version_condition
        else:
            raise NotImplementedError()

        files = [row[0] for row in self._execute(select_statement)]
        objects = []
        for filename in files:
            with open(self._main_dir + '/' + filename, 'r') as f:
                objects.append(json.load(
                    f, object_hook=RepoObjectDiskStorage.as_enum))
        if len(objects) == 1:
            return objects[0]
        return objects
