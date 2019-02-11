

import os
import sqlite3
from datetime import datetime, timedelta

import pathlib
import pailab.repo_objects as repo_objects
from pailab.repo_store import RepoInfoKey, _time_from_version
import pailab.repo as repo
from pailab.repo_store import RepoStore
from shutil import copy
import logging
logger = logging.getLogger(__name__)


def execute(cursor, cmd):
    """Execute a sql statement (sqlite) with logging of statement 

    Args:
        cursor (cursor): cursor used to execute statement
        cmd (str): sql statement which will be executed

    Returns:
        [type]: [description]
    """

    logger.info('Executing: ' + cmd)
    return cursor.execute(cmd)


class RepoObjectDiskStorage(RepoStore):

    # region private

    # region json encoding for enums
    PUBLIC_ENUMS = {
        'MLObjectType': repo.MLObjectType,
        # ...
    }

    # endregion

    def _sqlite_db_name(self):
        return self._main_dir + '/.version.sqlite'

    def _create_new_db(self):
        self._conn = sqlite3.connect(self._sqlite_db_name())
        # three tables: one with category->name mapping, one with category, version, and one with modification info
        # mapping
        cursor = self._conn.cursor()
        try:
            logger.info('Executing')
            execute(cursor,
                    '''CREATE TABLE mapping (name text PRIMARY KEY, category text)''')
            # versions
            execute(cursor,
                    '''CREATE TABLE versions (name TEXT NOT NULL, version TEXT NOT NULL, path TEXT NOT NULL, file TEXT NOT NULL, uuid_time TIMESTAMP,
                                        insert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, PRIMARY KEY(name, version) ) ''')

            # modification_info
            execute(cursor,
                    '''CREATE TABLE modification_info (name TEXT NOT NULL, version TEXT NOT NULL, modifier TEXT NOT NULL, modifier_version TEXT NOT NULL, modifier_uuid_time TIMESTAMP, PRIMARY KEY(name, version, modifier) ) ''')
            self._conn.commit()
        except:
            logger.error(
                'An error occured during creation of new db, rolling back.')
            self._conn.rollback()

    def _setup_new(self):
        if not os.path.exists(self._main_dir):
            os.makedirs(self._main_dir)
        if not os.path.exists(self._sqlite_db_name()):
            self._create_new_db()
        else:
            self._conn = sqlite3.connect(self._sqlite_db_name())

    # endregion

    def __init__(self, folder, file_format='pickle'):
        """Constructor

        Args:
            folder (str): directory used to store the objects in files as well as the sqlite database
            save_function (function, optional): Defaults to pickle_save. Function used to save the objects to disk.
            load_function (function, optional): Defaults to pickle_load. Function used to load the objects from disk.
        """

        self._main_dir = folder
        self._setup_new()
        self._file_format = file_format
        if self._file_format == 'pickle':
            import pickle

            def __pickle_save(file_prefix, obj):
                with open(file_prefix + '.pck', 'wb') as f:
                    pickle.dump(obj, f)

            def __pickle_load(file_prefix):
                with open(file_prefix+'.pck', 'rb') as f:
                    return pickle.load(f)

            self._save_function = __pickle_save
            self._load_function = __pickle_load
        elif self._file_format == 'json':
            import json

            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if type(obj) in RepoObjectDiskStorage.PUBLIC_ENUMS.values():
                        return {"__enum__": str(obj)}
                    else:
                        if isinstance(obj, datetime):
                            return {"__datetime__": str(obj)}
                    return json.JSONEncoder.default(self, obj)

            class CustomDecoder(json.JSONDecoder):
                def __init__(self, *args, **kwargs):
                    json.JSONDecoder.__init__(
                        self, object_hook=self.object_hook, *args, **kwargs)

                def object_hook(self, obj):
                    if "__enum__" in obj:
                        name, member = obj["__enum__"].split(".")
                        return getattr(RepoObjectDiskStorage.PUBLIC_ENUMS[name], member)
                    if "__datetime__" in obj:
                        return datetime.strptime(obj["__datetime__"], 'YYYY-MM-DD HH:MM:SS.mmmmmm ')
                    return obj

            def __json_load(file_prefix):
                with open(file_prefix+'.json', 'r') as f:
                    return json.load(f, cls=CustomDecoder)

            def __json_save(file_prefix, obj):
                with open(file_prefix + '.json', 'w') as f:
                    json.dump(obj, f, cls=CustomEncoder,
                              indent=4, separators=(',', ': '))

            self._save_function = __json_save
            self._load_function = __json_load
        else:
            raise Exception("Unknown file format " + file_format)

    def get_config(self):
        return {'folder': self._main_dir, 'file_format': self._file_format}

    def get_names(self, ml_obj_type):
        """Return the names of all objects belonging to the given category.

        Arguments:
            ml_obj_type {string} -- Value of MLObjectType-Enum specifying the category for which all names will be returned
        """
        cursor = self._conn.cursor()
        result = []
        for row in execute(cursor, "select name, category from  mapping where category = '" + ml_obj_type + "'"):
            result.append(row[0])
        return result

    def _add(self, obj):
        """Add an object to the storage.


        Arguments:
            obj {RepoObject} -- repository object

        Raises:
            Exception if an object with same name already exists.
        """
        cursor = self._conn.cursor()
        try:
            uid_time = _time_from_version(
                obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value])
            name = obj['repo_info'][repo_objects.RepoInfoKey.NAME.value]
            if isinstance(obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value], repo.MLObjectType):
                category = obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value].name
            else:
                category = obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value]
            exists = False
            # region write mapping
            for row in execute(cursor, 'select * from mapping where name = ' + "'" + name + "'"):
                exists = True
                break
            if not exists:
                execute(cursor,
                        "insert into mapping (name, category) VALUES ('" + name + "', '" + category + "')")
            # endregion
            # region write file info
            version = obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value]
            file_sub_dir = category + '/' + name + '/'
            os.makedirs(self._main_dir + '/' + file_sub_dir, exist_ok=True)
            filename = version
            execute(cursor, "insert into versions (name, version, path, file, uuid_time) VALUES('" +
                    name + "', '" + version + "','" + file_sub_dir + "','" + filename + "','" + str(uid_time) + "')")
            # endregion
            # region write modification info
            if repo_objects.RepoInfoKey.MODIFICATION_INFO.value in obj['repo_info']:
                for k, v in obj['repo_info'][repo_objects.RepoInfoKey.MODIFICATION_INFO.value].items():
                    tmp = _time_from_version(v)
                    execute(cursor, "insert into modification_info (name, version, modifier, modifier_version, modifier_uuid_time) VALUES ('"
                            + name + "','" + version + "','" + k + "','" + str(v) + "','" + str(tmp) + "')")
            # endregion
            self._conn.commit()
            # region write file
            logger.debug(
                'Write object as json file with filename ' + filename)

            self._save_function(self._main_dir + '/' +
                                file_sub_dir + '/' + filename, obj)
            # endregion
        except Exception as e:
            logger.error('Error: ' + str(e) + ', rolling back changes.')
            self._conn.rollback()

    def get_version_condition(self, name, versions, version_column, time_column):
        version_condition = ''
        if versions is not None:
            version_condition = ' and '
        if isinstance(versions, str):
            version_condition += version_column + " = '" + versions + "'"
        else:
            if isinstance(versions, tuple):
                start_time = _time_from_version(versions[0])
                end_time = _time_from_version(versions[1])
                version_condition += "'" + str(
                    start_time) + "' <= " + time_column + " and " + time_column + "<= '" + str(end_time) + "'"
            else:
                if isinstance(versions, list):
                    version_condition += version_column + ' in ('
                    tmp = "','"
                    tmp = tmp.join([v for v in versions])
                    version_condition += "'" + tmp + "')"
        return version_condition

    def _get(self, name, versions=None, modifier_versions=None, obj_fields=None,  repo_info_fields=None,
             throw_error_not_exist=True, throw_error_not_unique=True):
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
        cursor = self._conn.cursor()
        for row in execute(cursor, 'select category from mapping where name = ' + "'" + name + "'"):
            category = repo.MLObjectType(row[0])
        if category is None:
            if throw_error_not_exist:
                logger.error('no object ' + name + ' in storage.')
                raise Exception('no object ' + name + ' in storage.')
            else:
                return []

        version_condition = self.get_version_condition(
            name, versions, 'version', 'uuid_time')

        select_statement = "select path, file from versions where name = '" + \
            name + "'" + version_condition
        if modifier_versions is not None:
            for k, v in modifier_versions.items():
                tmp = self.get_version_condition(
                    k, v, 'modifier_version', 'modifier_uuid_time')
                if tmp != '':
                    select_statement += " and version in ( select version from modification_info where name ='" + \
                        name + "' and modifier = '" + k + "'" + tmp + ")"
        files = [row[0] + '/' + row[1]
                 for row in execute(cursor, select_statement)]
        objects = []
        for filename in files:
            objects.append(self._load_function(
                self._main_dir + '/' + filename))
        return objects

    def get_latest_version(self, name, throw_error_not_exist=True):
        cursor = self._conn.cursor()
        for row in execute(cursor, "select version from versions where name = '" + name + "' order by uuid_time DESC LIMIT 1"):
            return row[0]
        if throw_error_not_exist:
            logger.error('No object with name ' + name + ' exists.')
            raise Exception('No object with name ' + name + ' exists.')
        else:
            return []

    def get_first_version(self, name, throw_error_not_exist=True):
        cursor = self._conn.cursor()
        for row in execute(cursor, "select version from versions where name = '" + name + "' order by uuid_time ASC LIMIT 1"):
            return row[0]
        if throw_error_not_exist:
            logger.error('No object with name ' + name + ' exists.')
            raise Exception('No object with name ' + name + ' exists.')
        else:
            return []

    def get_version(self, name, offset, throw_error_not_exist=True):
        cursor = self._conn.cursor()
        if offset > 0:
            inner_select = "(select version, uuid_time from versions where name = '" + \
                name + "' order by uuid_time ASC LIMIT " + str(offset) + ")"
            stmt = "select version from " + inner_select + " order by uuid_time DESC LIMIT 1"
            for row in execute(cursor, stmt):
                return row[0]
        elif offset < 0:
            inner_select = "(select version, uuid_time from versions where name = '" + \
                name + "' order by uuid_time DESC LIMIT " + str(-offset) + ")"
            stmt = "select version from " + inner_select + " order by uuid_time ASC LIMIT 1"
            for row in execute(cursor, stmt):
                return row[0]
        if offset == 0:
            return self.get_first_version(name)
        if throw_error_not_exist:
            logger.error('No object with name ' + name + ' exists.')
            raise Exception('No object with name ' + name + ' exists.')
        else:
            return []

    def replace(self, obj):
        """Overwrite existing object without incrementing version

        Args:
            obj (RepoObject): repo object to be overwritten
        """
        logger.info('Replacing ' + obj["repo_info"][RepoInfoKey.NAME.value] +
                    ', version ' + str(obj["repo_info"][RepoInfoKey.VERSION.value]))
        select_statement = "select path, file from versions where name = '" +\
            obj["repo_info"][RepoInfoKey.NAME.value] + "' and version = '" +\
            str(obj["repo_info"][RepoInfoKey.VERSION.value]) + "'"
        cursor = self._conn.cursor()
        for row in execute(cursor, select_statement):
            self._save_function(self._main_dir + '/' +
                                str(row[0]) + '/' + str(row[1]), obj)
        # delete all modification infos
        execute(cursor, "delete from modification_info where name='" + obj["repo_info"][RepoInfoKey.NAME.value] + "' and version = '" +
                str(obj["repo_info"][RepoInfoKey.VERSION.value]) + "'")
        if repo_objects.RepoInfoKey.MODIFICATION_INFO.value in obj['repo_info']:
            for k, v in obj['repo_info'][repo_objects.RepoInfoKey.MODIFICATION_INFO.value].items():
                tmp = _time_from_version(v)
                execute(cursor, "insert into modification_info (name, version, modifier, modifier_version, modifier_uuid_time) VALUES ('"
                        + obj["repo_info"][RepoInfoKey.NAME.value] + "','" + str(obj["repo_info"][RepoInfoKey.VERSION.value]) + "','" + k + "','" + str(v) + "','" + str(tmp) + "')")

    def close_connection(self):
        """Closes the database connection
        """
        self._conn.close()

    def _get_all_files(self):
        """Returns set of all files in directory

        Returns:
            set: set with all filenames (including relative paths)
        """
        f = set()
        for path, subdirs, files in os.walk(self._main_dir):
            if '.git' in subdirs:
                subdirs.remove('.git')
            if '.gitignore' in files:
                files.remove('.gitignore')
            if '.version.sqlite' in files:
                files.remove('.version.sqlite')
            for name in files:
                f.add(os.path.splitext(name)[0])
        return f

    def check_integrity(self):
        """Checks if files are missing or have not yet been added

        Returns:
            dictionary: contains sets of missing files and/or set of files not yet added
        """

        # first get list of all files
        f = self._get_all_files()
        cursor = self._conn.cursor()
        repo_f = set()
        for row in execute(cursor, "select file from versions"):
            repo_f.add(row[0])
        tmp = f.copy()
        files_not_added = tmp-repo_f
        result = {}
        if(len(files_not_added) > 0):
            result = {'files not added to repo': files_not_added}
        files_missing = repo_f - f
        if(len(files_missing) > 0):
            result = {'files missing in repo': files_missing}
        return result
