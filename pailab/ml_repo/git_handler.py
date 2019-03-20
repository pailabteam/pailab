import os
import sqlite3
import pickle
from datetime import datetime, timedelta
import json
import pailab.ml_repo.repo_objects as repo_objects
from pailab.ml_repo.repo_store import RepoInfoKey, _time_from_version
import pailab.ml_repo.repo as repo
from pailab.ml_repo.disk_handler import RepoObjectDiskStorage
from git import Repo
import logging
logger = logging.getLogger(__name__)


class RepoObjectGitStorage(RepoObjectDiskStorage):
    """Object storage with git support.

    This storge stores all objects on disk  in a local git storage. It provides functionality to push and pull from 
    another git repo. Note that it handles all files in the same way as RepoObjectDiskStorage does (using methods from this storage).
    """

    @staticmethod
    def _is_git_repo(path):
        try:
            _ = Repo(path).git_dir
            return True
        except:
            return False

    def __init__(self, **kwargs):
        """Constructor

        Args:
            folder (str): directory used to store the objects in files as well as the sqlite database
            save_function (function, optional): Defaults to pickle_save. Function used to save the objects to disk.
            load_function (function, optional): Defaults to pickle_load. Function used to load the objects from disk.
        """
        super(RepoObjectGitStorage, self).__init__(**kwargs)
        # initialize git repo if it does not exist
        if not RepoObjectGitStorage._is_git_repo(self._main_dir):
            _ = Repo.init(self._main_dir)

    def _add(self, obj):
        super(RepoObjectGitStorage, self)._add(obj)
        message = 'adding ' + obj['repo_info']['name']
        if (obj['repo_info']['commit_message'] is not None) and (obj['repo_info']['commit_message'] != ""):
            message = obj['repo_info']['commit_message']
        self.commit(message)

    def _delete(self, name, version):
        super(RepoObjectGitStorage, self)._delete(name, version)
        self.commit('deleting ' + name + ', version ' + version)

    def replace(self, obj):
        """Overwrite existing object without incrementing version

        Args:
            obj (RepoObject): repo object to be overwritten
        """
        super(RepoObjectGitStorage, self).replace(obj)
        self.commit('Replace object ' + obj['repo_info']['name'] +
                    ', version ' + obj['repo_info']['version'] + '.')

    def commit(self, message):
        check = self.check_integrity()
        if len(check) > 0:
            raise Exception(
                "Integrity check fails, cannot commit: " + str(check))
        git_repo = Repo(self._main_dir)
        git_repo.git.add('-A')
        git_repo.git.commit('-m', message)

    def push(self, remote_name='origin'):
        _git_repo = Repo(self._main_dir)
        remote = None
        for r in _git_repo.remotes:
            if r.name == remote_name:
                remote = r
                break
        if remote is None:
            raise Exception('Remote ' + remote_name + ' does not exist.')
        remote.push()

    def _merge_from_db(self, sqlite_db_2):
        c = self._conn.cursor()
        c.execute('ATTACH DATABASE "' + sqlite_db_2 + '" AS db_2')
        statement = 'INSERT OR IGNORE INTO versions(name, version, file, uuid_time) SELECT name, version, file, uuid_time FROM db_2.versions;'
        c.execute(statement)
        statement = 'INSERT OR IGNORE INTO mapping(name, category) SELECT name, category FROM db_2.mapping;'
        c.execute(statement)
        statement = 'INSERT OR IGNORE INTO modification_info(name, version, modifier, modifier_version) SELECT name, version, modifier, modifier_version FROM db_2.modification_info;'
        c.execute(statement)
        self._conn.commit()
        c.execute("DETACH DATABASE 'db_2';")

    def pull(self, remote_name='origin'):
        # self._conn.close()
        remote = None
        _git_repo = Repo(self._main_dir)
        for r in _git_repo.remotes:
            if r.name == remote_name:
                remote = r
                break
        if remote is None:
            raise Exception('Remote ' + remote_name + ' does not exist.')
        self._conn.close()
        os.rename(self._sqlite_db_name(), self._sqlite_db_name() + '_old')
        try:
            remote.pull()
        except:
            os.rename(self._sqlite_db_name() + '_old', self._sqlite_db_name())
            raise Exception('An error occured during pull: ' + (str(e)))
        self._conn = sqlite3.connect(self._sqlite_db_name())
        self._merge_from_db(self._sqlite_db_name() + '_old')
        os.remove(self._sqlite_db_name() + '_old')
