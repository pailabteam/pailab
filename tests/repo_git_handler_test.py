import unittest
from pailab.ml_repo.numpy_handler_hdf import NumpyHDFStorage
import os
import shutil
import pailab.ml_repo.repo as repo
import pailab.ml_repo.repo_objects as repo_objects
from pailab.ml_repo.repo_store import RepoStore
import pailab.ml_repo.git_handler as git_handler
import time
import logging
from git import Repo
# since we also test for errors we switch off the logging in this level
logging.basicConfig(level=logging.FATAL)


class TestClass:
    @repo_objects.repo_object_init()
    def __init__(self):
        self.a = 1.0
        self.b = 2.0


class RepoGitStorageTest(unittest.TestCase):
    @staticmethod
    def onerror(func, path, exc_info):
        """
        Error handler for ``shutil.rmtree``.

        If the error is due to an access error (read only file)
        it attempts to add write permission and then retries.

        If the error is for another reason it re-raises the error.

        Usage : ``shutil.rmtree(path, onerror=onerror)``
        """
        import stat
        if not os.access(path, os.W_OK):
            # Is the error an access error ?
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise exc_info

    def setUp(self):
        self.git_dir = 'tmp_git_storage'
        try:
            if os.path.exists(self.git_dir):
                # since all files in .git are read only, we need the onerror method to successfully remove the directory
                shutil.rmtree(self.git_dir, onerror=RepoGitStorageTest.onerror)
        except OSError as e:
            print(e.message)
            pass
        self._storage = git_handler.RepoObjectGitStorage(
            folder=self.git_dir)
        self._modifier1_versions = []
        self._modifier2_versions = []
        self._object_versions = []
        self._object_names = ['modifier_1', 'modifier_2']
        for i in range(5):
            modifier = TestClass(repo_info={repo_objects.RepoInfoKey.NAME.value: 'modifier_1',
                                            repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TRAINING_DATA})
            self._modifier1_versions.append(
                self._storage.add(repo_objects.create_repo_obj_dict(modifier)))
            for j in range(2):
                modifier2 = TestClass(repo_info={repo_objects.RepoInfoKey.NAME.value: 'modifier_2',
                                                 repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TRAINING_DATA})
                self._modifier2_versions.append(
                    self._storage.add(repo_objects.create_repo_obj_dict(modifier2)))
                obj = TestClass(repo_info={repo_objects.RepoInfoKey.NAME.value: 'obj', repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TRAINING_DATA,
                                           repo_objects.RepoInfoKey.MODIFICATION_INFO.value: {'modifier_1': self._modifier1_versions[-1],
                                                                                              'modifier_2': self._modifier2_versions[-1]}})
                self._object_versions.append(self._storage.add(
                    repo_objects.create_repo_obj_dict(obj)))
            time.sleep(0.1)

    def remove_git_repo(name):
        try:
            if os.path.exists(name):
                shutil.rmtree(name, onerror=RepoGitStorageTest.onerror)
        except OSError as e:
            print(str(e))

    def tearDown(self):
        self._storage._conn.close()
        RepoGitStorageTest.remove_git_repo(self.git_dir)
        
    def test_get_by_version(self):
        '''Test get interface where only version of object is specified
        '''
        self.assertEqual(self._storage.get_latest_version(
            'obj'), self._object_versions[-1])
        self.assertEqual(self._storage.get_first_version('obj'),
                         self._object_versions[0])
        # single version numbers
        obj = self._storage.get('obj', versions=RepoStore.FIRST_VERSION)[0]
        self.assertEqual(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value],
                         self._object_versions[0])
        obj = self._storage.get('obj', versions=RepoStore.LAST_VERSION)[0]
        self.assertEqual(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value],
                         self._object_versions[-1])
        # range of version number
        obj = self._storage.get('obj', versions=(
            self._object_versions[1], self._object_versions[3]))
        self.assertEqual(len(obj), 3)
        for i in range(1, 4):
            self.assertEqual(
                obj[i-1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[i])
        # list of version number
        obj = self._storage.get(
            'obj', versions=[self._object_versions[1], self._object_versions[3]])
        self.assertEqual(len(obj), 2)
        self.assertEqual(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[1])
        self.assertEqual(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[3])

    def test_get_by_modifier_version(self):
        '''Test get interface where version of modifier objects are specified
        '''
        # one fixed modifier version
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': self._modifier1_versions[0]})
        self.assertEqual(len(obj), 2)
        self.assertEqual(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[0])
        self.assertEqual(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[1])
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': self._modifier1_versions[0]})
        # range of modifier versions
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': (self._modifier1_versions[0], self._modifier1_versions[1])})
        self.assertEqual(len(obj), 4)
        self.assertEqual(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[0])
        self.assertEqual(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[1])
        self.assertEqual(
            obj[2]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[2])
        self.assertEqual(
            obj[3]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[3])
        # list of modifier versions
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': [self._modifier1_versions[0], self._modifier1_versions[2]]})
        self.assertEqual(len(obj), 4)
        self.assertEqual(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[0])
        self.assertEqual(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[1])
        self.assertEqual(
            obj[2]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[4])
        self.assertEqual(
            obj[3]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[5])
        # list of modifier versions for two different modifiers
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': [self._modifier1_versions[0], self._modifier1_versions[2]],
                                'modifier_2': [self._modifier2_versions[0], self._modifier2_versions[1]]})
        self.assertEqual(len(obj), 2)
        self.assertEqual(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[0])
        self.assertEqual(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[1])

    def test_delete(self):
        def check_file_exists(obj, storage):
            if isinstance(obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value], repo.MLObjectType):
                category = obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value].name
            else:
                category = obj['repo_info'][repo_objects.RepoInfoKey.CATEGORY.value]
            name = obj['repo_info'][repo_objects.RepoInfoKey.NAME.value]
            filename = category + '/' + name + '/' + \
                obj['repo_info']['version'] + storage._extension
            return os.path.exists(filename)

        obj = self._storage.get(
            self._object_names[0], self._modifier1_versions[0])

        # delete and check if file has been deleted from disk
        self._storage._delete(
            self._object_names[0], self._modifier1_versions[0])
        self.assertFalse(check_file_exists(obj[0], self._storage))

    def test_pull(self):
        '''test if pull works correctly
        '''
        RepoGitStorageTest.remove_git_repo(self.git_dir + '_2') # remove directory of second repo
        cloned = Repo.clone_from(self.git_dir, self.git_dir + '_2')
        storage = git_handler.RepoObjectGitStorage(
            folder=self.git_dir)
        obj = TestClass(repo_info={repo_objects.RepoInfoKey.NAME.value: 'new_obj',
                                   repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TRAINING_DATA})
        storage.add(repo_objects.create_repo_obj_dict(obj))
        cloned_storage = git_handler.RepoObjectGitStorage(
            folder=self.git_dir + '_2')
        cloned_storage.pull()
        obj = cloned_storage.get('new_obj')
        cloned_storage._conn.close()
        RepoGitStorageTest.remove_git_repo(self.git_dir + '_2') # remove directory of second repo


if __name__ == '__main__':
    unittest.main()
