import unittest
from repo.numpy_handler_hdf import NumpyHDFStorage
import os
import shutil
import repo.repo as repo
import repo.repo_objects as repo_objects
from repo.repo_store import RepoStore
import repo.disk_handler as disk_handler
import time


class TestClass:
    @repo_objects.repo_object_init()
    def __init__(self):
        self.a = 1.0
        self.b = 2.0


class RepoDiskStorageTest(unittest.TestCase):

    def setUp(self):
        try:
            shutil.rmtree('tmp_disk_storage')
        except OSError:
            pass

        self._storage = disk_handler.RepoObjectDiskStorage('tmp_disk_storage')
        self._modifier1_versions = []
        self._modifier2_versions = []
        self._object_versions = []
        for i in range(5):
            modifier = TestClass(repo_info={repo_objects.RepoInfoKey.NAME.value: 'modifier_1',
                                            repo_objects.RepoInfoKey.CATEGORY.value: repo.MLObjectType.TRAINING_DATA})
            self._modifier1_versions.append(
                self._storage.add(repo_objects.create_repo_obj_dict(modifier)))
            for j in range(2):
                modifier2 = TestClass(repo_info={repo_objects.RepoInfoKey.NAME.value: 'modifier_2',
                                                 repo_objects.RepoInfoKey.CATEGORY.value: repo.MLObjectType.TRAINING_DATA})
                self._modifier2_versions.append(
                    self._storage.add(repo_objects.create_repo_obj_dict(modifier2)))
                obj = TestClass(repo_info={repo_objects.RepoInfoKey.NAME.value: 'obj', repo_objects.RepoInfoKey.CATEGORY.value: repo.MLObjectType.TRAINING_DATA,
                                           repo_objects.RepoInfoKey.MODIFICATION_INFO.value: {'modifier_1': self._modifier1_versions[-1],
                                                                                              'modifier_2': self._modifier2_versions[-1]}})
                self._object_versions.append(self._storage.add(
                    repo_objects.create_repo_obj_dict(obj)))
            time.sleep(0.1)

    def tearDown(self):
        try:
            shutil.rmtree('tmp_disk_storage')
        except OSError:
            pass

    def test_get_by_version(self):
        '''Test get interface where only version of object is specified
        '''
        self.assertEqual(self._storage.get_latest_version(
            'obj'), self._object_versions[-1])
        self.assertEqual(self._storage._get_first_version('obj'),
                         self._object_versions[0])
        # single version numbers
        obj = self._storage.get('obj', versions=RepoStore.FIRST_VERSION)
        self.assertEqual(obj['repo_info'][repo_objects.RepoInfoKey.VERSION.value],
                         self._object_versions[0])
        obj = self._storage.get('obj', versions=RepoStore.LAST_VERSION)
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
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': self._modifier1_versions[0]})
        self.assertEqual(len(obj), 2)
        self.assertEqual(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[0])
        self.assertEqual(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value], self._object_versions[1])


if __name__ == '__main__':
    unittest.main()
