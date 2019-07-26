import unittest
from pailab.ml_repo.numpy_handler_hdf import NumpyHDFStorage
import os
import shutil
import pailab.ml_repo.repo as repo
import pailab.ml_repo.repo_objects as repo_objects
from pailab.ml_repo.repo_store import RepoStore
import pailab.ml_repo.disk_handler as disk_handler
import time
import logging
# since we also test for errors we switch off the logging in this level
logging.basicConfig(level=logging.FATAL)


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
                time.sleep(2)

    def tearDown(self):
        try:
            self._storage._conn.close()
            shutil.rmtree('tmp_disk_storage')
        except OSError as e:
            print(str(e))
            pass

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
            self.assertTrue(
                obj[i-1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        # list of version number
        obj = self._storage.get(
            'obj', versions=[self._object_versions[1], self._object_versions[3]])
        self.assertEqual(len(obj), 2)
        self.assertTrue(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)

    def test_get_by_modifier_version(self):
        '''Test get interface where version of modifier objects are specified
        '''
        # one fixed modifier version
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': self._modifier1_versions[0]})
        self.assertEqual(len(obj), 2)
        self.assertTrue(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': self._modifier1_versions[0]})
        # range of modifier versions
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': (self._modifier1_versions[0], self._modifier1_versions[1])})
        self.assertEqual(len(obj), 4)
        self.assertTrue(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[2]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[3]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        # list of modifier versions
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': [self._modifier1_versions[0], self._modifier1_versions[2]]})
        self.assertEqual(len(obj), 4)
        self.assertTrue(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[2]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[3]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        # list of modifier versions for two different modifiers
        obj = self._storage.get('obj', modifier_versions={
                                'modifier_1': [self._modifier1_versions[0], self._modifier1_versions[2]],
                                'modifier_2': [self._modifier2_versions[0], self._modifier2_versions[1]]})
        self.assertEqual(len(obj), 2)
        self.assertTrue(
            obj[0]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions)
        self.assertTrue(
            obj[1]['repo_info'][repo_objects.RepoInfoKey.VERSION.value] in self._object_versions[1])

    def test_delete(self):
        """Test deletion
        """
        obj = self._storage.get('obj') 
        n_objs = len(self._storage.get('obj', version = (RepoStore.FIRST_VERSION, RepoStore.LAST_VERSION,)))
        self._storage._delete('obj', obj.repo_info.version)
        n_objs_new = len(self._storage.get('obj', version = (RepoStore.FIRST_VERSION, RepoStore.LAST_VERSION,)))
        self.assertEqual(objs-1, n_objs_new)
        





if __name__ == '__main__':
    unittest.main()
