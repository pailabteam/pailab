import unittest
import os
import numpy as np
import shutil
import pailab.repo as repo
import pailab.repo_objects as repo_objects
from pailab import RepoInfoKey, MeasureConfiguration, MLObjectType, RawData, DataSet, repo_object_init, JobState
from pailab.repo_store import RepoStore

from pailab.numpy_handler_hdf import NumpyHDFStorage
from pailab.disk_handler import RepoObjectDiskStorage
from pailab.job_runner.job_runner import SQLiteJobRunner

import time

import logging
# logging.basicConfig(level=logging.DEBUG)


class TestClass:
    @repo_object_init()
    def __init__(self, a, b):
        self.a = a
        self._b = b

    def a_plus_b(self):
        return self.a + self._b


class SQLiteJobRunner_Test(unittest.TestCase):

    def _setup_measure_config(self):
        """Add a measure configuration with two measures (both MAX) where one measure just uses the coordinate x0
        """

        measure_config = repo_objects.MeasureConfiguration(
            [(repo_objects.MeasureConfiguration.MAX, ['y0']),
             repo_objects.MeasureConfiguration.MAX],
            repo_info={RepoInfoKey.NAME: 'measure_config'}
        )
        self.repository.add(measure_config, category=MLObjectType.MEASURE_CONFIGURATION,
                            message='adding measure configuration')

    def _add_calibrated_model(self):
        dummy_model = TestClass(1, 2, repo_info={
                                RepoInfoKey.NAME.value: 'dummy_model'})  # pylint: disable=E1123
        self.repository.add(dummy_model, message='add dummy model',
                            category=MLObjectType.CALIBRATED_MODEL)

    def setUp(self):
        '''Setup a complete ML repo with two different test data objetcs, training data, model definition etc.
        '''
        try:
            shutil.rmtree('tmp')
            # os.path.
        except OSError:
            pass
        self.handler = RepoObjectDiskStorage('tmp')
        numpy_handler = NumpyHDFStorage('tmp/numpy')

        self.repository = repo.MLRepo(
            'doeltz', self.handler, numpy_handler, self.handler, None)
        self.job_runner = SQLiteJobRunner('tmp/job_runner.py', self.repository)
        self.repository._job_runner = self.job_runner
        # Setup dummy RawData
        raw_data = RawData(np.zeros([10, 1]), ['x0'], np.zeros(
            [10, 1]), ['y0'], repo_info={repo_objects.RepoInfoKey.NAME.value: 'raw_1'})
        self.repository.add(raw_data, category=repo.MLObjectType.RAW_DATA)
        raw_data = RawData(np.zeros([10, 1]), ['x0'], np.zeros(
            [10, 1]), ['y0'], repo_info={repo_objects.RepoInfoKey.NAME.value: 'raw_2'})
        self.repository.add(raw_data, category=repo.MLObjectType.RAW_DATA)
        raw_data = RawData(np.zeros([10, 1]), ['x0'], np.zeros(
            [10, 1]), ['y0'], repo_info={repo_objects.RepoInfoKey.NAME.value: 'raw_3'})
        self.repository.add(raw_data, category=repo.MLObjectType.RAW_DATA)
        # Setup dummy Test and Training DataSets on RawData
        training_data = DataSet('raw_1', 0, None,
                                repo_info={repo_objects.RepoInfoKey.NAME.value: 'training_data_1', repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TRAINING_DATA})
        test_data_1 = DataSet('raw_2', 0, None,
                              repo_info={repo_objects.RepoInfoKey.NAME.value: 'test_data_1',  repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TEST_DATA})
        test_data_2 = DataSet('raw_3', 0, 2,
                              repo_info={repo_objects.RepoInfoKey.NAME.value: 'test_data_2',  repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TEST_DATA})
        self.repository.add([training_data, test_data_1, test_data_2])

        self.repository.add_eval_function('tests.repo_test', 'eval_func_test')
        self.repository.add_training_function(
            'tests.repo_test', 'train_func_test')
        self.repository.add(TestClass(1, 2, repo_info={repo_objects.RepoInfoKey.NAME.value: 'training_param',  # pylint: disable=E1123
                                                       repo_objects.RepoInfoKey.CATEGORY: repo.MLObjectType.TRAINING_PARAM}))
        # setup dummy model definition
        self.repository.add_model('model')
        # setup measure configuration
        self._setup_measure_config()
        # add dummy calibrated model
        self._add_calibrated_model()

    def tearDown(self):
        try:
            self.job_runner.close_connection()
            self.handler.close_connection()
            shutil.rmtree('tmp')
        except OSError as e:
            print(str(e))
            pass

    def test_job_runner(self):
        job = self.repository.run_training()
        tmp = self.job_runner.get_info(job[0], job[1])
        # check if traning job is in waiting state
        self.assertEqual(tmp['job_state'], JobState.WAITING.value)
        self.job_runner.run(1)
        tmp = self.job_runner.get_info(job[0], job[1])
        self.assertEqual(tmp['job_state'],
                         JobState.SUCCESSFULLY_FINISHED.value)

        self.assertEqual(0, 0)


if __name__ == '__main__':
    unittest.main()
