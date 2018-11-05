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

# import all things you need to get startes
import time
import pandas as pd
import logging as logging

# Here start the repository specific imports
#import pailab.repo as repo
import pailab.memory_handler as memory_handler
from pailab import RepoInfoKey, MeasureConfiguration, MLRepo, DataSet, MLObjectType
from pailab.job_runner.job_runner import SimpleJobRunner, JobState, SQLiteJobRunner


class TutorialTest(unittest.TestCase):

    def test_tutorial(self):
        # cleanup disk before running
        repo_path = './tmp_tutorial'
        try:
            shutil.rmtree(repo_path)
            # os.path.
        except OSError:
            pass

        # creating new repository
        from pailab.disk_handler import RepoObjectDiskStorage
        from pailab.numpy_handler_hdf import NumpyHDFStorage
        handler = RepoObjectDiskStorage(repo_path)
        numpy_handler = NumpyHDFStorage(repo_path)
        job_runner = SimpleJobRunner(None)
        ml_repo = MLRepo('test_user', handler,
                         numpy_handler, handler, job_runner)
        job_runner.set_repo(ml_repo)
        ml_repo._job_runner = job_runner
        # end creating new repository

        # add RawData
        # A convenient way to add RawData is simply to use the method add_data.
        # This method just takes a pandas dataframe and the specification, which columns belong to the input
        # and which to the targets.
        data = pd.read_csv('./examples/boston_housing/housing.csv')
        ml_repo.add_data('boston_housing', data, input_variables=[
                         'RM', 'LSTAT', 'PTRATIO'], target_variables=['MEDV'])
        # end adding RawData

        # add DataSet
        # create DataSet objects for training and test data
        training_data = DataSet('boston_housing', 0, 300,
                                repo_info={RepoInfoKey.NAME: 'training_data', RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        test_data = DataSet('boston_housing', 301, None,
                            repo_info={RepoInfoKey.NAME: 'test_data',  RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        # add the objects to the repository. The method returns a dictionary of object names to version numbers of the added objects.
        version_list = ml_repo.add(
            [training_data, test_data], message='add training and test data')
        # end adding DataSet
        # add model
        import externals.sklearn_interface as sklearn_interface
        from sklearn.tree import DecisionTreeRegressor
        sklearn_interface.add_model(
            ml_repo, DecisionTreeRegressor(), model_param={'max_depth': 5})
        # end adding model
        # run training
        job_id = ml_repo.run_training()
        # end running training

        # run evaluation
        job_id = ml_repo.run_evaluation()
        # end running evaluation
        # run measures
        ml_repo.add_measure(MeasureConfiguration.MAX)
        ml_repo.add_measure(MeasureConfiguration.R2)
        job_ids = ml_repo.run_measures()
        # end running measures

        # cleanup after running
        # job_runner.close_connection()
        handler.close_connection()
        try:
            shutil.rmtree(repo_path)
            # os.path.
        except OSError:
            pass


if __name__ == '__main__':
    unittest.main()
