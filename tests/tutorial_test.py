import unittest
import os
import numpy as np
import shutil
import logging as logging
import pailab.ml_repo.repo as repo
import pailab.ml_repo.repo_objects as repo_objects
from pailab import RepoInfoKey, MeasureConfiguration, MLObjectType, RawData, DataSet, repo_object_init, JobState
from pailab.ml_repo.repo_store import RepoStore

from pailab.ml_repo.numpy_handler_hdf import NumpyHDFStorage
from pailab.ml_repo.disk_handler import RepoObjectDiskStorage
from pailab.job_runner.job_runner import SQLiteJobRunner

# import all things you need to get started
import time
import pandas as pd
import logging as logging

# Here start the repository specific imports
# import pailab.ml_repo.repo as repo
import pailab.ml_repo.memory_handler as memory_handler
from pailab import RepoInfoKey, MeasureConfiguration, MLRepo, DataSet, MLObjectType
from pailab.job_runner.job_runner import SimpleJobRunner, JobState, SQLiteJobRunner

logging.basicConfig(level=logging.ERROR)


class TutorialTest(unittest.TestCase):

    def test_tutorial(self):
        # cleanup disk before running
        repo_path = './tmp_tutorial'
        try:
            shutil.rmtree(repo_path)
            # os.path.
        except OSError:
            pass

        # creating in memory storage
        ml_repo = MLRepo(user='test_user')
        # end creating in memory storage

        # creating new repository
        config = {'user': 'test_user',
                  'workspace': repo_path,
                  'repo_store':
                  {
                      'type': 'disk_handler',
                      'config': {
                          'folder': repo_path,
                          'file_format': 'pickle'
                      }
                  },
                  'numpy_store':
                  {
                      'type': 'hdf_handler',
                      'config': {
                          'folder': repo_path,
                          'version_files': True
                      }
                  }
                  }
        ml_repo = MLRepo(user='test_user', config=config)
        # end creating new repository
        # specifying job runner
        job_runner = SimpleJobRunner(None)
        job_runner.set_repo(ml_repo)
        ml_repo._job_runner = job_runner
        # end specifying job runner
        job_runner._throw_job_error = True

        from pailab.tools.tools import MLTree
        MLTree.add_tree(ml_repo)

        # A convenient way to add RawData is simply to use the method add on the raw_data collection.
        # This method just takes a pandas dataframe and the specification, which columns belong to the input
        # and which to the targets.

        try:
            # read pandas
            import pandas as pd
            data = pd.read_csv('./examples/boston_housing/housing.csv')
            # end read pandas
        except:
            data = pd.read_csv('../examples/boston_housing/housing.csv')

        # extract data
        input_variables = ['RM', 'LSTAT', 'PTRATIO']
        target_variables = ['MEDV']
        x = data.loc[:, input_variables].values
        y = data.loc[:, target_variables].values
        # end extract data

        # add RawData snippet
        from pailab import RawData, RepoInfoKey

        raw_data = RawData(x, input_variables, y, target_variables, repo_info={
                           RepoInfoKey.NAME: 'raw_data/boston_housing'})
        ml_repo.add(raw_data)

        # end adding RawData snippet
        # ml_repo.tree.raw_data.add('boston_housing', data, input_variables=[
        #    'RM', 'LSTAT', 'PTRATIO'], target_variables=['MEDV'])

        # add DataSet
        # create DataSet objects for training and test data
        training_data = DataSet('raw_data/boston_housing', 0, 300,
                                repo_info={RepoInfoKey.NAME: 'training_data', RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        test_data = DataSet('raw_data/boston_housing', 301, None,
                            repo_info={RepoInfoKey.NAME: 'test_data',  RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        # add the objects to the repository
        version_list = ml_repo.add(
            [training_data, test_data], message='add training and test data')
        # end adding DataSet

        # add model
        import pailab.externals.sklearn_interface as sklearn_interface
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

        # add measures snippet
        ml_repo.add_measure(MeasureConfiguration.MAX)
        ml_repo.add_measure(MeasureConfiguration.R2)
        # end add measure snippet

        # run measures snippet
        job_ids = ml_repo.run_measures()
        # end run measures snippet

        print(ml_repo.get_names(MLObjectType.MEASURE))

        # get measures
        max_measure = ml_repo.get(
            'DecisionTreeRegressor/measure/training_data/max')
        print(str(max_measure.value))
        # end getting measures

        # label snippet
        from pailab import LAST_VERSION
        ml_repo.set_label('prod', 'DecisionTreeRegressor/model',
                          model_version=LAST_VERSION, message='we found our first production model')
        # end label snippet

        # test definition snippet
        import pailab.tools.tests
        reg_test = pailab.tools.tests.RegressionTestDefinition(
            reference='prod', models=None, data=None, labels=None, measures=None,  tol=1e-3)
        reg_test.repo_info.name = 'reg_test'
        ml_repo.add(reg_test, message='regression test definition')
        # end test definition snippet

        # add test snippet
        tests = ml_repo.run_tests()
        # end add test snippet
        print(tests)

        # run check snippet
        import pailab.tools.checker as checker
        inconsistencies = checker.run(ml_repo)
        # end run check snippet

        print(inconsistencies)

        # add inconsistency snippet
        param = ml_repo.get('DecisionTreeRegressor/model_param')
        param.sklearn_params['max_depth'] = 2
        version = ml_repo.add(param)
        # end add inconsistency snippet

        inconsistencies = checker.run(ml_repo)
        print(inconsistencies)

        ml_repo.run_training()

        inconsistencies = checker.run(ml_repo)
        print(inconsistencies)

        ml_repo.run_evaluation(run_descendants=True)

        print(checker.run(ml_repo))

        # add second test data snippet
        test_data_2 = DataSet('raw_data/boston_housing', 0, 50,
                              repo_info={RepoInfoKey.NAME: 'test_data_2',
                                         RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA}
                              )
        ml_repo.add(test_data_2)
        ml_repo.run_evaluation(run_descendants=True)
        # end add second test data snippet

        ml_repo.run_tests()

        print(checker.run(ml_repo))

        # check tests
        print(checker.Tests.run(ml_repo))
        # end check tests

        # cleanup after running
        # job_runner.close_connection()
        ml_repo._ml_repo.close_connection()
        try:
            shutil.rmtree(repo_path)
            # os.path.
        except OSError:
            pass


if __name__ == '__main__':
    unittest.main()
