import unittest
import os
import numpy as np

from pailab import RepoInfoKey, MLObjectType, repo_object_init, RepoInfoKey, DataSet, RawData, MLRepo  # pylint: disable=E0401
from pailab.ml_repo.repo import NamingConventions
from pailab.ml_repo.repo_objects import RepoInfo
import pailab.tools.tests as ml_tests
import pailab.ml_repo.repo_objects as repo_objects
import pailab.ml_repo.memory_handler as memory_handler
import pailab.ml_repo.repo_store as repo_store
from pailab.job_runner.job_runner import SimpleJobRunner # pylint: disable=E0401
import pailab.ml_repo.repo_store_factory as repo_store_factory
from pailab.ml_repo.numpy_handler_hdf import NumpyHDFStorage, NumpyHDFRemoteStorage, _get_all_files
import logging
logging.basicConfig(level=logging.FATAL) # since we also test for errors we switch off the logging in this level

class TestClass:
    @repo_object_init(['mat'])
    def __init__(self, a, b, mat=np.zeros([10,1]),):
        self.a = a
        self._b = b
        self.mat = mat

    def a_plus_b(self):
        return self.a + self._b


class RepoInfoTest(unittest.TestCase):

    def test_repo_info(self):
        name = 'dummy'
        version = '1.0.2'
        repo_info = RepoInfo({'version': version, 'NAME': name})
        self.assertEqual(repo_info.name, 'dummy')
        self.assertEqual(repo_info['name'], 'dummy')
        self.assertEqual(repo_info['NAME'], 'dummy')
        self.assertEqual(repo_info[RepoInfoKey.NAME], 'dummy')
        repo_info.set_fields({RepoInfoKey.BIG_OBJECTS: {'dummy': 'dummy2'}})
        self.assertEqual(repo_info.name, 'dummy')
        self.assertEqual(repo_info.big_objects['dummy'], 'dummy2')
        self.assertEqual(repo_info.big_objects['dummy'], 'dummy2')
        repo_info[RepoInfoKey.NAME] = 'dummy2'
        self.assertEqual(repo_info.name, 'dummy2')
        repo_info['NAME'] = 'dummy2'
        self.assertEqual(repo_info['NAME'], 'dummy2')
        repo_info['name'] = 'dummy2'
        self.assertEqual(repo_info['NAME'], 'dummy2')
        self.assertEqual(repo_info['dummy'], None)
        repo_info_dict = repo_info.get_dictionary()
        self.assertEqual(repo_info_dict['name'], 'dummy2')
        # now test if dictionary contains only values for RepoInfoKeys
        for k in RepoInfoKey:
            self.assertEqual(repo_info_dict[k.value], repo_info[k])


class RepoObjectTest(unittest.TestCase):

    def test_repo_object_helper(self):
        orig = TestClass(5, 3)
        self.assertEqual(orig.a_plus_b(), 8)
        orig = TestClass(5, 3, repo_info={  # pylint: disable=E1123
                         'name': 'test'})
        tmp = repo_objects.create_repo_obj_dict(orig)
        new = repo_objects.create_repo_obj(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())
        # test if different constructor call work correctly after applying decorator
        orig = TestClass(5, b=3, repo_info={  # pylint: disable=E1123
                         'name': 'test'})
        tmp = repo_objects.create_repo_obj_dict(orig)
        new = repo_objects.create_repo_obj(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())

        orig = TestClass(**{'a': 5, 'b': 3},  # pylint: disable=E1123
                         repo_info={'name': 'test'})  # pylint: disable=E1123
        tmp = repo_objects.create_repo_obj_dict(orig)
        new = repo_objects.create_repo_obj(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())

    def test_repo_object(self):
        # initialize repo object from test class
        version = 5
        obj = TestClass(5, 3,  # pylint: disable=E1123
                        repo_info={'name': 'dummy', 'version': version})  # pylint: disable=E1123
        self.assertEqual(obj.repo_info.name, 'dummy')  # pylint: disable=E1101
        self.assertEqual(obj.repo_info.version,  # pylint: disable=E1101
                         version)
        repo_dict = obj.to_dict()  # pylint: disable=E1101
        self.assertEqual(repo_dict['a'], obj.a)
        self.assertEqual(repo_dict['_b'], obj._b)
        obj2 = TestClass(
            **repo_dict, repo_info={'name': 'dummy', 'version': version}, _init_from_dict=True)
        self.assertEqual(obj2.a, obj.a)
        self.assertEqual(obj2.a_plus_b(), obj.a_plus_b())


class RawDataTest(unittest.TestCase):
    """Simple RawData objec tests
    """

    def test_validation(self):
        # test if validation works
        x_data = np.zeros([100, 2])

        # exception because number of coord_names does not equal number of x_coords
        with self.assertRaises(Exception):
            test_data = repo_objects.RawData(  # pylint: disable=W0612
                x_data, x_coord_names=[])  # pylint: disable=W0612
        # exception because number of y-coords does not equal number of x-coords
        y_data = np.zeros([99, 2])
        with self.assertRaises(Exception):
            test_data = repo_objects.RawData(  # pylint: disable=W0612
                x_data, x_coord_names=['x1', 'x2'], y_data=y_data, y_coord_names=['y1', 'y2'])
        # exception because number of y-coordnamess does not equal number of y-coords
        y_data = np.zeros([100, 2])
        with self.assertRaises(Exception):
            test_data = repo_objects.RawData(  # pylint: disable=W0612
                x_data, x_coord_names=['x1', 'x2'], y_data=y_data, y_coord_names=['y1'])

    def test_constructor(self):
        # simple construction
        x_data = np.zeros([100, 4])
        x_names = ['x1', 'x2', 'x3', 'x4']
        test_data = repo_objects.RawData(x_data, x_names)
        self.assertEqual(test_data.x_data.shape[0], x_data.shape[0])
        self.assertEqual(test_data.n_data, 100)
        # construction from array
        x_data = np.zeros([100])
        test_data = repo_objects.RawData(x_data, ['x1'])
        self.assertEqual(len(test_data.x_data.shape), 2)
        self.assertEqual(test_data.x_data.shape[1], 1)
        # construction from list
        test_data = repo_objects.RawData([100, 200, 300], ['x1'])
        self.assertEqual(test_data.x_data.shape[0], 3)
        self.assertEqual(test_data.x_data.shape[1], 1)

def eval_func_test(model, data):
    '''Dummy model eval function for testing
    
        Function retrns independent of data and model a simple numpy array with zeros
    Args:
        model ():dummy model, not used
        data ():dummy data, not used
    '''
    return np.zeros([data.x_data.shape[0],1])

def train_func_test(training_param, data):
    '''Dummy model training function for testing
    
        Function returns independent of data and model a simple numpy array with zeros
    Args:
        model_param ():dummy model parameter
        training_param ():dummy training parameter, not used
        data ():dummy trainig data, not used
    '''
    return TestClass(2,3, repo_info = {}) # pylint: disable=E1123


def preprocessor_transforming_function_test(preprocessor_param, data_x, x_coord_names, fitted_preprocessor=None):
    '''Dummy preprocessor transforming function for testing
    
        Function returns the input data
    Args:
        preprocessor_param ():dummy preprocessor parameter
        data_x ():dummy input data for transforming
        x_coord_names ():dummy coordinates names for transforming
        fitted_preprocessor ():dummy for a fitted preprocessor
    ''' 
    return data_x, x_coord_names

def preprocessor_fitting_function_test(preprocessor_param, data_x, x_coord_names):
    return TestClass(4,5, repo_info = {}) # pylint: disable=E1123

class RepoTest(unittest.TestCase):

    def _setup_measure_config(self):
        """Add a measure configuration with two measures (both MAX) where one measure just uses the coordinate x0
        """

        measure_config = repo_objects.MeasureConfiguration(
                                    [(repo_objects.MeasureConfiguration.MAX, ['y0']),
                                        repo_objects.MeasureConfiguration.MAX],
                                        repo_info={RepoInfoKey.NAME.value: 'measure_config'}
                                    )
        self.repository.add(measure_config, category=MLObjectType.MEASURE_CONFIGURATION, message = 'adding measure configuration')

    def _add_calibrated_model(self):
        self.repository.run_training()
        self.repository.set_label('prod')
        
    def setUp(self):
        '''Setup a complete ML repo with two different test data objetcs, training data, model definition etc.
        '''
        self.repository = MLRepo(user = 'unittestuser')
        job_runner = SimpleJobRunner(self.repository)
        self.repository._job_runner = job_runner
        #### Setup dummy RawData
        raw_data = repo_objects.RawData(np.zeros([10,1]), ['x0'], np.zeros([10,1]), ['y0'], repo_info = {repo_objects.RepoInfoKey.NAME.value: 'raw_1'})
        self.repository.add(raw_data, category=MLObjectType.RAW_DATA)
        raw_data = repo_objects.RawData(np.zeros([10,1]), ['x0'], np.zeros([10,1]), ['y0'], repo_info = {repo_objects.RepoInfoKey.NAME.value: 'raw_2'})
        self.repository.add(raw_data, category=MLObjectType.RAW_DATA)
        raw_data = repo_objects.RawData(np.zeros([10,1]), ['x0'], np.zeros([10,1]), ['y0'], repo_info = {repo_objects.RepoInfoKey.NAME.value: 'raw_3'})
        self.repository.add(raw_data, category=MLObjectType.RAW_DATA)
        ## Setup dummy Test and Training DataSets on RawData
        training_data = DataSet('raw_1', 0, None, 
                                    repo_info = {repo_objects.RepoInfoKey.NAME.value: 'training_data_1', repo_objects.RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        test_data_1 = DataSet('raw_2', 0, None, 
                                    repo_info = {repo_objects.RepoInfoKey.NAME.value: 'test_data_1',  repo_objects.RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        test_data_2 = DataSet('raw_3', 0, 2, 
                                    repo_info = {repo_objects.RepoInfoKey.NAME.value: 'test_data_2',  repo_objects.RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        self.repository.add([training_data, test_data_1, test_data_2])

        ## setup dummy preprocessor
        self.repository.add_preprocessing_transforming_function(preprocessor_transforming_function_test,
                        repo_name = 'transform_func')
        self.repository.add_preprocessing_fitting_function(preprocessor_fitting_function_test, 
                        repo_name = 'fit_func')
        self.repository.add_preprocessor('test_preprocessor_with_fitting', 'transform_func', 'fit_func', 
                         preprocessor_param = None)

        self.repository.add_eval_function(eval_func_test, 'eval_func')
        self.repository.add_training_function(train_func_test, 'train_func')
        self.repository.add(TestClass(1,2, repo_info={repo_objects.RepoInfoKey.NAME.value: 'training_param', # pylint: disable=E1123
                                            repo_objects.RepoInfoKey.CATEGORY: MLObjectType.TRAINING_PARAM}))
        ## setup dummy model definition
        self.repository.add_model('model', 'eval_func', 'train_func', preprocessors = ['test_preprocessor_with_fitting'])
        # setup measure configuration
        self._setup_measure_config()
        # add dummy calibrated model
        self._add_calibrated_model()
        
    def test_commit_increase_update(self):
        '''Check if updating an object in repository increases commit but does not change mapping
        '''
        obj = self.repository.get('raw_1')
        old_num_commits = len(self.repository.get_commits())
        old_version_mapping = self.repository.get('repo_mapping').repo_info[RepoInfoKey.VERSION]
        self.repository.add(obj)
        new_num_commits = len(self.repository.get_commits())
        new_version_mapping = self.repository.get('repo_mapping').repo_info[RepoInfoKey.VERSION]
        self.assertEqual(old_num_commits+1, new_num_commits)
        self.assertEqual(old_version_mapping, new_version_mapping)
        
    def test_commit_increase_add(self):
        '''Check if adding a new object in repository increases commit and does also change the mapping
        '''
        obj = DataSet('raw_data_1', 0, None, 
            repo_info={RepoInfoKey.NAME.value: 'test...', RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        old_num_commits = len(self.repository.get_commits())
        old_version_mapping = self.repository.get('repo_mapping').repo_info.version
        self.repository.add(obj)
        new_num_commits = len(self.repository.get_commits())
        new_version_mapping = self.repository.get('repo_mapping').repo_info.version
        self.assertEqual(old_num_commits+1, new_num_commits)
        commits = self.repository.get_commits()
        
    def test_DataSet_get(self):
        '''Test if getting a DataSet does include all informations from the underlying RawData (excluding numpy data)
        '''
        obj = self.repository.get('test_data_1')
        raw_obj = self.repository.get(obj.raw_data)
        for i in range(len(raw_obj.x_coord_names)):
            self.assertEqual(raw_obj.x_coord_names[i], obj.x_coord_names[i])
        for i in range(len(raw_obj.y_coord_names)):
            self.assertEqual(raw_obj.y_coord_names[i], obj.y_coord_names[i])
    
    def test_DataSet_get_full(self):
        '''Test if getting a DataSet does include all informations from the underlying RawData (including numpy data)
        '''
        obj = self.repository.get('test_data_1', version = repo_store.RepoStore.LAST_VERSION, full_object = True)
        raw_obj = self.repository.get(obj.raw_data, version = repo_store.RepoStore.LAST_VERSION, full_object = True)
        for i in range(len(raw_obj.x_coord_names)):
            self.assertEqual(raw_obj.x_coord_names[i], obj.x_coord_names[i])
        for i in range(len(raw_obj.y_coord_names)):
            self.assertEqual(raw_obj.y_coord_names[i], obj.y_coord_names[i])
        self.assertEqual(raw_obj.x_data.shape[0], obj.x_data.shape[0])

        obj = self.repository.get('test_data_2', version = repo_store.RepoStore.LAST_VERSION, full_object = True)
        self.assertEqual(obj.x_data.shape[0], 2)
    
        obj = self.repository.get('training_data_1', version = repo_store.RepoStore.LAST_VERSION, full_object = True)
        self.assertEqual(obj.x_data.shape[0], 10)
    
    def test_repo_RawData(self):
        """Test RawData within repo
        """
        repository = MLRepo(user = 'unittestuser')
        job_runner = SimpleJobRunner(repository)
        repository._job_runner = job_runner
        raw_data = repo_objects.RawData(np.zeros([10, 1]), ['test_coord'], repo_info={  # pylint: disable=E0602
            repo_objects.RepoInfoKey.NAME.value: 'RawData_Test'})
        repository.add(raw_data, 'test commit', MLObjectType.RAW_DATA)
        raw_data_2 = repository.get('RawData_Test')
        self.assertEqual(len(raw_data.x_coord_names),
                         len(raw_data_2.x_coord_names))
        self.assertEqual(
            raw_data.x_coord_names[0], raw_data_2.x_coord_names[0])
        commits = repository.get_commits()
        self.assertEqual(len(commits), 1)
        self.assertEqual(len(commits[0].objects), 1)
        
    def test_add_model_defaults(self):
        """test add_model using defaults to check whether default logic applies correctly
        """
        model_param = TestClass(3,4, repo_info={RepoInfoKey.NAME.value: 'model_param', RepoInfoKey.CATEGORY: MLObjectType.MODEL_PARAM.value}) # pylint: disable=E1123
        self.repository.add(model_param)
        training_param = TestClass(3,4, repo_info={RepoInfoKey.NAME.value: 'training_param', RepoInfoKey.CATEGORY: MLObjectType.TRAINING_PARAM.value}) # pylint: disable=E1123
        self.repository.add(training_param)
        self.repository.add_model('model1')
        model = self.repository.get('model1')
        self.assertEqual(model.eval_function, 'eval_func')
        self.assertEqual(model.training_function, 'train_func')
        self.assertEqual(model.training_param, 'training_param')
        self.assertEqual(model.model_param, 'model_param')
        
    def test_get_history(self):
        training_data_history = self.repository.get_history('training_data_1')
        self.assertEqual(len(training_data_history), 1)
        training_data = self.repository.get('training_data_1')
        self.repository.add(training_data)
        training_data_history = self.repository.get_history('training_data_1')
        self.assertEqual(len(training_data_history), 2)

    def test_run_eval_defaults(self):
        '''Test running evaluation with default arguments
        '''
        self.repository.run_evaluation()

    def test_run_train_defaults(self):
        '''Test running training with default arguments
        '''
        self.repository.run_training()
        model_calib = self.repository.get(str(NamingConventions.CalibratedModel(NamingConventions.Model('model'))))
        self.assertTrue('training_data_1' in model_calib.repo_info.modification_info.keys())

    def test_run_train_defaults_two_training_data(self):
        '''Test running training with default arguments where repo contains two models with different training data
        '''
        training_data = DataSet('raw_1', 0, None, 
                                    repo_info = {repo_objects.RepoInfoKey.NAME.value: 'training_data_2', repo_objects.RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})

        model =  self.repository.get('model')
        model.training_data = 'training_data_2'
        self.repository.add(model)
        self.repository.add(training_data)
        self.repository.run_training()
        model_calib = self.repository.get(str(NamingConventions.CalibratedModel(NamingConventions.Model('model'))))
        self.assertTrue('training_data_2' in model_calib.repo_info.modification_info.keys())

    def test_run_model_on_model(self):
        """Test model which is calibrated on output of another model.
        """        
        model =  self.repository.get('model')
        model.training_data = 'training_data_1'
        self.repository.add(model)
        # define the training data of second model as the eval data of first model
        training_data = DataSet('model/eval/training_data_1', 0, None, 
                                    repo_info = {repo_objects.RepoInfoKey.NAME.value: 'training_data_2', repo_objects.RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        self.repository.add(training_data)
        model_2 = repo_objects.Model(repo_info = {})
        model_2.training_data = 'training_data_2'
        model_2.test_data = 'model/eval/.*'
        model_2.repo_info.name = 'model_2'
        model_2.eval_function = 'eval_func'
        model_2.training_function = 'train_func'
        self.repository.add(model_2)

        # now run model 1
        self.repository.run_training(model='model')
        self.repository.run_evaluation(model='model/model')
        # now train model 2 on th eval data of model 1
        self.repository.run_training(model='model_2')
        
    def test_run_measure_defaults(self):
        self.repository.run_evaluation() # run first the evaluation so that there is at least one evaluation
        self.repository.run_measures()

    def test_run_measure_defaults_restrict_testdata(self):
        model =  self.repository.get('model')
        model.test_data = 'test_data_2'
        self.repository.add(model)
        self.repository.run_evaluation() # run first the evaluation so that there is at least one evaluation
        self.repository.run_measures()
        evals = self.repository.get_names(MLObjectType.EVAL_DATA)
        self.assertEqual(len(evals), 2)
        measures = self.repository.get_names(MLObjectType.MEASURE)
        self.assertEqual(len(measures), 2)
        

    def test_repo_training_test_data(self):
        # init repository with sample in memory handler
        repository = MLRepo(user = 'unittestuser')
        job_runner = SimpleJobRunner(repository)
        repository._job_runner = job_runner
        training_data = RawData(np.zeros([10,1]), ['x_values'], np.zeros([10,1]), ['y_values'], repo_info = {repo_objects.RepoInfoKey.NAME.value: 'training_data'})
        repository.add(training_data, category=MLObjectType.TRAINING_DATA)
        
        training_data_2 = repository.get_training_data()
        self.assertEqual(training_data_2.repo_info[repo_objects.RepoInfoKey.NAME], training_data.repo_info[repo_objects.RepoInfoKey.NAME])
        
        test_data = RawData(np.zeros([10,1]), ['x_values'], np.zeros([10,1]), ['y_values'], repo_info = {repo_objects.RepoInfoKey.NAME.value: 'test_data'})
        repository.add(test_data, category=MLObjectType.TEST_DATA)  
        test_data_ref = repository.get('test_data')
        self.assertEqual(test_data_ref.repo_info[repo_objects.RepoInfoKey.NAME], test_data.repo_info[repo_objects.RepoInfoKey.NAME])
        self.assertEqual(test_data_ref.repo_info[repo_objects.RepoInfoKey.VERSION], test_data.repo_info[repo_objects.RepoInfoKey.VERSION])
        
        test_data_2 = RawData(np.zeros([10,1]), ['x_values'], np.zeros([10,1]), ['y_values'], repo_info = {repo_objects.RepoInfoKey.NAME.value: 'test_data_2'})
        repository.add(test_data_2, category = MLObjectType.TEST_DATA)
        test_data_2_ref = repository.get('test_data_2')
        self.assertEqual(test_data_2.repo_info[repo_objects.RepoInfoKey.NAME], test_data_2_ref.repo_info[repo_objects.RepoInfoKey.NAME])
        
        commits = repository.get_commits()
        self.assertEqual(len(commits), 3)
        self.assertEqual(commits[1].objects['test_data'], test_data.repo_info.version)
        #self.assertEqual(commits[1].objects['repo_mapping'], 1)
        self.assertEqual(commits[2].objects['test_data_2'], test_data_2.repo_info.version)
        #self.assertEqual(commits[2].objects['repo_mapping'], 2)
        
    def test_repo_RegressionTest(self):
        regression_test_def = ml_tests.RegressionTestDefinition(repo_info = {RepoInfoKey.NAME: 'regression_test', RepoInfoKey.CATEGORY:MLObjectType.TEST_DEFINITION.name})
        tests = regression_test_def.create(self.repository)
        self.assertEqual(len(tests), 3)
        self.repository.add(regression_test_def)
        self.repository.run_evaluation()
        self.repository.run_measures()
        self.repository.run_tests()

    def test_add_multiple(self):
        """Test adding multiple objects at once
        """
        obj1 = TestClass(5,4, repo_info={})
        obj1.repo_info.name = 'obj1'
        v1 = self.repository.add(obj1, category = MLObjectType.CALIBRATED_MODEL)
        obj2 = TestClass(2,3, repo_info={})
        obj2.repo_info.name = 'obj2'
        self.repository.add([obj1, obj2], category = MLObjectType.CALIBRATED_MODEL)
        new_obj1 = self.repository.get('obj1')
        self.assertEqual(new_obj1.repo_info.name, 'obj1')
        new_obj2 = self.repository.get('obj2')
        self.assertEqual(new_obj2.repo_info.name, 'obj2')
        

    def test_delete(self):
        """Test if deletion works and if it considers if there are dependencies to other objects
        """

        obj1 = TestClass(5,4, repo_info={})
        obj1.repo_info.name = 'obj1'
        v1 = self.repository.add(obj1, category = MLObjectType.CALIBRATED_MODEL)
        obj2 = TestClass(2,3, repo_info={})
        obj2.repo_info.name = 'obj2'
        obj2.repo_info.modification_info = {'obj1': v1}
        v2 = self.repository.add(obj2, category = MLObjectType.CALIBRATED_MODEL)
        # check if an exception is thrown if one tries to delete obj1 although obj2 has
        # a dependency on obj1
        try:
            self.repository.delete('obj1', v1)
            self.assertEqual(0,1)
        except:
            pass
        # now first delete obj2
        self.repository.delete('obj2', v2)
        # check if obj2 has really been deleted
        try:
            obj2 = self.repository.get('obj2')
            self.assertEqual(0,1)
        except:
            pass
        
        #now, deletion of obj 1 should work
        try:
            self.repository.delete('obj1', v1)
        except:
            self.assertEqual(0,1)
        try: #check if object really has been deleted
            obj1 = self.repository.get('obj1')
            self.assertEqual(0,1)
        except:
            pass

    def test_add_raw_data(self):
        """Test the method add_raw_data
        """
        raw_data_input = np.zeros((10,3,))
        # test to add just x_data
        ml_repo = MLRepo(user = 'unittestuser')
        ml_repo.add_raw_data('test1', raw_data_input, ['x0', 'x1', 'x2'])
        raw_data = ml_repo.get('raw_data/test1', full_object = True)
        self.assertTrue('x0' in raw_data.x_coord_names)
        self.assertEqual(raw_data.x_data.shape[1], 3)
        # test to add x and y data from data only
        ml_repo.add_raw_data('test1', raw_data_input, ['x0', 'x1'], target_names = ['x2'])
        raw_data = ml_repo.get('raw_data/test1', full_object = True)
        self.assertTrue('x2' in raw_data.y_coord_names)
        self.assertEqual(raw_data.x_data.shape[1], 2)
        self.assertEqual(raw_data.y_data.shape[1], 1)
        # test to add x and y data with two different input matrices
        raw_data_y_input = np.zeros( (10,1, ) )
        ml_repo.add_raw_data('test1', raw_data_input, ['x0', 'x1', 'x2'], raw_data_y_input, target_names = ['y1'])
        raw_data = ml_repo.get('raw_data/test1', full_object = True)
        self.assertTrue('y1' in raw_data.y_coord_names)
        self.assertTrue('x0' in raw_data.x_coord_names)
        self.assertEqual(raw_data.y_data.shape[1], 1)
        self.assertEqual(raw_data.x_data.shape[1], 3)

        #test to add data from pandas dataframe
        import pandas as pd
        ml_repo.add_raw_data('test1', pd.DataFrame(data = raw_data_input, columns = ['x0', 'x1', 'x2']), ['x0', 'x1'], target_names = ['x2'])
        raw_data = ml_repo.get('raw_data/test1', full_object = True)
        self.assertTrue('x2' in raw_data.y_coord_names)
        self.assertEqual(raw_data.y_data.shape[1], 1)

        # try to read from file
        try:
            shutil.rmtree('test_add_raw_data')
            os.makedirs('test_add_raw_data')
        except OSError:
            os.makedirs('test_add_raw_data')
        tmp = pd.DataFrame(data = raw_data_input, columns = ['x0', 'x1', 'x2'])
        tmp.to_csv('test_add_raw_data/dummy.csv')
        ml_repo.add_raw_data('test1', 'test_add_raw_data/dummy.csv', ['x0', 'x1'], target_names = ['x2'], file_format = 'csv')
        raw_data = ml_repo.get('raw_data/test1', full_object = True)
        self.assertTrue('x2' in raw_data.y_coord_names)
        self.assertEqual(raw_data.y_data.shape[1], 1)
        self.assertEqual(raw_data.x_data.shape[1], 2)

        #try to read from numpy file
        np.save('test_add_raw_data/dummy.npy', np.zeros((10,3, ) ))
        ml_repo.add_raw_data('test1', 'test_add_raw_data/dummy.npy', ['x0', 'x1'], target_names = ['x2'], file_format = 'numpy')
        raw_data = ml_repo.get('raw_data/test1', full_object = True)
        self.assertTrue('x2' in raw_data.y_coord_names)
        self.assertEqual(raw_data.y_data.shape[1], 1)
        self.assertEqual(raw_data.x_data.shape[1], 2)
        try:
            shutil.rmtree('test_add_raw_data')
        except OSError:
            pass
        # now test add_training_data
        ml_repo.add_training_data('training_data_dummy', 'raw_data/test1')
        # now test add_test_data
        ml_repo.add_test_data('test_data_dummy', 'raw_data/test1')

    def test_add_label(self):
        """Test adding a label and if adding same label does not change anything
        """
        names = self.repository.get_names(MLObjectType.CALIBRATED_MODEL)
        model = self.repository.get(names[0])
        self.repository.set_label('test_label', model.repo_info.name, model.repo_info.version)
        label = self.repository.get('test_label')
        if isinstance(label, list):
            self.assertEqual(len(label), 1)
        if label is None:
            self.assertEqual(1,0)
        # add label again althogu neither model nor version have been changed
        self.repository.set_label('test_label', model.repo_info.name, model.repo_info.version)
        
        if isinstance(label, list):
            self.assertEqual(len(label), 1)
        if label is None:
            self.assertEqual(1,0)
        

class MLRepoConstructorTest(unittest.TestCase):
    def test_default_constructor(self):
        #example with default
        ml_repo = MLRepo(user = 'test_user')
        #end example with default
        # If on of these test fail asince the logic has been modified, please update the documentation in basics.rst
        self.assertTrue(isinstance(ml_repo._ml_repo, memory_handler.RepoObjectMemoryStorage))
        self.assertTrue(isinstance(ml_repo._numpy_repo, memory_handler.NumpyMemoryStorage))
        self.assertTrue(isinstance(ml_repo._job_runner, SimpleJobRunner))

    def test_config_disk_handler(self):
        #diskhandlerconfig
        config = {
          'user': 'test_user',
          'workspace': 'tmp',
          'repo_store': 
          {
              'type': 'disk_handler',  
              'config': {
                  'folder': 'tmp/objects', 
                  'file_format': 'json'
              }
          },
          'numpy_store':
          {
              'type': 'hdf_handler',
              'config':{
                  'folder': 'tmp/repo_data',
                  'version_files': True
              }
          },
          'job_runner':
          {
              'type': 'simple',
              'config':{}
          }
        }
        # end diskhandlerconfig

        # instantiate diskhandler
        ml_repo = MLRepo(config = config)
        # end instantiate diskhandler
        
        # instantiate diskhandler save config
        ml_repo = MLRepo(config = config, save_config=True)
        # end instantiate diskhandler save config
        
        # instantiate with workspace
        ml_repo = MLRepo(workspace = 'tmp')
        # end instantiate with workspace


class NumpyMemoryHandlerTest(unittest.TestCase):
    def test_append(self):
        numpy_store = memory_handler.NumpyMemoryStorage()
        numpy_dict = {'a':np.zeros([10,2]), 'b': np.zeros([5])}
        numpy_store.add('test_data', 'v1', numpy_dict)
        numpy_dict_2 = {'a':np.zeros([1,2]), 'b': np.zeros([1])}
        numpy_dict_2['b'][0] = 5.0
        numpy_dict_2['a'][0,0] = 3.0
        numpy_dict_2['a'][0,1] = 2.0
        numpy_store.append('test_data', 'v1', 'v2', numpy_dict_2)

        numpy_dict_3 = numpy_store.get('test_data', 'v2')
        self.assertEqual(numpy_dict_3['a'].shape[0],11)
        self.assertEqual(numpy_dict_3['b'].shape[0],6)
        self.assertEqual(numpy_dict_3['b'][5],5.0)
        self.assertEqual(numpy_dict_3['b'][0],0.0)

# define model
class SuperML:
    @repo_object_init()
    def __init__(self):
        self._value = None

    def train(self, data_x, data_y, median = True):
        if median:
            self._value = np.median(data_y)
        else:
            self._value = np.mean(data_y)

    def eval(self, data):
        return self._value
# end define model
# define training param
class SuperMLTrainingParam:
    @repo_object_init()
    def __init__(self):
        self.median = True
# end define training param

# define training function
def train(training_param, data_x, data_y):
    result =  SuperML()
    result.train(data_x, data_y, training_param.median)
    return result
# end define training function

# define eval function
def eval(model, data):
    return model.eval(data)
# end define eval function

class ModelIntegrationTest(unittest.TestCase):
    def test_model_integration(self):
        """Simple test of code snippets for documentation around model integration
        """
        def test_integration(self):
            ml_repo = MLRepo(user = 'test_user')
            x=np.zeros([10,1])
            y=np.zeros([10])
            # define eval and train
            ml_repo.add_eval_function(train,
                           repo_name='my_eval_func')
            ml_repo.add_training_function(eval, repo_name='my_eval_func')
            # end define eval and train

            # define add training parameter
            training_param = SuperMLTrainingParam()
            training_param.median = True
            ml_repo.add(training_param, message='my first training parameter for my own super ml algorithm')
            # end define add training parameter

            # add own model
            ml_repo.add_model('my_model')
            # end add own model
            ml_repo.run_training()
            
            #self.assertEqual

import shutil
class NumpyHDFStorageTest(unittest.TestCase):
    def setUp(self):
        try:
            shutil.rmtree('test_numpy_hdf5')
            os.makedirs('test_numpy_hdf5')
        except OSError:
            os.makedirs('test_numpy_hdf5')
        self.store = NumpyHDFStorage('test_numpy_hdf5')
        
    def tearDown(self):
        try:
            shutil.rmtree('test_numpy_hdf5')
        except OSError:
            pass
        
    def test_add(self):
        # add martix
        test_data = np.full((1,5), 1.0)
        self.store.add('test_2d', '1', {'test_data': test_data})
        test_data_get = self.store.get('test_2d', '1')
        self.assertEqual(test_data_get['test_data'].shape, test_data.shape)
        self.assertEqual(test_data[0,0], test_data_get['test_data'][0,0])
        # add array
        test_data = np.full((1,), 1.0)
        self.store.add('test_1d', '1', {'test_data': test_data})
        test_data_get = self.store.get('test_1d', '1')
        self.assertEqual(test_data_get['test_data'].shape, test_data.shape)
        self.assertEqual(test_data[0], test_data_get['test_data'][0])
        # add 3d array
        test_data = np.full((1,5,5), 1.0)
        self.store.add('test_3d', '1', {'test_data': test_data})
        test_data_get = self.store.get('test_3d', '1')
        self.assertEqual(test_data_get['test_data'].shape, test_data.shape)
        self.assertEqual(test_data[0,0,0], test_data_get['test_data'][0,0,0])
        # add 4d array
        test_data = np.full((1,5,5,6), 1.0)
        self.store.add('test_4d', '1', {'test_data': test_data})
        test_data_get = self.store.get('test_4d', '1')
        self.assertEqual(test_data_get['test_data'].shape, test_data.shape)
        self.assertEqual(test_data[0,0,0,1], test_data_get['test_data'][0,0,0,1])
        

    def test_append(self):
        """test appending data to existing numpy data (using one hdf file)
        """

        # add matrix
        test_data = np.full((1,5), 1.0)
        self.store.add('test_2d', '1', {'test_data': test_data})
        self.store.append('test_2d', '1', '2', {'test_data': np.full((1,5), 2.0)})
        self.store.append('test_2d', '2', '3', {'test_data': np.full((1,5), 3.0)})
        test_data_get = self.store.get('test_2d', '1')
        self.assertEqual(test_data_get['test_data'].shape, (1,5))
        self.assertEqual(test_data_get['test_data'][0,0], 1.0)
        test_data_get = self.store.get('test_2d', '2')
        self.assertEqual(test_data_get['test_data'].shape, (2,5))
        self.assertEqual(test_data_get['test_data'][0,0], 1.0)
        self.assertEqual(test_data_get['test_data'][1,0], 2.0)
        test_data_get = self.store.get('test_2d', '3')
        self.assertEqual(test_data_get['test_data'].shape, (3,5))
        self.assertEqual(test_data_get['test_data'][0,0], 1.0)
        self.assertEqual(test_data_get['test_data'][1,0], 2.0)
        self.assertEqual(test_data_get['test_data'][2,0], 3.0)

        # add array
        test_data = np.full((1,), 1.0)
        self.store.add('test_1d', '1', {'test_data': test_data})
        self.store.append('test_1d', '1', '2', {'test_data': np.full((1, ), 2.0)})
        self.store.append('test_1d', '2', '3', {'test_data': np.full((1, ), 3.0)})
        test_data_get = self.store.get('test_1d', '1')
        self.assertEqual(test_data_get['test_data'].shape, (1, ))
        self.assertEqual(test_data_get['test_data'][0], 1.0)
        test_data_get = self.store.get('test_1d', '2')
        self.assertEqual(test_data_get['test_data'].shape, (2, ))
        self.assertEqual(test_data_get['test_data'][0], 1.0)
        self.assertEqual(test_data_get['test_data'][1], 2.0)
        test_data_get = self.store.get('test_1d', '3')
        self.assertEqual(test_data_get['test_data'].shape, (3,))
        self.assertEqual(test_data_get['test_data'][0], 1.0)
        self.assertEqual(test_data_get['test_data'][1], 2.0)
        self.assertEqual(test_data_get['test_data'][2], 3.0)
        
        # add 3d array
        test_data = np.full((1,5,5), 1.0)
        self.store.add('test_3d', '1', {'test_data': test_data})
        test_data_get = self.store.get('test_3d', '1')
        self.assertEqual(test_data_get['test_data'].shape, test_data.shape)
        self.assertEqual(test_data[0,0,0], test_data_get['test_data'][0,0,0])
        self.store.append('test_3d', '1', '2', {'test_data': np.full((1,5,5), 2.0)})
        test_data_get = self.store.get('test_3d', '2')
        self.assertEqual(test_data_get['test_data'].shape, (2,5,5,) )
        self.assertEqual(test_data[0,0,0], test_data_get['test_data'][0,0,0])
        self.assertEqual(2.0, test_data_get['test_data'][1,1,0])

        # add 4d array
        test_data = np.full((1,5,5, 7), 1.0)
        self.store.add('test_4d', '1', {'test_data': test_data})
        test_data_get = self.store.get('test_4d', '1')
        self.assertEqual(test_data_get['test_data'].shape, test_data.shape)
        self.assertEqual(test_data[0,0,0,0], test_data_get['test_data'][0,0,0,0])
        self.store.append('test_4d', '1', '2', {'test_data': np.full((1,5,5,7), 2.0)})
        test_data_get = self.store.get('test_4d', '2')
        self.assertEqual(test_data_get['test_data'].shape, (2,5,5,7,) )
        self.assertEqual(test_data[0,0,0, 0], test_data_get['test_data'][0,0,0,0])
        self.assertEqual(2.0, test_data_get['test_data'][1,1,0,0])


    def test_append_single_files(self):
        """test appending data to existing numpy data (using deifferent hdf files for different versions)
        """
        self.store = NumpyHDFStorage('test_numpy_hdf5', True)
        # add martix
        test_data = np.full((1,5), 1.0)
        self.store.add('test_2d', '1', {'test_data': test_data})
        self.store.append('test_2d', '1', '2', {'test_data': np.full((1,5), 2.0)})
        self.store.append('test_2d', '2', '3', {'test_data': np.full((1,5), 3.0)})
        test_data_get = self.store.get('test_2d', '1')
        self.assertEqual(test_data_get['test_data'].shape, (1,5))
        self.assertEqual(test_data_get['test_data'][0,0], 1.0)
        test_data_get = self.store.get('test_2d', '2')
        self.assertEqual(test_data_get['test_data'].shape, (2,5))
        self.assertEqual(test_data_get['test_data'][0,0], 1.0)
        self.assertEqual(test_data_get['test_data'][1,0], 2.0)
        test_data_get = self.store.get('test_2d', '3')
        self.assertEqual(test_data_get['test_data'].shape, (3,5))
        self.assertEqual(test_data_get['test_data'][0,0], 1.0)
        self.assertEqual(test_data_get['test_data'][1,0], 2.0)
        self.assertEqual(test_data_get['test_data'][2,0], 3.0)

        # add array
        test_data = np.full((1,), 1.0)
        self.store.add('test_1d', '1', {'test_data': test_data})
        self.store.append('test_1d', '1', '2', {'test_data': np.full((1, ), 2.0)})
        self.store.append('test_1d', '2', '3', {'test_data': np.full((1, ), 3.0)})
        test_data_get = self.store.get('test_1d', '1')
        self.assertEqual(test_data_get['test_data'].shape, (1, ))
        self.assertEqual(test_data_get['test_data'][0], 1.0)
        test_data_get = self.store.get('test_1d', '2')
        self.assertEqual(test_data_get['test_data'].shape, (2, ))
        self.assertEqual(test_data_get['test_data'][0], 1.0)
        self.assertEqual(test_data_get['test_data'][1], 2.0)
        test_data_get = self.store.get('test_1d', '3')
        self.assertEqual(test_data_get['test_data'].shape, (3,))
        self.assertEqual(test_data_get['test_data'][0], 1.0)
        self.assertEqual(test_data_get['test_data'][1], 2.0)
        self.assertEqual(test_data_get['test_data'][2], 3.0)
        
        # add 3d array
        test_data = np.full((1,5,5), 1.0)
        self.store.add('test_3d', '1', {'test_data': test_data})
        test_data_get = self.store.get('test_3d', '1')
        self.assertEqual(test_data_get['test_data'].shape, test_data.shape)
        self.assertEqual(test_data[0,0,0], test_data_get['test_data'][0,0,0])

    def test_delete(self):
        """Test deletion of data.
        """
        test_data = np.full((1,5,5,6), 1.0)
        self.store.add('test_4d', '1', {'test_data': test_data})
        self.store._delete('test_4d', '1')
        succeeded = False
        try:
            test_data_get = self.store.get('test_4d', '1')
        except:
            succeeded = True
        self.assertTrue(succeeded)
        self.store._version_files = True
        # now test branch with different files
        self.store.add('test_4d', '1', {'test_data': test_data})
        self.store._delete('test_4d', '1')
        succeeded = False
        try:
            test_data_get = self.store.get('test_4d', '1')
        except:
            succeeded = True
        self.assertTrue(succeeded)

class NumpyHDFRemoteStorageTest(unittest.TestCase):
    class RemoteDummy:
        """Dummy remote class to test NumpyHDFRemoteStorage.
        """
        def __init__(self, directory):
            self.directory = directory

        def _remote_file_list(self):
            return _get_all_files(self.directory)

        def _download_file(self, local_filename, remote_filename):
            shutil.copyfile(self.directory + '/' + remote_filename, local_filename)

        def _upload_file(self,  local_filename, remote_filename):
            shutil.copyfile(local_filename, self.directory + '/' + remote_filename)

        def _delete_file(self, filename):
            pass

    def setUp(self):
        try:
            shutil.rmtree('test_numpy_hdf5_remote')
            os.makedirs('test_numpy_hdf5_remote')
        except OSError:
            os.makedirs('test_numpy_hdf5_remote')
        try:
            shutil.rmtree('test_numpy_hdf5_remote_remote')
            os.makedirs('test_numpy_hdf5_remote_remote')
        except OSError:
            os.makedirs('test_numpy_hdf5_remote_remote')
        self.store = repo_store_factory.NumpyStoreFactory.get('hdf_remote_handler', folder = 'test_numpy_hdf5_remote')
        self.remote = NumpyHDFRemoteStorageTest.RemoteDummy('test_numpy_hdf5_remote_remote')
        self.store.set_remote(self.remote)

    def tearDown(self):
        try:
            shutil.rmtree('test_numpy_hdf5_remote')
        except OSError:
            pass
        try:
            shutil.rmtree('test_numpy_hdf5_remote_remote')
        except OSError:
            pass

    def test_add(self):
        test_data = np.full((1,5), 1.0)
        self.store.add('test_1', '1', {'test_data': test_data})
        self.assertTrue(os.path.exists('test_numpy_hdf5_remote/test_1_1.hdf5'))
        self.assertFalse(os.path.exists('test_numpy_hdf5_remote_remote/test_1_1.hdf5'))
        self.store.push()
        self.assertTrue(os.path.exists('test_numpy_hdf5_remote_remote/test_1_1.hdf5'))
        self.store._sync_add = True
        self.store.add('test_2', '1', {'test_data': test_data})
        self.assertTrue(os.path.exists('test_numpy_hdf5_remote_remote/test_2_1.hdf5'))

    def test_get(self):
        test_data = np.full((1,5), 1.0)
        self.store._sync_add = True
        self.store._sync_get = True
        self.store.add('test_1', '1', {'test_data': test_data})
        os.remove('test_numpy_hdf5_remote/test_1_1.hdf5')
        self.assertFalse(os.path.exists('test_numpy_hdf5_remote/test_1_1.hdf5'))
        self.assertTrue(os.path.exists('test_numpy_hdf5_remote_remote/test_1_1.hdf5'))
        
        self.store.pull()
        self.assertTrue(os.path.exists('test_numpy_hdf5_remote/test_1_1.hdf5'))
        
        os.remove('test_numpy_hdf5_remote/test_1_1.hdf5')
        self.assertFalse(os.path.exists('test_numpy_hdf5_remote/test_1_1.hdf5'))
        self.store.get('test_1', '1')
        self.assertTrue(os.path.exists('test_numpy_hdf5_remote/test_1_1.hdf5'))
        
    def test_delete(self):
        test_data = np.full((1,5), 1.0)
        self.store._sync_add = True
        self.store.add('test_1', '1', {'test_data': test_data})
        self.store._delete('test_1', '1')
        succeeded = False
        try:
            test_data_get = self.store.get('test_1', '1')
        except:
            succeeded = True
        self.assertTrue(succeeded)


        


from pailab.tools.tests import RegressionTestDefinition
class RegressionTestTest(unittest.TestCase):
    """Test tools.RegressionTestDefinition

    """
    def test_regression_test(self):
        """Test the regression test framework
        """
        repo = MLRepo(user = 'unittestuser')
        model_1 = TestClass(1.0, 2.0, repo_info={'name': 'model'})
        model_1.repo_info.category = MLObjectType.CALIBRATED_MODEL.value
        model_version = repo.add(model_1)
        measure_1 = repo_objects.Measure(1.0, repo_info = {'name':'model/measure/test_data/max', 'modification_info': {'model'}})

if __name__ == '__main__':
    unittest.main()
