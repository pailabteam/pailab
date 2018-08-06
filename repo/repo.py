"""
Machine learning repository
"""
from enum import Enum
import logging
from repo.repo_objects import repo_object_init # pylint: disable=E0401
LOGGER = logging.getLogger('repo')


class MLObjectType(Enum):
    """Enums describing all ml object types.
    """
    TRAINING_DATA = 'training_data'
    TEST_DATA = 'test_data'
    TEST_RESULT = 'test_result'
    MODEL_PARAM = 'model_param'
    TRAINING_PARAM = 'training_param'
    TRAINING_FUNCTION = 'training_function'
    MODEL_EVAL_FUNCTION = 'model_eval_function'
    MODEL_CALIBRATOR = 'model_calibrator'
    MODEL_EVALUATOR = 'model_evaluator'
    PREP_PARAM = 'prep_param'
    PREPROCESSOR = 'preprocessor'
    MODEL_INFO = 'model_info'
    LABEL = 'label'

class ModelInfo:
    """ This class summarizes information of a special model version.

        This class summarizes information of a special model version,
        such as version numbers of all objects used such as trainin data, training parametere, etc..
    """
    @repo_object_init()
    def __init__(self):
        self.model_version = None
        self.training_data_version = {}
        self.test_version = {}
        self.test_result_version = {}
        self.model_param_version = None
        self.training_param_version = None
        self.prep_param_version = None
        self.model_training_function_version = None

class MLRepo:
    """ Repository for doing machine learning

        The repository and his extensions provide a solid fundament to do machine learning science supporting features such as:
            - auditing/versioning of the data, models and tests
            - best practice standardized plotting for investigating model performance and model behaviour
            - automated quality checks
            
        The repository needs three different handlers/repositories 

    """
    def __init__(self, script_repo, numpy_repo, ml_repo):
        """ Constructor of MLRepo

            :param script_repo: repository for the user's modules providing the customized model evaluation, raining, calibration and preprocessing methods
            :param numpy_repo: repository where the numpy data is stored in versions
            :param ml_repo: repository where the repo_objects are stored
        """
        pass

    def add(self, repo_object, message = ''):
        """ Add a repo_object to the repository.

            This method raises an exception, if an object with the same name already exists. It also checks for consistencies such as if a DataSet is added,
            if the underlying RawData object is already contained in the repository or if an object with the same name already exists and raises exceptions.

            :param repo_object: repo_object to be added
            :param message: commit message
        """
        pass
        
    def update(self, repo_object, message = ''):
        """ Update a repo_object in the repository.

            This method raises an exception, if an object with the same name does not already exist.
             It also checks for consistencies such as if a DataSet is updated,
            if the underlying RawData object is already contained in the repository.

            :param repo_object: repo_object to be added
            :param message: commit message
        """
        pass

    def get(self, name, version = -1, full_object = False):
        """ Get a repo objects. It throws an exception, if an object with the name does not exist.

            :param name: Object name
            :param version: object version, default is latest (-1)
            :param full_object: flag to determine whether also the numpy objects are loaded (True->load)
        """
        pass

    def get_names(self, ml_obj_type):
        """ Get the list of names of all repo_objects from a given repo_object_type in the repository.

            :param ml_obj_type: MLObjectType specifying the types of objects names are returned for.

            :return list of object names for the gigven category.
        """
        pass

    def get_history(self, name, repo_info_fields= [], obj_member_fields = [], version_start = 0, version_end = -1, label = None):
        """ Return a list of histories of object member variables.

        :param repo_info_field: List of fields from repo_info which will be returned in the dictionary. 
                                If List contains flag 'ALL', all fields will be returned.
        :param obj_member_fields: List of member atributes from repo_object which will be returned in the dictionary. 
                                If List contains flag 'ALL', all attributes will be returned.
        """
        pass

    def run_training(self, message = '', job_runner = None ):
        """ Run the training algorithm. 

        :param message: commit message
        :param job_runner: job runner executing the raining script. Default is single threaded local jobrunner.

        :return ticket number and new model version
        """
        pass

    def run_evaluation(self, message = '', model_version = -1, datasets = {}):
        """ Evaluate the model on all datasets. 

            :param message: Commit message for this operation.
            :param model_version: Version of model to be evaluated.
            :datasets: Dictionary of datasets (names and version numbers) on which the model is evaluated. 
                Default is all datasets on latest version.
        """
        pass

    def run_tests(self, message = '', model_version = -1, tests = {}, job_runner = None):
        """ Run tests for a specific model version.

            :param message: Commit message for this operation.
            :param model_version: Version or label of model for which the tests are executed.
            :param tests: Dictionary of tests (names and version numbers) run. Default is all tests on latest version. 
            :param job_runner: job runner executing the tests. Default is single threaded local jobrunner.

            :return ticket number of job
        """
        pass

    def get_model_info(self, model_version = -1):
        """ Return model info for the given model version.

        :param model_version: Version or label of model, default is latest.
        """
        pass

    def set_label(self, model_version, message = '', force = True):
        """ Label a certain model version.

            This method labels a certain model version. If force == False it checks if the label 
            already exists an raises in this case an exception.

            :param message: Commit message
            :param model_version: model version for which the label is set.
            :param force: lag determining if ann exception is raise if the label is already used (force->False, exception is raised)
        """
        pass
