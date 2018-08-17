"""
Machine learning repository
"""
from enum import Enum
import logging
import repo.repo_objects as repo_objects
from repo.repo_objects import repo_object_init  # pylint: disable=E0401
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
    MODEL_EVALUATION = 'model_evaluator'
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
    class Mapping:
        """This class maps all repo objects to their respective ids
        """
        @repo_object_init()
        def __init__(self):
            self.training_data = None
            self.test_data = None
            self.preprocessing = None
            self.training_parameter = None
            self.model_parameter = None
            self.unit_tests = None

    def __init__(self, script_repo, numpy_repo, ml_repo):
        """ Constructor of MLRepo

            :param script_repo: repository for the user's modules providing the customized model evaluation, raining, calibration and preprocessing methods
            :param numpy_repo: repository where the numpy data is stored in versions
            :param ml_repo: repository where the repo_objects are stored
        """
        self._script_repo = script_repo
        self._numpy_repo = numpy_repo
        self._ml_repo = ml_repo
        # check if the ml mapping is already contained in the repo, otherwise add it
        try:
            self._mapping = self._ml_repo.get('repo_mapping')
        except:
            self._mapping = MLRepo.Mapping(  # pylint: disable=E1123
                repo_info={repo_objects.RepoInfoKey.NAME.value: 'repo_mapping'})
            self._add(self._mapping, 'Adding empty mapping for repo creation.')

    def _adjust_version(self, version, name):
        """Checks if version is negative and then adjust it according to the typical python 
        way for list where -1 is the last element of the list, -2 the second last etc.

        :param version: version number
        :param name: name of object for which version is adjusted
        :return adjusted version
        """
        if version >= 0:
            return version
        return self._ml_repo.get_latest_version(name) + 1 + version

    def _get_data(self, name, version=-1, full_object=True):
        """Return data object

        This method returns a data object, i.e. either a DataSet or RawData and is internaly used.

        Keyword Arguments:
            name {string} -- name of data object to be returned
            version {int} -- version of data object (default: {-1})
            full_object {bool} -- complete data if set to True (default: {True})
        """
        result = self._get(name, version, full_object)
        if isinstance(result, repo_objects.DataSet):
            raw_data = self._get(
                result.raw_data, result.raw_data_version, full_object)
            setattr(result, 'x_data', raw_data.x_data)
            setattr(result, 'x_coord_names', raw_data.x_coord_names)
            setattr(result, 'y_data', raw_data.y_data)
            setattr(result, 'y_coord_names', raw_data.y_coord_names)
        return result

    def _add(self, repo_object, message=''):
        """ Add a repo_object to the repository.

            :param repo_object: repo_object to be added, will be modified so that it contains the version number
            :param message: commit message

            :return version number of object added
        """
        repo_object.repo_info[repo_objects.RepoInfoKey.COMMIT_MESSAGE.value] = message
        obj_dict = repo_objects.create_repo_obj_dict(repo_object)
        version = self._ml_repo.add(obj_dict)
        repo_object.repo_info[repo_objects.RepoInfoKey.VERSION.value] = version
        if len(repo_object.repo_info[repo_objects.RepoInfoKey.BIG_OBJECTS]) > 0:
            np_dict = repo_object.numpy_to_dict()
            self._numpy_repo.add(repo_object.repo_info[repo_objects.RepoInfoKey.NAME],
                                 repo_object.repo_info[repo_objects.RepoInfoKey.VERSION],
                                 np_dict)
        return version

    def add_training_data(self, data):
        """Add training data, must be either RawData or DataSet

        This method adds training data. If training data is already contained in the repository, it raises an exception.
        To update training data call update_training_data method.

        Arguments:
            data {RawData or DataSet} -- training data to be added.
        """
        if self._mapping.training_data is not None:
            raise Exception(
                'Training data already contained in repository, use update_training_data to update the data.')
        self._mapping.training_data = data.repo_info[repo_objects.RepoInfoKey.Name.value]
        self._add(data)

    def get_training_data(self, version=-1, full_object=True):
        """Returns training data 

        Keyword Arguments:
            version {integer} -- version of data object
            full_object {bool} -- if True, the complete data is returned including numpy data (default: {True})
        """
        if self._mapping.training_data is None:
            raise Exception("No training_data in repository.")
        return self._get_data(self._mapping.training_data, version, full_object)

    def add_test_data(self, data):
        """Add test data to repository.

        It adds the test data and raises an exception if testdata with the same name already exists.

        Arguments:
            data {RawData or DataSet} -- [description]

        Raises:
            Exception -- [description]
        """
        if self._mapping.test_data is not None and data.repo_info[repo_objects.RepoInfoKey.NAME.value] in self._mapping.test_data:
            raise Exception(
                'Test data already contained in repository, use update_test_data to update the data.')
        if self._mapping.test_data is None:
            self._mapping.test_data = [
                data.repo_info[repo_objects.RepoInfoKey.Name.value]]
        else:
            self._mapping.test_data.append(
                data.repo_info[repo_objects.RepoInfoKey.Name.value])

    def _get(self, name, version=-1, full_object=False):
        """ Get a repo objects. It throws an exception, if an object with the name does not exist.

            :param name: Object name
            :param version: object version, default is latest (-1)
            :param full_object: flag to determine whether the numpy objects are loaded (True->load)
        """
        repo_dict = self._ml_repo.get(name, version)
        result = repo_objects.create_repo_obj(repo_dict)
        numpy_dict = {}
        if len(result.repo_info[repo_objects.RepoInfoKey.BIG_OBJECTS]) > 0 and full_object:
            numpy_dict = self._numpy_repo.get(
                result.repo_info[repo_objects.RepoInfoKey.NAME], result.repo_info[repo_objects.RepoInfoKey.VERSION])
        for x in result.repo_info[repo_objects.RepoInfoKey.BIG_OBJECTS]:
            if not x in numpy_dict:
                numpy_dict[x] = None
        result.numpy_from_dict(numpy_dict)
        return result

    def get_names(self, ml_obj_type):
        """ Get the list of names of all repo_objects from a given repo_object_type in the repository.

            :param ml_obj_type: MLObjectType specifying the types of objects names are returned for.

            :return list of object names for the gigven category.
        """
        return self._ml_repo.get_names(ml_obj_type)

    def get_history(self, name, repo_info_fields=[], obj_member_fields=[], version_start=0, version_end=-1, label=None):
        """ Return a list of histories of object member variables.

        :param repo_info_field: List of fields from repo_info which will be returned in the dictionary. 
                                If List contains flag 'ALL', all fields will be returned.
        :param obj_member_fields: List of member atributes from repo_object which will be returned in the dictionary. 
                                If List contains flag 'ALL', all attributes will be returned.
        """
        version_list = []
        if not label is None:
            raise Exception('Not yet implemented.')
        version_list = range(self._adjust_version(
            version_start, name), self._adjust_version(version_end, name))
        fields = [x for x in repo_info_fields]
        if len(repo_info_fields) == 0:
            fields = [x.value for x in repo_objects.RepoInfoKey]
        fields.extend(obj_member_fields)
        return self._ml_repo.get_history(name, fields, version_list)

    def run_training(self, message='', job_runner=None):
        """ Run the training algorithm. 

        :param message: commit message
        :param job_runner: job runner executing the raining script. Default is single threaded local jobrunner.

        :return ticket number and new model version
        """
        pass

    def run_evaluation(self, message='', model_version=-1, datasets={}):
        """ Evaluate the model on all datasets. 

            :param message: Commit message for this operation.
            :param model_version: Version of model to be evaluated.
            :datasets: Dictionary of datasets (names and version numbers) on which the model is evaluated. 
                Default is all datasets on latest version.
        """
        pass

    def run_tests(self, message='', model_version=-1, tests={}, job_runner=None):
        """ Run tests for a specific model version.

            :param message: Commit message for this operation.
            :param model_version: Version or label of model for which the tests are executed.
            :param tests: Dictionary of tests (names and version numbers) run. Default is all tests on latest version. 
            :param job_runner: job runner executing the tests. Default is single threaded local jobrunner.

            :return ticket number of job
        """
        pass

    def get_model_info(self, model_version=-1):
        """ Return model info for the given model version.

        :param model_version: Version or label of model, default is latest.
        """
        pass

    def set_label(self, model_version, message='', force=True):
        """ Label a certain model version.

            This method labels a certain model version. If force == False it checks if the label 
            already exists an raises in this case an exception.

            :param message: Commit message
            :param model_version: model version for which the label is set.
            :param force: lag determining if ann exception is raise if the label is already used (force->False, exception is raised)
        """
        pass
