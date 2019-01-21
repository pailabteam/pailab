import abc
from collections import defaultdict
import logging
from pailab import repo_object_init, MLRepo, RepoInfoKey, MLObjectType, MeasureConfiguration
from pailab.repo_store import LAST_VERSION, FIRST_VERSION
from pailab.repo import Job, NamingConventions, _add_modification_info

logger = logging.getLogger(__name__)


class TestDefinition(abc.ABC):
    """Abstract base class for all test definitions.

    A test definition defines the framework such as models and data the tests are applied to. It also provides a create method
    which creates the test cases for a special model and data version.

    """

    @repo_object_init()
    def __init__(self, models=None, data=None, labels=[]):
        self.models = models
        self.data = data
        self.labels = None

    def __get_models(self, ml_repo: MLRepo):
        models_test = defaultdict(set)
        if self.models is None:
            tmp = ml_repo.get_names(MLObjectType.CALIBRATED_MODEL)
            for k in tmp:
                m = ml_repo.get(k, full_object=False)
                models_test[k].add(m.repo_info[RepoInfoKey.VERSION])

        else:
            for k, v in self.models.items():
                models_test[k].add(v)
        if self.labels is None:
            labels = ml_repo.get_names(MLObjectType.LABEL)
        else:
            labels = self.labels
        for l in labels:
            tmp = ml_repo.get(l)
            models_test[tmp.name].add(tmp.version)
        return models_test

    def __get_data(self, ml_repo: MLRepo):
        data = []
        if self.data is None:
            for k in ml_repo.get_names(MLObjectType.TEST_DATA):
                data.append(k)
            tmp = ml_repo.get_names(MLObjectType.TRAINING_DATA)
            for k in tmp:
                data.append(k)
        else:
            data = self.data
        return data

    def create(self, ml_repo: MLRepo):
        """Create a set of tests for models of the repository.

        Args:
            ml_repo (MLRepo): ml repo
            models (dict, optional): Defaults to {}. Dictionary of model names to version numbers to apply tests for. If empty, all latest models are used.
            data (dict, optional): Defaults to {}. Dictionary of data the tests are applied to. If empty, all latest test- and train data will be used.
            labels (list, optional): Defaults to []. List of labels to which the tests are applied.

        Returns:
            [type]: [description]
        """
        models_test = self.__get_models(ml_repo)
        result = []
        data_test = self.__get_data(ml_repo)
        for model, v in models_test.items():
            for version in v:
                for d in data_test:
                    tmp = self._create(model, d, version, version)
                    tmp.test_definition = self.repo_info.name
                    tmp.repo_info[RepoInfoKey.NAME] = str(
                        NamingConventions.Test(model=model, test_name=self.repo_info[RepoInfoKey.NAME],
                                               data=d))
                    result.append(tmp)
        return result

    @abc.abstractmethod
    def _create(self, model, data, model_version, data_version):
        """Create a specific test for the given model and data

        Args:
            model (str): name of model
            data (str): [name of data
            model_version (str): version of model
            data_version (str): version of data
        """

        pass


class Test(Job):
    def __init__(self, model, data, test_definition_version=LAST_VERSION, model_version=LAST_VERSION, data_version=LAST_VERSION):
        super(Test, self).__init__()
        self.test_definition = None
        self.test_definition_version = test_definition_version
        self.model_version = model_version
        self.model = model
        self.data_version = data_version
        self.data = data
        self.result = 'not run'
        self.details = {}

    def _run(self, ml_repo: MLRepo, jobid):
        result = self._run_test(ml_repo, jobid)
        if len(result) > 0:
            self.result = 'failed'
            self.details = result
        else:
            self.result = 'succeeded'
            self.details = {}

    @abc.abstractmethod
    def _run_test(self, ml_repo: MLRepo, jobid):
        pass


class RegressionTestDefinition(TestDefinition):
    @repo_object_init()
    def __init__(self, reference='prod', models=None, data=None, labels=None, measures=None,  tol=1e-3):
        """Regression test definition

        It defines a test where measures are compared against values from a reference model. The test fails if the new measure exceeds the reference measure
        with 

        Args:
            model (str, optional): Defaults to '.*'. Regular expression defining the models the test should be applied to.
            data (str, optional): Defaults to '.*'. Regular expression defining the data the test should be applied to.
            measures ([type], optional): Defaults to None. List of measures used in the test
            reference (str, optional): Defaults to 'prod'. Label defining th reference model to which the measures are compared.
            tol ([type], optional): Defaults to 1e-3. Tolerance, if new_value-ref_value < tol, the test fails.
        """

        super(RegressionTestDefinition, self).__init__(models, data, labels)
        self.measures = measures
        self.reference = reference
        self.tol = tol

    def _create(self, model, data, model_version, data_version):
        return RegressionTest(model, data, self.repo_info[RepoInfoKey.VERSION], model_version, data_version, repo_info={})


class RegressionTest(Test):
    @repo_object_init()
    def __init__(self,  model, data, test_definition_version=LAST_VERSION, model_version=LAST_VERSION, data_version=LAST_VERSION):
        super(RegressionTest, self).__init__(
            model, data, model_version, data_version)

    def _run_test(self, ml_repo: MLRepo, jobid):
        regression_test = ml_repo.get(
            self.test_definition, version=LAST_VERSION)
        label = ml_repo.get(regression_test.reference)
        result = {}
        measure_types = regression_test.measures
        if measure_types is None:
            tmp = ml_repo.get_names(MLObjectType.MEASURE_CONFIGURATION)
            if len(tmp) == 0:
                raise Exception(
                    'No regression test possible since no measure defined.')
            m_config = ml_repo.get(tmp[0])
            measure_types = [MeasureConfiguration.get_name(
                x) for k, x in m_config.measures.items()]
        for measure_type in measure_types:
            measure_name = str(NamingConventions.Measure(
                {'model': self.model.split('/')[0], 'data': self.data, 'measure_type': measure_type}))
            measure = ml_repo.get(measure_name,
                                  modifier_versions={
                                      str(NamingConventions.CalibratedModel(self.model)): self.model_version,
                                      self.data: self.data_version
                                  }
                                  )
            measure_name = str(NamingConventions.Measure(
                {'model': label.name.split('/')[0], 'data': self.data, 'measure_type': measure_type}))
            reference_value = ml_repo.get(measure_name,
                                          modifier_versions={str(NamingConventions.CalibratedModel(
                                              label.name)): label.version,
                                              self.data: self.data_version})
            if measure.value-reference_value.value > regression_test.tol:
                result[measure_type] = {
                    'reference_value': reference_value.value, 'value': measure.value}
        return result
