import abc
import logging
from pailab import repo_object_init, MLRepo, RepoInfoKey, MLObjectType
from pailab.repo_store import LAST_VERSION, FIRST_VERSION
from pailab.repo import Job, NamingConventions, _add_modification_info

logger = logging.getLogger(__name__)


class TestDefinition(abc.ABC):
    @repo_object_init()
    def __init__(self, model='.*', data='.*'):
        self.model = model
        self.data = data

    def create(self, model_version, data_version):
        result = self._create(model_version, data_version)
        result.repo_info[RepoInfoKey.NAME] = str(
            NamingConventions.Test(self.repo_info[RepoInfoKey.NAME]))
        return result

    @abc.abstractmethod
    def _create(self, model_version, data_version):
        pass


class Test(Job):
    def __init__(self, test_definition=LAST_VERSION, model_version=LAST_VERSION, data_version=LAST_VERSION):
        super(Test, self).__init__()
        self.test_definition = test_definition
        self.model_version = model_version
        self.data_version = data_version
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
    def __init__(self, model='.*', data='.*', measures=None, reference='prod', tol=1e-3):
        """Regression test definition

        It defines a test where measures are compared against values from a reference model. The test fails if the new measure exceeds the reference measure
        with 
        
        Args:
            TestDefinition ([type]): [description]
            model (str, optional): Defaults to '.*'. [description]
            data (str, optional): Defaults to '.*'. [description]
            measures ([type], optional): Defaults to None. [description]
            reference (str, optional): Defaults to 'prod'. [description]
            tol ([type], optional): Defaults to 1e-3. [description]
        """

        super(RegressionTestDefinition, self).__init__()
        self.measures = measures
        self.reference = reference
        self.tol = tol

    def _create(self, model_version, data_version):
        return RegressionTest(self.model, self.data, self.repo_info[RepoInfoKey.VERSION], model_version, data_version)


class RegressionTest(Test):
    @repo_object_init()
    def __init__(self,  model='.*', data='.*', test_definition=LAST_VERSION, model_version=LAST_VERSION, data_version=LAST_VERSION):
        super(RegressionTest, self).__init__(
            model, data, model_version, data_version)

    def _run_test(self, ml_repo: MLRepo, jobid):
        regression_test = ml_repo.get(
            self.test_definition, version=LAST_VERSION)
        label = ml_repo.get(regression_test.reference)
        result = {}
        for measure_type in regression_test.measures:
            measure_name = str(NamingConventions.Measure(
                {'model': regression_test.model, 'data': regression_test.data, 'measure_type': measure_type}))
            measure = ml_repo.get(measure_name,
                                  modifier_versions={
                                      str(NamingConventions.CalibratedModel(regression_test.model)): self.model_version,
                                      regression_test.data: self.data_version
                                  }
                                  )
            measure_name = str(NamingConventions.Measure(
                {'model': label.name, 'data': regression_test.data, 'measure_type': measure_type}))
            reference_value = ml_repo.get(measure_name,
                                          modifier_versions={NamingConventions.CalibratedModel(
                                              regression_test.model): label.version,
                                              regression_test.data: self.data_version})
            if measure.value-reference_value.value > regression_test.tol:
                result[measure_type] = {
                    'reference_value': reference_value.value, 'value': measure.value}
        return result
