# -*- coding: utf-8 -*-
"""This module contains all tests.


"""

import abc
from collections import defaultdict
import logging
from pailab.ml_repo.repo_objects import RepoInfo, RepoObject
from pailab import repo_object_init, MLRepo, RepoInfoKey, MLObjectType, MeasureConfiguration
from pailab.ml_repo.repo_store import LAST_VERSION, FIRST_VERSION
from pailab.ml_repo.repo import Job, NamingConventions, _add_modification_info

logger = logging.getLogger(__name__)


class TestDefinition(RepoObject, abc.ABC):
    """Abstract base class for all test definitions.

    A test definition defines the framework such as models and data the tests are applied to. It also provides a create method
    which creates the test cases for a special model and data version.

    Args:
        models (iterable with str items, optional): Defaults to None. Iterable (e.g. list of str) returning names of the models to be tested.
        data (iterable with str items, optional): Defaults to None. Iterable (e.g. list of str)  returning names of the data used for testing.
        labels (iterable with str items, optional): Defaults to []. Iterable returning labels defining models to be tested.
        repo_info (RepoInfo, optional): Defaults to RepoInfo(). 

    Attributes:
        models (list of str): List of strings defining the models to be tested.
        labels (list of str): List of strings defining the labels to be tested.
        data (list of str): List of strings defining the names of the data to be tested

    """

    def __init__(self, models=None, data=None, labels=[], repo_info=RepoInfo()):
        super(TestDefinition, self).__init__(repo_info)
        self.repo_info.category = MLObjectType.TEST_DEFINITION
        self.models = models
        self.data = data
        self.labels = None

    def _get_models(self, ml_repo: MLRepo):
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

    def _get_data(self, ml_repo: MLRepo):
        data = []
        if self.data is None:
            data.extend(ml_repo.get_names(MLObjectType.TEST_DATA))
            data.extend(ml_repo.get_names(MLObjectType.TRAINING_DATA))
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
        models_test = self._get_models(ml_repo)
        result = []
        data_test = self._get_data(ml_repo)
        for model, v in models_test.items():
            for version in v:
                for d in data_test:
                    tmp = self._create(model, d, version, LAST_VERSION)
                    tmp.test_definition = self.repo_info.name
                    tmp.repo_info[RepoInfoKey.NAME] = str(
                        NamingConventions.Test(model=NamingConventions.get_model_from_name(model), test_name=self.repo_info[RepoInfoKey.NAME],
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
    """Base class for all tests.

    Note: 
        In general, tests are automatically constructed and run using :py:meth:`pailab.ml_repo.repo.run_tests`. As a user, there is nearly no need to
        construct a test by hand.

    Args:
        model (str): Name of model for which the test is applied.
        data (str): Name of dataset used in the test.
        test_definition_version (str, optional): Defaults to latest version. Version of the tests's underlying :py:class:`pailab.tools.tests.TestDefinition` that is used as basis for the test.
        model_version (str, optional): Defaults to latest version. Version of the model the test is applied to.
        data_version (str, optional): Defaults to latest version. Version of the data used in the test.

    Args:
        model (str): Name of model for which the test is applied.
        data (str): Name of dataset used in the test.
        test_definition_version (str, optional): Defaults to latest version. Version of the tests's underlying :py:class:`pailab.tools.tests.TestDefinition` that is used as basis for the test.
        model_version (str, optional): Defaults to latest version. Version of the model the test is applied to.
        data_version (str, optional): Defaults to latest version. Version of the data used in the test.

    Attributes:
        test_definition (str): Name of underlying  :py:class:`pailab.tools.tests.TestDefinition`.
        model (str): Name of model for which the test is applied.
        data (str): Name of dataset used in the test.
        test_definition_version (str): Version of the tests's underlying :py:class:`pailab.tools.tests.TestDefinition` that is used as basis for the test.
        model_version (str): Version of the model the test is applied to.
        data_version (str): Version of the data used in the test.
        result (str, 'not run', 'failed', 'succeeded'):  Describes the state of the test.
        details (dict): Contains details when test fails, otherwise empty dict.
    """

    def __init__(self, model, data, test_definition_version=LAST_VERSION, model_version=LAST_VERSION, data_version=LAST_VERSION, repo_info=RepoInfo()):
        super(Test, self).__init__(repo_info)
        self.test_definition = None
        self.test_definition_version = test_definition_version
        self.model_version = model_version
        self.model = model
        self.data_version = data_version
        self.data = data
        self.result = 'not run'
        self.details = {}

    def _run(self, ml_repo: MLRepo, jobid):
        """Method called from Job base class 

        Args:
            ml_repo (MLRepo): Repo used to retrieve and store data.
            jobid (str): Id of job
        """

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

    @abc.abstractmethod
    def _check(self, ml_repo: MLRepo):
        """Checks if current test is up-to-date and successfully finished in repo.

        Args:
            ml_repo (MLRepo): repository

        Returns:
            None or string with description. 
        """
        pass


class RegressionTestDefinition(TestDefinition):
    """Definition of a regression test.

    A regression test compares a specified measure of a reference model described by a label to the respective measure of the model
    to be tested. It fails, if the measure of the tested model is greater then a given tolerance of the reference measure, i.e. the test fails if

        - measure-measure_ref < tol and an absolute tolerance is defined,
        - measure-measure_ref < tol*measure_ref if a relative tolerance is used.

    Note:
        The tests needs the chosen measure(s) to be computed, therefore you have to take care that the measure has 
        been added to the repo (using :py:meth:`pailab.ml_repo.repo.MLRepo.add_measure`)

    Examples:
        Add a test for the model `'my_model'` on a data set named `'test_data'` which checks if the 
        maximum error of the model is not greater than 10% in relation to the error of the reference model defined by the label `'production_model'`

        >>> test_def = RegressionTestDefinition(models=['my_model'], reference ='production_model', data = ['test_data'], measures = ['max'], tol = 0.1, relative = True)
        >>> ml_repo.add(test_def)

        Add a test applied to all models in the repo (always the latest versions of the models are used within the tests)


        >>> test_def = RegressionTestDefinition(models=None, reference ='production_model', data = ['test_data'], measures = ['max'], tol = 0.1, relative = True)
        >>> ml_repo.add(test_def)

    Args:
        models (iterable with str items, optional): Defaults to None. Iterable (e.g. list of str) returning names of the models to be tested.
        data (iterable with str items, optional): Defaults to None. Iterable (e.g. list of str)  returning names of the data used for testing.
        labels (iterable with str items, optional): Defaults to []. Iterable returning labels defining models to be tested.
        measures ([type], optional): Defaults to None. List of measures used in the test
        reference (str, optional): Defaults to 'prod'. Label defining the reference model to which the measures are compared.
        tol (float, optional): Defaults to 1e-3. Tolerance, if relative is False, the test fails if new_value-ref_value < tol, otherwise if new_value-ref_value < tol*ref_value.
        relative (bool, optional): Defaults to False. 

        repo_info (RepoInfo, optional): Defaults to RepoInfo(). 

    Attributes:
        models (list of str): List of strings defining the models to be tested.
        labels (list of str): List of strings defining the labels to be tested.
        data (list of str): List of strings defining the names of the data to be tested

    """

    def __init__(self, reference='prod', models=None, data=None, labels=None,
                 measures=None,  tol=1e-3, repo_info=RepoInfo(), relative=False):

        super(RegressionTestDefinition, self).__init__(
            models, data, labels, repo_info=repo_info)
        self.measures = measures
        self.reference = reference
        self.tol = tol
        self.repo_info.category = MLObjectType.TEST_DEFINITION
        self.relative = relative

    def _create(self, model, data, model_version, data_version):
        return RegressionTest(model, data, self.repo_info[RepoInfoKey.VERSION], model_version, data_version, repo_info={})


class RegressionTest(Test):
    """Regression test.

    Note: 
        In general, tests are automatically constructed and run using :py:meth:`pailab.ml_repo.repo.run_tests`. As a user, there is nearly no need to
        construct a test by hand.

    A regression test compares a specified measure of a reference model described by a label to the respective measure of the model
    to be tested. It fails, if the measure of the tested model is greater then a given tolerance of the reference measure, i.e. the test fails if

        - measure-measure_ref < tol and an absolute tolerance is defined,
        - measure-measure_ref < tol*measure_ref if a relative tolerance is used.

    All the attributes specific for the regression test (i.e. not contained in the base class) are retrieved during the run of the test
    from the underlying testdefinition.

    Args:
        model (str): Name of model for which the test is applied.
        data (str): Name of dataset used in the test.
        test_definition_version (str, optional): Defaults to latest version. Version of the tests's underlying :py:class:`pailab.tools.tests.TestDefinition` that is used as basis for the test.
        model_version (str, optional): Defaults to latest version. Version of the model the test is applied to.
        data_version (str, optional): Defaults to latest version. Version of the data used in the test.

    Attributes:
        test_definition (str): Name of underlying  :py:class:`pailab.tools.tests.TestDefinition`.
        model (str): Name of model for which the test is applied.
        data (str): Name of dataset used in the test.
        test_definition_version (str): Version of the tests's underlying :py:class:`pailab.tools.tests.TestDefinition` that is used as basis for the test.
        model_version (str): Version of the model the test is applied to.
        data_version (str): Version of the data used in the test.
        result (str, 'not run', 'failed', 'succeeded'):  Describes the state of the test.
        details (dict): Contains details when test fails, otherwise empty dict.
    """

    def __init__(self,  model, data, test_definition_version=LAST_VERSION,
                 model_version=LAST_VERSION, data_version=LAST_VERSION, repo_info=RepoInfo()):
        super(RegressionTest, self).__init__(
            model, data, test_definition_version, model_version, data_version, repo_info=repo_info)

    def _get_measure_types(self, ml_repo: MLRepo, reg_test=None):
        if reg_test is None:
            reg_test = ml_repo.get(
                self.test_definition, version=LAST_VERSION)
        measure_types = reg_test.measures
        if measure_types is None:
            tmp = ml_repo.get_names(MLObjectType.MEASURE_CONFIGURATION)
            if len(tmp) == 0:
                raise Exception(
                    'No regression test possible since no measure defined.')
            m_config = ml_repo.get(tmp[0], version=LAST_VERSION)
            measure_types = [MeasureConfiguration.get_name(
                x) for k, x in m_config.measures.items()]
        return measure_types

    def _run_test(self, ml_repo: MLRepo, jobid):
        logger.debug('Running regression test ' + self.repo_info.name + ' on model ' +
                     str(NamingConventions.CalibratedModel(self.model)) + ', version ' + self.model_version)
        regression_test = ml_repo.get(
            self.test_definition, version=LAST_VERSION)
        label = ml_repo.get(regression_test.reference, version=LAST_VERSION)
        result = {}
        measure_types = self._get_measure_types(ml_repo, regression_test)
        for measure_type in measure_types:
            measure_name = str(NamingConventions.Measure(
                {'model': self.model.split('/')[0], 'data': self.data, 'measure_type': measure_type}))
            measure = ml_repo.get(measure_name, version=None,
                                  modifier_versions={
                                      str(NamingConventions.CalibratedModel(self.model)): self.model_version,
                                      self.data: self.data_version
                                  }, throw_error_not_exist=False, throw_error_not_unique=True
                                  )
            if measure == []:
                continue
            measure_name = str(NamingConventions.Measure(
                {'model': label.name.split('/')[0], 'data': self.data, 'measure_type': measure_type}))
            reference_value = ml_repo.get(measure_name, version=None,
                                          modifier_versions={str(NamingConventions.CalibratedModel(
                                              label.name)): label.version,
                                              self.data: self.data_version}, adjust_modification_info=False)
            if regression_test.relative:
                if measure.value-reference_value.value < regression_test.tol*reference_value.value:
                    result[measure_type] = {
                        'reference_value': reference_value.value, 'value': measure.value}
            else:
                if measure.value-reference_value.value < regression_test.tol:
                    result[measure_type] = {
                        'reference_value': reference_value.value, 'value': measure.value}
        return result

    def get_modifier_versions(self, ml_repo):
        modifiers = {}
        modifiers[str(NamingConventions.CalibratedModel(
            self.model))] = self.model_version
        modifiers[self.test_definition] = self.test_definition_version
        regression_test = ml_repo.get(
            self.test_definition, version=LAST_VERSION)
        label = ml_repo.get(regression_test.reference, version=LAST_VERSION)
        modifiers[label.repo_info.name] = label.repo_info.version
        modifiers[self.data] = self.data_version
        return self.repo_info.name, modifiers

    def _check(self, ml_repo: MLRepo):
        # check if test is based on latest test definition
        regression_test = ml_repo.get(
            self.test_definition, version=LAST_VERSION)
        if regression_test.repo_info.version != self.repo_info.modification_info[self.test_definition]:
            return 'Test is not based on latest test definition, latest version: ' + regression_test.repo_info.version + ', version used for test: ' + self.modification_info[self.test_definition]
        # check if measure config did not change
        if regression_test.measures is None:
            tmp = ml_repo.get_names(MLObjectType.MEASURE_CONFIGURATION)
            if len(tmp) == 0:
                raise Exception(
                    'No check possible since no measure defined.')
            m_config = ml_repo.get(tmp[0], version=LAST_VERSION)
            if m_config.repo_info.version != self.repo_info.modification_info[m_config.repo_info.name]:
                return 'Test is not based on latest measure configuration, latest version: ' + m_config.repo_info.version + ', version used for test: ' + self.modification_info[m_config.repo_info.name]
        #  check if ref model did not change
        label = ml_repo.get(regression_test.reference, version=LAST_VERSION)
        if not label.repo_info.name in self.repo_info.modification_info.keys():
            return 'Test on different reference model.'
        if not label.repo_info.version == self.repo_info.modification_info[label.repo_info.name]:
            return 'Test on old reference model.'
        # check if test was on latest data version
        if not self.data in self.repo_info.modification_info.keys():
            return 'Data of test has changed since last test.'
        version = self.data_version
        if version == LAST_VERSION:
            version = ml_repo._ml_repo.get_latest_version(self.data)
        elif version == FIRST_VERSION:
            version = ml_repo._ml_repo.get_first_version(self.data)
        if not version == self.repo_info.modification_info[self.data]:
            return 'Data of test has changed since last test.'
        return None
