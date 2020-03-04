"""Machine Learning Repository

This module contains pailab's machine learning repository.
"""
import abc
import json
import os
from datetime import datetime
from numpy import linalg
from numpy import inf, load
import numpy as np
from enum import Enum
from copy import deepcopy
from types import SimpleNamespace
import logging
import pailab.ml_repo.repo_objects as repo_objects
from pailab.ml_repo.repo_objects import RepoInfoKey, RawData, DataSet, MeasureConfiguration
from pailab.ml_repo.repo_objects import repo_object_init, RepoInfo, RepoObject  # pylint: disable=E0401
import pailab.ml_repo.repo_store as repo_store
from pailab.ml_repo.repo_store_factory import RepoStoreFactory, NumpyStoreFactory
from pailab.job_runner.job_runner_factory import JobRunnerFactory

logger = logging.getLogger(__name__)


class MLObjectType(Enum):
    """Enum describing all ml object types.

    The MLObjectType is assigned to each object in the MLRepo. It is used to structure all objects and to support consistency checks and 
    automatic pipelines, the following types are defined:

        - EVAL_DATA: evaluation data (result from evaluation of a model)
        - RAW_DATA: raw data, i.e. simple numpy structures most often used to derive test or training data from the RawData
        - TRAINING_DATA: training data used for model training
        - TEST_DATA: data used for model testing
        - TEST: concrete test
        - TEST_DEFINITION: definition of a test which is applied to respective data and models to obtain a test
        - MODEL_PARAM: model parameter
        - TRAINING_PARAM: training parameter
        - TRAINING_FUNCTION: function to train e certain model
        - MODEL_EVAL_FUNCTION: function to evaluate a certain model
        - PREPROCESSOR_PARAM: preprocessing parameter
        - PREPROCESSOR: definition of a preprocessor
        - PREPROCESSING_FITTING_FUNCTION: function to fit the preprocessor
        - PREPROCESSING_TRANSFORMING_FUNCTION: function to apply preprocessing to data
        - LABEL: model label
        - MODEL: definition of a model
        - CALIBRATED_MODEL: object containing a calibrated instande of a model
        - COMMIT_INFO: internally used to store commit messages
        - MAPPING: internally used mapping object to map an object's name to the object's category
        - MEASURE: computed measure (e.g. norm of error)
        - MEASURE_CONFIGURATION: the configuration of all measures applied to the model
        - RESULT: object holding results
        - JOB: a job
        - TRAINING_STATISTIC: object holding training statistics, e.g. training history
        - CACHED_VALUE: cached return values of time consuming functions
    """
    EVAL_DATA = 'EVAL_DATA'
    RAW_DATA = 'RAW_DATA'
    TRAINING_DATA = 'TRAINING_DATA'
    TEST_DATA = 'TEST_DATA'
    TEST = 'TEST'
    TEST_DEFINITION = 'TEST_DEFINITION'
    MODEL_PARAM = 'MODEL_PARAM'
    TRAINING_PARAM = 'TRAINING_PARAM'
    TRAINING_FUNCTION = 'TRAINING_FUNCTION'
    MODEL_EVAL_FUNCTION = 'MODEL_EVAL_FUNCTION'
    PREPROCESSOR_PARAM = 'PREPROCESSOR_PARAM'
    PREPROCESSOR = 'PREPROCESSOR'
    PREPROCESSING_FITTING_FUNCTION = 'PREPROCESSING_FITTING_FUNCTION'
    PREPROCESSING_TRANSFORMING_FUNCTION = 'PREPROCESSING_TRANSFORMING_FUNCTION'
    LABEL = 'LABEL'
    MODEL = 'MODEL'
    CALIBRATED_MODEL = 'CALIBRATED_MODEL'
    COMMIT_INFO = 'COMMIT_INFO'
    MAPPING = 'MAPPING'
    MEASURE = 'MEASURE'
    MEASURE_CONFIGURATION = 'MEASURE_CONFIGURATION'
    RESULT = 'RESULT'
    JOB = 'JOB'
    TRAINING_STATISTIC = 'TRAINING_STATISTIC'
    CACHED_VALUE = 'CACHED_VALUE'

    @staticmethod
    def _get_key(category):
        """ Returns a standardized key for the given category

        Args:
            category (MLObjectType or str (name or value of MLObjectType)): MLObjectType or string defining the enum

        Raises:
            Exception: If no MLObjectType exists matching the given category or if category is of wrong type

        Returns:
            str -- string which can be used e.g. in a dictionary
        """

        category_name = None
        if isinstance(category, MLObjectType):
            category_name = category.value
        if isinstance(category, str):
            for k in MLObjectType:
                if k.value == category or k.name == category:
                    category_name = k.value
                    break
        if category_name is None:
            raise Exception('No category ' + str(category) +
                            ' exists.')  # pragma: no cover
        return category_name


class Mapping(RepoObject):
    """ Provides a mapping from MLObjectType to all objects in the repo belonging to this type
    """

    def __init__(self, repo_info=RepoInfo(), **kwargs):
        super(Mapping, self).__init__(repo_info)
        logging.debug('Initializing map with kwargs: ' + str(kwargs))
        for key in MLObjectType:
            setattr(self, key.value, [])
        self.set_fields(kwargs)

    def set_fields(self, kwargs):
        """ Set  fields from a dictionary

        Args:
            kwargs (dictionary): sets the fields in the mapping
        """

        if not kwargs is None:
            for key in MLObjectType:
                if key.name in kwargs.keys():
                    setattr(self, key.value, kwargs[key.name])
                else:
                    if key.value in kwargs.keys():
                        setattr(self, key.value, kwargs[key.value])
                    else:
                        if key in kwargs.keys():
                            setattr(self, key.value, kwargs[key])

    def add(self, category, name):
        """ Add a object to the category mapping if it does not already exists

        Args:
            category (MLObjectType): The category of the object to be added
            name (str): The name of the object to be added

        Raises:
            Exception: Raises an exception if a second training_data set should be added

        Returns:
            bool -- true, if mapping has changed, false otherwise
        """

        category_name = MLObjectType._get_key(category)
        mapping = getattr(self, category_name)
        if not name in mapping:
            mapping.append(name)
            return True
        return False

    def __getitem__(self, category):
        """ Get an item

        Args:
            category (MLObjectType): The category of the object to return

        Returns:
            [object] -- Key of specified category.
        """

        category_name = MLObjectType._get_key(category)
        return getattr(self, category_name)

    def __str__(self):
        """ Returns a string representation of the object

        Returns:
            str -- the return string
        """

        result = ''
        for k in MLObjectType:
            result += k.value + ': '
            if hasattr(self, k.value):
                result += str(getattr(self, k.value)) + ','
            else:
                result += '[],'
        return result


def _add_modification_info(repo_obj, *args):
    """ Add/update modificaion info to a repo object from a list if repo objects which were used to create it.

    Args:
        repo_obj (repoobject): the object the modification info is added to
    """

    mod_info = repo_obj.repo_info[RepoInfoKey.MODIFICATION_INFO]
    for v in args:
        if v is not None:
            mod_info[v.repo_info[RepoInfoKey.NAME]
                     ] = v.repo_info[RepoInfoKey.VERSION]
            # for k, w in v.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
            # check if there is a bad inconsistency: if an object modifying the current object has already been used with a different version
            # number, this must be annotated
            # if k in mod_info.keys() and mod_info[k] != w:
            #    mod_info['inconsistency'] = 'object ' + k + \
            #        ' has already been used with a different version number'
            #mod_info[k] = w

# region Jobs


class Job(RepoObject, abc.ABC):
    """ A class for the jobs
    """

    def __init__(self, repo_info):
        super(Job, self).__init__(repo_info)
        self.state = 'created'
        self.started = 'not yet started'
        self.finished = 'not yet finished'

    class RepoWrapper:
        """ Abstract class defining the interfaces needed for a job to be used in the JobRunner
        """

        def __init__(self, job, ml_repo):
            self.versions = {}
            self.ml_repo = ml_repo
            for k in job.get_predecessor_jobs():
                job = ml_repo.get(k[0], version=k[1])
                for k, v in job.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
                    self.versions[k] = v
            self.modification_info = {}

        def _get_version(self, name, orig_version):
            """ Get the version of the object

            Args:
                name (str): The name of the object
                orig_version ([type]): [description]

            Returns:
                [type] -- [description]
            """

            if orig_version is None:
                return orig_version
            if name in self.versions.keys():
                return self.versions[name]
            return orig_version

        def get(self, name, version=None, full_object=False,
                modifier_versions=None, obj_fields=None,  repo_info_fields=None,
                throw_error_not_exist=True, throw_error_not_unique=True, adjust_modification_info=True):
            """ Get function

            Args:
                name (str): The name of the object to return
                version (str): An explicit version of the object can be returned. Defaults to None.
                full_object (bool): Determines whether to return the full object - meaning including large object parts. Defaults to False.
                modifier_versions (str): [description]. Defaults to None.
                obj_fields ([type]): [description]. Defaults to None.
                repo_info_fields ([type]): [description]. Defaults to None.
                throw_error_not_exist (bool): true - throw error if not exists, else return []. Defaults to True.
                throw_error_not_unique (bool): true - throw error if item is not unique, else return []. Defaults to True.
                adjust_modification_info (bool): [description]. Defaults to True.
            """

            new_v = self._get_version(name, version)
            m_v = None
            if modifier_versions is not None:
                m_v = deepcopy(modifier_versions)
                for k, v in m_v.items():
                    m_v[k] = self._get_version(k, v)
            obj = self.ml_repo.get(name, version=new_v, full_object=full_object,
                                   modifier_versions=m_v, obj_fields=obj_fields,  repo_info_fields=repo_info_fields,
                                   throw_error_not_exist=throw_error_not_exist, throw_error_not_unique=throw_error_not_unique)
            if isinstance(obj, list):
                if len(obj) == 0:
                    if throw_error_not_exist:
                        raise Exception('More than one object with name ' + name +
                                        ' found meeting the conditions.')  # pragma: no cover
                    else:
                        return []
                else:
                    if throw_error_not_unique:
                        raise Exception(
                            'More than one object with name ' + name + ' found meeting the conditions.')
                    else:
                        return []
            if adjust_modification_info:
                self.modification_info[name] = obj.repo_info[RepoInfoKey.VERSION]
                for k, v in obj.repo_info.modification_info.items():
                    self.modification_info[k] = v
            return obj

        def add(self, obj, message, category=None):
            """ Add an object

            Args:
                obj ([type]): The object to add
                message (str): A commit message
                category (MLObjectType): The object type, if None the type is determined by the object. Defaults to None.
            """

            self.ml_repo.add(obj, message, category=category)
            if isinstance(obj, list):
                for o in obj:
                    self.modification_info[o.repo_info[RepoInfoKey.NAME]
                                           ] = o.repo_info[RepoInfoKey.VERSION]
            else:
                self.modification_info[obj.repo_info[RepoInfoKey.NAME]
                                       ] = obj.repo_info[RepoInfoKey.VERSION]

        def get_training_data(self, obj_version, full_object, model=None, model_version=repo_store.RepoStore.LAST_VERSION):
            """ Get the training data

            Args:
                obj_version (str): The object version to return
                full_object (bool):  Determines whether to return the full object - meaning including large object parts

            Returns:
                [type] -- training data
            """

            # TODO: replace get_training data by get using the training data name
            tmp = self.ml_repo.get_training_data(
                obj_version, full_object=False, model=model, model_version=model_version)
            return self.get(tmp.repo_info[RepoInfoKey.NAME], obj_version, full_object=full_object)

        def get_names(self, ml_obj_type):
            return self.ml_repo.get_names(ml_obj_type)

    def add_predecessor_job(self, job):
        """ Add job as predecessor which must have been run successfull before the job can be started

        Args:
            predecessors (list o jobids): predecessors
        """

        if isinstance(job, tuple):
            self._predecessors.append(job)
        else:
            self._predecessors.append(
                (job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION]))

    def set_predecessor_jobs(self, predecessors):
        """ Sets the predecessor jobs

        Args:
            predecessors (list of strings): the predecessors to be added to the job
        """

        self._predecessors = []
        for obj in predecessors:
            self.add_predecessor_job(obj)

    def get_predecessor_jobs(self):
        """ Returns a list of jobids which must have been run sucessfully before the job will be executed

        Returns:
            list of strings -- list of jobids
        """

        if hasattr(self, '_predecessors'):
            return self._predecessors
        return []

    def run(self, ml_repo, jobid):
        """ Run a job

        Args:
            ml_repo (MLrepository): repository used to get and store the data
            jobid (str): the job id to be executed

        Raises:
            e -- The execution error to be raised
        """

        wrapper = Job.RepoWrapper(self, ml_repo)
        self.state = 'running'
        self.started = str(datetime.now())
        ml_repo._update_job(self)
        try:
            self._run(wrapper,  jobid)
            self.finished = str(datetime.now())
            self.state = 'finished'
            self.repo_info[RepoInfoKey.MODIFICATION_INFO] = wrapper.modification_info
            ml_repo._update_job(self)
        except Exception as e:
            self.finished = str(datetime.now())
            self.state = 'error'
            self.repo_info[RepoInfoKey.MODIFICATION_INFO] = wrapper.modification_info
            self.error_message = str(e)
            ml_repo._update_job(self)
            raise e from None

    def check_rerun(self, ml_repo):
        """ Check whether the job must be executed again

        Args:
            ml_repo (MLrepository): repository used to get and store the data

        Returns:
            bool -- Info whether the job must be rerun
        """

        name, modifier_versions = self.get_modifier_versions(ml_repo)
        job = ml_repo.get(self.repo_info.name, version=None,
                          modifier_versions=modifier_versions, throw_error_not_exist=False)
        if job == []:
            return True
        if job.state == 'error':
            return True
        return False

    @abc.abstractmethod
    def _run(self, ml_repo, jobid):
        """ Run the job with data from the given repo

        Args:
            repo (MLrepository): repository used to get and store the data
            jobid (str): the job id to be executed
        """

        pass

    @abc.abstractmethod
    def get_modifier_versions(self, ml_repo):
        """ Get the modifier versions

        Args:
            repo (MLrepository): repository used to get and store the data

        Returns:
            tuple of string, dict -- return the object name and the modifiers
        """

        pass


class EvalJob(Job):
    """ Definition of a model evaluation job
    """

    def __init__(self, model, data, user, eval_function_version=repo_store.RepoStore.LAST_VERSION,
                 model_version=repo_store.RepoStore.LAST_VERSION, data_version=repo_store.RepoStore.LAST_VERSION,
                 repo_info=RepoInfo()):
        """ Init function for the EvalJob

        Args:
            model ([type]): [description]
            data ([type]): [description]
            user (str): The user
            eval_function_version (str): version of the evaluation function. Defaults to repo_store.RepoStore.LAST_VERSION.
            model_version (str): version of the model. Defaults to repo_store.RepoStore.LAST_VERSION.
            data_version (str): version of the data. Defaults to repo_store.RepoStore.LAST_VERSION.
            repo_info ([type]): [description]}). Defaults to RepoInfo().
        """

        super(EvalJob, self).__init__(repo_info)
        self.model = model
        self.data = data
        self.user = user
        self.eval_function_version = eval_function_version
        self.model_version = model_version
        self.data_version = data_version
        # list of jobids which must have been run before this job should be excuted
        self.predecessors = []

    def _run(self, repo, jobid):
        """ Run the job with data from the given repo

        Args:
            repo (MLrepository): repository used to get and store the data
            jobid (str): the job id to be executed
        """

        logging.info('Start evaluation job ' + str(jobid) + ' on model ' +
                     self.model + ' ' + str(self.model_version) + ' ')
        model = repo.get(self.model, self.model_version, full_object=True)
        model_definition_name = self.model.split('/')[0]
        model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO][model_definition_name]
        model_definition = repo.get(model_definition_name, model_def_version)

        data = repo.get(self.data, self.data_version, full_object=True)
        eval_func = repo.get(model_definition.eval_function,
                             self.eval_function_version)

        x_data = data.x_data
        x_coord_names = data.x_coord_names
        if model.preprocessors is not None:
            for k in range(len(model.preprocessors)):
                prepro = model.preprocessors[k]
                transforming_func = repo.get(
                    prepro.transforming_function, model.repo_info.modification_info[prepro.transforming_function])
                prepro_param = None
                if not prepro.preprocessing_param == None:
                    prepro_param = repo.get(
                        prepro.preprocessing_param, model.repo_info.modification_info[prepro.preprocessing_param])
                if prepro.fitting_function is not None:
                    x_data, x_coord_names = transforming_func.create()(
                        prepro_param, x_data, x_coord_names, model.fitted_preprocessors[k])
                else:
                    x_data, x_coord_names = transforming_func.create()(
                        prepro_param, x_data, x_coord_names)
            data.x_data = x_data
            data.x_coord_names = x_coord_names
        y = eval_func.create()(model, data)
        y_name = data.y_coord_names
        if y_name is None:
            # if y_name is None, we just use the x-coord names (maybe we have an autoencoder?)
            if len(x_coord_names) == y.shape[1]:
                y_name = x_coord_names
            else:
                raise Exception('No y_coord_names defined.')
        result = repo_objects.RawData(y, y_name, repo_info={
            RepoInfoKey.NAME: MLRepo.get_eval_name(model_definition, data),
            RepoInfoKey.CATEGORY: MLObjectType.EVAL_DATA.value
        }
        )

        result.repo_info.category = MLObjectType.EVAL_DATA.value
        # create modification info
        _add_modification_info(
            result, model, model_definition, data, eval_func)
        model_param_name = str(
            NamingConventions.ModelParam(model=model_definition_name))
        if model_param_name in model.repo_info.modification_info.keys():
            result.repo_info.modification_info[model_param_name] = model.repo_info.modification_info[model_param_name]
        training_param_name = str(
            NamingConventions.TrainingParam(model=model_definition_name))
        if training_param_name in model.repo_info.modification_info.keys():
            result.repo_info.modification_info[training_param_name] = model.repo_info.modification_info[training_param_name]
        repo.add(result, 'evaluate data ' +
                 self.data + ' with model ' + self.model)
        logging.info('Finished evaluation job ' + str(jobid))

    def get_modifier_versions(self, repo):
        """ Get the modifier versions

        Args:
            repo (MLrepository): repository used to get and store the data

        Returns:
            tuple of string, dict -- return the object name and the modifiers
        """

        # get if evaluation with given inputs has already been computed
        model_definition_name = self.model.split('/')[0]
        model = repo.get(self.model, self.model_version,
                         full_object=False, throw_error_not_exist=False)
        if model == []:  # no model has been calibrated so far
            model_def_version = repo_store.RepoStore.LAST_VERSION
        else:
            model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO][model_definition_name]
        model_definition = repo.get(model_definition_name, model_def_version)
        result_name = str(NamingConventions.EvalData(
            model=self.model, data=self.data))
        return result_name, {self.model: self.model_version, self.data: self.data_version, model_definition.eval_function: self.eval_function_version}


class TrainingJob(Job):
    """ Definition of a model training job
    """

    @repo_object_init()
    def __init__(self, model, user, training_function_version=repo_store.RepoStore.LAST_VERSION, model_version=repo_store.RepoStore.LAST_VERSION,
                 training_data_version=repo_store.RepoStore.LAST_VERSION, training_param_version=repo_store.RepoStore.LAST_VERSION,
                 model_param_version=repo_store.RepoStore.LAST_VERSION,
                 preprocessor_versions=repo_store.RepoStore.LAST_VERSION,
                 preprocessor_fitting_function_versions=repo_store.RepoStore.LAST_VERSION,
                 preprocessor_transforming_function_versions=repo_store.RepoStore.LAST_VERSION,
                 preprocessor_param_versions=repo_store.RepoStore.LAST_VERSION,
                 repo_info=RepoInfo()):
        super(TrainingJob, self).__init__(repo_info)
        self.model = model
        self.user = user
        self.training_function_version = training_function_version
        self.model_version = model_version
        self.training_param_version = training_param_version
        self.model_param_version = model_param_version
        self.training_data_version = training_data_version
        self.preprocessor_versions = preprocessor_versions
        self.preprocessor_fitting_function_versions = preprocessor_fitting_function_versions
        self.preprocessor_transforming_function_versions = preprocessor_transforming_function_versions
        self.preprocessor_param_versions = preprocessor_param_versions
        # list of jobids which must have been run before this job should be excuted
        self.predecessors = []

    def _run(self, repo, jobid):
        """ Run the job with data from the given repo

        Args:
            repo (MLrepository): repository used to get and store the data
            jobid (str): the job id to be executed
        """

        model = repo.get(self.model, self.model_version)
        if model.training_data is None:
            train_data = repo.get_training_data(
                self.training_data_version, full_object=True)
        else:
            train_data = repo.get(model.training_data,
                                  self.training_data_version, full_object=True)
        train_func = repo.get(model.training_function,
                              self.training_function_version)
        train_param = None
        if not model.training_param == '':
            train_param = repo.get(model.training_param,
                                   self.training_param_version)
        model_param = None
        if not model.model_param is None:
            model_param = repo.get(
                model.model_param, self.model_param_version)

        # preprocessing
        preprocessors_modification_info = {}
        if model.preprocessors is None:
            fitted_preprocessors = None
            list_preprocessors = None
        else:
            x_data = train_data.x_data
            x_coord_names = train_data.x_coord_names
            list_preprocessors = []
            fitted_preprocessors = []
            preprocessor_output_columns = []
            num_preprocessors = len(model.preprocessors)
            # checking preprocessor versions
            if isinstance(self.preprocessor_versions, list):
                preprocessor_versions = self.preprocessor_versions
            else:
                preprocessor_versions = [
                    self.preprocessor_versions]*num_preprocessors
            if isinstance(self.preprocessor_fitting_function_versions, list):
                preprocessor_fitting_function_versions = self.preprocessor_fitting_function_versions
            else:
                preprocessor_fitting_function_versions = [
                    self.preprocessor_fitting_function_versions]*num_preprocessors
            if isinstance(self.preprocessor_transforming_function_versions, list):
                preprocessor_transforming_function_versions = self.preprocessor_transforming_function_versions
            else:
                preprocessor_transforming_function_versions = [
                    self.preprocessor_transforming_function_versions]*num_preprocessors
            if isinstance(self.preprocessor_param_versions, list):
                preprocessor_param_versions = self.preprocessor_param_versions
            else:
                preprocessor_param_versions = [
                    self.preprocessor_param_versions]*num_preprocessors

            if len(preprocessor_versions) != num_preprocessors:
                raise Exception(
                    'Number of preprocessors and their versions does not match.')
            if len(preprocessor_fitting_function_versions) != num_preprocessors:
                raise Exception(
                    'Number of preprocessors and their fitting function versions does not match.')
            if len(preprocessor_transforming_function_versions) != num_preprocessors:
                raise Exception(
                    'Number of preprocessors and their transforming function versions does not match.')
            if len(preprocessor_param_versions) != num_preprocessors:
                raise Exception(
                    'Number of preprocessors and their parameter versions does not match.')

            for k in range(num_preprocessors):
                logger.info('Apply preprocessor ' + model.preprocessors[k])
                preprocessor = repo.get(
                    model.preprocessors[k], preprocessor_versions[k])
                preprocessors_modification_info[preprocessor.repo_info.name] = preprocessor.repo_info.version
                transforming_func = repo.get(
                    preprocessor.transforming_function, preprocessor_transforming_function_versions[k])
                preprocessors_modification_info[transforming_func.repo_info.name] = transforming_func.repo_info.version
                prepro_param = None
                if not preprocessor.preprocessing_param == None:
                    prepro_param = repo.get(
                        preprocessor.preprocessing_param, preprocessor_param_versions[k])
                    preprocessors_modification_info[prepro_param.repo_info.name] = prepro_param.repo_info.version
                else:
                    prepro_param = None

                if preprocessor.fitting_function is not None:
                    fitting_func = repo.get(
                        preprocessor.fitting_function, preprocessor_fitting_function_versions[k])
                    preprocessors_modification_info[fitting_func.repo_info.name] = fitting_func.repo_info.version
                    fitted_preprocessor = fitting_func.create()(prepro_param, x_data, x_coord_names)
                    x_data, x_coord_names_new = transforming_func.create()(
                        prepro_param, x_data, x_coord_names, fitted_preprocessor)
                    fitted_preprocessors.append(fitted_preprocessor)
                else:
                    fitting_func = None
                    x_data, x_coord_names_new = transforming_func.create()(
                        prepro_param, x_data, x_coord_names)
                    fitted_preprocessors.append(None)
                if set(x_coord_names_new) == set(x_coord_names):
                    preprocessor_output_columns.append(None)
                else:
                    preprocessor_output_columns.append(x_coord_names_new)
                    x_coord_names = x_coord_names_new
                #_add_modification_info(preprocessor, transforming_func, prepro_param, fitting_func)
                list_preprocessors.append(preprocessor)
            train_data.x_data = x_data
            train_data.x_coord_names = x_coord_names
        # calibration
        if model_param is None:
            result = train_func.create()(train_param, train_data)
        else:
            if train_param is None:
                result = train_func.create()(model_param, train_data)
            else:
                result = train_func.create()(model_param, train_param, train_data)

        if isinstance(result, tuple):
            calibrated_model = result[0]
            training_stat = result[1]
            training_stat.repo_info[RepoInfoKey.NAME] = self.model + \
                '/training_stat'
            training_stat.repo_info[RepoInfoKey.CATEGORY] = MLObjectType.TRAINING_STATISTIC.value
            _add_modification_info(
                training_stat, model_param, train_param, train_data, model, train_func)
        else:
            calibrated_model = result
            training_stat = None
        calibrated_model.preprocessors = list_preprocessors
        calibrated_model.fitted_preprocessors = fitted_preprocessors
        calibrated_model.repo_info[RepoInfoKey.NAME] = self.model + '/model'
        calibrated_model.repo_info[RepoInfoKey.CATEGORY] = MLObjectType.CALIBRATED_MODEL.value
        # create modification info
        _add_modification_info(calibrated_model, model_param,
                               train_param, train_data, model, train_func)
        # add the preprocessor modification info
        if calibrated_model.preprocessors is not None:
            calibrated_model.repo_info[RepoInfoKey.MODIFICATION_INFO].update(
                preprocessors_modification_info)

        if training_stat is not None:
            repo.add([training_stat, calibrated_model],
                     'training of model ' + self.model)
        else:
            repo.add(calibrated_model, 'training of model ' + self.model)

    def get_modifier_versions(self, repo):
        """ Get the modifier versions

        Args:
            repo (MLrepository): repository used to get and store the data

        Returns:
            tuple of string, dict -- return the object name and the modifiers
        """

        modifiers = {}
        modifiers[self.model] = self.model_version
        model = repo.get(self.model, self.model_version)
        if model.training_data is None:
            train_data = repo.get_training_data(
                self.training_data_version, full_object=False)
        else:
            train_data = repo.get(
                model.training_data, self.training_data_version, full_object=False)
        modifiers[train_data.repo_info.name] = self.training_data_version
        modifiers[model.training_function] = self.training_function_version
        if not model.training_param == '':
            modifiers[model.training_param] = self.training_param_version
        if not model.model_param is None:
            modifiers[model.model_param] = self.model_param_version
        if not model.preprocessors is None:
            for k in range(len(model.preprocessors)):
                if isinstance(self.preprocessor_versions, list):
                    prepro = repo.get(
                        model.preprocessors[k], self.preprocessor_versions[k])
                    modifiers[model.preprocessors[k]
                              ] = self.preprocessor_versions[k]
                else:
                    prepro = repo.get(
                        model.preprocessors[k], self.preprocessor_versions)
                    modifiers[model.preprocessors[k]
                              ] = self.preprocessor_versions
                if isinstance(self.preprocessor_transforming_function_versions, list):
                    modifiers[prepro.transforming_function] = self.preprocessor_transforming_function_versions[k]
                else:
                    modifiers[prepro.transforming_function] = self.preprocessor_transforming_function_versions
                if not prepro.fitting_function is None:
                    if isinstance(self.preprocessor_fitting_function_versions, list):
                        modifiers[prepro.fitting_function] = self.preprocessor_fitting_function_versions[k]
                    else:
                        modifiers[prepro.fitting_function] = self.preprocessor_fitting_function_versions
                if not prepro.preprocessing_param is None:
                    if isinstance(self.preprocessor_param_versions, list):
                        modifiers[prepro.preprocessing_param] = self.preprocessor_param_versions[k]
                    else:
                        modifiers[prepro.preprocessing_param] = self.preprocessor_param_versions
        return self.model + '/model', modifiers


class MeasureJob(Job):
    """ Defintion of a job to measure the error
    """

    @repo_object_init()
    def __init__(self, result_name, measure_type, coordinates, data_name, model_name, data_version=repo_store.RepoStore.LAST_VERSION,
                 model_version=repo_store.RepoStore.LAST_VERSION, repo_info=RepoInfo()):
        """ Constructor for the MeasureJob class

        Args:
            measure_type (str): string describing the measure type
            coordinates (str or list): either a string describing the coordinate (or simply all coordinates if the string equals MeasureConfiguration._ALL_COORDINATES) or a list of strings
            data_name (str): name of data for which the measure shall be calculated
            model_name (str): name of model for which the measure shall be calculated
            data_version (versionnumber): version of data to be used. Defaults to repo_store.RepoStore.LAST_VERSION.
            model_version (versionnumber): version of model to be used. Defaults to repo_store.RepoStore.LAST_VERSION.
        """

        super(MeasureJob, self).__init__(repo_info)
        self.measure_type = measure_type
        self.coordinates = coordinates
        self.model_name = model_name
        self.model_version = model_version
        self.data_name = data_name
        self.data_version = data_version

    def _run(self, repo, jobid):
        """ Run the job with data from the given repo

        Args:
            repo (MLrepository): repository used to get and store the data
            jobid (str): the job id to be executed
        """

        logging.info('Start measure job ' + self.repo_info.name)
        target = repo.get(self.data_name, self.data_version, full_object=True)
        # if given model name is a name of calibrated model, split to find the evaluation
        m_name = self.model_name.split('/')[0]
        eval_data_name = NamingConventions.EvalData(
            data=self.data_name, model=m_name)
        eval_data = repo.get(str(eval_data_name), modifier_versions={
                             self.model_name: self.model_version, self.data_name: self.data_version}, full_object=True)
        logger.info('run MeasureJob on data ' + self.data_name + ':' + str(self.data_version)
                    + ', ' + str(eval_data_name) + ':' +
                    str(eval_data.repo_info[RepoInfoKey.VERSION])
                    )
        columns = []
        measure_name = MeasureConfiguration.get_name(self.measure_type)
        if not repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
            measure_name = MeasureConfiguration.get_name(
                (self.measure_type, self.coordinates))
            for x in self.coordinates:
                columns.append(target.y_coord_names.index(x))
        if len(columns) == 0:
            v = self._compute(target.y_data, eval_data.x_data)
        else:
            v = self._compute(
                target.y_data[:, columns], eval_data.x_data[:, columns])
        result_name = str(NamingConventions.Measure(
            eval_data_name, measure_type=measure_name))
        result = repo_objects.Measure(v,
                                      repo_info={RepoInfoKey.NAME: result_name, RepoInfoKey.CATEGORY: MLObjectType.MEASURE.value})

        # create modification info
        result.repo_info.modification_info[self.model_name] = eval_data.repo_info.modification_info[self.model_name]
        result.repo_info.modification_info[self.data_name] = eval_data.repo_info.modification_info[self.data_name]
        result.repo_info.modification_info[str(
            eval_data_name)] = eval_data.repo_info.version
        result.repo_info.modification_info[m_name] = eval_data.repo_info.modification_info[m_name]
        model_param_name = str(NamingConventions.ModelParam(model=m_name))
        if model_param_name in eval_data.repo_info.modification_info.keys():
            result.repo_info.modification_info[model_param_name] = eval_data.repo_info.modification_info[model_param_name]
        training_param_name = str(
            NamingConventions.TrainingParam(model=m_name))
        if training_param_name in eval_data.repo_info.modification_info.keys():
            result.repo_info.modification_info[training_param_name] = eval_data.repo_info.modification_info[training_param_name]
        ################

        logging.debug('Add result ' + result_name)

        repo.add(result, 'computing  measure ' +
                 self.measure_type + ' on data ' + self.data_name)
        logging.info('Finished measure job ' + self.repo_info.name)

    def get_modifier_versions(self, repo):
        """ Get the modifier versions

        Args:
            repo (MLrepository): repository used to get and store the data

        Returns:
            tuple of string, dict -- return the object name and the modifiers
        """

        modifiers = {}
        modifiers[self.data_name] = self.data_version
        modifiers[self.model_name] = self.model_version
        measure_name = MeasureConfiguration.get_name(self.measure_type)
        if not repo_objects.MeasureConfiguration._ALL_COORDINATES in self.coordinates:
            measure_name = MeasureConfiguration.get_name(
                (self.measure_type, self.coordinates))
        return measure_name, modifiers

    def _compute(self, target_data, eval_data):
        """ Computes the specified error measure

        Args:
            target_data ([type]): the target data
            eval_data ([type]): the evaluated data

        Raises:
            NotImplementedError -- If the measuretype is not implemented, raise an exception

        Returns:
            double -- the error
        """

        if self.measure_type == repo_objects.MeasureConfiguration.MAX:
            return self._compute_max(target_data, eval_data)
        if self.measure_type == repo_objects.MeasureConfiguration.R2:
            return self._compute_r2(target_data, eval_data)
        if self.measure_type == repo_objects.MeasureConfiguration.MSE:
            return self._compute_mse(target_data, eval_data)
        else:
            raise NotImplementedError

    def _compute_max(self, target_data, eval_data):
        """ Computes the maximum error measure

        Args:
            target_data ([type]): the target data
            eval_data ([type]): the evaluated data

        Returns:
            double -- the error
        """

        logger.debug('computing maximum error')
        return linalg.norm(target_data-eval_data, inf)

    def _compute_r2(self, target_data, eval_data):
        """ Computes the r2 error measure

        Args:
            target_data ([type]): the target data
            eval_data ([type]): the evaluated data

        Returns:
            double -- the error
        """

        from sklearn.metrics import r2_score
        return r2_score(target_data, eval_data)

    def _compute_mse(self, target_data, eval_data):
        """ Computes the mean squared error error measure

        Args:
            target_data ([type]): the target data
            eval_data ([type]): the evaluated data

        Returns:
            double -- the error
        """

        from sklearn.metrics import mean_squared_error
        return mean_squared_error(target_data, eval_data)

# endregion


class Name:
    def __init__(self, name_order, tag):
        self.name_order = name_order.split('/')
        self.tag = tag
        self.model = None

    def __str__(self):
        result = self.values[self.name_order[0]]
        for i in range(1, len(self.name_order)):
            result = result + '/' + self.values[self.name_order[i]]
        return result

    def _set_from_string(self, v):
        tmp = v.split('/')
        if len(tmp) > len(self.name_order):
            raise Exception('Incorrect name depth: ' +
                            str(len(tmp)) + ' != ' + str(len(self.name_order)))
        for i in range(len(tmp)):
            if self.name_order[i] != '*':
                self.values[self.name_order[i]] = tmp[i]

    def _set_from_dictionary(self, values):
        for k, v in values.items():
            if k in self.name_order and k != '*':
                self.values[k] = v

    def _set_from(self, v):
        if isinstance(v, Name):
            self._set_from_dictionary(v.values)
        if isinstance(v, list):
            for x in v:
                self._set_from(x)
        if isinstance(v, str):
            self._set_from_string(v)
        if isinstance(v, dict):
            self._set_from_dictionary(v)

    def __call__(self, name=None, **args):
        self.values = {'*': self.tag}
        if args is not None:
            self._set_from([name, args])
        else:
            self._set_from(name)
        return self


class NamingConventions:
    @staticmethod
    def _get_object_name(name):
        if not isinstance(name, str):
            return name.repo_info[RepoInfoKey.NAME]
        return name

    @staticmethod
    def get_model_from_name(name):
        return name.split('/')[0]

    @staticmethod
    def get_model_param_name(model_name):
        return NamingConventions.get_model_from_name(model_name) + '/model_param'

    @staticmethod
    def get_calibrated_model_name(model):
        return model.split('/')[0]+'/model'

    @staticmethod
    def get_eval_name(model_name, data_name):
        return model_name + '/eval/' + data_name

    @staticmethod
    def get_eval_name_from_measure_name(measure_name):
        model = NamingConventions.get_model_from_name(measure_name)
        data_name = measure_name.split('/')[2]
        return NamingConventions.get_eval_name(model, data_name)

    Measure = Name('model/*/data/measure_type', 'measure')
    EvalData = Name('model/*/data', 'eval')
    Data = Name('data', '')
    CalibratedModel = Name('model/*', 'model')
    Model = Name('model', '')
    ModelParam = Name('model/*', 'model_param')
    TrainingParam = Name('model/*', 'training_param')
    Test = Name('model/*/test_name/data', 'tests')


class MLRepo:

    """ Repository for doing machine learning

        The repository and his extensions provide a solid fundament to do machine learning science supporting features such as:
            - auditing/versioning of the data, models and tests
            - best practice standardized plotting for investigating model performance and model behaviour
            - automated quality checks

        Args:
            workspace ([type]): [description]. Defaults to None.
            user (str): the user. Defaults to None.
            config (dict): the configuration to use. Defaults to None.
            save_config (bool): determines whether to save the configuration or not. Defaults to False.

    """

    @staticmethod
    def __create_default_config(user, workspace):
        if user is None:
            raise Exception('Please specify a user.')
        return {'name': 'NONE',
                'user': user, 'workspace': workspace,
                'repo_store': {
                    'type': 'memory_handler',
                    'config': {}
                },
                'numpy_store': {
                    'type': 'memory_handler',
                    'config': {}
                },
                'job_runner': {
                    'type': 'simple',
                    'config': {
                        'throw_job_error': True
                    }
                }
                }

    def _save_config(self):
        """ Save the configuration
        """

        if 'workspace' in self._config.keys():
            if self._config['workspace'] is not None:
                if not os.path.exists(self._config['workspace']):
                    os.makedirs(self._config['workspace'])
                with open(self._config['workspace'] + '/.config.json', 'w') as f:
                    json.dump(self._config, f, indent=4,
                              separators=(',', ': '))

    def __init__(self,  workspace=None, user=None, config=None, save_config=False, name='NONE'):
        """ Constructor of MLRepo

        Args:
            workspace ([type]): [description]. Defaults to None.
            user (str): the user. Defaults to None.
            config (dict): the configuration to use. Defaults to None.
            save_config (bool): determines whether to save the configuration or not. Defaults to False.

        Raises:
            Exception: Raises an exception if more than one mapping is found
        """

        self._config = config
        if config is None:
            if workspace is not None:
                try:
                    logger.info('Reading config from workspace ' + workspace)
                    with open(workspace + '/.config.json', 'r') as f:
                        self._config = json.load(f)
                except:
                    logger.info('Reading config from workspace ' + workspace +
                                ' has not been successfull, using defaults.')
                    self._config = MLRepo.__create_default_config(
                        user, workspace)
            else:
                self._config = MLRepo.__create_default_config(user, workspace)
        else:
            if 'workspace' in self._config.keys():
                self._save_config()  # we save the config

        if 'name' not in self._config.keys():
            self._config['name'] = name
            save_config = True
        else:
            if name != 'NONE':
                if not self._config['name'] == 'NONE':
                    raise Exception(
                        'A name for the repository has already been specified.')
                self._config['name'] = name
                self._save_config()  # we save the config

        self._numpy_repo = NumpyStoreFactory.get(
            self._config['numpy_store']['type'], **self._config['numpy_store']['config'])
        self._ml_repo = RepoStoreFactory.get(
            self._config['repo_store']['type'], **self._config['repo_store']['config'])
        self._job_runner = JobRunnerFactory.get(
            self._config['job_runner']['type'], self, **self._config['job_runner']['config'])
        self._user = self._config['user']

        # check if the ml mapping is already contained in the repo, otherwise add it
        logging.info('Get mapping.')
        repo_dict = []
        if self._ml_repo.object_exists('repo_mapping', version=repo_store.RepoStore.LAST_VERSION):
            repo_dict = self._ml_repo.get(
                'repo_mapping', versions=repo_store.RepoStore.LAST_VERSION)
        else:
            logging.info('No mapping found, creating new mapping.')
        if len(repo_dict) > 1:
            raise Exception('More than one mapping found.')
        if len(repo_dict) == 1:
            self._mapping = repo_objects.create_repo_obj(repo_dict[0])
        else:
            version = repo_store._version_str()
            self._mapping = Mapping(  # pylint: disable=E1123
                repo_info={RepoInfoKey.NAME: 'repo_mapping',
                           RepoInfoKey.CATEGORY: MLObjectType.MAPPING.value,  RepoInfoKey.VERSION: version})
            repo_obj = repo_objects.create_repo_obj_dict(self._mapping)
            self._ml_repo.add(repo_obj)

        self._add_triggers = []

        if save_config:
            self._save_config()

    def _add(self, repo_object, message='', category=None):
        """ Add a repo_object to the repository.

        Args:
            repo_object (RepoObject): repo_object to be added, will be modified so that it contains the version number
            message (str): commit message. Defaults to ''.
            category (MLObjectType): Category of repo_object which overwrites the objects category.. Defaults to None.

        Returns:
            tuple of string and bool -- version number of object added and boolean if mapping has changed
        """

        if category is not None:
            repo_object.repo_info[RepoInfoKey.CATEGORY] = category

        mapping_changed = self._mapping.add(
            repo_object.repo_info[RepoInfoKey.CATEGORY], repo_object.repo_info[RepoInfoKey.NAME])

        repo_object.repo_info[RepoInfoKey.COMMIT_MESSAGE] = message
        repo_object.repo_info[RepoInfoKey.COMMIT_DATE] = str(datetime.now())
        repo_object.repo_info[RepoInfoKey.AUTHOR] = self._user
        obj_dict = repo_objects.create_repo_obj_dict(repo_object)
        version = self._ml_repo.add(obj_dict)
        if len(repo_object.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0:
            np_dict = repo_object.numpy_to_dict()
            self._numpy_repo.add(repo_object.repo_info[RepoInfoKey.NAME],
                                 repo_object.repo_info[RepoInfoKey.VERSION],
                                 np_dict)
        for trigger in self._add_triggers:
            trigger()
        return version, mapping_changed

    def _update_job(self, job_object):
        """ Update a job object without incrementing version number

        Updates a repo object without incrementing version number or adding commit message. This should be only used internally by the jobs
            to update the job objects according to their state, modification info and runtime.

        Args:
            job_object (RepoObject): the job object to be updated

        Raises:
            Exception: If the provided object is not a job object an exception is raised.
        """

        obj_dict = repo_objects.create_repo_obj_dict(job_object)
        self._ml_repo.replace(obj_dict)
        if len(job_object.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0:
            raise Exception('Jobs with big objects cannot be updated.')

    def add(self, repo_object, message='', category=None):
        """ Add a repo_object or list of repo objects to the repository.

        Raises an exception if the category of the object is not defined in the object and if it is not defined with the category argument.
        It raises an exception if an object with this id does already exist.

        Args:
            repo_object (RepoObject): repo_object or list of repo_objects to be added, will be modified so that it contains the version number
            message (str): commit message. Defaults to ''.
            category (MLObjectType): Category of repo_object which overwrites the objects category.. Defaults to None.

        Returns:
            str or dictionary -- version number of object added or dictionary of names and versions of objects added
        """

        version = repo_store._version_str()
        result = {}
        repo_list = repo_object
        mapping_changed = False
        if not isinstance(repo_list, list):
            repo_list = [repo_object]
        if isinstance(repo_list, list):
            for obj in repo_list:
                obj.repo_info.version = version
                result[obj.repo_info[RepoInfoKey.NAME]], mapping_changed_tmp = self._add(
                    obj, message, category)
                mapping_changed = mapping_changed or mapping_changed_tmp
        if mapping_changed:
            obj_dict = repo_objects.create_repo_obj_dict(self._mapping)
            self._ml_repo.replace(obj_dict)

        commit_message = repo_objects.CommitInfo(message, self._user, result, repo_info={RepoInfoKey.CATEGORY: MLObjectType.COMMIT_INFO.value,
                                                                                         RepoInfoKey.NAME: 'CommitInfo', RepoInfoKey.VERSION: version})
        self._add(commit_message)
        if not isinstance(repo_object, list):
            if len(result) == 1 or (mapping_changed and len(result) == 2):
                return result[repo_object.repo_info[RepoInfoKey.NAME]]
        return result

    def get_training_data(self, version=repo_store.RepoStore.LAST_VERSION, full_object=True, model=None, model_version=repo_store.RepoStore.LAST_VERSION):
        """ Returns training data for a model.

        It returns the training data in the repo for a specified model. If there is only one set of training data in the repo, this set will be returned.
        Otherwise, the model is loaded and the training data is used as defined in the model. If in this case a model is not specified the
        method throws an exception.

        Args:
            version (str): version of data object. Defaults to repo_store.RepoStore.LAST_VERSION.
            full_object (bool): if True, the complete data is returned including numpy data. Defaults to True.
            model (str): Name of model definition for which the training data will be returned.
            model_version (str): Version of model definition for which teh trainin data will be returned.
        """

        if self._mapping[MLObjectType.TRAINING_DATA] is None or len(self._mapping[MLObjectType.TRAINING_DATA]) == 0:
            raise Exception("No training_data in repository.")
        training_data = None
        if len(self._mapping[MLObjectType.TRAINING_DATA]) > 1:
            if model is None:
                raise Exception(
                    "More then one training_data in repository, please use method get and specify the name of the training data.")
            m = self.get(model, model_version)
            if m.training_data is None:
                raise Exception(
                    "More then one training_data in repository and the model does not explicitely specify a training data set.")
            training_data = m.training_data
        else:
            training_data = self._mapping[MLObjectType.TRAINING_DATA][0]
        return self.get(training_data, version, full_object)

    def add_raw_data(self, name, data, input_names=None, data_y=None, target_names=None, file_format=None, axis=1):
        """Adds a RawData object to the repository.

        This methods creates/reads from the given data/file a RawData object and adds it to the repository. 

        Examples:
            Read data from csv file 'test_data.csv' and use columns with headers 'x0', 'x1' as input data and column with label 'x2' as target, store the results under name 'my_data'::

                >>ml_repo.add_raw_data('my_data', 'test_data.csv', ['x0', 'x1], file_format = 'csv')

            Create data from a DataFrame test where the columns 'x0', 'x1' are used as input and no target is specified::

                >>ml_repo.add_raw_data('my_data', test, ['x0', 'x1])



        Args:
            name (str): Name of RawData in repository (if name does not start with 'raw_data/' this is added.
            data (str, numpy ndarray or pandas DataFrame): Eithr a pandas DataFarme, a numpy ndarray or a string that is interpreted as filename of the underling data.
            input_names (iterable of str, optional): List of the input variables names. Defaults to None.
            data_y (str or numpy ndarray, optional):Either a numpy ndarray or a string defining the filename of th y-data (not valid if file_format=='csv'). Defaults to None.
            target_names (iterable of str, optional): List of the target variables names. Defaults to None.
            file_format ('csv' or 'numpy', optional): File type which can be either csv or numpy (numpy means an ndarray stored with numpy.save). Defaults to None.
            axis (int, optional): If only an ndarray is given but target variables are defined, this array will b split into weo arrays (one for input, one for target) along this axis. Defaults to 1.

        Returns:
             str: version number of RawData object added
        """
        _data_x = None
        _data_y = None
        # read from file
        if isinstance(data, str):
            if file_format is None:
                raise Exception(
                    'Please specify a file format.')  # pragma: no cover
            # we assume that data represents a filename of the file containing the data
            if file_format == 'csv':
                try:
                    import pandas as pd
                    data = pd.read_csv(data)
                    if data_y is not None:
                        raise Exception(
                            'Cannot define separate y-data using csv file to read RawData from.')
                except ImportError:
                    raise Exception(
                        'Cannot add RawData: To read csv pandas needs to be installed.')  # pragma: no cover
            elif file_format == 'numpy':
                data = np.load(data)
                if isinstance(data_y, str):
                    data_y = np.load(data_y)
            else:
                raise Exception('Unknown file format ' + file_format)
            return self.add_raw_data(name, data, input_names, data_y, target_names, axis=axis)
        # use objects directly
        else:
            # check if pandas is installed and if the given data object is a DataFrame
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    _data_x = data.loc[:, input_names].values
                    if target_names is not None:
                        _data_y = data.loc[:, target_names].values
            except ImportError:
                pass  # pragma: no cover
            if isinstance(data, np.ndarray):
                if data_y is not None:
                    _data_x = data
                    _data_y = data_y
                elif target_names is not None:
                    tmp = np.split(data, [len(input_names)], axis=1)
                    _data_x = tmp[0]
                    _data_y = tmp[1]
                else:
                    _data_x = data
        if _data_x is None:
            raise Exception(
                'Cannot add given data: The data is neither a filename, nor a pandas DataFrame nor a numpy ndarray.')  # pragma: no cover
        path = name
        if not 'raw_data' in path:
            path = 'raw_data/' + name
        raw_data = RawData(_data_x, input_names, _data_y,
                           target_names, repo_info={RepoInfoKey.NAME: path})
        return self.add(raw_data)

    def add_training_data(self, name, raw_data_name, start_index=0, end_index=None, raw_data_version='last'):
        """Add training data as a DataSet to the repository.

            This method defines a DataSet and adds it to the repository. A DataSet is a logical unit based on a RawData object and defines the range of data
            that is taken from the respective RawData data.

        Args:
            name (str): Name of respective object in repository.
            raw_data_name (str): Name of the underlying RawData object that is used as basis.
            start_index (int, optional): Start index where training data starts from underlying RawData. Defaults to 0.
            end_index (int, optional): End index where training data end. Defaults to None.
            raw_data_version (str): Version of underlying RawData (if 'last', always the latest RawData will be used to derive the respective DataSet). Defaults to 'last'
        """
        training_data = repo_objects.DataSet(raw_data_name, start_index, end_index,
                                             raw_data_version, repo_info={RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        return self.add(training_data)

    def add_test_data(self, name, raw_data_name, start_index=0, end_index=None, raw_data_version='last'):
        """Add test data as a DataSet to the repository.

            This method defines a DataSet and adds it to the repository. A DataSet is a logical unit based on a RawData object and defines the range of data
            that is taken from the respective RawData data.

        Args:
            name (str): Name of respective object in repository.
            raw_data_name (str): Name of the underlying RawData object that is used as basis.
            start_index (int, optional): Start index where test data starts from underlying RawData. Defaults to 0.
            end_index (int, optional): End index where test data end. Defaults to None.
            raw_data_version (str): Version of underlying RawData (if 'last', always the latest RawData will be used to derive the respective DataSet). Defaults to 'last'
        """
        test_data = repo_objects.DataSet(raw_data_name, start_index, end_index,
                                         raw_data_version, repo_info={RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        return self.add(test_data)

    def add_eval_function(self, f, repo_name=None):
        """ Add the function to evaluate the model

        Args:
            module_name (str): module where function is located
            function_name (str): function name
            repo_name (str): identifier of the repo object used to store the information, if None, the name is set to module_name.function_name. Defaults to None.
        """

        self._add_function(f, MLObjectType.MODEL_EVAL_FUNCTION, repo_name)

    def add_training_function(self, f, repo_name=None):
        """ Add function to train a model

        Args:
            module_name (str): module where function is located
            function_name (str): function name
            repo_name (tring): identifier of the repo object used to store the information, if None, the name is set to module_name.function_name. Defaults to None.
        """

        self._add_function(f, MLObjectType.TRAINING_FUNCTION, repo_name)

    def add_preprocessing_fitting_function(self, f, repo_name=None):
        """ Add function to fit a preprocessor

        Args:
            module_name (str): module where function is located
            function_name (str): function name
            repo_name (tring): identifier of the repo object used to store the information, if None, the name is set to module_name.function_name. Defaults to None.
        """

        self._add_function(
            f, MLObjectType.PREPROCESSING_FITTING_FUNCTION, repo_name)

    def add_preprocessing_transforming_function(self, f, repo_name=None):
        """ Add function to transform the data by a preprocessor

        Args:
            module_name (str): module where function is located
            function_name (str): function name
            repo_name (str): identifier of the repo object used to store the information, if None, the name is set to module_name.function_name. Defaults to None.
        """

        self._add_function(
            f, MLObjectType.PREPROCESSING_TRANSFORMING_FUNCTION, repo_name)

    def _add_function(self, f, category, repo_name=None):
        """ Add function to the repository
        private function to reduce repeating code

        Args:
            module_name (str): module where function is located
            function_name (str): function name
            category{MLObjectType} -- the function category
            repo_name (str): identifier of the repo object used to store the information, if None, the name is set to module_name.function_name. Defaults to None.
        """

        name = repo_name
        if name is None:
            name = f.__module__ + "." + f.__name__
        func = repo_objects.Function(f, repo_info={
                                     RepoInfoKey.NAME: name,
                                     RepoInfoKey.CATEGORY: category.value})
        self.add(func, 'add function ' + name + ' of category ' +
                 category.value + ' to the repo')

    def add_model(self, model_name, model_eval=None, model_training=None, model_param=None, training_param=None,
                  preprocessors=None):
        """ Add a new model to the repo

        Args:
            model_name (str): identifier of the model
            model_eval (str): identifier of the evaluation function in the repo to evaluate the model, 
                                    if None and there is only one evaluation function in the repo, this function will be used
            model_training (str): identifier of the training function in the repo to train the model, 
                                    if None and there is only one evaluation function in the repo, this function will be used
            model_param (str): identifier of the model parameter in the repo, if None and there is exactly one ModelParameter in teh repo, this will be used,. Defaults to None.
                                    otherwise it is assumed that no model_params are needed
            training_param (str): identifier of the training parameter, if None and there is only one training_parameter object in the repo, . Defaults to None.
                                    this will be used. If an empty string is given as training parameter, we assume that the algorithm does not need a training pram.
            preprocessors (list): list of preprocessors to be execute. Defaults to None.
                                    this is a list of strings
        """

        model = repo_objects.Model(preprocessors=preprocessors,
                                   repo_info={RepoInfoKey.CATEGORY: MLObjectType.MODEL.value,
                                              RepoInfoKey.NAME: model_name})
        model.eval_function = model_eval
        if model.eval_function is None:
            mapping = self._mapping[MLObjectType.MODEL_EVAL_FUNCTION]
            if len(mapping) == 1:
                model.eval_function = mapping[0]
            else:
                raise Exception(
                    'More than one or no eval function in repo, therefore you must explicitely specify an eval function.')

        model.training_function = model_training
        if model.training_function is None:
            mapping = self._mapping[MLObjectType.TRAINING_FUNCTION]
            if len(mapping) == 1:
                model.training_function = mapping[0]
            else:
                raise Exception(
                    'More than one or no training function in repo, therefore you must explicitely specify a training function.')

        model.training_param = training_param
        if model.training_param is None:
            mapping = self._mapping[MLObjectType.TRAINING_PARAM]
            if len(mapping) == 1:
                model.training_param = mapping[0]
            else:
                raise Exception(
                    'More than one or no training parameter in repo, therefore you must explicitely specify a training parameter.')

        model.model_param = model_param
        if model.model_param is None:
            mapping = self._mapping[MLObjectType.MODEL_PARAM]
            if len(mapping) == 1:
                model.model_param = mapping[0]
            else:
                if len(mapping) > 1:
                    raise Exception(
                        'More than one model parameter in repo, therefore you must explicitely specify a model parameter.')
        self.add(model, 'add model ' + model_name)

    def add_measure(self, measure, coordinates=None):
        """ Add a measure to the repository

            If the measure already exists, it returns the message

        Args:
            measure (str): string defining the measure, i.e MAX,...
            coordinates (list of str): list of strings defining the coordinates (by name) used for the measure, if None, all coordinates will be used. Defaults to None.
        """

        # if a measure configuration already exists, use this, otherwise creat a new one
        measure_config = None
        tmp = self.get_names(MLObjectType.MEASURE_CONFIGURATION)
        if len(tmp) > 0:
            measure_config = self.get(tmp[0])
        else:
            measure_config = repo_objects.MeasureConfiguration([], repo_info={RepoInfoKey.NAME: 'measure_config',
                                                                              RepoInfoKey.CATEGORY: MLObjectType.MEASURE_CONFIGURATION.value})
        measure_config.add_measure(measure, coordinates)
        coord = 'all'
        if not coordinates is None:
            coord = str(coordinates)
        self.add(measure_config, message='added measure ' +
                 measure + ' for coordinates ' + coord)

    def add_preprocessor(self, preprocessor_name, transforming_function=None, fitting_function=None,
                         preprocessor_param=None):
        """ Add a new preprocessor to the repo

        Args:
            preprocessor_name (str): identifier of the preprocessor
            transforming_function (str): identifier of the transforming function in the repo, 
                                    if None and there is only one transforming function in the repo, this function will be used
            fitting_function (str): identifier of the fitting function in the repo to fit the preprocessor,
                                    if None the preprocessor does not need to be fitted
            preprocessor_param (str): identifier of the preprocessor parameter. Defaults to None.

        Raises:
            Exception: Raises an error if the preprocessing transforming function is not in repo
        """

        transforming_func = transforming_function
        if transforming_func is None:
            mapping = self._mapping[MLObjectType.PREPROCESSING_TRANSFORMING_FUNCTION]
            if len(mapping) == 1:
                transforming_func = mapping[0]
            else:
                raise Exception(
                    'More than one or no preprocessing transforming function in repo, therefore you must explicitely specify an eval function.')

        prepro = repo_objects.Preprocessor(transforming_function=transforming_func, fitting_function=fitting_function,
                                           preprocessing_param=preprocessor_param,
                                           repo_info={RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSOR.value,
                                                      RepoInfoKey.NAME: preprocessor_name})

        self.add(prepro, 'add preprocessor ' + preprocessor_name)

    def get_ml_repo_store(self):
        """ Return the storage for the ml repo

        Returns:
            RepoStore -- the storage for the RepoObjects
        """
        return self._ml_repo

    def get_numpy_data_store(self):
        """ Return the numpy data store of the ml repo

        Returns:
            numpy_handler -- the numpy repo
        """

        return self._numpy_repo

    def get(self, name, version=repo_store.RepoStore.LAST_VERSION, full_object=False,
            modifier_versions=None, obj_fields=None,  repo_info_fields=None,
            throw_error_not_exist=True, throw_error_not_unique=True):
        """ Get repo objects. It throws an exception, if an object with the name does not exist.

        Args:
            name (str): the object name
            version (str): object version, default is latest (-1). If the fields are nested (an element of a dictionary which is an element of a 
                    dictionary, use path notation to the element, i.e. p/elem1/elem2 to get p[elem1][elem2]). Defaults to repo_store.RepoStore.LAST_VERSION.
            full_object (bool): flag to determine whether the numpy objects are loaded (True->load). Defaults to False.
            modifier_versions ([type]): [description]. Defaults to None.
            obj_fields ([type]): [description]. Defaults to None.
            repo_info_fields ([type]): [description]. Defaults to None.
            throw_error_not_exist (bool): true - throw error if not exists, else return []. Defaults to True.
            throw_error_not_unique (bool): true - throw error if item is not unique, else return []. Defaults to True.

        Raises:
            Exception: raises an exception if no object with the specific name is found

        Returns:
            RepoObject or list thereof -- The repo object
        """
        if version is not None:
            logging.debug('Getting ' + name + ', version ' + str(version))
        else:
            logging.debug('Getting ' + name + ', version is None.')
        repo_dict = self._ml_repo.get(name, version, modifier_versions, obj_fields, repo_info_fields,
                                      throw_error_not_exist, throw_error_not_unique)
        if len(repo_dict) == 0:
            if throw_error_not_exist:
                logger.error('No object found with name ' + name + ' and version ' +
                             str(version) + ', modifier_versions: ' + str(modifier_versions))
                raise Exception('No object found with name ' +
                                name + ' and version ' + str(version))
            else:
                return []

        tmp = []
        for x in repo_dict:
            result = repo_objects.create_repo_obj(x)
            if isinstance(result, DataSet):
                raw_data = self.get(
                    result.raw_data, result.raw_data_version, False)
                if full_object:
                    numpy_data = self._numpy_repo.get(result.raw_data, raw_data.repo_info[RepoInfoKey.VERSION],
                                                      result.start_index, result.end_index)
                    repo_objects.repo_object_init.numpy_from_dict(
                        raw_data, numpy_data)
                result.set_data(raw_data)

            numpy_dict = {}
            if len(result.repo_info[RepoInfoKey.BIG_OBJECTS]) > 0 and full_object:
                numpy_dict = self._numpy_repo.get(
                    result.repo_info[RepoInfoKey.NAME], result.repo_info[RepoInfoKey.VERSION])
            # for x in result.repo_info[RepoInfoKey.BIG_OBJECTS]:
            #    if not x in numpy_dict:
            #        numpy_dict[x] = None
            result.numpy_from_dict(numpy_dict)
            tmp.append(result)
        if len(tmp) == 1:
            return tmp[0]
        return tmp

    def delete(self, name, version):
        """ Delete a specific object. 

        It deletes the object. If other objects were modified by this object, it throws an exception
        that first the modified objects must be deleted.

        Args:
            name (str): name of the object
            version (str): version of the object

        Raises:
            Exception: If the object has depending objects, it can not be deleted and an error is thrown.
        """

        dependent_objects = self._ml_repo._get_by_modification_info(
            name, version, [k.value for k in MLObjectType])
        if len(dependent_objects) > 0:
            obj_list = ';'
            obj_list.join([k.repo_info.name + ': ' +
                           k.repo_info.version for k in dependent_objects])
            raise Exception(
                "Objects dependending on the object to be deleted, please delete these objects first, objects: " + obj_list)
        self._ml_repo._delete(name, version)
        self._numpy_repo._delete(name, version)

    @staticmethod
    def get_calibrated_model_name(model_name):
        """ For a model name the calibrated model name is returned

        Args:
            model_name (str): model name

        Returns:
            string -- the calibrated model name
        """

        tmp = model_name.split('/')
        return tmp[0] + '/model'

    @staticmethod
    def get_eval_name(model, data):
        """ Return name of the object containing evaluation results

        Args:
            model (ModelDefinition object or str): 
            data {RawData or DataSet object or str} --

        Returns:
            string -- name of valuation results
        """

        model_name = model
        if not isinstance(model, str):
            model_name = model.repo_info[RepoInfoKey.NAME]
        data_name = data
        if not isinstance(data, str):
            data_name = data.repo_info[RepoInfoKey.NAME]
        return model_name + '/eval/' + data_name

    def get_names(self, ml_obj_type):
        """ Get the list of names of all repo_objects from a given repo_object_type in the repository.

        Args:
            ml_obj_type (MLObjectType): MLObjectType specifying the types of objects names are returned for.

        Returns:
            list of strings -- list of object names for the given category.
        """

        if isinstance(ml_obj_type, MLObjectType):
            return self._ml_repo.get_names(ml_obj_type.value)
        else:
            return self._ml_repo.get_names(ml_obj_type)

    def get_history(self, name, repo_info_fields=None, obj_member_fields=None, version_start=repo_store.RepoStore.FIRST_VERSION,
                    version_end=repo_store.RepoStore.LAST_VERSION):
        """ Return a list of histories of object member variables without bigobjects

        Args:
            name (str): the object name
            repo_info_fields (list of strings): List of fields from repo_info which will be returned in the dictionary. 
                                If List contains flag 'ALL', all fields will be returned.. Defaults to None.
            obj_member_fields (list of strings): List of member atributes from repo_object which will be returned in the dictionary. 
                                If List contains flag 'ALL', all attributes will be returned.. Defaults to None.
            version_start (str): only display versions after version_start. Defaults to repo_store.RepoStore.FIRST_VERSION.
            version_end (str): only display versions up to version_end. Defaults to repo_store.RepoStore.LAST_VERSION.

        Returns:
            str or list of strings -- returns a list of the objects
        """

        tmp = self._ml_repo.get(name, versions=(version_start, version_end),
                                obj_fields=obj_member_fields,  repo_info_fields=repo_info_fields)
        if isinstance(tmp, list):
            return tmp
        return [tmp]

    def get_commits(self,  version_start=repo_store.RepoStore.FIRST_VERSION, version_end=repo_store.RepoStore.LAST_VERSION):
        """ gets the commits

        Args:
            version_start (str): only display versions after version_start. Defaults to repo_store.RepoStore.FIRST_VERSION.
            version_end (str): only display versions up to version_end. Defaults to repo_store.RepoStore.LAST_VERSION.

        Returns:
            list of commit infos -- returns a list of commit infots
        """

        tmp = self.get('CommitInfo', (version_start, version_end))
        if isinstance(tmp, list):
            return tmp
        return [tmp]

    def __obj_latest(self, name, modification_info={}):
        """ Returns version of object created on latest data (modulo specified versions)
            If the object does not exist, it returns None.

        Args:
            name (str): name of object checked
            modification_info (dict): dictionay containing objects and their version used within the check (objects not contained are assumed to be LAST_VERSION). Defaults to {}.

        Returns:
            object -- the latests repo object
        """

        tmp = None
        try:
            tmp = self.get(name, version=repo_store.RepoStore.LAST_VERSION)
        except:
            return None
        # now loop over all modified objects
        modifiers = {}
        for k, v in tmp.repo_info.modification_info.items():
            if k in modification_info.keys():
                modifiers[k] = v
            else:
                modifiers[k] = repo_store.RepoStore.LAST_VERSION
        try:
            tmp = self.get(name, version=None, modifier_versions=modifiers)
        except:
            return None
        return tmp

    def run(self, job):
        """ Executes a job

        Args:
            job (Job): The job object to be executed

        Returns:
            [type] -- Return the name and version of the job or a message that the job does not need to be rerun
        """

        if job.check_rerun(self):
            self.add(job)
            self._job_runner.add(job.repo_info.name,
                                 job.repo_info.version, self._user)
            return job.repo_info.name, job.repo_info.version
        else:
            return 'No input changed since last run, do not start job..'

    def run_training(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION,
                     training_function_version=repo_store.RepoStore.LAST_VERSION,
                     training_data_version=repo_store.RepoStore.LAST_VERSION, training_param_version=repo_store.RepoStore.LAST_VERSION,
                     model_param_version=repo_store.RepoStore.LAST_VERSION, run_descendants=False):
        """ Run the training algorithm. 

        Args:
            model (str): the identifyer of the model. Defaults to None.
            message (str): the commit message. Defaults to None.
            model_version (str): the version of the model. Defaults to repo_store.RepoStore.LAST_VERSION.
            training_function_version (str): the version of the training function. Defaults to repo_store.RepoStore.LAST_VERSION.
            training_data_version (str): the version of the training data. Defaults to repo_store.RepoStore.LAST_VERSION.
            training_param_version (str): the version of the training parameter. Defaults to repo_store.RepoStore.LAST_VERSION.
            model_param_version {str} --the version of the model parameter. Defaults to repo_store.RepoStore.LAST_VERSION.
            run_descendants (bool): if True also run all decendant jobs. Defaults to False.

        Returns:
            [type] -- return name and version or message
        """

        if model is None:
            m_names = self.get_names(MLObjectType.MODEL.value)
            if len(m_names) == 0:
                Exception(
                    'No model definition exists, please define a model first.')
            if len(m_names) > 1:
                Exception(
                    'More than one model in repository, please specify a model to evaluate.')
            model = m_names[0]
        train_job = TrainingJob(model, self._user, training_function_version=training_function_version, model_version=model_version,
                                training_data_version=training_data_version, training_param_version=training_param_version,
                                model_param_version=model_param_version, repo_info={RepoInfoKey.NAME: model + '/jobs/training',
                                                                                    RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
        # Only start training if input data has changed since last run
        if train_job.check_rerun(self):
            self.add(train_job)
            self._job_runner.add(
                train_job.repo_info[RepoInfoKey.NAME], train_job.repo_info[RepoInfoKey.VERSION], self._user)
            logging.info('Training job ' + train_job.repo_info[RepoInfoKey.NAME] + ', version: '
                         + str(train_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
            if run_descendants:
                self.run_evaluation(model + '/model', 'evaluation triggered as descendant of run_training',
                                    predecessors=[(train_job.repo_info[RepoInfoKey.NAME], train_job.repo_info[RepoInfoKey.VERSION])], run_descendants=True)
            return train_job.repo_info[RepoInfoKey.NAME], str(train_job.repo_info[RepoInfoKey.VERSION])
        else:
            return 'No new training started: A model has already been trained on the latest data.'

    def _get_default_object_name(self, obj_name, obj_category):
        """ Returns unique object name of given category if given object_name is None

        Args:
            obj_name (str or None): if not None, the given string will simply be returned
            obj_category (MLObjectType): category of object

        Raises:
            Exception: If not exactly one object of desired type exists in repo

        Returns:
            str -- Resulting object name
        """

        if obj_name is not None:
            return obj_name
        names = self.get_names(obj_category)
        if len(names) == 0:
            raise Exception('No object of type ' +
                            obj_category.value + ' exists.')
        if len(names) > 1:
            raise Exception('More than one object of type ' + obj_category.value +
                            ' exist, please specify a specific object name.')
        return names[0]

    def _get_datasets(self, model, model_version=repo_store.RepoStore.LAST_VERSION):
        """Return name of datasets relevant for evaluation, measurement or tests.

        Args:
            model (str): The name of the calibrated model for which the datasets will be returned.
            model_version (str, optional): Version of model definition for which the dataset will be returned. Defaults to repo_store.RepoStore.LAST_VERSION.

        Returns:
            [dictionary]: The dictionary from dataset name to latest dataset version.
        """
        result = {}
        model_def = str(NamingConventions.Model(
            NamingConventions.CalibratedModel(model)))
        train_data = self.get_training_data(
            model=model_def, model_version=model_version, full_object=False)
        result[train_data.repo_info.name] = train_data.repo_info.version
        m = self.get(model_def, model_version, full_object=False)
        test_data = m.get_test_data(self)
        for t in test_data:
            tmp = self.get(t, full_object=False)
            result[tmp.repo_info.name] = tmp.repo_info.version
        return result

    def _create_evaluation_jobs(self, model=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, predecessors=[], labels=None):
        models = [(self._get_default_object_name(
            model, MLObjectType.CALIBRATED_MODEL), model_version)]
        if model is None and labels is None:
            labels = self.get_names(MLObjectType.LABEL)
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            for l in labels:
                tmp = self.get(l)
                models.append((tmp.name, tmp.version))
        datasets_ = deepcopy(datasets)
        jobs = []
        for m in models:
            if len(datasets) == 0:
                datasets_ = self._get_datasets(m[0])
            for n, v in datasets_.items():
                eval_job = EvalJob(m[0], n, self._user, model_version=m[1], data_version=v,
                                   repo_info={RepoInfoKey.NAME: m[0] + '/jobs/eval_job/' + n,
                                              RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
                if eval_job.check_rerun(self):
                    eval_job.set_predecessor_jobs(predecessors)
                    jobs.append(eval_job)
                # else:
                #    logger.info('Skip running ' + eval_job.repo_info.name + ' has already been run on the specified data, job version: ' + tmp.repo_info.version)
        return jobs

    # def run_jobs(self, job, modifier_versions={}):
    #     """Runs jobs matching a regular expression.

    #     It checks if the job has already been run on the same input data.

    #     Args:
    #         job (str): Regular expression defining used to define the jobs to run
    #         modifier_versions (dict, optional): Defaults to {}. Only jobs which have not already been run on the input data are started.
    #             The dictionary defines the object version which shall be used within the job run, if an object is not specified in this dict, the latest version is used.
    #     """
    #     tmp = self.get_names(MLObjectType.JOB)
    #     import re
    #     jobs = [x for x in tmp and re.match(job, x)]
    #     jobs_to_run = []
    #     for j in jobs:
    #         ref_job = self.get(j) #get job to get all modification infos
    #         mod_info = deepcopy(ref_job.repo_info.modification_info)
    #         for k,v in mod_info.items():
    #             if k in modifier_versions.keys():
    #                 mod_info[k] = v
    #         try:
    #             job = self.get(j, modifier_versions=mod_info)
    #         except:
    #             jobs_to_run.append(job)

    def run_evaluation(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION,
                       datasets={}, predecessors=[], run_descendants=False, labels=None):
        """ Evaluate the model on all datasets. 

        Args:
            model (str): name of model to evaluate, if None and only one model exists. Defaults to None.
            message (str): message inserted into commit, if None: an automated message is created. Defaults to None.
            model_version (str): version of model to be evaluated.. Defaults to repo_store.RepoStore.LAST_VERSION.
            datasets (dict): dictionary of datasets (names and version numbers) on which the model is evaluated. . Defaults to {}.
            predecessors (list): list of jobs which shall have been completed successfull before the evaluation is started. Default is all datasets from testdata on latest version.. Defaults to [].
            run_descendants (bool): if True also run all decendant jobs. Defaults to False.
            labels ([type]): [description]. Defaults to None.

        Returns:
            list of strings -- a list of the job ids
        """

        jobs = self._create_evaluation_jobs(
            model, model_version, datasets, predecessors, labels=labels)
        job_ids = []
        for job in jobs:
            if job.check_rerun(self):
                self.add(job)
                self._job_runner.add(
                    job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION], self._user)
                logging.info('Eval job ' + job.repo_info[RepoInfoKey.NAME] + ', version: '
                             + str(job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
                job_ids.append((job.repo_info[RepoInfoKey.NAME], str(
                    job.repo_info[RepoInfoKey.VERSION])))
                if run_descendants:
                    self.run_measures(job.model,  'run_measures started as predecessor of run_evaluation', model_version=job.model_version, datasets={job.data: repo_store.RepoStore.LAST_VERSION},
                                      predecessors=[(job.repo_info[RepoInfoKey.NAME], job.repo_info[RepoInfoKey.VERSION])])

        return job_ids

    def run_measures(self, model=None, message=None, model_version=repo_store.RepoStore.LAST_VERSION, datasets={}, measures={}, predecessors=[], labels=None):
        """ Run the measures

        Args:
            model (str): name of model to evaluate, if None and only one model exists. Defaults to None.
            message (str): message inserted into commit, if None: an automated message is created. Defaults to None.
            model_version (str): version of model to be evaluated.. Defaults to repo_store.RepoStore.LAST_VERSION.
            datasets (dict): dictionary of datasets (names and version numbers) on which the model is evaluated. . Defaults to {}.
            predecessors (list): list of jobs which shall have been completed successfull before the evaluation is started. Default is all datasets from testdata on latest version.. Defaults to [].
            run_descendants (bool): if True also run all decendant jobs. Defaults to False.
            labels ([type]): [description]. Defaults to None.

        Returns:
            list of strings -- a list of the job ids
        """

        models = [(self._get_default_object_name(
            model, MLObjectType.CALIBRATED_MODEL), model_version)]
        if model is None and labels is None:
            labels = self.get_names(MLObjectType.LABEL)
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            for l in labels:
                tmp = self.get(l)
                models.append((tmp.name, tmp.version))

        datasets_ = deepcopy(datasets)
        measure_names = self.get_names(MLObjectType.MEASURE_CONFIGURATION)
        if len(measure_names) == 0:
            logger.warning('No measures defined.')
            return
        measure_config = measure_names[0]
        measure_config = self.get(measure_config)
        measures_to_run = {}
        if len(measures) == 0:  # if no measures are specified, use all from configuration
            measures_to_run = measure_config.measures
        else:
            for k, v in measures.items():
                measures_to_run[k] = v
        job_ids = []
        for mod in models:
            if len(datasets) == 0:
                datasets_ = self._get_datasets(mod[0])
                for k in datasets_.keys():
                    datasets_[k] = repo_store.RepoStore.LAST_VERSION
            for n, v in datasets_.items():
                for m_name, m in measures_to_run.items():
                    measure_job = MeasureJob(m_name, m[0], m[1], n, mod[0], v, mod[1],
                                             repo_info={RepoInfoKey.NAME: mod[0] + '/jobs/measure/' + n + '/' + m[0],
                                                        RepoInfoKey.CATEGORY: MLObjectType.JOB.value})
                    if measure_job.check_rerun(self):
                        measure_job.set_predecessor_jobs(predecessors)
                        self.add(measure_job)
                        self._job_runner.add(
                            measure_job.repo_info[RepoInfoKey.NAME], measure_job.repo_info[RepoInfoKey.VERSION], self._user)
                        job_ids.append(
                            (measure_job.repo_info[RepoInfoKey.NAME], measure_job.repo_info[RepoInfoKey.VERSION]))
                        logging.info('Measure job ' + measure_job.repo_info[RepoInfoKey.NAME] + ', version: '
                                     + str(measure_job.repo_info[RepoInfoKey.VERSION]) + ' added to jobrunner.')
        return job_ids

    def run_tests(self, test_definitions=None, predecessors=[]):
        """ Run tests for a specific model version.

        Args:
            test_definitions (list or set): List or set of names of the test definitions which shall be executed. If None, all test definitions are executed.. Defaults to None.
            predecessors (list): list of jobs which shall have been completed successfull before the evaluation is started. Default is all datasets from testdata on latest version.. Defaults to [].

        Returns:
            str -- ticket number of job
        """

        test_defs = test_definitions
        if test_defs is None:
            test_defs = self._ml_repo.get_names(
                MLObjectType.TEST_DEFINITION.value)
        job_ids = []
        for t in test_defs:
            tmp = self.get(t)
            tests = tmp.create(self)
            for tt in tests:
                if tt.check_rerun(self):
                    self.add(tt, category=MLObjectType.TEST)
                    self._job_runner.add(
                        tt.repo_info.name, tt.repo_info.version, self._user)
                    job_ids.append((tt.repo_info.name, tt.repo_info.version))
        return job_ids

    def set_label(self, label_name, model=None, model_version=repo_store.RepoStore.LAST_VERSION, message=''):
        """ Label a certain model version.

            It checks if a model with this version really exists and throws an exception if such a model does not exist.
            This method labels a certain model version.

        Args:
            label_name (str): the label name
            model (str): the identifyer of the model. Defaults to None.
            model_version (str): model version for which the label is set.. Defaults to repo_store.RepoStore.LAST_VERSION.
            message (str): commit message. Defaults to ''.
        """
        # check if label with same model and model version already exists
        model = self._get_default_object_name(
            model, MLObjectType.CALIBRATED_MODEL)
        tmp = self.get(label_name, throw_error_not_exist=False)

        # check if a model with this version exists
        m = self.get(model)
        if model_version == repo_store.RepoStore.LAST_VERSION:
            model_version = m.repo_info[RepoInfoKey.VERSION]

        if not tmp == []:
            if isinstance(tmp, list):
                tmp = tmp[0]
            if tmp.name == model and model_version == tmp.version:
                return
        label = repo_objects.Label(model, model_version, repo_info={RepoInfoKey.NAME: label_name,
                                                                    RepoInfoKey.CATEGORY: MLObjectType.LABEL.value})
        return self.add(label)

    def push(self):
        """ Push changes to an external repo.
        """

        self._ml_repo.push()
        self._numpy_repo.push()

    def pull(self):
        """ Pull changes from an external repo
        """

        self._ml_repo.pull()
        self._numpy_repo.pull()

    def _object_exists(self, name):
        """ checks whether an object exists (True) or not (False)

        Args:
            name (str): the name of the object

        Returns:
            bool -- True: object exists, False: object does not exist
        """

        return self._ml_repo.object_exists(name)
