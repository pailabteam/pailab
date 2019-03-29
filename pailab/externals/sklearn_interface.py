"""Module for pailab to sklearn

    This module defines all necessary objects and functions to use sklearn from within pailab.
"""

__version__ = '0.0.1'

from pailab.ml_repo.repo_objects import repo_object_init
from pailab.ml_repo.repo_objects import RepoInfoKey
from pailab.ml_repo.repo_objects import Preprocessor
from pailab.ml_repo.repo import MLObjectType


def get_classname(x):
    return x.__class__.__module__ + '.' + x.__class__.__name__


class SKLearnModelParam:
    """Interfaces the parameters of the sklearn algorithms

    """

    @repo_object_init()
    def __init__(self, model, sklearn_param):
        self.sklearn_module_name = model.__class__.__module__
        self.sklearn_class_name = model.__class__.__name__
        self.sklearn_params = sklearn_param

    def get_params(self):
        return self.sklearn_params


class SKLearnModel:
    """Class to store all sklearn models in pailab's MLRepo
    """

    @repo_object_init()
    def __init__(self, model, preprocessors=None):
        self.model = model
        self.preprocessors = preprocessors


def eval_sklearn(model, data):
    """Function to evaluate an sklearn model

    Args:
        model ([type]): [description]
        data ([type]): [description]

    Returns:
        [type]: [description]
    """

    return model.model.predict(data)


def train_sklearn(model_param, data_x, data_y, preprocessors=None):
    def get_class(full_class_name):
        parts = full_class_name.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    m = get_class(model_param.sklearn_module_name +
                  '.' + model_param.sklearn_class_name)

    model = m(**model_param.sklearn_params)
    model.fit(data_x, data_y)
    result = SKLearnModel(model, preprocessors=preprocessors, repo_info={})
    return result


def add_model(repo, skl_learner, model_name=None, model_param=None,
              preprocessors=None):
    """Adds a new sklearn model to a pailab MLRepo

    Args:
        repo ([type]): [description]
        skl_learner ([type]): [description]
        model_name ([type], optional): Defaults to None. [description]
        model_param ([type], optional): Defaults to None. [description]
        preprocessors (list of strings, optional): List of used preprocessors
    """

    # model name
    m_name = model_name
    if m_name is None:
        m_name = skl_learner.__class__.__name__

    # check whether the preprocessors have been added

    # adding model functions
    repo.add_eval_function(eval_sklearn, repo_name='eval_sklearn')
    repo.add_training_function(train_sklearn, repo_name='train_sklearn')

    param = skl_learner.get_params(True)
    if model_param is not None:
        for k, v in model_param.items():
            param[k] = v
    model_p = SKLearnModelParam(
        skl_learner, param, repo_info={RepoInfoKey.NAME.value: m_name + '/model_param',
                                       RepoInfoKey.CATEGORY: MLObjectType.MODEL_PARAM.value})

    repo.add(model_p, 'adding model and training parameter')

    repo.add_model(m_name, model_eval='eval_sklearn', model_training='train_sklearn',
                   model_param=m_name + '/model_param', training_param='',
                   preprocessors=preprocessors)

# region Preprocessing


class SKLearnPreprocessingParam:
    """Interfaces the parameters of the sklearn algorithms

    """

    @repo_object_init()
    def __init__(self, preprocessor, sklearn_param):
        self.sklearn_module_name = preprocessor.__class__.__module__
        self.sklearn_class_name = preprocessor.__class__.__name__
        self.sklearn_params = sklearn_param

    def get_params(self):
        return self.sklearn_params


class SKLearnPreprocessor:
    """Class to store all sklearn preprocessor
    """

    @repo_object_init()
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


def transform_sklearn(preprocessor_param, data_x, x_coord_names, fitted_preprocessor=None):
    if fitted_preprocessor is None:
        def get_class(full_class_name):
            parts = full_class_name.split('.')
            module = ".".join(parts[:-1])
            m = __import__(module)
            for comp in parts[1:]:
                m = getattr(m, comp)
            return m
        m = get_class(preprocessor_param.sklearn_module_name +
                      '.' + preprocessor_param.sklearn_class_name)

        prepro = m(**preprocessor_param.sklearn_params)
    else:
        prepro = fitted_preprocessor
    return prepro.preprocessor.transform(X=data_x), x_coord_names


def fit_sklearn(preprocessor_param, data_x, x_coord_names):
    def get_class(full_class_name):
        parts = full_class_name.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    m = get_class(preprocessor_param.sklearn_module_name +
                  '.' + preprocessor_param.sklearn_class_name)

    if preprocessor_param.sklearn_params is not None:
        prepro = m(**preprocessor_param.sklearn_params)
    else:
        prepro = m()
    prepro.fit(X=data_x)
    return SKLearnPreprocessor(prepro, repo_info={})


def add_preprocessor(repo, skl_preprocessor, preprocessor_name=None, preprocessor_param=None):
    """Adds a new sklearn preprocessor to a pailab MLRepo

    Args:
        repo ([type]): [description]
        preprocessor (Class): The sklearn preprocessor class
        preprocessor_name ([type], optional): Defaults to None. [description]
        preprocessor_param ([type], optional): Defaults to None. [description]
    """

    # preprocessor name
    p_name = preprocessor_name
    if p_name is None:
        p_name = skl_preprocessor.__class__.__name__

    # add preprocessing
    repo.add_preprocessing_transforming_function(
        transform_sklearn, repo_name = p_name + '/transform_sklearn')
    repo.add_preprocessing_fitting_function(
        fit_sklearn, repo_name = p_name + '/fit_sklearn')
    param = skl_preprocessor.get_params(True)
    if preprocessor_param is not None:
        for k, v in preprocessor_param.items():
            param[k] = v
    skl_param = SKLearnPreprocessingParam(skl_preprocessor, param,
                                          repo_info = {RepoInfoKey.NAME.value: p_name + '/preprocessor_param',
                                                     RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSOR_PARAM.value})

    if skl_param is not None:
        repo.add(skl_param, 'adding preprocessor parameter')

    repo.add_preprocessor(p_name, transforming_function = p_name+'/transform_sklearn', fitting_function = p_name+'/fit_sklearn',
                          preprocessor_param = p_name + '/preprocessor_param')

# endregion
