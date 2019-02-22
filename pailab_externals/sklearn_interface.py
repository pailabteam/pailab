"""Module for pailab to sklearn

    This module defines all necessary objects and functions to use sklearn from within pailab.
"""

__version__ = '0.0.1'

from pailab.repo_objects import repo_object_init
from pailab.repo_objects import RepoInfoKey
from pailab.repo_objects import Preprocessor
from pailab.repo import MLObjectType


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
    """

    # model name
    m_name = model_name
    if m_name is None:
        m_name = skl_learner.__class__.__name__

    # add preprocessing
    if preprocessors is not None:
        for k in preprocessors:
            repo.add_preprocessing_transforming_function(
                transform_sklearn, repo_name='transform_sklearn')
            if k.fitting_function is not None:
                repo.add_preprocessing_fitting_function(
                    fit_sklearn, repo_name='fit_sklearn')
            fit_param = SKLearnFittingParam(k.preprocessor, k.fitting_param,
                                            repo_info={RepoInfoKey.NAME.value: m_name + '/preprocessor_param',
                                                       RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSING_PARAM.value})
            if k.fitting_param is not None:
                repo.add(fit_param, 'adding preprocessor')

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


class SKLearnFittingParam:
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


def create_preprocessor(preprocessor, name=None):
    p_name = name
    if p_name is None:
        p_name = preprocessor.__class__.__name__
    fitting_param = SKLearnFittingParam(preprocessor, sklearn_param=None,
                                        repo_info={RepoInfoKey.NAME.value: p_name + '/preprocessor_param',
                                                   RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSING_PARAM.value})
    return Preprocessor(preprocessor, fitting_function='fit_sklearn', transforming_function='transform_sklearn',
                        fitting_param=fitting_param)


def transform_sklearn(fitting_param, data_x, fitted_preprocessor=None):
    if fitted_preprocessor is None:
        def get_class(full_class_name):
            parts = full_class_name.split('.')
            module = ".".join(parts[:-1])
            m = __import__(module)
            for comp in parts[1:]:
                m = getattr(m, comp)
            return m
        m = get_class(fitting_param.sklearn_module_name +
                      '.' + fitting_param.sklearn_class_name)

        prepro = m(**fitting_param.sklearn_params)
    else:
        prepro = fitted_preprocessor
    return prepro.preprocessor.transform(X=data_x)


def fit_sklearn(fitting_param, data_x):
    def get_class(full_class_name):
        parts = full_class_name.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    m = get_class(fitting_param.sklearn_module_name +
                  '.' + fitting_param.sklearn_class_name)

    if fitting_param.sklearn_params is not None:
        prepro = m(**fitting_param.sklearn_params)
    else:
        prepro = m()
    prepro.fit(X=data_x)
    return SKLearnPreprocessor(prepro, repo_info={})

# endregion
