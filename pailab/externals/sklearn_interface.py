"""Module for pailab to sklearn

    This module defines all necessary objects and functions to use sklearn from within pailab.
"""

__version__ = '0.0.1'

import numpy as np
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

    return model.model.predict(data.x_data)


def train_sklearn(model_param, data, preprocessors=None):
    def get_class(full_class_name):
        parts = full_class_name.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    m = get_class(model_param.sklearn_module_name +
                  '.' + model_param.sklearn_class_name)
    data_x = data.x_data
    data_y = data.y_data
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
    def __init__(self, preprocessor, sklearn_param, columns = None):
        """Constructor
        
        Args:
            preprocessor (obj): Object of sklearn preprocessor
            sklearn_param (dict ): Dictionary of preprocessor parameter
            columns (list of str optional): If set, preprocessor is applied only to columns defined by the respective list of names. Defaults to None.
        """
        self.sklearn_module_name = preprocessor.__class__.__module__
        self.sklearn_class_name = preprocessor.__class__.__name__
        self.sklearn_params = sklearn_param
        self.columns =  columns

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
    if preprocessor_param.columns is None:
        return prepro.preprocessor.transform(X=data_x), x_coord_names
    
    # get submatrix with columns used for this preprocessor
    columns = [x_coord_names.index(x) for x in preprocessor_param.columns] # columns which will be replaced by transformation
    x_trans = prepro.preprocessor.transform(X=data_x[:, columns])
    if x_trans.shape[1] == len(columns): #in this case we replace the column by their trasformed vesions
        result = data_x.copy()
        for i in columns:
            result[:,i] = x_trans[:,i]
        x_coord_names_new = x_coord_names.copy()
        if hasattr(prepro.preprocessor,'get_feature_names'):
            new_names = prepro.preprocessor.get_feature_names(preprocessor_param.columns)
            for i in range(len(columns)):
                x_coord_names_new[columns[i]] = new_names[i]
        return result, x_coord_names_new
    elif x_trans.shape[1] > len(columns): # append new columns from preprocessing to the end
        x_coord_names_new = [x for x in x_coord_names if x not in preprocessor_param.columns]
        if hasattr(prepro.preprocessor,'get_feature_names'):
            x_coord_names_new.extend(prepro.preprocessor.get_feature_names(preprocessor_param.columns))
        else:
            x_coord_names_new.extend(['trans_'+str(i) for i in x_trans.shape[1]])
        remaining_columns = [ i for i in data_x.shape[1] if x_coord_names[i] not in preprocessor_param.columns]
        return np.concatenate((data_x[:,remaining_columns], x_trans,), axis=1), x_coord_names_new
    else:
        raise Exception('Only implemented for preprocessors that do not shrink number of columns.')



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
    if preprocessor_param.columns is not None:
        columns = [x_coord_names.index(x) for x in preprocessor_param.columns]
        prepro.fit(X=data_x[:, columns])
    else:
        prepro.fit(X=data_x)
    return SKLearnPreprocessor(prepro, repo_info={})


def add_preprocessor(repo, skl_preprocessor, preprocessor_name=None, preprocessor_param=None, columns = None):
    """Adds a new sklearn preprocessor to a pailab MLRepo

    Args:
        repo (MLRepo): MLRepo to which preprocessor will be added
        preprocessor (obj): An object to the sklearn preprocessor class.
        preprocessor_name (str, optional): Name of the preprocessor in repo. If None, a default name will be generated. Defaults to None.
        preprocessor_param (dict, optional): Dictionary of parameters for the SKLearn preprocessor. Ths elements will be used to overwrite 
                                            the parameters in the given preprocessor object. Defaults to None. 
        columns (list(str)): List of string defining the columns the preprocessor will be applied to. If None, all columns are used. Defaults to None.
    """

    # preprocessor name
    p_name = preprocessor_name
    if p_name is None:
        p_name = skl_preprocessor.__class__.__name__

    # add preprocessing
    repo.add_preprocessing_transforming_function(
        transform_sklearn, repo_name=p_name + '/transform_sklearn')
    repo.add_preprocessing_fitting_function(
        fit_sklearn, repo_name=p_name + '/fit_sklearn')
    param = skl_preprocessor.get_params(True)
    if preprocessor_param is not None:
        for k, v in preprocessor_param.items():
            param[k] = v
    skl_param = SKLearnPreprocessingParam(skl_preprocessor, param, columns, 
                                          repo_info={RepoInfoKey.NAME.value: p_name + '/preprocessor_param',
                                                     RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSOR_PARAM.value})

    if skl_param is not None:
        repo.add(skl_param, 'adding preprocessor parameter')

    repo.add_preprocessor(p_name, transforming_function=p_name+'/transform_sklearn', fitting_function=p_name+'/fit_sklearn',
                          preprocessor_param=p_name + '/preprocessor_param')

# endregion
