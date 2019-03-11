"""Module for pailab to pandas

    This module defines all necessary objects and functions to use pandas for preprocessing within pailab
    Currently only preprocessing functions without a previous fit to the training data is supported.
"""

__version__ = '0.0.1'

import pandas as pd

from pailab.ml_repo.repo_objects import repo_object_init
from pailab.ml_repo.repo_objects import RepoInfoKey
from pailab.ml_repo.repo_objects import Preprocessor
from pailab.ml_repo.repo import MLObjectType


def get_classname(x):
    return x.__class__.__module__ + '.' + x.__class__.__name__

# region Preprocessing


class PandasPreprocessingParam:
    """Interfaces the parameters of the pandas preprocessing algorithms

    """

    @repo_object_init()
    def __init__(self, preprocessor, pandas_param):
        self.pandas_module_name = preprocessor.__module__
        self.pandas_function_name = preprocessor.__name__
        self.pandas_params = pandas_param

    def get_params(self):
        if self.pandas_params is None:
            return {}
        else:
            return self.pandas_params


class PandasPreprocessor:
    """Class to store all pandas preprocessor
    """

    @repo_object_init()
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


def transform_pandas(preprocessor_param, data_x, column_names):
    def get_function(full_function_name):
        parts = full_function_name.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    m = get_function(preprocessor_param.pandas_module_name +
                     '.' + preprocessor_param.pandas_function_name)

    # tranform to pandas
    pandas_data = pd.DataFrame(data=data_x, columns=column_names)

    # transform
    pandas_data = m(data = pandas_data, **preprocessor_param.get_params())

    # transform back
    x_data = pandas_data.values
    x_coordinates = list(pandas_data)

    return x_data, x_coordinates


def add_preprocessor(repo, pandas_preprocessor, preprocessor_name=None, preprocessor_param=None):
    """Adds a new pandas preprocessor to a pailab MLRepo

    Args:
        repo ([type]): [description]
        preprocessor (Class): The pandas preprocessor class
        preprocessor_name ([type], optional): Defaults to None. [description]
        preprocessor_param ([type], optional): Defaults to None. [description]
    """

    # preprocessor name
    if not callable(pandas_preprocessor):
        raise Exception(
            'Currently only function are supported as pandas preprocessors, type was ' + type(pandas_preprocessor))

    p_name = preprocessor_name
    if p_name is None:
        p_name = pandas_preprocessor.__name__

    # add preprocessing
    repo.add_preprocessing_transforming_function(
        transform_pandas, repo_name=p_name + '/transform_pandas')

    pandas_param = PandasPreprocessingParam(pandas_preprocessor, preprocessor_param,
                                            repo_info={RepoInfoKey.NAME.value: p_name + '/preprocessor_param',
                                                       RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSOR_PARAM.value})

    if pandas_param is not None:
        repo.add(pandas_param, 'adding preprocessor parameter')

    repo.add_preprocessor(p_name, transforming_function=p_name+'/transform_pandas', fitting_function=None,
                          preprocessor_param=p_name + '/preprocessor_param')

# endregion
