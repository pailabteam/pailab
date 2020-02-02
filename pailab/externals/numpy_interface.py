"""Module for pailab to numpy

    This module defines all necessary objects and functions to use numpy for preprocessing within pailab
    Currently only preprocessing functions without a previous fit to the training data is supported.
"""

__version__ = '0.0.1'

import numpy as np

from pailab.ml_repo.repo_objects import repo_object_init
from pailab.ml_repo.repo_objects import RepoInfoKey
from pailab.ml_repo.repo_objects import Preprocessor
from pailab.ml_repo.repo import MLObjectType


def get_classname(x):
    return x.__class__.__module__ + '.' + x.__class__.__name__

# region Preprocessing


class NumpyPreprocessingParam:
    """Interfaces the parameters of the numpy preprocessing algorithms

    """

    @repo_object_init()
    def __init__(self, numpy_param):
        self.numpy_params = numpy_param

    def get_params(self):
        return self.numpy_params


class NumpyPreprocessor:
    """Class to store all numpy preprocessor
    """

    @repo_object_init()
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


def select_columns(preprocessor_param, data_x, column_names):
    selected_columns = preprocessor_param.get_params()['columns']
    # first find the indices of the columns to select

    idx = []
    for column in selected_columns:
        try:
            idx.append(column_names.index(column))
        except:
            raise Exception('Column ' + column +
                            ' was not found and can not be selected.')

    # select columns
    x_data = data_x[:, idx]
    x_coordinates = [column_names[i] for i in idx]

    return x_data, x_coordinates


def remove_rows_nan(data_x, column_names):
    return data_x[~np.isnan(data_x).any(axis=1)], column_names
    

def add_preprocessor_select_columns(repo, preprocessor_name='Numpy_Select_Columns', preprocessor_param={}):
    """Adds a new numpy preprocessor to a pailab MLRepo
    This is a specific preprocessor to select certain columns

    Args:
        repo ([type]): [description]
        preprocessor_name ([type], optional): Defaults to None. [description]
        preprocessor_param ([type], optional): Defaults to None. [description]
    """

    if not 'columns' in preprocessor_param:
        raise Exception(
            'The preprocessor parameters must have an entry columns')

    # add preprocessing
    repo.add_preprocessing_transforming_function(
        select_columns, repo_name=preprocessor_name + '/select_columns')

    numpy_param = NumpyPreprocessingParam(preprocessor_param,
                                          repo_info={RepoInfoKey.NAME.value: preprocessor_name + '/preprocessor_param',
                                                     RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSOR_PARAM.value})

    repo.add(numpy_param, 'adding preprocessor parameter')

    repo.add_preprocessor(preprocessor_name, transforming_function=preprocessor_name+'/select_columns', fitting_function=None,
                          preprocessor_param=preprocessor_name + '/preprocessor_param')


def add_preprocessor_remove_rows_nan(repo, preprocessor_name='Numpy_Select_Columns'):
    """Adds a new numpy preprocessor to a pailab MLRepo
    This is a specific preprocessor to remove rows containing nan
    Args:
        repo (MLRepo): repository
        preprocessor_name (str, optional): Defaults to 'Numpy_Select_Columns'. Defins preprocessor name.
       
    """
    # add preprocessing
    repo.add_preprocessing_transforming_function(
        remove_rows_nan, repo_name=preprocessor_name + '/select_columns')
    numpy_param = NumpyPreprocessingParam({},
                                          repo_info={RepoInfoKey.NAME.value: preprocessor_name + '/preprocessor_param',
                                                     RepoInfoKey.CATEGORY: MLObjectType.PREPROCESSOR_PARAM.value})
    repo.add(numpy_param, 'adding preprocessor parameter')

    repo.add_preprocessor(preprocessor_name, transforming_function=preprocessor_name+'/select_columns', fitting_function=None,
                          preprocessor_param=preprocessor_name + '/preprocessor_param')
# endregion
