# -*- coding: utf-8 -*-
"""This module contains functions for model agnostic interpretation methods. 
"""

import numpy as np
from pailab.ml_repo.repo_objects import RepoObject, RawData
from pailab.tools.tools import ml_cache
from pailab.ml_repo.repo_store import RepoStore  # pylint: disable=E0401

has_sklearn = True
try:
    from sklearn.cluster import KMeans
except ImportError:
    import warnings
    warnings.warn('No sklearn installed, some functions of this submodule may not be usable.')
    has_sklearn = False


@ml_cache
def _compute_ice(data, model_eval_function, model,
                 y_coordinate, x_coordinate,
                 x_values, start_index = 0, end_index = -1, scale=True):
    """Independent conditional expectation plot

    Args:
        data ([type]): [description]
        model_eval_function ([type]): [description]
        model ([type]): [description]
        direction ([type]): [description]
        y_coordinate ([type]): [description]
        x_coordinate ([type]): [description]
        x_values ([type]): [description]
        scale (non-zero int, inf, -inf, ''): If specified, the scaling parameter is used to scale each projection by the respective numpy.linalg.norm.
        Scaling may be useful to compare ice at different data points.
    Returns:
        [type]: [description]
    """
    x_data = data.x_data[start_index:end_index, :]
    
    # compute input for evaluation
    shape = (x_data.shape[0], len(x_values),)  # x_data.shape[1:]
    _x_data = np.empty(shape=(len(x_values), ) + x_data.shape[1:])
    eval_data = RawData(_x_data, [str(i) for i in range(x_data.shape[1])])
    result = np.empty(shape)
    eval_f = model_eval_function.create()
    nom = 1.0
    if isinstance(scale, str):
        use_scaling = False
    else:
        use_scaling = True

    for i in range(x_data.shape[0]):
        for j in range(len(x_values)):
            eval_data.x_data[j] = x_data[i]
            eval_data.x_data[j, x_coordinate] = x_values[j]
        tmp = eval_f(model, eval_data)
        if len(tmp.shape) > 1:
            y = tmp[:, y_coordinate]
        elif y_coordinate == 0:
            y = tmp[:]
        else:
            raise Exception('Evaluation data is just an array but y_coordinate > 0.')
        if use_scaling:
            nom = 1.0 / max(np.linalg.norm(y, ord = scale), 1e-10)
        result[i] = nom * y
    return result


def _get_model_eval(model, model_version, ml_repo, model_label=None):
    """Return model and evaluation function for given model and model version

    Args:
        model (str): The name of the (calibrated) model.
        model_version (str): The version of the calibrated model.
        ml_repo (MLRepo): The ml repo used to retrieve the model.
        model_label (str): A label defining the model to investigate (instead of defining the model).
    """

    model_version_ = model_version
    if model_label is not None:
        label = ml_repo.get(model_label)
        model_ = label.name
        model_version_ = label.version

    if isinstance(model, str):
        if len(model.split('/')) == 1:
            model_ = model + '/model'
        else:
            model_ = model
        result_model = ml_repo.get(model_, model_version_)
    else:
        result_model = model

    # now the eval function
    model_definition_name = result_model.repo_info.name.split('/')[0]
    model_definition = ml_repo.get(
        model_definition_name, result_model.repo_info.modification_info[model_definition_name])
    result_eval_f = ml_repo.get(model_definition.eval_function)
    return result_model, result_eval_f



class ICE_Results:
    def __init__(self):
        """Class holding the results of the compute_ice function.

        Attributes:
            ice (numpy matrix): Each row contains the ICE values at each datapoint.
            x_values (list): List of the used x values for the ICE projection.
            x_coord_name (str): Name of x-coordinate used for ICE.
            y_coord_name (str): Name of y-coordinate used for ICE.
            labels (numpy array): If functional clustering of ICE curves has been performed, it contains the cluster label for each datapoint.
            cluster_centers (numpy matriy): Each row contains one cluster center of the functional clustering.
            distance_to_clusters (numpy matrix): Each row contains the distance of the respective ICE graph to each of the clusters.
        """
        self.ice = None
        self.x_values = None
        self.x_coord_name = ''
        self.y_coord_name = ''
        self.labels = None
        self.cluster_centers = None
        self.distance_to_clusters = None

@ml_cache
def _compute_and_cluster_ice(data, model_eval_function, model,
                 y_coordinate, x_coordinate,
                 x_values, start_index = 0, end_index = -1, scale=True,
                 n_clusters=20, random_state=42):
    """[summary]
    
    Args:
        data ([type]): [description]
        model_eval_function ([type]): [description]
        model ([type]): [description]
        y_coordinate ([type]): [description]
        x_coordinate ([type]): [description]
        x_values ([type]): [description]
        start_index (int, optional): [description]. Defaults to 0.
        end_index (int, optional): [description]. Defaults to -1.
        scale (bool, optional): [description]. Defaults to True.
        n_clusters (int, optional): [description]. Defaults to 20.
        random_state (int, optional): [description]. Defaults to 42.
    
    Returns:
        list of double: the x-values used to compute ICE
        numpy matrix: Each row contains the ice values
        numpy vector: Each entry contains the cluster of the respective point
        numpy matrix: Each row contains a cluster center
        numpy matrix: Each row contains the distances to the different clusters (i-th column contains distance to i-th cluster)
    """
    if isinstance(x_coordinate, str):
        x_coordinate = data.x_coord_names.index(x_coordinate)
    if isinstance(y_coordinate, str):
        y_coordinate = data.y_coord_names.index(y_coordinate)
    ice = _compute_ice(data, model_eval_function, model, y_coordinate, x_coordinate, x_values,
                                 start_index=start_index, end_index=end_index, scale=scale)
    labels = None
    cluster_centers = None
    distance_to_clusters = None
    if n_clusters > 0:
        if not has_sklearn:
            raise Exception('Cannot compute ice clusters: No sklearn installed. Please either install sklearn or set n_clusters to zero.')
        k_means = KMeans(init='k-means++', n_clusters=n_clusters,
                        n_init=10, random_state=random_state)
        labels = k_means.fit_predict(ice)
        # now store for each data point the distance to the respectiv cluster center
        distance_to_clusters = k_means.transform(ice)
        cluster_centers =  k_means.cluster_centers_
        #distance_to_center = np.empty((x.shape[0],))

    #for i in range(x.shape[0]):
    #    distance_to_center[i] = distance_to_clusters[i, labels[i]]
    #big_obj['distance_to_center'] = distance_to_center
    return x_values, ice, labels, cluster_centers, distance_to_clusters

def compute_ice(ml_repo, x_values, data, model=None, model_label=None, model_version=RepoStore.LAST_VERSION,
                data_version=RepoStore.LAST_VERSION, y_coordinate=0, x_coordinate = 0,
                start_index=0, end_index=-1, scale='', cache = False,
                n_clusters=0, random_state=42):
    """Compute individual conditional expectation (ice) for a given dataset and model
    
    Args:
        ml_repo (MLRepo): MLRepo used to retrieve model and data and be used in caching.
        x_values (list): List of x values for the ICE.
        data (str, DataSet, RawData): Either name of data or directly the data object which is used as basis for ICE (an ICE is computed at each datapoint of the data).
        model (str, optional): Name of model in the MLRepo for which the ICE will be computed. If None, model_label must be specified, defining the model to be used. Defaults to None.
        model_label (str, optional): Label defining the model to be used. Defaults to None.
        model_version (str, optional): Version of model to be used for ICE. Only needed if model is specified. Defaults to RepoStore.LAST_VERSION.
        data_version (str, optional): Version of data used. Defaults to RepoStore.LAST_VERSION.
        y_coordinate (int or str, optional): Defines y-coordinate (either by name or coordinate index) for which the ICE is computed. Defaults to 0.
        x_coordinate (int or str, optional): Defines x-coordinate (either by name or coordinate index) for which the ICE is computed. Defaults to 0.
        start_index (int, optional): Defines the start index of the data to be used in ICE computation (data[start_index:end_index] will be used). Defaults to 0.
        end_index (int, optional): Defines the end index of the data to be used in ICE computation (data[start_index:end_index] will be used). Defaults to -1.
        n_clusters (int, optional): Number of clusters for the functional clustering of the ICE curves. Defaults to 0.
        scale (str, optional): String defining the scaling for the functions before functional clustering is applied. Scaling is perfomred by
                                dividing the vector of the y-values of the ICE by the respective vector norm defined by scaling.
                                The scaling must be one of numpy's valid strings for linalg.norm's ord parameter. If string is empty, no scaling will be applied.
                                Defaults to ''. 
        cache (bool, optional): If True, results will be cached. Defaults to False.
        random_state (int, optional): Random state for random generator used to initialize randam centroids. Defaults to 42.
        
    Returns:
        ICE_Results: result object containing all relevant data (including functional clustering)
    """
    data_ = data
    if isinstance(data, str):
        data_ = ml_repo.get(data, data_version, full_object = True)
    if isinstance(x_coordinate, str):
        x_coordinate = data_.x_coord_names.index(x_coordinate)
    if isinstance(y_coordinate, str):
        y_coordinate = data_.y_coord_names.index(y_coordinate)
    model_, model_eval_f = _get_model_eval(
        model, model_version, ml_repo, model_label)

    cache_ = None
    if cache:
        cache_ = ml_repo

    result = ICE_Results()
    result.x_values, result.ice, result.labels, result.cluster_centers, result.distance_to_clusters = _compute_and_cluster_ice(data_, model_eval_f, model_,  y_coordinate,
                     x_coordinate=x_coordinate, x_values=x_values, scale=scale, 
                     start_index = start_index, end_index = end_index, cache=cache_,
                     n_clusters=n_clusters, random_state=random_state)
    result.x_coord_name = data_.x_coord_names[x_coordinate]
    result.y_coord_name = data_.y_coord_names[y_coordinate]
    return result