# -*- coding: utf-8 -*-
"""This module contains functions for model agnostic interpretation methods. 
"""

import numpy as np
from pailab.ml_repo.repo_objects import RepoObject, RawData, RepoInfoKey
from pailab.tools.tools import ml_cache
from pailab.ml_repo.repo_store import RepoStore  # pylint: disable=E0401
from pailab.ml_repo.repo import MLObjectType


has_sklearn = True
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_kernels
    from sklearn import preprocessing
except ImportError:
    import warnings
    warnings.warn(
        'No sklearn installed, some functions of this submodule may not be usable.')
    has_sklearn = False


@ml_cache
def _compute_ice(data, model_eval_function, model,
                 y_coordinate, x_coordinate,
                 x_values, start_index=0, end_index=-1, scale=''):
    """Independent conditional expectation plot

    Args:
        data ([type]): [description]
        model_eval_function ([type]): [description]
        model ([type]): [description]
        direction ([type]): [description]
        y_coordinate ([type]): [description]
        x_coordinate ([type]): [description]
        x_values ([type]): [description]
        scale (str or int, optional): String defining the scaling for the functions before functional clustering is applied. Scaling is perfomred by
                                dividing the vector of the y-values of the ICE by the respective vector norm defined by scaling.
                                The scaling must be one of numpy's valid strings for linalg.norm's ord parameter. If string is empty, no scaling will be applied.
                                Defaults to ''. 
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
            raise Exception(
                'Evaluation data is just an array but y_coordinate > 0.')
        if use_scaling:
            nom = 1.0 / max(np.linalg.norm(y, ord=scale), 1e-10)
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
        model = label.name
        model_version = label.version

    if isinstance(model, str):
        if len(model.split('/')) == 1:
            model_ = model + '/model'
        else:
            model_ = model
        result_model = ml_repo.get(model_, model_version_, full_object = True)
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
            data_name (str): Name of input data  used for ICE curves.
            start_index (int): Start index of input data.
            model (str): Name of model used for the ICE curves.
            model_version (str): Version of model used for the ICE curves.
            data_name (str): Name of underlying data.
            data_version (str): Version of underlying data.
        """
        self.ice = None
        self.x_values = None
        self.x_coord_name = ''
        self.y_coord_name = ''
        self.labels = None
        self.cluster_centers = None
        self.distance_to_clusters = None
        self.data_name = ''
        self.data_version = ''
        self.start_index = 0
        self.model = ''
        self.model_version = ''

    def _validate_for_comparison(self, ice_results_2):
        if self.data_name != ice_results_2.data_name:
            raise Exception('Cannot compare two ice results on different data, ice_results_1 : ' +
                            self.data_name + ', ice_results_2: ' + ice_results_2.data_name)
        if self.data_version != ice_results_2.data_version:
            raise Exception('Cannot compare two ice results on different data, ice_results_1 : ' +
                            self.data_version + ', ice_results_2: ' + ice_results_2.data_version)
        if self.start_index != ice_results_2.start_index:
            raise Exception('Cannot compare two ice results with different start_index, ice_results_1 : ' +
                            str(self.start_index) + ', ice_results_2: ' + str(ice_results_2.start_index))

    def compute_cluster_average(self, ice_results_2,):
        """[summary]

        Args:
            ice_results_2 ([type]): [description]

        Returns:
            numpy matrix: Matrix containing the average of values from ice_results_2 over the different clusters from this result. 
        """
        self._validate_for_comparison(ice_results_2)
        if self.cluster_centers is None:
            raise Exception("No clusters have been computed yet.")
        result = np.empty(self.cluster_centers.shape)

        for i in range(self.cluster_centers.shape[0]):
            tmp = ice_results_2.ice[self.labels == i]
            result[i] = np.mean(tmp, axis=0)
        return result


def functional_clustering(x, scale='',
                          n_clusters=20, random_state=42):
    """Given a set of different 1D functions (represented in matrix form where each row contains the function values on a given grid), this
        method clusters those functions and tries to find typical function structures.

    Args:
        x (numpy matrix): Matrix containing in each row the function values at a datapoint.
        n_clusters (int, optional): Number of clusters for the functional clustering of the ICE curves. Defaults to 0.
        random_state (int, optional): [description]. Defaults to 42.
        scale (str or int, optional): String defining the scaling for the functions before functional clustering is applied. Scaling is perfomred by
                                dividing the vector of the y-values of the ICE by the respective vector norm defined by scaling.
                                The scaling must be one of numpy's valid strings for linalg.norm's ord parameter. If string is empty, no scaling will be applied.
                                Defaults to ''. 
    Returns:
        numpy vector: Vector where each value defines the corresponding cluster of the respective function (x[i] is the cluster the i-th function belongs to).
        numpy matrix: Contains in each row th distance to each cluster for the respective function.
        numpy matrix: Contains in each row a cluster centroid.
    """
    if not has_sklearn:
        raise Exception(
            'Cannot apply functional clustering: No sklearn installed. Please either install sklearn or set n_clusters to zero.')
    k_means = KMeans(init='k-means++', n_clusters=n_clusters,
                     n_init=10, random_state=random_state)
    labels = k_means.fit_predict(x)
    # now store for each data point the distance to the respectiv cluster center
    distance_to_clusters = k_means.transform(x)
    cluster_centers = k_means.cluster_centers_
    return labels, distance_to_clusters, cluster_centers


@ml_cache
def _compute_and_cluster_ice(data, model_eval_function, model,
                             y_coordinate, x_coordinate,
                             x_values, start_index=0, end_index=-1, scale='',
                             n_clusters=20, random_state=42, clustering_param=None):
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
        clustering_param (dict or None, optional): Default to None

    Returns:
        list of doubles: the x-values used to compute ICE
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
    if clustering_param is not None:
        labels, distance_to_clusters, cluster_centers = functional_clustering(
            ice, **clustering_param)
    return x_values, ice, labels, cluster_centers, distance_to_clusters


def compute_ice(ml_repo, x_values, data, model=None, model_label=None, model_version=RepoStore.LAST_VERSION,
                data_version=RepoStore.LAST_VERSION, y_coordinate=0, x_coordinate=0,
                start_index=0, end_index=-1, cache=False,
                clustering_param=None, scale=''):
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
        cache (bool, optional): If True, results will be cached. Defaults to False.
        clustering_param (dict or None, optional): Dictionary of parameters for method functional_clustering that is called if the parameter is not None and applies 
            functional clustering to the ICE curves.
        scale (str or int, optional): String defining the scaling for the functions before functional clustering is applied. Scaling is perfomred by
                                dividing the vector of the y-values of the ICE by the respective vector norm defined by scaling.
                                The scaling must be one of numpy's valid strings for linalg.norm's ord parameter. If string is empty, no scaling will be applied.
                                Defaults to ''. 

    Returns:
        ICE_Results: result object containing all relevant data (including functional clustering)
    """
    data_ = data
    if isinstance(data, str):
        data_ = ml_repo.get(data, data_version, full_object=True)
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
                                                                                                                               x_coordinate=x_coordinate, x_values=x_values,
                                                                                                                               start_index=start_index, end_index=end_index, cache=cache_,
                                                                                                                               clustering_param=clustering_param, scale=scale)
    result.x_coord_name = data_.x_coord_names[x_coordinate]
    result.y_coord_name = data_.y_coord_names[y_coordinate]
    result.data_name = data_.repo_info.name
    result.start_index = start_index
    result.model = model_.repo_info.name
    result.model_version = model_.repo_info.version
    result.data_version = data_.repo_info.version
    return result


def _compute_MMD_square(X, Y=None, metric='rbf', k_XX=None, k_YY=None, **kwds):
    """Compute the Maximum Mean Dicreapency (MMD) to measure how close two distributions are.

    Args:
        X (numpy nd-array): X points used to approximate first distribution. 
        Y (numpy nd-array): Y points used to approximate second distribution. 
        metric (str or callable, optional): The metric to use when calculating kernel between instances in a feature array.
            If metric is a string, it must be one of the metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS. 
            If metric is precomputed, X is assumed to be a kernel matrix. Alternatively, if metric is a callable function, 
            it is called on each pair of instances (rows) and the resulting value recorded. 
            The callable should take two arrays from X as input and return a value indicating the distance between them. 
            Currently, sklearn provides the following strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
                                                ‘laplacian’, ‘sigmoid’, ‘cosine’ 
        **kwds: optional keyword parameters
            Any further parameters are passed directly to the kernel function.

    Returns:
        float: The squared MDM computed by [description]
    """
    if not has_sklearn:
        raise Exception(
            'This method needs functionality form sklearn but sklearn is not installed.')
    m = float(X.shape[0])

    if k_XX is None:
        k_XX = pairwise_kernels(X, metric=metric, **kwds).sum()/(m**2)
    if Y is None:
        return k_XX
    n = float(Y.shape[0])
    if k_YY is None:
        k_YY = pairwise_kernels(Y, metric=metric, **kwds).sum()/(n**2)
    k_XY = pairwise_kernels(X, Y=Y, metric=metric, **kwds).sum()/(m*n)
    return k_XX + k_YY - 2.0*k_XY


def _get_MMD2_X_vs_X_percentile(X, Y,  percentile=0.1, scale=True, metric='rbf', **kwds):
    """This method compares the distribution of X and of values of X whose indices belong to the respective percentile of Y.

    Args:
        X (ndarray): X values for which the squared Maximum Mean Discrepancy to the subset of X described by indices defined as percentile of Y is computed.
        Y (ndarray): Y values used to compute indices belonging to the percentile.
        percentile (float, optional): The percentile used. Defaults to 0.1.
        scale (bool, optional): If True, the x-values are scaled to zero mean and unit variance. Defaults to True.
        metric (str or callable, optional): The metric to use when calculating kernel between instances in a feature array.
            If metric is a string, it must be one of the metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS. 
            If metric is precomputed, X is assumed to be a kernel matrix. Alternatively, if metric is a callable function, 
            it is called on each pair of instances (rows) and the resulting value recorded. 
            The callable should take two arrays from X as input and return a value indicating the distance between them. 
            Currently, sklearn provides the following strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
                                                ‘laplacian’, ‘sigmoid’, ‘cosine’ 
        **kwds: optional keyword parameters that are passed directly to the kernel function.

    Returns:
        ndarray: Numpy nd array containing the respective squared MMD between the diffrent distributions.
    """
    result = np.empty([X.shape[1], Y.shape[1]])
    if scale:
        _X = preprocessing.StandardScaler().fit_transform(X)
    else:
        _X = X
    k_xx = [0]*X.shape[1]
    for i in range(X.shape[1]):
        tmp = np.reshape(_X[:, i], (X.shape[0], 1, ))
        #print('tmp: ' + str(tmp))
        k_xx[i] = _compute_MMD_square(tmp, metric=metric, **kwds)

    for i in range(Y.shape[1]):
        sorted_indices = np.argsort(Y[:, i])
        i_start = int((1.0-percentile)*len(sorted_indices))
        indices = sorted_indices[i_start:]
        #print('x_percentile: ' + str(x_percentile))
        for j in range(X.shape[1]):
            x_percentile = np.reshape(_X[indices, j], (len(indices), 1, ))
            #k_yy = _compute_MMD_square(x_percentile, metric = metric, **kwds)
            x = np.reshape(_X[:, j], (len(sorted_indices), 1, ))
            #print('k_xx: ' + str(k_xx[i]) + '   k_yy: ' + str(k_yy))
            result[j, i] = _compute_MMD_square(
                x, x_percentile, k_XX=k_xx[j], metric=metric, **kwds)
    return result


@ml_cache
def _get_MMD2_X_vs_abs_ptw_error_percentile(X, Y_pred, x_coords=None, y_coords=None, percentile=0.1, scale=True, metric='rbf', **kwds):
    """This method compares the distribution of X and of values of the pointwise absolut errors between Y_target and Y_pred.

    Args:
        X (DataSet/RawData): X values for which the squared Maximum Mean Discrepancy to the subset of X described by indices defined as percentile of Y is computed.
        Y_pred (DataSet/RawData): predicted Y values
        percentile (float, optional): The percentile used. Defaults to 0.1.
        scale (bool, optional): If True, the x-values are scaled to zero mean and unit variance. Defaults to True.
        metric (str or callable, optional): The metric to use when calculating kernel between instances in a feature array.
            If metric is a string, it must be one of the metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS. 
            If metric is precomputed, X is assumed to be a kernel matrix. Alternatively, if metric is a callable function, 
            it is called on each pair of instances (rows) and the resulting value recorded. 
            The callable should take two arrays from X as input and return a value indicating the distance between them. 
            Currently, sklearn provides the following strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
                                                ‘laplacian’, ‘sigmoid’, ‘cosine’ 
        **kwds: optional keyword parameters that are passed directly to the kernel function.

    Returns:
        ndarray: Numpy nd array containing the respective squared MMD between the diffrent distributions.
    """
    if y_coords is None:
        ptw_error = np.abs(X.y_data - Y_pred.x_data)
    else:
        ptw_error = np.abs(X.y_data[:, y_coords] - Y_pred.x_data[:, y_coords])
    if x_coords is None:
        return _get_MMD2_X_vs_X_percentile(X.x_data, ptw_error, percentile=percentile, scale=scale, metric=metric, **kwds)
    else:
        return _get_MMD2_X_vs_X_percentile(X.x_data[:, x_coords], ptw_error, percentile=percentile, scale=scale, metric=metric, **kwds)


def _compute_MMD2(X, prototypes, metric='rbf', **kwds):
    kernel_matrix = pairwise_kernels(X, metric=metric, **kwds)
    m = float(len(prototypes))
    n = float(kernel_matrix.shape[0])
    #print('kernel_matrix.sum()/(n**2): ' + str(kernel_matrix.sum()/(n**2)))
    #print('kernel_matrix[np.ix_(prototypes, prototypes)].sum()/(m**2): ' + str(kernel_matrix[np.ix_(prototypes, prototypes)].sum()/(m**2)))
    #print('2.0*kernel_matrix[prototypes, :].sum()/(m*n): ' + str(2.0*kernel_matrix[prototypes, :].sum()/(m*n)))
    return kernel_matrix.sum()/(n**2) + kernel_matrix[np.ix_(prototypes, prototypes)].sum()/(m**2) - 2.0*kernel_matrix[prototypes, :].sum()/(m*n)


def _compute_prototypes(X, n_prototypes, n_criticisms, metric='rbf', witness_penalty=1.0, **kwds):
    """This methods computes for given datapoints prototypes and criticisms.

    This methods computes for given datapoints prototypes and criticisms, i.e. datapoints from th given set that are typical representatives (prototypes) and datapoints
    that are not well representatives (criticisms). Here, a simple greedy algorithm using MDM2 is used to compute the prototypes and a witness function together
    with som simple penalty are used to compute the criticisms (see e.g. C. Molnar, Interpretable Machine Learning).

    Args:
        X (numpy matrix): Set of datapoints (each row representing one datapoint)
        n_prototypes (int): Number of prototypes.
        n_criticisms (int): Number of criticisms.
        metric (str or callable, optional): The metric to use when calculating kernel between instances in a feature array.
            If metric is a string, it must be one of the metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS. 
            If metric is precomputed, X is assumed to be a kernel matrix. Alternatively, if metric is a callable function, 
            it is called on each pair of instances (rows) and the resulting value recorded. 
            The callable should take two arrays from X as input and return a value indicating the distance between them. 
            Currently, sklearn provides the following strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
                                                ‘laplacian’, ‘sigmoid’, ‘cosine’ 
        witness_penalty (float): Penalty parameter to include some penalty to avoid to close criticisms. 
        **kwds: optional keyword parameters
            Any further parameters are passed directly to the kernel function.

    Raises:
        Exception: If sklearn is not installed

    Returns:
        list of int: List of indices defining the datapoints which are the resulting prototypes.
        list of int: List of indices defining the datapoints which are the resulting criticisms.
    """
    if not has_sklearn:
        raise Exception(
            'This method needs functionality form sklearn but sklearn is not installed.')
    kernel_matrix = pairwise_kernels(X, metric=metric, **kwds)
    prototypes = []
    n = float(kernel_matrix.shape[0])
    if n_prototypes >= n:
        raise Exception(
            'Number of prototypes must be less then number of datapoints.')
    # To compute  prototypes we minimize the Maximum Mean Discrepancy (MDM), .. math::
    #     MDM^2 = \frac{1}{m^2}\sum_{i=1}^m \sum_{j=1}^m k(z_i,z_j) - \frac{2}{mn}\sum_{i=1}^m \sum_{j=1}^n k(z_i,x_j) + \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n k(x_i,x_j)
    #     We are doing this using a greedy search, looking for the next prototype by simply computing which next datapoint reduces the current MDM most.
    #     For this we compute simply
    #     the impact on the MDM if a new point x is used as a prototype. The impact is computed by .. math::
    #     \frac{2}{(m+1)^2}(k(x,x) + \sum{i=1}^mk(z_i,x)) -  \frac{2}{(m+1)n}\sum_{j=1}^n k(x,x_j)for i in range(n_prototypes):
    #max_impact  = 1.0e8
    for i in range(n_prototypes):
        m = float(len(prototypes))
        max_impact = 1.0e8
        new_prototype = None
        for candidate in range(kernel_matrix.shape[0]):
            if candidate not in prototypes:
                impact = kernel_matrix[candidate][prototypes].sum(
                )/((m+1)**2) - kernel_matrix[candidate, :].sum()/((m+1)*n)
                if impact < max_impact:
                    new_prototype = candidate
                    max_impact = impact
        if new_prototype is None:
            print(str(prototypes))
            raise Exception('Cannot find a new prototype.')
        prototypes.append(new_prototype)

    m = float(len(prototypes))
    n = float(kernel_matrix.shape[0])
    criticisms = []
    for i in range(n_criticisms):
        m_criticisms = float(len(criticisms))
        max_witness = -1.0e8
        new_criticism = None
        for candidate in range(kernel_matrix.shape[0]):
            if candidate not in criticisms:
                witness = kernel_matrix[candidate].sum(
                )/n - kernel_matrix[candidate][prototypes].sum()/m
                if m_criticisms > 0:
                    regularizer = kernel_matrix[candidate][criticisms].max()
                else:
                    regularizer = 0
                cost = abs(witness) - witness_penalty*regularizer
                if cost > max_witness:
                    max_witness = cost
                    new_criticism = candidate
        if new_criticism is None:
            raise Exception('Cannot find a new criticism.')
        criticisms.append(new_criticism)

    return prototypes, criticisms


def generate_prototypes(ml_repo, data, n_prototypes, n_criticisms, data_version=RepoStore.LAST_VERSION,
                        use_x=True, data_start_index=0, data_end_index=-1, metric='rbf', witness_penalty=1.0,
                        **kwds):
    """This methods computes for a given test/training dataset prototypes and criticisms and adds them as separate test data sets to the repository. 

    This methods computes for given test/training dataset prototypes and criticisms, i.e. datapoints from th given set that are typical representatives (prototypes) 
    and datapoints that are not well representatives (criticisms). Here, a simple greedy algorithm using MDM2 is used to compute the prototypes and a witness 
    function together with some simple penalty are used to compute the criticisms (see e.g. C. Molnar, Interpretable Machine Learning).

    Args:
        ml_repo (MLRepo): The repository used to retrieve data and store prototypes/criticisms.
        data (str): Name of data used for computation.
        n_prototypes (int): Number of prototypes.
        n_criticisms (int): Number of criticisms.
        data_version (str): Version of data to be used. Defaults to RepoStore.LAST_VERSION.
        use_x (bool): Flags that determine if prototypes are computed w.r.t. x or y coordinates. Defaults to True.
        data_start_index (int): Startindex of data used.
        data_end_index (int): Endindex of data used.
        metric (str or callable, optional): The metric to use when calculating kernel between instances in a feature array.
            If metric is a string, it must be one of the metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS. 
            If metric is precomputed, X is assumed to be a kernel matrix. Alternatively, if metric is a callable function, 
            it is called on each pair of instances (rows) and the resulting value recorded. 
            The callable should take two arrays from X as input and return a value indicating the distance between them. 
            Currently, sklearn provides the following strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
                                                ‘laplacian’, ‘sigmoid’, ‘cosine’ 
        witness_penalty (float): Penalty parameter to include some penalty to avoid to close criticisms. 
        **kwds: optional keyword parameters
            Any further parameters are passed directly to the kernel function.

    Raises:
        Exception: If sklearn is not installed

    Returns:
        list of int: List of indices defining the datapoints which are the resulting prototypes.
        list of int: List of indices defining the datapoints which are the resulting criticisms.
    """
    if isinstance(data, str):
        data = ml_repo.get(data, data_version, full_object=True)
    if use_x:
        d = data.x_data[data_start_index:data_end_index]
        if d is None:
            raise Exception('No x-data defined.')
    else:
        d = data.y_data[data_start_index:data_end_index]
        if d is None:
            raise Exception('No y-data defined.')

    std_scale = preprocessing.StandardScaler().fit(d)
    d = std_scale.transform(d)
    prototypes, criticisms = _compute_prototypes(
        d, n_prototypes, n_criticisms, metric=metric, witness_penalty=witness_penalty, **kwds)

    result_name = data.repo_info.name+'_'+'prototypes'
    if data.y_data is None:
        result = RawData(data.x_data[data_start_index:data_end_index][prototypes], data.x_coord_names, repo_info={RepoInfoKey.NAME: result_name,
                                                                                                                  RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA,
                                                                                                                  RepoInfoKey.MODIFICATION_INFO: {
                                                                                                                      data.repo_info.name: data.repo_info.version}
                                                                                                                  })
    else:
        result = RawData(data.x_data[data_start_index:data_end_index][prototypes], data.x_coord_names,
                         data.y_data[data_start_index:data_end_index][prototypes], data.y_coord_names,
                         repo_info={RepoInfoKey.NAME: result_name,
                                    RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA,
                                    RepoInfoKey.MODIFICATION_INFO: {
                                        data.repo_info.name: data.repo_info.version}
                                    })
    ml_repo.add(result)

    result_name = data.repo_info.name+'_'+'criticisms'
    if data.y_data is None:
        result = RawData(data.x_data[data_start_index:data_end_index][criticisms], data.x_coord_names, repo_info={RepoInfoKey.NAME: result_name,
                                                                                                                  RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA,
                                                                                                                  RepoInfoKey.MODIFICATION_INFO: {
                                                                                                                      data.repo_info.name: data.repo_info.version}
                                                                                                                  })
    else:
        result = RawData(data.x_data[data_start_index:data_end_index][criticisms], data.x_coord_names,
                         data.y_data[data_start_index:data_end_index][criticisms], data.y_coord_names,
                         repo_info={RepoInfoKey.NAME: result_name,
                                    RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA,
                                    RepoInfoKey.MODIFICATION_INFO: {
                                        data.repo_info.name: data.repo_info.version}
                                    })
    ml_repo.add(result)
