# -*- coding: utf-8 -*-
"""This module contains functions for functional clustering. 
"""
import logging
import numpy as np
from collections import defaultdict
has_sklearn = True
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering

except ImportError:
    import warnings
    warnings.warn(
        'No sklearn installed, some functions of this submodule may not be usable.')
    has_sklearn = False


logger = logging.getLogger(__name__)


def _transform_standard_grid(x_coords, f, gridwidth):
    x_coords_uniform = np.arange(0, 1.0, gridwidth)
    if len(f.shape) > 2:
        raise Exception(
            'f must either be a one dimensional array (representing only one function) or a two dimensional array where each row represents one function.')
    if len(x_coords.shape) > 2:
        raise Exception(
            'x_coords must either be a one dimensional array (representing a singl grid or a two dimensional array where each row represents one grid.')
    f_transformed = np.empty(shape=f.shape, )
    if len(f.shape) == 1:
        x_coords_inter = np.min(
            x_coords) + x_coords_uniform*(np.max(x_coords)-np.min(x_coords))
        f_transformed = np.interp(x_coords_uniform, x_coords, f)
    else:
        for i in range(f.shape[0]):
            if len(x_coords.shape) == 1:
                x_coords_inter = np.min(
                    x_coords) + x_coords_uniform*(np.max(x_coords)-np.min(x_coords))
            else:
                x_coords_inter = np.min(
                    x_coords[i]) + x_coords_uniform*(np.max(x_coords[i])-np.min(x_coords[i]))
            f_transformed[i, :] = np.interp(x_coords_inter, x_coords, f[i, :])
    return f_transformed


def _get_landmarks(f, landmark_type, rel_tol=0.001):
    """Return the landmarks (i.e. minimums, maximums) of the function(s)

    This method returns the landmark according to their landmark type
    Args:
        f (1-D or 2D array): Either a 1-D array representing a function via the function values at certain gridpoints (implicitely assumed that the gridpoints are uniformly distributed). 
                            2-D array to represent a set of functions where each row consists of function values of a different function.
        landmark (str): Definition of landmark type. 'MIN': Local minima are returned, 'MAX': Local maxima are returned

    Returns:
        List of indices of the function values representing the specified landmark
    """
    def get_local_minimum(f):
        abs_tol = rel_tol*(np.max(f)-np.min(f))
        return np.nonzero((f[1:-1]+abs_tol < f[0:-2]) & (f[1:-1]+abs_tol < f[2:]))[0]

    def get_local_maximum(f):
        abs_tol = rel_tol*(np.max(f)-np.min(f))
        return np.nonzero((f[1:-1]+abs_tol > f[0:-2]) & (f[1:-1]+abs_tol > f[2:]))[0]

    if len(f.shape) == 1:
        f_ = np.reshape(f, (1, -1,))
    else:
        f_ = f
    result = []
    if landmark_type == 'MIN':
        for i in range(f.shape[0]):
            result.append(get_local_minimum(f_[i, :]))
    else:
        for i in range(f.shape[0]):
            result.append(get_local_maximum(f_[i, :]))
    return result


def _compute_similarity_matrix(f):
    result = np.ones((f.shape[0], f.shape[0],))
    dx_f = f[:, 1:]-f[:, :-1]
    d = np.sqrt(np.sum(dx_f*dx_f, axis=1))

    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):
            if d[i] < 0.00001 or d[j] < 0.00001:
                if d[j] > 0.0001 or d[i] > 0.0001:
                    # no similarity: Constant and non-constant
                    result[i, j] = 0.0
                else:
                    # very similar since both seem to be constants
                    result[i, j] = 0.98
            else:
                result[i, j] = np.minimum(
                    np.dot(dx_f[i, :], dx_f[j, :])/(d[i]*d[j]), 1.0)
                result[j, i] = result[i, j]
    return result


def _compute_similarity(dx_f1, dx_f2, d1, d2):
    if d1 < 0.00001 or d2 < 0.00001:
        if d1 > 0.0001 or d2 > 0.0001:
            # no similarity: Constant and non-constant
            return 0.0
        else:
            # very similar since both seem to be constants
            return 0.98
    return np.minimum(np.dot(dx_f1, dx_f2)/(d1*d2), 1.0)


def agglomerative(f,
                  n_clusters=10,
                  min_cluster_size=10,
                  random_state=42,
                  gridwidth=0.01,
                  rel_tol=0.001,
                  **sklearnArg):
    result = np.empty((f.shape[0],))

    local_minimum = defaultdict(list)
    tmp = _get_landmarks(f, 'MIN', rel_tol)
    for i in range(f.shape[0]):
        local_minimum[tmp[i].shape[0]].append(i)
    cluster_offset = 0
    logger.debug('Found ' + str(len(local_minimum)) +
                 ' local minima configuration(s).')
    c_centers = []
    for k, v in local_minimum.items():
        logger.debug('Start clustering for ' + str(len(v)) + ' functions with ' +
                     str(k) + ' local minima.')
        f_sub = f[v, :]
        similarity_matrix = _compute_similarity_matrix(f_sub)
        distance_matrix = np.sqrt(1.0-similarity_matrix)

        if len(v) > n_clusters:
            clustering = AgglomerativeClustering(
                affinity='precomputed', linkage='average',
                n_clusters=n_clusters, **sklearnArg)
            clusters = clustering.fit_predict(distance_matrix)
        else:
            clusters = np.zeros(shape=(len(v),))
        for i in range(len(v)):
            result[v[i]] = clusters[i] + cluster_offset
        cluster_offset += np.max(clusters)+1
        logger.debug('Finished clustering for functions with ' +
                     str(k) + ' local minima.')

        # determine cluster centroids and distances to centroid
        cluster_centers = np.empty((int(np.max(clusters))+1,), dtype=int)
        for i in range(cluster_centers.shape[0]):
            ind = np.nonzero(clusters == i)[0]
            dist_sub = distance_matrix[np.ix_(ind, ind)]
            # take the first index with smallest sum of distances
            cluster_centers[i] = v[ind[np.argmin(np.sum(dist_sub, axis=1))]]
        c_centers.append(cluster_centers)
    cluster_centers = np.concatenate(c_centers)

    cluster_distance = np.empty((f.shape[0], cluster_centers.shape[0],))
    dx_f = f[:, 1:]-f[:, :-1]
    d = np.sqrt(np.sum(dx_f*dx_f, axis=1))
    for i in range(f.shape[0]):
        for j in range(cluster_centers.shape[0]):
            cluster_distance[i, j] = _compute_similarity(
                dx_f[i, :], dx_f[j, :], d[i], d[j])
    return result, cluster_distance, f[cluster_centers, :]


def kmeans(x, n_clusters=20, random_state=42, **sklearnArg):
    """Given a set of different 1D functions (represented in matrix form where each row contains the function values on a given grid), this
        method clusters those functions and tries to find typical function structures.

    Args:
        x (numpy matrix): Matrix containing in each row the function values at a datapoint.
        n_clusters (int, optional): Number of clusters for the functional clustering of the ICE curves. Defaults to 0.
        random_state (int, optional): [description]. Defaults to 42.

    Returns:
        numpy vector: Vector where each value defines the corresponding cluster of the respective function (x[i] is the cluster the i-th function belongs to).
        numpy matrix: Contains in each row the distance to each cluster for the respective function.
        numpy matrix: Contains in each row a cluster centroid.
    """
    if not has_sklearn:
        raise Exception(
            'Cannot apply functional clustering: No sklearn installed. Please either install sklearn or set n_clusters to zero.')
    k_means = KMeans(init='k-means++', n_clusters=n_clusters,
                     n_init=10, random_state=random_state, **sklearnArg)
    labels = k_means.fit_predict(x)
    # now store for each data point the distance to the respectiv cluster center
    distance_to_clusters = k_means.transform(x)
    cluster_centers = k_means.cluster_centers_
    return labels, distance_to_clusters, cluster_centers


def fit_predict(f,
                method='euclidean',
                x_coords=None,
                n_clusters=10,
                min_cluster_size=10,
                random_state=42,
                gridwidth=0.01,
                rel_tol=0.001,
                **sklearnArg):
    """Given a set of different 1D functions (represented in matrix form where each row contains the function values on a given grid), this
        method clusters those functions and tries to find typical function structures.

    Args:
        x (numpy matrix): Matrix containing in each row the function values at a datapoint.
        method (str): Method to use for clustering: 
            - kmeans: k-means clustering where vector of function values is used to compare functions (see :meth:`pailab.tools.functional_clustering.kmeans`)
            - agglomerative: Agglomerative clusering is used (see :meth:`pailab.tools.functional_clustering.agglomerative`)
        n_clusters (int, optional): Number of clusters for the functional clustering of the ICE curves. Defaults to 0.
        random_state (int, optional): [description]. Defaults to 42.

    Returns:
        numpy vector: Vector where each value defines the corresponding cluster of the respective function (x[i] is the cluster the i-th function belongs to).
        numpy matrix: Contains in each row th distance to each cluster for the respective function.
        numpy matrix: Contains in each row a cluster centroid.
    """
    logger.info('Start functional clustering with n_clusters=' +
                str(n_clusters) + ' and method ' + method)
    # apply transformation to uniform [0,1] grid to make functions comparable
    if x_coords is not None:
        logger.debug(
            'Transform functions to uniform [0,1] grid width gridwidth ' + str(gridwidth))
        f_transformed = _transform_standard_grid(x_coords, f, gridwidth)
    else:
        f_transformed = f
    if method == 'euclidean':
        return kmeans(f_transformed, n_clusters=n_clusters, random_state=random_state, **sklearnArg)
    elif method == 'agglomerative':
        return agglomerative(f_transformed, n_clusters=n_clusters, random_state=random_state, rel_tol=rel_tol, **sklearnArg)
    else:
        raise ValueError('Unknown method ' + method)
    logger.info('Finished functional clustering')
