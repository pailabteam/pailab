# -*- coding: utf-8 -*-
"""This module contains functions for functional clustering. 
"""
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import numpy as np


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


def functional_clustering(x_coords, f,
                          n_clusters=10,
                          min_cluster_size=10,
                          random_state=42,
                          gridwidth=0.01,
                          rel_tol=0.001,
                          **sklearnArg):
    result = np.empty((f.shape[0],))
    # apply transformation to uniform [0,1] grid to make functions comparable
    f_transformed = _transform_standard_grid(x_coords, f, gridwidth)

    local_minimum = defaultdict(list)
    tmp = _get_landmarks(f_transformed, 'MIN', rel_tol)
    for i in range(f.shape[0]):
        local_minimum[tmp[i].shape[0]].append(i)
    cluster_offset = 0

    for k, v in local_minimum.items():
        f_transformed_sub = f_transformed[v, :]
        similarity_matrix = _compute_similarity_matrix(f_transformed_sub)
        # return similarity_matrix
        # since clustering algorithm needs distance matrix, we first transform similarity to distance
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

    return result
