import numpy as np
from pailab.ml_repo.repo_objects import RepoObject, RawData
from pailab.tools.tools import ml_cache
from pailab.ml_repo.repo_store import RepoStore  # pylint: disable=E0401

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
    if isinstance(x_coordinate, str):
        x_coordinate = data.x_coord_names.index(x_coordinate)
    if isinstance(y_coordinate, str):
        y_coordinate = data.y_coord_names.index(y_coordinate)
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



def compute_ice(ml_repo, x_values, data, model=None, model_label=None, model_version=RepoStore.LAST_VERSION,
                data_version=RepoStore.LAST_VERSION, y_coordinate=0, x_coordinate = 0,
                start_index=0, end_index=-1, n_steps=20, scale='', cache = False):
    """Compute the curve for an individual conditional expectation plot at different locations.

    This method computes at differet x-points the values of the model w.r.t. a projection along a defined x-coordinate (projection of the model). 
    The x-values used within this projection must be given by the user. It returns a matrix where each row i contains the projected values along the
    x-coordinate for the i-th datapoint.

    Args:
        ml_repo ([type]): [description]
        x_values ([type]): [description]
        data ([type]): [description]
        model ([type], optional): [description]. Defaults to None.
        model_label ([type], optional): [description]. Defaults to None.
        model_version ([type], optional): [description]. Defaults to RepoStore.LAST_VERSION.
        data_version ([type], optional): [description]. Defaults to RepoStore.LAST_VERSION.
        y_coordinate (int, optional): [description]. Defaults to 0.
        start_index (int, optional): Index defining the first point of the dataset for which an ice will be computed. Defaults to 0.
        end_index (int, optional): Index defining the last point of the dataset for which an ice will be computed. Defaults to -1.
        scale (non-zero int, inf, -inf, ''): If specified, the scaling parameter is used to scale each projection by the respective numpy.linalg.norm.
            Scaling may be useful to compare ice at different data points.
        cache (bool): If True, the ice computation will be cached in the ml_reo
    """
    data_ = data
    if isinstance(data, str):
        data_ = ml_repo.get(data, data_version, full_object = True)
    model_, model_eval_f = _get_model_eval(
        model, model_version, ml_repo, model_label)
    if cache:
        x = _compute_ice(data_, model_eval_f, model_,  y_coordinate,
                     x_coordinate=x_coordinate, x_values=x_values, scale=scale, 
                     start_index = start_index, end_index = end_index, cache=ml_repo)
    else:    
        x = _compute_ice(data_, model_eval_f, model_,  y_coordinate,
                        x_coordinate=x_coordinate, x_values=x_values, 
                        start_index = start_index, end_index = end_index,
                        scale=scale)
    return x_values, x


def analyze_ice(model,  data, x_values, x_coordinate,
                version=RepoStore.LAST_VERSION, data_version=RepoStore.LAST_VERSION,
                y_coordinate=None, start_index=0, end_index=100, full_object=True, n_steps=20,
                n_clusters=20, scale=True, random_state=42, percentile=90):
    """[summary]

    Args:
        model ([type]): [description]
        data ([type]): [description]
        x_values ([type]): [description]
        x_coordinate ([type]): [description]
        version ([type], optional): [description]. Defaults to RepoStore.LAST_VERSION.
        data_version ([type], optional): [description]. Defaults to RepoStore.LAST_VERSION.
        y_coordinate ([type], optional): [description]. Defaults to None.
        start_index (int, optional): [description]. Defaults to 0.
        end_index (int, optional): [description]. Defaults to 100.
        full_object (bool, optional): [description]. Defaults to True.
        n_steps (int, optional): [description]. Defaults to 20.
        n_clusters (int, optional): [description]. Defaults to 20.
        scale (bool, optional): [description]. Defaults to True.
        random_state (int, optional): [description]. Defaults to 42.
        percentile (int, optional): [description]. Defaults to 90.
    """
    if y_coordinate is None:
        y_coordinate = 0
    if isinstance(y_coordinate, str):
        raise NotImplementedError()
    if isinstance(x_coordinate, str):
        raise NotImplementedError()

    param = {
        'y_coordinate': y_coordinate, 'start_index': start_index, 'end_index': end_index,
        'n_steps': n_steps,
        'x_values': x_values,
        'x_coodrinate': x_coordinate,
        'n_clusters': n_clusters,
        'scale': scale, 'random_state': random_state,
        'percentile': percentile}

    param_hash = hashlib.md5(json.dumps(
        param, sort_keys=True).encode('utf-8')).hexdigest()

    # check if results of analysis are already stored in the repo
    model_ = self._ml_repo.get(model, version, full_object=False)
    data_ = self._ml_repo.get(data, data_version, full_object=False)
    result_name = 'analyzer_ice_' + model_.repo_info.name + '_' + data_.repo_info.name
    result = self._ml_repo.get(result_name, None,
                               modifier_versions={model_.repo_info.name: model_.repo_info.version,
                                                  data_.repo_info.name: data_.repo_info.version,
                                                  'param_hash': param_hash},
                               throw_error_not_exist=False, full_object=full_object)
    if result != []:
        if isinstance(result, list):
            return result[0]
        else:
            return result

    model_definition_name = model.split('/')[0]
    model = self._ml_repo.get(model, version, full_object=True)
    model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO][model_definition_name]
    model_definition = self._ml_repo.get(
        model_definition_name, model_def_version)
    data = self._ml_repo.get(data, data_version, full_object=True)

    eval_func = self._ml_repo.get(
        model_definition.eval_function, RepoStore.LAST_VERSION)

    data_ = data.x_data[start_index:end_index, :]

    x = ModelAnalyzer._compute_ice(data_, eval_func, model,  y_coordinate,
                                   x_coordinate=x_coordinate, x_values=x_values, scale=scale)
    big_obj = {}
    big_obj['ice'] = x
    big_obj['x_values'] = np.array(x_values)
    # now apply a clustering algorithm to search for good representations of all ice results
    k_means = KMeans(init='k-means++', n_clusters=n_clusters,
                     n_init=10, random_state=42)
    labels = k_means.fit_predict(x)
    big_obj['labels'] = labels
    big_obj['cluster_centers'] = k_means.cluster_centers_
    # now store for each data point the distance to the respectiv cluster center
    tmp = k_means.transform(x)
    distance_to_center = np.empty((x.shape[0],))

    for i in range(x.shape[0]):
        distance_to_center[i] = tmp[i, labels[i]]
    big_obj['distance_to_center'] = distance_to_center
    #perc = np.percentile(distance_to_center, percentile)
    #percentile_ice = np.extract(distance_to_center > perc, )
    # for i in range(distance_to_center.shape[0]):
    #    if distance_to_center[i] > perc:
    #        percentile_ice.append(distance_to_center[i])
    #big_obj['percentiles'] = percentile_ice
    result = ModelAnalyzer._create_result(result_name, model, data, param, {
                                          'param': param, 'data': data.repo_info.name, 'model': model.repo_info.name}, big_obj)
    self._ml_repo.add(result)
    return result
