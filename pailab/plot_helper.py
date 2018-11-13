import logging
from pailab.repo import MLObjectType, MLRepo, NamingConventions  # pylint: disable=E0401,E0611
from pailab.repo_objects import RepoInfoKey  # pylint: disable=E0401
from pailab.repo_store import RepoStore  # pylint: disable=E0401
logger = logging.getLogger('repo.plot')


def get_measure_by_model_parameter(ml_repo, measure_names, param_name, data_versions=None):
    """Returns for a (list of) measure(s) the measures and corresponding param values for a certain parameter 

    Args:
        ml_repo (MLRepo): the ml repo
        measure_names (str, list(str)): string or list of strings of measure names  
        param_name (str): name of parameter
        data_versions (version number, optional): Defaults to None. If not None, only values on measures on dta with this version number are used

    Raises:
        NotImplementedError: raises exception if data_versiosn is not None

    Returns:
        [dict]: dictionary of measure name to list of dictionaries containing the result, i.e. 
        model_version: version of model parameter
        param_version: version of the parameter of this data point
        param_name: the parameter value
        data_version: version of the underlying data
        train_data_version: version number of trainin data used to calibrate the model leading to this measure
        measure_version: version of measure
        value: measure value
    """

    if isinstance(measure_names, str):
        measure_names = [measure_names]
    if data_versions is not None:
        raise NotImplementedError()
    result_all = {}
    for measure_name in measure_names:
        measures = ml_repo.get(measure_name, version=(
            RepoStore.FIRST_VERSION, RepoStore.LAST_VERSION))
        model_name = NamingConventions.CalibratedModel(
            NamingConventions.Measure(measure_name)
        )
        data = str(NamingConventions.Data(NamingConventions.EvalData(
            NamingConventions.Measure(measure_name))))
        model_param_name = str(NamingConventions.ModelParam(model_name))
        train_data = ml_repo.get_names(MLObjectType.TRAINING_DATA)[0]
        model_name = str(model_name)
        # eval_name

        result = []
        for x in measures:
            p = ml_repo.get(
                model_param_name, version=x.repo_info[RepoInfoKey.MODIFICATION_INFO][model_param_name])
            param_value = p.get_params()[param_name]
            info = {'model_version': x.repo_info[RepoInfoKey.MODIFICATION_INFO][model_name],
                    param_name: param_value, 'param_version': p.repo_info[RepoInfoKey.VERSION],
                    'data_version': x.repo_info[RepoInfoKey.MODIFICATION_INFO][data],
                    'train_data_version': x.repo_info[RepoInfoKey.MODIFICATION_INFO][train_data],
                    'measure_version': x.repo_info[RepoInfoKey.VERSION], 'value': x.value}
            result.append(info)
        result_all[measure_name] = result
    return result_all
