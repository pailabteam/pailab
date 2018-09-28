import logging
from repo.repo import MLObjectType, MLRepo, NamingConventions
from repo.repo_objects import RepoInfoKey
from repo.repo_store import RepoStore
logger = logging.getLogger('repo.plot')


def get_measure_by_model_parameter(ml_repo, measure_name, param_name):

    measures = ml_repo._get(measure_name, version=(
        RepoStore.FIRST_VERSION, RepoStore.LAST_VERSION))
    model_name = NamingConventions.CalibratedModel(
        NamingConventions.Measure(measure_name)
    )
    data = str(NamingConventions.Data(NamingConventions.EvalData(
        NamingConventions.Measure(measure_name))))
    model_param_name = str(NamingConventions.ModelParam(model_name))
    model_name = str(model_name)
    # eval_name

    result = []

    for x in measures:
        p = ml_repo._get(
            model_param_name, version=x.repo_info[RepoInfoKey.MODIFICATION_INFO][model_param_name])
        param_value = p.get_params()[param_name]
        info = {'model_version': x.repo_info[RepoInfoKey.MODIFICATION_INFO][model_name],
                param_name: param_value, 'param_version': p.repo_info[RepoInfoKey.VERSION],
                'data_version': x.repo_info[RepoInfoKey.MODIFICATION_INFO][data],
                'measure_version': x.repo_info[RepoInfoKey.VERSION], 'value': x.value}
        result.append(info)
    return result
