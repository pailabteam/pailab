import logging
from pailab.repo import MLObjectType, MLRepo, NamingConventions  # pylint: disable=E0401,E0611
from pailab.repo_objects import RepoInfoKey  # pylint: disable=E0401
from pailab.repo_store import RepoStore  # pylint: disable=E0401
logger = logging.getLogger(__name__)


class _LabelChecker:
    """Checks wether a certain object and object version has a label
    """

    def __init__(self, ml_repo):
        self._labels = {}
        labels = ml_repo.get_names(MLObjectType.LABEL)
        for label in labels:
            l = ml_repo.get(label)
            if l.name in self._labels:
                self._labels[l.name][str(
                    l.version)] = l.repo_info[RepoInfoKey.NAME]
            else:
                self._labels[l.name] = {
                    str(l.version): l.repo_info[RepoInfoKey.NAME]}

    def get_label(self, name, version):
        if name in self._labels.keys():
            if str(version) in self._labels[name].keys():
                return self._labels[name][str(version)]
        return None


def get_measure_by_model_parameter(ml_repo, measure_names, param_name, data_versions=None):
    """Returns for a (list of) measure(s) the measures and corresponding param values for a certain parameter 

    Args:
        ml_repo (MLRepo): the ml repo
        measure_names (str, list(str)): string or list of strings of measure names  
        param_name (str): name of parameter
        data_versions (version number, optional): Defaults to None. If not None, only values on measures on dta with this version number are used

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
    label_checker = _LabelChecker(ml_repo)

    if isinstance(measure_names, str):
        measure_names = [measure_names]

    result_all = {}
    for measure_name in measure_names:
        data = str(NamingConventions.Data(NamingConventions.EvalData(
            NamingConventions.Measure(measure_name))))
        measures = ml_repo.get(measure_name, version=(
            RepoStore.FIRST_VERSION, RepoStore.LAST_VERSION), modifier_versions={data: data_versions})
        model_name = NamingConventions.CalibratedModel(
            NamingConventions.Measure(measure_name)
        )

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
            label = label_checker.get_label(
                model_name, x.repo_info[RepoInfoKey.MODIFICATION_INFO][model_name])
            if label is not None:
                info['model_label'] = label
            result.append(info)
        result_all[measure_name] = result
    return result_all
