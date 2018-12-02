"""This module creaes the data for plotting used by plot.py. The different methods return the respective data and information for plotting 
such as axes labels, titles, additional info which may be shown on hover in the plot encapsulated in a dictionary.
"""

import logging
from pailab.repo import MLObjectType, MLRepo, NamingConventions  # pylint: disable=E0401,E0611
from pailab.repo_objects import RepoInfoKey  # pylint: disable=E0401
from pailab.repo_store import RepoStore, LAST_VERSION, FIRST_VERSION  # pylint: disable=E0401

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
                self._labels[l.name][
                    l.version] = l.repo_info[RepoInfoKey.NAME]
            else:
                self._labels[l.name] = {
                    l.version: l.repo_info[RepoInfoKey.NAME]}

    def get_label(self, name, version):
        _name = name
        if len(_name.split('/')) == 1:
            _name = name + '/model'
        if name in self._labels.keys():
            if version in self._labels[name].keys():
                return self._labels[name][version]
        return None


def get_measure_by_model_parameter(ml_repo, measure_names, param_name, data_versions=LAST_VERSION):
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


def get_pointwise_model_errors(ml_repo, models, data, coord_name=None, data_version=LAST_VERSION, x_coord_name=None):
    label_checker = _LabelChecker(ml_repo)

    def get_model_dict(ml_repo, models, label_checker):
        # first determine models (including their respective version) to be plotted
        _models = {}  # dictionary containing model names together with model versions to be plotted
        if models is None:  # if models is None, use all labeled models
            logging.info(
                'No model specified, use all labeled models.')
            for k, v in label_checker._labels.items():
                if v.name in _models.keys():
                    _models[v.name].append(v.version)
                else:
                    _models[v.name] = [v.version]
        if isinstance(models, list):
            for m in models:
                if m in label_checker._labels.keys():
                    label = label_checker._labels[m]
                    _models[label.name] = label.version
                else:
                    _models[str(NamingConventions.CalibratedModel(
                        model=m))] = LAST_VERSION
        # if just a string is given, use all labels on this model and the latest model
        if isinstance(models, str):
            _models[models] = [LAST_VERSION]
            if models in label_checker._labels.keys():
                for k in label_checker._labels[models].keys():
                    _models[models].append(k)
            logging.info(logging.info(
                'Only a model name given, using last version and ' + str(len(_models[models])-1) + ' labeled versions of this model.'))
        if isinstance(models, dict):
            _models = models
        return _models

    _data = data
    if isinstance(_data, str):
        _data = [data]

    _models = get_model_dict(ml_repo, models, label_checker)
    ref_data = ml_repo.get(_data[0], version=data_version, full_object=False)
    coord = 0
    if coord_name is None:
        coord_name = ref_data.y_coord_names[0]

    coord = ref_data.y_coord_names.index(coord_name)
    result = {'title': 'pointwise error (' + coord_name + ')', 'data': {}}
    if x_coord_name is None:
        result['x0_name'] = 'model-target  [' + coord_name + ']'
    else:
        result['x0_name'] = x_coord_name
        result['x1_name'] = 'model-target  [' + coord_name + ']'

    for d in _data:
        ref_data = ml_repo.get(d, version=data_version, full_object=True)

        for m_name, m_versions in _models.items():
            tmp = m_name.split('/')[0]
            eval_data_name = str(
                NamingConventions.EvalData(data=d, model=tmp))
            logging.info('Retrieving eval data for model ' + tmp + ', versions ' +
                         str(m_versions) + ' and data ' + d + ', versions ' + str(data_version))
            eval_data = ml_repo.get(
                eval_data_name, version=(FIRST_VERSION, LAST_VERSION), modifier_versions={m_name: m_versions, d: data_version}, full_object=True)
            if not isinstance(eval_data, list):
                eval_data = [eval_data]
            for eval_d in eval_data:
                error = ref_data.y_data[:, coord] - eval_d.x_data[:, coord]
                tmp = {}
                if x_coord_name is None:
                    tmp['x0'] = error
                else:
                    tmp['x1'] = error
                    tmp['x0_name'] = x_coord_name
                    tmp['x0'] = ref_data.x_data[:,
                                                ref_data.x_coord_names.index(x_coord_name)]
                tmp['info'] = {d: str(data_version),
                               m_name: str(eval_d.repo_info[RepoInfoKey.MODIFICATION_INFO][m_name])}

                model_label = label_checker.get_label(
                    m_name, eval_d.repo_info[RepoInfoKey.MODIFICATION_INFO][m_name])
                if model_label is not None:
                    tmp['label'] = model_label

                result['data'][eval_data_name + ': ' +
                               str(eval_d.repo_info[RepoInfoKey.VERSION])] = tmp
    return result


def get_data(ml_repo, data, x0_coord_name, x1_coord_name=None):

    data_dict = data
    result = {'title': 'data distribution', 'data': {}}
    result['x0_name'] = x0_coord_name
    result['data'] = {}
    if x1_coord_name is not None:
        result['x1_name'] = x1_coord_name

    if isinstance(data, str):
        data_dict[data] = LAST_VERSION

    x_coord = None
    y_coord = None
    for k, v in data_dict.items():
        ref_data = ml_repo.get(k, version=v, full_object=True)

        if not isinstance(ref_data, list):
            ref_data = [ref_data]
        for d in ref_data:
            if x_coord is None:
                x_coord = d.x_coord_names.index(x0_coord_name)
            if x1_coord_name is not None:
                y_coord = d.x_coord_names.index(x1_coord_name)
            tmp = {'info': {}}
            tmp['x0'] = d.x_data[:, x_coord]
            if y_coord is not None:
                tmp['x1'] = d.x_data[:, y_coord]
            result['data'][d.repo_info[RepoInfoKey.NAME] + ': ' +
                           str(d.repo_info[RepoInfoKey.VERSION])] = tmp
    return result
