from repo.repo_objects import repo_object_init
from repo.repo_objects import RepoInfoKey
from repo.repo import MLObjectType


def get_classname(x):
    return x.__class__.__module__ + '.' + x.__class__.__name__


class SKLearnModelParam:
    @repo_object_init()
    def __init__(self, model, sklearn_param):
        self.sklearn_module_name = model.__class__.__module__
        self.sklearn_class_name = model.__class__.__name__
        self.params = sklearn_param


class SKLearnTrainingParam:
    @repo_object_init()
    def __init__(self, sklearn_param={}):
        self.params = sklearn_param


class SKLearnModel:
    @repo_object_init()
    def __init__(self, model):
        self.model = model


def eval_sklearn(model, data):
    return model.model.predict(data)


def train_sklearn(model_param, training_param, data):
    def get_class(full_class_name):
        parts = full_class_name.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    for k, v in training_param.params.items():
        model_param.params[k] = v
    m = get_class(model_param.sklearn_module_name +
                  '.' + model_param.sklearn_class_name)

    model = m(**model_param.params)
    model.fit(data._x_data, data.y_data)
    result = SKLearnModel(model)
    return result


def add_sklearn_model(repo, skl_learner, params=None, model_name=None, model_param=None):
    repo.add_eval_function('repo.externals.sklearn_interface',
                           'eval_sklearn', repo_name='eval_sklearn')
    repo.add_training_function(
        'repo.externals.sklearn_interface', 'train_sklearn', repo_name='train_sklearn')
    m_name = model_name
    if m_name is None:
        m_name = skl_learner.__class__.__name__

    param = skl_learner.get_params(True)
    if model_param is not None:
        for k, v in param.items():
            param[k] = v
    model_p = SKLearnModelParam(
        skl_learner, param, repo_info={RepoInfoKey.CLASSNAME.value: m_name + '.model_param',
                                       RepoInfoKey.CATEGORY.value: MLObjectType.MODEL_PARAM.value})
    repo.add(model_p, 'adding model parameter for ' + m_name)

    training_p = SKLearnTrainingParam(
        param, repo_info={RepoInfoKey.CLASSNAME.value: m_name + '.training_param',
                          RepoInfoKey.CATEGORY.value: MLObjectType.TRAINING_PARAM})
    repo.add(training_p, 'adding training parameter for ' + m_name)

    repo.add_model(m_name, model_eval='eval_sklearn',
                   model_training='train_sklearn', model_param=m_name + '.model_param', training_param=None)
