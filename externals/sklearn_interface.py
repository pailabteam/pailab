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
        self.sklearn_params = sklearn_param

    def get_params(self):
        return self.sklearn_params


class SKLearnModel:
    @repo_object_init()
    def __init__(self, model):
        self.model = model


def eval_sklearn(model, data):
    return model.model.predict(data)


def train_sklearn(model_param, data_x, data_y):
    def get_class(full_class_name):
        parts = full_class_name.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    m = get_class(model_param.sklearn_module_name +
                  '.' + model_param.sklearn_class_name)

    model = m(**model_param.sklearn_params)
    model.fit(data_x, data_y)
    result = SKLearnModel(model, repo_info={})
    return result


def add_model(repo, skl_learner, model_name=None, model_param=None):
    repo.add_eval_function('externals.sklearn_interface',
                           'eval_sklearn', repo_name='eval_sklearn')
    repo.add_training_function(
        'externals.sklearn_interface', 'train_sklearn', repo_name='train_sklearn')
    m_name = model_name
    if m_name is None:
        m_name = skl_learner.__class__.__name__

    param = skl_learner.get_params(True)
    if model_param is not None:
        for k, v in model_param.items():
            param[k] = v
    model_p = SKLearnModelParam(
        skl_learner, param, repo_info={RepoInfoKey.NAME.value: m_name + '/model_param',
                                       RepoInfoKey.CATEGORY.value: MLObjectType.MODEL_PARAM.value})

    repo.add(model_p, 'adding model and training parameter')

    repo.add_model(m_name, model_eval='eval_sklearn',
                   model_training='train_sklearn', model_param=m_name + '/model_param', training_param='')
