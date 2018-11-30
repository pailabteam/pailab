import tensorflow as tf
from tensorflow.keras.models import model_from_config  # model_from_json
from tensorflow.keras.models import Sequential
from pailab.repo_objects import repo_object_init
from pailab.repo_objects import RepoInfoKey
from pailab.repo import MLObjectType


class TensorflowKerasModel:
    @repo_object_init()
    def __init__(self, model=None):
        self.keras_model = model

    def numpy_to_dict(self, repo_obj):
        model_weights = self.keras_model.get_weights()
        result = {}
        for i in range(len(model_weights)):
            result[str(i)] = model_weights[i]

    def numpy_from_dict(self, repo_numpy_dict):
        weights = [None] * len(repo_numpy_dict)
        for k, v in repo_numpy_dict.items():
            weights[int(k)] = v
        self.keras_model.set_weights(weights)


class TensorflowKerasTrainingParameter:
    @repo_object_init()
    def __init__(self, loss, epochs, batch_size, optimizer='ADAM', optimizer_param={}):
        self.optimizer = optimizer

        tmp = tf.keras.optimizers.get(optimizer)
        # gt default parameters of choen optimizer
        self.optimizer_parameter = tmp.get_config()
        for k, v in optimizer_param.items():  # overwrite defaults where values are specified
            self.optimizer_parameter[k] = v
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size

    def get_optimizer(self):
        tmp = tf.keras.optimizers.get(self.optimizer)
        return tmp.from_config(self.optimizer_parameter)


class TensorflowKerasModelParameter:
    @repo_object_init()
    def __init__(self, model):
        self.param = model.get_config()  # model.to_json()


def eval_keras_tensorflow(model, data):
    """Function to evaluate a keras-tensorflowmodel

    Args:
        model (TensorflowKerasModel): model to evaluate
        data (DataSet): dataset on which model is evaluated

    Returns:
        numpy-data: evaluated data
    """

    return model.keras_model.predict(data)


def train_keras_tensorflow(model_param, train_param, data_x, data_y):
    # stopping = EarlyStopping(monitor = 'loss', min_delta = training_parameter.metadata['param']['delta'],
    #                             patience=training_parameter.metadata['param']['patience'], mode='auto')
    #reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.8, patience=100, min_lr=0.00001)

    # model_from_json(param.param)

    model = Sequential.from_config(model_param.param)
    model.compile(loss=train_param.loss, optimizer=train_param.get_optimizer())
    history = model.fit(data_x, data_y, epochs=train_param.epochs,
                        batch_size=train_param.batch_size,
                        #callbacks=[reduce_lr, stopping],
                        verbose=0)
    return TensorflowKerasModel(model)


def add_model(repo, tensorflow_keras_model, model_name, loss, epochs, batch_size, optimizer='ADAM', optimizer_param={}):
    """Adds a new tensorflow-keras model to a pailab MLRepo

    Args:
        :param repo (MLRepo): ml repo
        :param tensorflow_keras_model (keras model): the model created with tensorflows keras (not yet compiled)
        :param model_name (str): name of model used in repo
        :param loss (str): lossfunction
        :param epochs (int): number of epochs used
        :param batch_size (int): batch size
    """

    repo.add_eval_function(eval_keras_tensorflow,
                           repo_name='eval_keras_tensorflow')
    repo.add_training_function(
        train_keras_tensorflow, repo_name='train_keras_tensorflow')
    model_p = TensorflowKerasModelParameter(
        tensorflow_keras_model, repo_info={RepoInfoKey.NAME.value: model_name + '/model_param',
                                           RepoInfoKey.CATEGORY: MLObjectType.MODEL_PARAM})
    train_p = TensorflowKerasTrainingParameter(loss, epochs, batch_size, optimizer, optimizer_param, repo_info={RepoInfoKey.NAME.value: model_name + '/training_param',
                                                                                                                RepoInfoKey.CATEGORY: MLObjectType.TRAINING_PARAM})

    repo.add([model_p, train_p], 'adding model and training parameter')

    repo.add_model(model_name, model_eval='eval_keras_tensorflow',
                   model_training='train_keras_tensorflow', model_param=model_name + '/model_param', training_param=model_name + '/training_param')
