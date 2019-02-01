import tensorflow as tf
import numpy
from tensorflow.keras.models import model_from_config  # model_from_json
from tensorflow.keras.models import Sequential
from pailab.repo_objects import repo_object_init
from pailab.repo_objects import RepoInfoKey
from pailab.repo import MLObjectType
import logging
logger = logging.getLogger(__name__)


class TensorflowKerasModel:
    @repo_object_init(['model_weights'])
    def __init__(self, model, model_weights):
        #logger.error("In constructor")
        self.keras_model_config = model
        self.model_weights = model_weights
        #logger.error("Leaving constructor, model: " + str(model))

    def numpy_to_dict(self):
        result = {}
        for i in range(len(self.model_weights)):
            result[str(i)] = self.model_weights[i]
        return result  # {'model_weights': result}

    def numpy_from_dict(self, repo_numpy_dict):
        #logger.error("In numpy_from_dict")
        self.model_weights = [None] * len(repo_numpy_dict)
        # logger.error(str(repo_numpy_dict))
        for i in range(len(repo_numpy_dict)):
            self.model_weights[i] = repo_numpy_dict[str(i)]

    def get_model(self):
        model = Sequential.from_config(self.keras_model_config)
        model.set_weights(self.model_weights)
        return model


class TensorflowKerasHistory:
    @repo_object_init(['loss'])
    def __init__(self, loss):
        self.loss = loss
        self.n_epochs = 0
        if len(loss) > 0:
            self.n_epochs = loss.shape[0]


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
        self.random_seed = 7
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = 0.0
        self.early_stopping = {'monitor': 'loss', 'min_delta': 0,
                               'patience': 500, 'verbose': 1, 'baseline': None}

    def get_params(self):
        return self.__dict__

    def get_optimizer(self):
        tmp = tf.keras.optimizers.get(self.optimizer)
        return tmp.from_config(self.optimizer_parameter)


class TensorflowKerasModelParameter:
    @repo_object_init()
    def __init__(self, model):
        self.param = model.get_config()  # model.to_json()

    def get_params(self):
        return self.param


def eval_keras_tensorflow(model, data):
    """Function to evaluate a keras-tensorflowmodel

    Args:
        model (TensorflowKerasModel): model to evaluate
        data (DataSet): dataset on which model is evaluated

    Returns:
        numpy-data: evaluated data
    """

    return model.get_model().predict(data)


def train_keras_tensorflow(model_param, train_param, data_x, data_y, verbose=0):
    cb = []
    if train_param.early_stopping != {}:
        cb.append(tf.keras.callbacks.EarlyStopping(
            **train_param.early_stopping))
    # stopping = EarlyStopping(monitor = 'loss', min_delta = training_parameter.metadata['param']['delta'],
    #                             patience=training_parameter.metadata['param']['patience'], mode='auto')
    #reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.8, patience=100, min_lr=0.00001)

    # model_from_json(param.param)
    # fix random seed for reproducibility
    numpy.random.seed(train_param.random_seed)
    model = Sequential.from_config(model_param.param)
    logger.info("Compiling model.")
    model.compile(loss=train_param.loss, optimizer=train_param.get_optimizer())
    logger.info("Start training, epochs: " + str(train_param.epochs) +
                ", batch_size: " + str(train_param.batch_size))
    history = model.fit(data_x, data_y, epochs=train_param.epochs,
                        batch_size=train_param.batch_size,
                        callbacks=cb,
                        verbose=verbose,
                        validation_split=train_param.validation_split)

    logger.info("Finished training")
    h = None
    if 'loss' in history.history:
        loss = numpy.asarray(history.history['loss'])
        return TensorflowKerasModel(model.get_config(), model.get_weights(), repo_info={}), TensorflowKerasHistory(loss, repo_info={})
    else:
        return TensorflowKerasModel(model.get_config(), model.get_weights(), repo_info={})


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
