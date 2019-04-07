"""This file contains a template how to develop/wrap a model to be used in pailab.

"""


from pailab.ml_repo.repo_objects import repo_object_init
import logging
logger = logging.getLogger(__name__)


class PailabCompatibleModel:
    """This class is a simple template showing what you may implement to use it as a calibrated model in pailab.

    """
    # The decorator repo_object_init makes the class compatible with pailab. If you want the big objects (numpy) that your model contains
    # be stored in  the NumpyStore, list them in the decorator.
    @repo_object_init(['big_data_1', 'big_data_2'])
    def __init__(self, data_1, data_2, big_data_1, big_data_2):
        self.data_1 = data_1
        self.big_data_1 = big_data_1
        self.big_data_2 = big_data_2

    ###########################
    # This method must be implemented only if your objects listed in the repo_object_init decorator
    # need some special treatment and are not simple numpy object (so that you have to apply some transformation).
    ###########################
    # def numpy_to_dict(self):
    #    return {'big_data_1': self.big_data_1, 'big_data_2': self.big_data_2} # this is just a simple example

    ###########################
    # This method must be implemented only if your objects listed in the repo_object_init decorator
    # need some special treatment and are not simple numpy object (so that you have to apply some transformation)
    ###########################
    # def numpy_from_dict(self, repo_numpy_dict):
    #    self.big_data_1 = repo_numpy_dict['big_data_1']
    #    self.big_data_2 = repo_numpy_dict['big_data_2']


class PailabCompatibleTrainingParameter:
    """This class is a simple template showing what you may implement to use it 
       as storage for your model's training parameter in pailab.
    """

    # The decorator repo_object_init makes the class compatible with pailab.
    @repo_object_init()
    def __init__(self, n_iter=0):
        self.n_iter = n_iter

    # This method is used to plot errors vs. training parameter.
    # It is up to you which parameters you include.
    def get_params(self):
        return {'n_iter': self.n_iter}


class PailabCompatibleModelParameter:
    # The decorator repo_object_init makes the class compatible with pailab. You could also inherit from the RepoObject base class which is a good solution
    # if you need to specialize some of the RepoObject interface functions.
    @repo_object_init()
    def __init__(self, n_layers):
        self.n_layers = n_layers

    # This method is used to plot errors vs. training parameter.
    # It is up to you which parameters you include.
    def get_params(self):
        return {'n_layers': self.n_layers}


# function to evaluate your model on given data
def eval(model, data):
    """Function to evaluate a your model
    Args:
        model (PailabCompatibleModel): model to evaluate
        data (DataSet): dataset on which model is evaluated
    """
    pass

# function to train your model


def train(model_param, train_param, data_x, data_y):
    """function to train your model

    Args:
        model_param (PailabCompatibleModelParameter): model parameter
        train_param (PailabCompatibleTrainingParameter): training parameter
        data_x (numpy array/matrix): x data
        data_y (numpy array/matrix): y_data


    Returns:
        PailabCompatibleModel: trained model
    """
    pass
