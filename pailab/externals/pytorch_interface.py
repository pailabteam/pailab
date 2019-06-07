# -*- coding: utf-8 -*-
"""Module with interfaces for using PyTorch in pailab.

This module provides some useful methods and code snippets to integrate PyTorch projects into pailab.
The main entry point is the method `add_model` that setups all methods and objects needed to train the individual 
PyTorch model. 

Note:
    Currently, pailab supports only numpy data which is connected via a DataSet object to the training and test functions 
    in the repository. Pailab's DataSet object has nothing to do with PyTorch's DataSet object and to use all functionality out of the box 
    one has to convert the given PyTorch DataSet object into a numpy array (this module provides the method `convert_to_numpy` to do this for simple DataSets).
    It may not be difficult to integrate PyTorch DataSets (e.g. using a specialization of class NumpyStore) in a more clever and adapted way which will be subject 
    of future work.
"""
import numpy as np
from collections2 import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pailab.ml_repo.repo_objects import repo_object_init

from pailab.ml_repo.repo_objects import RepoInfoKey
from pailab.ml_repo.repo import MLObjectType
import logging
logger = logging.getLogger(__name__)


def _get_object_from_classname(classname, data):
    """ Returns an object instance for given classname and data dictionary.

    Arguments:
        classname {str} -- Full classname as string including the modules, e.g. repo.Y if class Y is defined in module repo.
        data {dict} -- dictionary of data used to initialize the object instance.

    Returns:
        [type] -- Instance object of class.
    """

    parts = classname.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    result = m(**data)  # ._create_from_dict(**data)
    return result


class PytorchModelWrapper:
    @repo_object_init(['state_dict'])
    def __init__(self, pytorch_model, param):
        self.classname = pytorch_model.__class__.__module__ + \
            '.' + pytorch_model.__class__.__name__
        self.state_dict_order = []
        self.state_dict = {}
        self.model_param = param
        for k, v in pytorch_model.state_dict().items():
            self.state_dict_order.append(k)
            self.state_dict[k] = v.numpy()

    def numpy_to_dict(self):
        return self.state_dict

    def numpy_from_dict(self, numpy_dict):
        self.state_dict = numpy_dict

    def get_model(self):
        state_dict = OrderedDict()
        for k in self.state_dict_order:
            state_dict[k] = torch.from_numpy(self.state_dict[k])
        model = _get_object_from_classname(self.classname, self.model_param)
        model.load_state_dict(state_dict)
        # model.eval()
        return model


class _PytorchDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data.x_data).float()
        self.target = None
        if hasattr(data, 'y_data'):
            self.target = torch.from_numpy(data.y_data).float()
        self.transform = None

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        if self.target is not None:
            y = self.target[index]
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data)


class PytorchTrainingParameter:
    @repo_object_init()
    def __init__(self, batch_size, epochs, loss='MSE', optimizer='Adam', optim_param={}):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.loss = 'MSE'
        self.optimizer = optimizer
        # create all default parameter for the chosen optimizer
        optimizer_ = _get_object_from_classname(
            'torch.optim.' + optimizer, {'params': (torch.empty(1, 1, dtype=torch.float),)})
        optimizer_.defaults.update(optim_param)
        self.optim_param = optimizer_.defaults


class PytorchModelParameter:
    @repo_object_init()
    def __init__(self, pytorch_model, **kwargs):
        self.model_classname = pytorch_model.__class__.__module__ + \
            '.' + pytorch_model.__class__.__name__
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_param(self):
        model_param = {name: attr for name, attr in self.__dict__.items()
                       if not name.startswith("__")
                       and not callable(attr)
                       and not type(attr) is staticmethod
                       and not name == 'model_classname'
                       and not name == 'repo_info'
                       }
        return model_param

    def create_model(self):
        model_param = self.get_param()
        model = _get_object_from_classname(self.model_classname, model_param)
        return model


def eval_pytorch(model: PytorchModelWrapper, data, num_workers=0):
    d = _PytorchDataset(data)
    loader = torch.utils.data.DataLoader(
        d, batch_size=1, num_workers=num_workers)
    m = model.get_model()
    m.eval()
    # determine number of data
    n_data = len(loader)
    result = None
    is_tuple = False
    for x in loader:
        if isinstance(x, list):
            output = m(x[0])
        else:
            output = m(x)
        if isinstance(output, list):
            result = np.empty((n_data, ) + tuple(output[0].shape[1:]))
            is_tuple = True
        else:
            result = np.empty((n_data, ) + tuple(output.shape[1:]))
        break
    if is_tuple:
        for i, x in enumerate(loader):
            if isinstance(x, list):
                output = m(x[0])
            else:
                output = m(x)
            result[i] = output.data
    else:
        for i, x in enumerate(loader):
            if isinstance(x, list):
                output = m(x[0])
            else:
                output = m(x)
            result[i] = output.data  # .detach.numpy()
    return result


def train_pytorch(model_param: PytorchModelParameter, train_param: PytorchTrainingParameter, data):
    if train_param.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise Exception('Unknown loss function ' + train_param.loss)

    # specify loss function
    logger.debug('Setting up data loader.')
    train_data = _PytorchDataset(data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_param.batch_size, num_workers=0)
    logger.debug('Finished setting up data loader.')
    # create model
    logger.debug('Setting up model.')
    model = model_param.create_model()
    model.train()
    logger.debug('Finished setting up model.')

    optim_args = {'params': model.parameters()}
    optim_args.update(train_param.optim_param)
    optimizer = _get_object_from_classname('torch.optim.'+train_param.optimizer,
                                           optim_args)
    logger.info('Start training with ' + str(train_param.epochs) + ' epochs.')
    train_loss = 0.0
    for epoch in range(1, train_param.epochs+1):
        train_loss = 0.0
        for data in train_loader:
            #logger.debug('Start training')
            if isinstance(data, list):
                x, target = data
            else:
                x = data
                target = data
            optimizer.zero_grad()
            outputs = model(x)
            # calculate the loss
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*target.size(0)  # funktioniert nur für mean-values

        # print avg training statistics
        train_loss = train_loss/len(train_loader)
        logger.debug('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))

    logger.info('Finished training with train loss ' + str(train_loss))
    result = PytorchModelWrapper(model, model_param.get_param(), repo_info={})
    return result


def add_model(repo, pytorch_model, model_param, model_name,
              batch_size, epochs, loss='MSE', optimizer='Adam', optim_param={}):
    """Add a PyTorch model.

    This method adds all relevant objects to train the PyTorch model: The training and evaluation functions as well as training- and model parameter.
    The objects created for the training and model parameter are instances of the internally defined  PytorchTrainingParameter and PytorchModelParameter
    classes.

    Args:
        repo ([type]): [description]
        pytorch_model ([type]): [description]
        model_param ([type]): [description]
        model_name ([type]): [description]
        batch_size ([type]): [description]
        epochs ([type]): [description]
        loss (str, optional): [description]. Defaults to 'MSE'.
        optimizer (str, optional): [description]. Defaults to 'Adam'.
        optim_param (dict, optional): [description]. Defaults to {}.
    """
    repo.add_eval_function(eval_pytorch,
                           repo_name='eval_pytorch')
    repo.add_training_function(
        train_pytorch, repo_name='train_pytorch')

    train_param = PytorchTrainingParameter(
        batch_size, epochs, loss, optimizer, optim_param,  repo_info={RepoInfoKey.NAME.value: model_name + '/training_param',
                                                                      RepoInfoKey.CATEGORY: MLObjectType.TRAINING_PARAM})
    model_param = PytorchModelParameter(
        pytorch_model, **model_param, repo_info={RepoInfoKey.NAME.value: model_name + '/model_param',
                                                 RepoInfoKey.CATEGORY: MLObjectType.MODEL_PARAM})

    repo.add([train_param, model_param])
    repo.add_model(model_name, model_eval='eval_pytorch',
                   model_training='train_pytorch', model_param=model_name + '/model_param', training_param=model_name + '/training_param')


def convert_to_numpy(pt_data, tuple_index=0):
    """Convert a PyTorch DataSet into a numpy array.

    Args:
        pt_data (torch.utils.data.DataSet): The DataSet to be converted to numpy.
        tuple_index (int): If each item of the dataset is a tuple (e.g. for images there may be the image and a label), 
            this index describes which of the tuple elements enters into the resulting numpy array.

    Returns:
        nparray: numpy array containing the data
    """
    if isinstance(pt_data[0], tuple):
        result = np.empty((len(pt_data), ) +
                          tuple(pt_data[0][tuple_index].shape))
        for i in range(len(pt_data)):
            result[i] = pt_data[i][tuple_index]
            return result
    else:
        result = np.empty((len(pt_data), ) + tuple(pt_data[0].shape))
        for i in range(len(pt_data)):
            result[i] = pt_data[i]
        return result
