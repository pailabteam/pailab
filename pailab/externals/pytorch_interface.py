import numpy
from collections import OrderedDict

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
        self.state_dict = pytorch_model.state_dict()
        self.classname = pytorch_model.__class__.__module__ + \
            '.' + pytorch_model.__class__.__name__
        self.state_dict_order = []
        self.model_param = param
        for k, v in pytorch_model.items():
            self.state_dict_order.append(k)
            self.state_dict = v.numpy()

    def get_model(self):
        state_dict = OrderedDict()
        for k in self.state_dict_order:
            state_dict[k] = torch.from_numpy(self.state_dict[k])
        model = _get_object_from_classname(self.classname, self.model_param)
        model.load_state_dict(state_dict)
        model.eval()
        return model


class _PytorchDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data.x_data).float()
        self.target = None
        if hasattr(data, 'y_data'):
            if data.y_data:
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


def eval_pytorch(model: PytorchModelWrapper, data, batch_size=1, num_workers=0):
    d = _PytorchDataset(data)
    loader = torch.utils.data.DataLoader(
        d, batch_size=batch_size, num_workers=num_workers)
    m = model.get_model()
    output = m(loader)
    return torch.to_numpy(output)


def train_pytorch(model_param: PytorchModelParameter, train_param: PytorchTrainingParameter, data):
    if train_param.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise Exception('Unknown loss function ' + train_param.loss)

    # specify loss function

    train_data = _PytorchDataset(data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=train_param.batch_size, num_workers=0)

    # create model
    model = model_param.create_model()
    optim_args = {'params': model.parameters()}
    optim_args.update(train_param.optim_param)
    optimizer = _get_object_from_classname('torch.optim.'+train_param.optimizer,
                                           optim_args)

    for epoch in range(1, train_param.epochs+1):
        train_loss = 0.0
        for data in train_loader:
            target = data
            optimizer.zero_grad()
            outputs = model(target)
            # calculate the loss
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*target.size(0)

        # print avg training statistics
        train_loss = train_loss/len(train_loader)
        logger.debug('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))

    result = PytorchModelWrapper(model, model_param.get_param(), repo_info={})
    return result


def add_model(repo, pytorch_model, model_param, model_name,
              batch_size, epochs, loss='MSE', optimizer='Adam', optim_param={}):
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
