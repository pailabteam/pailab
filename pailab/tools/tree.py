# -*- coding: utf-8 -*-
"""This module contains all functions and classes for the MLTree. The MLTree buils a tree-like
structure of the objects in a given repository. This allows the user to access objects in a
comfortable way allowing for autocompletion (i.e. in Jupyter notebooks).

To use it one can simply call the :py:meth:`pailab.tools.tree.MLTree.add_tree` method to 
add such a tree to the current repository::

    >>from pailab.tools.tree import MLTree
    >>MLTree.add_tree(ml_repo)

After the tree has been added, one can simply use the tree. Here, using autocompletion makes the basic work wih repo objects quite simply.
Each tree node provides useful functions that can be applied:

- ``load`` loads the object of the given tree node or the child tree nodes of the current node. a
  After calling load the respective nodes have a new attribute ``obj`` that contains the respective loaded object. To load all objects belonging to the models subtree like 
  parameters, evaluations or measures one can call::

   >> ml_repo.tree.models.load()

- ``history`` lists the history of all objects of the respective subtree, where history excepts certain parameters such as a range of versions or 
  which repo object information to include. To list th history of all training data just use::

    >> ml_repo.tree.training_data.history()

- ``modifications`` lists all objects of the respective subtree that have been modified and no yet been committed.

There are also node dependent function (depending on what object the node represents).
"""
import logging
from numpy import load
from deepdiff import DeepDiff
from pailab.ml_repo.repo import MLObjectType, MLRepo
from pailab.ml_repo.repo_objects import RepoInfoKey, DataSet  # pylint: disable=E0401
from pailab.ml_repo.repo_store import RepoStore  # pylint: disable=E0401
import pailab.ml_repo.repo_store as repo_store
import pailab.ml_repo.repo_objects as repo_objects
logger = logging.getLogger(__name__)

#region collections and items



class _RepoObjectItem:

    def __init__(self, name, ml_repo, repo_obj = None):
        self._name = name
        self._repo = ml_repo
        if repo_obj is not None:
            self.obj = repo_obj
     
    def _set(self, path, items):
        if len(path) > 0:
            if len(path) == 1:
                setattr(self, path[0], items[0])
                return
            if hasattr(self, path[0]):
                getattr(self, path[0])._set(path[1:], items[1:])
            else:
                setattr(self, path[0], items[0])
                items[0]._set(path[1:], items[1:])

    def load(self, version=repo_store.LAST_VERSION, full_object=False,
            modifier_versions=None, containing_str=None):
            if containing_str is None or containing_str in self._name:
                if self._repo is not None:
                    self.obj = self._repo.get(self._name, version, full_object, modifier_versions, throw_error_not_exist = False)
            for v in self.__dict__.values():
                if hasattr(v,'load'):
                    v.load(version, full_object, modifier_versions, containing_str)

    def modifications(self, commit=False, commit_message=''):
        result = {}
        if self._name is not None:
            try:
                if self._repo is not None:
                    obj_orig = self._repo.get(
                        self.obj.repo_info[RepoInfoKey.NAME], version=self.obj.repo_info[RepoInfoKey.VERSION])
                    diff = DeepDiff(obj_orig, self.obj,
                                ignore_order=True)
            except AttributeError:
                return None
            if len(diff) == 0:
                return None
            else:
                if commit and (self._repo is not None):
                    version = self._repo.add(
                        self.obj, message=commit_message)
                    self.obj = self._repo.get(self._name, version=version)
                result = {self._name: diff}
        for v in self.__dict__.values():
            if hasattr(v, 'modifications'):
                tmp = v.modifications(commit, commit_message)
                if tmp is not None:
                    result.update(tmp)
        return result

    def history(self, version = (repo_store.FIRST_VERSION,repo_store.LAST_VERSION), 
                repo_info = [RepoInfoKey.NAME, RepoInfoKey.AUTHOR, RepoInfoKey.COMMIT_DATE, RepoInfoKey.COMMIT_MESSAGE], 
                obj_data = []):
        history = []
        if self._repo is not None:
            history = self._repo.get(self._name, version = version,  throw_error_not_exist=False)
        if not isinstance(history, list):
            history = [history]
        result = {}
        tmp = []
        for h in history:
            r = {}
            for r_info in repo_info:
                r[str(r_info)] = h.repo_info[r_info]
            for o_info in obj_data:
                r[o_info] = obj_data.__dict__[o_info]
            tmp.append(r)
        result[self._name] = tmp
        for v in self.__dict__.values():
            if isinstance(v, _RepoObjectItem):
                tmp2 = v.history(version, repo_info, obj_data)
                if tmp2 is not None:
                    result.update(tmp2)
        if len(result) > 0:
            return result
        

    def __call__(self, containing_str=None):
        # if len(self.__dict__) == 1:
        if containing_str is not None:
            result = []
            if containing_str in self._name:
                result.append(self._name)
            for v in self.__dict__.values():
                if isinstance(v, _RepoObjectItem):
                    d = v(containing_str)
                    if isinstance(d, str):
                        result.append(d)
                    else:
                        result.extend(d)
            return [x for x in result if containing_str in x]
        else:
            return self._name
        return result


class _RawDataItem(_RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(_RawDataItem,self).__init__(name, ml_repo, repo_obj)

    def append(self, x_data, y_data = None):
        """Append data to a RawData object

        It appends data to the given RawData object and updates all training and test DataSets which implicitely changed by this update.

        Args:
            name (string): name of RawData object
            x_data (numpy matrix): the x_data to append
            y_data (numpy matrix, optional): Defaults to None. The y_data to append
        
        Raises:
            Exception: If the data is not consistent to the RawData (e.g. different number of x-coordinates) it throws an exception.
        """
        logger.info('Start appending ' + str(x_data.shape[0]) + ' datapoints to RawData' + self._name)
        raw_data = self._repo.get(self._name)
        if len(raw_data.x_coord_names) != x_data.shape[1]:
            raise Exception('Number of columns of x_data of RawData object is not equal to number of columns of additional x_data.')
        if raw_data.y_coord_names is None and y_data is not None:
            raise Exception('RawData object does not contain y_data but y_data is given')
        if raw_data.y_coord_names is not None:
            if y_data is None:
                raise Exception('RawData object has y_data but no y_data is given')
            if y_data.shape[1] != len(raw_data.y_coord_names ):
                raise Exception('Number of columns of y_data of RawData object is not equal to number of columns of additional y_data.')
        numpy_dict = {'x_data' : x_data}
        if raw_data.y_coord_names is not None:
            numpy_dict['y_data'] =  y_data
        raw_data.n_data += x_data.shape[0]
        old_version = raw_data.repo_info[RepoInfoKey.VERSION]
        new_version = self._repo.add(raw_data)
        self._repo._numpy_repo.append(self._name, old_version, new_version, numpy_dict)
        # now find all datasets which are affected by the updated data
        changed_data_sets = []
        training_data = self._repo.get_training_data(full_object = False)
        if isinstance(training_data, DataSet):
            if training_data.raw_data == self._name and training_data.raw_data_version == repo_store.RepoStore.LAST_VERSION:
                if training_data.end_index is None or training_data.end_index < 0:
                    training_data.raw_data_version = new_version
                    changed_data_sets.append(training_data)
        test_data = self._repo.get_names(MLObjectType.TEST_DATA)
        for d in test_data:
            data = self._repo.get(d)
            if isinstance(data, DataSet):
                if data.raw_data == self._name and data.raw_data_version == repo_store.RepoStore.LAST_VERSION:
                    if data.end_index is None or data.end_index < 0:
                        data.raw_data_version = new_version
                        changed_data_sets.append(data)
        self._repo.add(changed_data_sets, 'RawData ' + self._name + ' updated, add DataSets depending om the updated RawData.')
        if hasattr(self, 'obj'):#update current object
            self.obj = self._repo.get(self._name, version=new_version)
        logger.info('Finished appending data to RawData' + self._name)

class _RawDataCollection(_RepoObjectItem):
    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]

    def __init__(self, repo):
        super(_RawDataCollection, self).__init__('raw_data', repo)
        names = repo.get_names(MLObjectType.RAW_DATA)
        for n in names:
            setattr(self, _RawDataCollection.__get_name_from_path(n), _RawDataItem(n, repo))
        
    def add(self, name, data, input_variables = None, target_variables = None):
        """Add raw data to the repository

        Arguments:
            data_name {name of data} -- the name of the data added
            data {pandas datatable} -- the data as pandas datatable
        
        Keyword Arguments:
            input_variables {list of strings} -- list of column names defining the input variables for the machine learning (default: {None}). If None, all variables are used as input
            target_variables {list of strings} -- list of column names defining the target variables for the machine learning (default: {None}). If None, no target data is added from the table.
        """
        path = 'raw_data/' + name

        if input_variables is None:
            input_variables = list(data)
            if not target_variables is None:
                [input_variables.remove(x) for x in target_variables]
        else:
            # check whether the input_variables are included in the data
            if not [item for item in input_variables if item in list(data)] == input_variables:
                raise Exception('RawData does not include at least one column included in input_variables')
      
        if target_variables is not None:
            # check if target variables are in list
            if not [item for item in target_variables if item in list(data)] == target_variables:
                raise Exception('RawData does not include at least one column included in target_variables')
            raw_data = repo_objects.RawData(data.loc[:, input_variables].values, input_variables, data.loc[:, target_variables].values, 
                target_variables, repo_info = {RepoInfoKey.NAME: path})
        else:
            raw_data = repo_objects.RawData(data.loc[:, input_variables].values, input_variables, repo_info = {RepoInfoKey.NAME: path})
        v = self._repo.add(raw_data, 'data ' + path + ' added to repository' , category = MLObjectType.RAW_DATA)
        obj = self._repo.get(path, version=v, full_object = False)
        setattr(self, name, _RawDataItem(path, self._repo, obj))

    def add_from_numpy_file(self, name, filename_X, x_names, filename_Y=None, y_names = None):
        path = name
        X = load(filename_X)
        Y = None
        if filename_Y is not None:
            Y = load(filename_Y)
        raw_data =  repo_objects.RawData(X, x_names, Y, y_names, repo_info = {RepoInfoKey.NAME: path})
        v = self._repo.add(raw_data, 'data ' + path + ' added to repository' , category = MLObjectType.RAW_DATA)
        obj = self._repo.get(path, version=v, full_object = False)
        setattr(self, name, _RawDataItem(path, self._repo, obj))

class _TrainingDataCollection(_RepoObjectItem):

    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]
    
    def __init__(self, repo):
        super(_TrainingDataCollection, self).__init__('training_data', None)
        self.__repo = repo # we store ml_repo in __repo to circumvent that obj is loaded from eneric base class
        names = repo.get_names(MLObjectType.TRAINING_DATA)
        for n in names:
            setattr(self, _TrainingDataCollection.__get_name_from_path(n), _RepoObjectItem(n, repo))
        
    def add(self, name, raw_data, start_index=0, 
        end_index=None, raw_data_version='last'):
        #path = 'training_data/' + name
        data_set = repo_objects.DataSet(raw_data, start_index, end_index, 
                raw_data_version, repo_info = {RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        v = self.__repo.add(data_set)
        tmp = self.__repo.get(name, version=v)
        item = _RepoObjectItem(name, self.__repo, tmp)
        setattr(self, name, item)

class _TestDataCollection(_RepoObjectItem):
    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]
    
    def __init__(self, repo):
        super(_TestDataCollection, self).__init__('test_data', None)
        self.__repo = repo # we store ml_repo in __repo to circumvent that obj is loaded from eneric base class
        names = repo.get_names(MLObjectType.TEST_DATA)
        for n in names:
            setattr(self, _TestDataCollection.__get_name_from_path(n), _RepoObjectItem(n,repo))
        
    def add(self, name, raw_data, start_index=0, 
        end_index=None, raw_data_version='last'):
        data_set = repo_objects.DataSet(raw_data, start_index, end_index, 
                raw_data_version, repo_info = {RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        v = self.__repo.add(data_set)
        tmp = self.__repo.get(name, version=v)
        item = _RepoObjectItem(name, self.__repo, tmp)
        setattr(self, name, item)

class _MeasureItem(_RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(_MeasureItem, self).__init__(name, ml_repo, repo_obj) 

class _JobItem(_RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(_JobItem, self).__init__(name, ml_repo, repo_obj) 

class _MeasureCollection(_RepoObjectItem):
    def __init__(self, name, ml_repo):
        super(_MeasureCollection, self).__init__('measures', None)
        names = ml_repo.get_names(MLObjectType.MEASURE)
        for n in names:
            path = n.split('/')[2:]
            items = [None] * len(path)
            for i in range(len(items)-1):
                items[i] = _RepoObjectItem(path[i], None)
            items[-1] = _MeasureItem(n, ml_repo)
            self._set(path, items)
            #items[-2] = MeasuresOnDataItem

class _EvalCollection(_RepoObjectItem):
    def __init__(self, name, ml_repo):
        super(_EvalCollection, self).__init__('eval', None)
        names = ml_repo.get_names(MLObjectType.EVAL_DATA)
        for n in names:
            path = n.split('/')[2:]
            items = [None] * len(path)
            for i in range(len(items)-1):
                items[i] = _RepoObjectItem(path[i], None)
            items[-1] = _MeasureItem(n, ml_repo)
            self._set(path, items)

class _TestCollection(_RepoObjectItem):
    def __init__(self, name, ml_repo):
        super(_TestCollection, self).__init__('tests', None)
        names = ml_repo.get_names(MLObjectType.TEST)
        for n in names:
            path = n.split('/')[2:]
            items = [None] * len(path)
            for i in range(len(items)-1):
                items[i] = _RepoObjectItem(path[i], None)
            items[-1] = _RepoObjectItem(n, ml_repo)
            self._set(path, items)

class _JobCollection(_RepoObjectItem):
    def __init__(self, name, ml_repo, model_name):
        super(_JobCollection, self).__init__('jobs', None)
        names = ml_repo.get_names(MLObjectType.JOB)
        for n in names:
            if model_name in n:
                path = n.split('/')
                path = path[path.index('jobs')+1:]
                items = [None] * len(path)
                for i in range(len(items)-1):
                    items[i] = _RepoObjectItem(path[i], None)
                items[-1] = _JobItem(n, ml_repo)
                self._set(path, items)

class _ModelItem(_RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(_ModelItem,self).__init__(name, ml_repo, repo_obj)
        self.model = _RepoObjectItem(name + '/model', ml_repo)
        self.eval = _EvalCollection(name + '/eval', ml_repo)
        self.model_param = _RepoObjectItem(name + '/model_param', ml_repo)
        self.tests = _TestCollection(name + '/tests', ml_repo)
        self.measures = _MeasureCollection(name+ '/measure', ml_repo)
        self.jobs = _JobCollection(name+'/jobs', ml_repo, name)
        if ml_repo._object_exists(name+'/training_stat'):
            self.training_statistic = _RepoObjectItem(name+'/training_stat', ml_repo)
        if ml_repo._object_exists(name+'/training_param'):
            self.training_param = _RepoObjectItem(name + '/training_param', ml_repo)


    def set_label(self, label_name, version = repo_store.RepoStore.LAST_VERSION, message=''):
        self._repo.set_label(label_name, self._name+ '/model', version, message)

class _LabelCollection(_RepoObjectItem):
    def __init__(self, repo):
        super(_LabelCollection,self).__init__('labels', None)
        names = repo.get_names(MLObjectType.LABEL)
        for n in names:
            #label = ml_repo.get()
            setattr(self, n, _RepoObjectItem(n, repo))
        
class _ModelCollection(_RepoObjectItem):
    @staticmethod
    def __get_name_from_path(name):
        return name

    def __init__(self, repo):
        super(_ModelCollection,self).__init__('models', None)
        names = repo.get_names(MLObjectType.MODEL)
        for n in names:
            setattr(self, _ModelCollection.__get_name_from_path(n), _ModelItem(n, repo))
        self.labels = _LabelCollection(repo)
        
    def add(self, name):
        setattr(self, name, _ModelItem(name,self._repo))



class _CacheDataCollection(_RepoObjectItem):

    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]
    
    def __init__(self, repo):
        super(_CacheDataCollection, self).__init__('cache', None)
        self.__repo = repo # we store ml_repo in __repo to circumvent that obj is loaded from eneric base class
        names = repo.get_names(MLObjectType.CACHED_VALUE)
        for n in names:
            setattr(self, _CacheDataCollection.__get_name_from_path(n), _RepoObjectItem(n, repo))
#endregion



class MLTree:

    @staticmethod
    def add_tree(ml_repo):
        """Adds an MLTree to a repository.

        Args:
            ml_repo (MLRepo): the repository the tre is added
        """
        setattr(ml_repo, 'tree', MLTree(ml_repo))
        ml_repo._add_triggers.append(ml_repo.tree.reload)
        
    def __create(self):
        self.raw_data = _RawDataCollection(self.__ml_repo)
        self.training_data = _TrainingDataCollection(self.__ml_repo)
        self.test_data = _TestDataCollection(self.__ml_repo)
        self.models = _ModelCollection(self.__ml_repo)
        self.cache = _CacheDataCollection(self.__ml_repo)

    def __init__(self, ml_repo):
        self.__ml_repo = ml_repo
        self.__create()

    def reload(self, **kwargs):
        """Method to reload the tree after objects have been added or deleted from the repository.
        """
        self.__create()  # todo make this more efficient by just updating collections and items which are affected by this

    def modifications(self):
        """Return a dictionary of all objects that were modified but no yet 
        commited to the repository.
        
        Returns:
            dict: dictionary mapping object ids to dictionary of the modified attributes 
        """
        result = {}
        tmp = self.raw_data.modifications()
        if tmp is not None:
            result.update(tmp)
        tmp = self.training_data.modifications()
        if tmp is not None:
            result.update(tmp)
        tmp = self.test_data.modifications()
        if tmp is not None:
            result.update(stmp)
        tmp = self.models.modifications()
        if tmp is not None:
            result.update(tmp)
        if len(result) == 0:
            return None
        return result

        
