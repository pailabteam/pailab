import logging
import collections
from numpy import load
from deepdiff import DeepDiff
from pailab.ml_repo.repo import MLObjectType, MLRepo, NamingConventions
from pailab.repo_objects import RepoInfoKey, DataSet  # pylint: disable=E0401
from pailab.repo_store import RepoStore  # pylint: disable=E0401
import pailab.repo_store as repo_store
import pailab.repo_objects as repo_objects
logger = logging.getLogger(__name__)

#region collections and items

   
def get_test_summary(repo:MLRepo):
    """Return test summary for all labeled models and all latest models
    
    Args:
        repo (MLRepo): repo
    """
    pass

class RepoObjectItem:

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
                self.obj = self._repo.get(self._name, version, full_object, modifier_versions, throw_error_not_exist = False)
                if self.obj == []:
                    pass
            for v in self.__dict__.values():
                if hasattr(v,'load'):
                    v.load(version, full_object, modifier_versions, containing_str)

    def modifications(self, commit=False, commit_message=''):
        result = {}
        if self._name is not None:
            try:
                obj_orig = self._repo.get(
                    self.obj.repo_info[RepoInfoKey.NAME], version=self.obj.repo_info[RepoInfoKey.VERSION])
                diff = DeepDiff(obj_orig, self.obj,
                                ignore_order=True)
            except AttributeError:
                return None
            if len(diff) == 0:
                return None
            else:
                if commit:
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

    def history(self, version = (repo_store.FIRST_VERSION,repo_store.LAST_VERSION), repo_info = [RepoInfoKey.AUTHOR, RepoInfoKey.COMMIT_DATE, RepoInfoKey.COMMIT_MESSAGE], obj_data = []):
        history = self._repo.get(self._name, version = version)
        if not isinstance(history, list):
            history = [history]
        result = {}
        for h in history:
            r = {}
            for r_info in repo_info:
                r[str(r_info)] = h.repo_info[r_info]
            for o_info in obj_data:
                r[o_info] = obj_data.__dict__[o_info]
            result[h.repo_info[RepoInfoKey.VERSION]] = r
        return result

    def __call__(self, containing_str=None):
        # if len(self.__dict__) == 1:
        if containing_str is not None:
            result = []
            if containing_str in self._name:
                result.append(self._name)
            for v in self.__dict__.values():
                if isinstance(v, RepoObjectItem):
                    d = v(containing_str)
                    if isinstance(d, str):
                        result.append(d)
                    else:
                        result.extend(d)
            return [x for x in result if containing_str in x]
        else:
            return self._name
        return result

class RawDataItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(RawDataItem,self).__init__(name, ml_repo, repo_obj)

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

class RawDataCollection(RepoObjectItem):
    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]

    def __init__(self, repo):
        super(RawDataCollection, self).__init__('raw_data', repo)
        names = repo.get_names(MLObjectType.RAW_DATA)
        for n in names:
            setattr(self, RawDataCollection.__get_name_from_path(n), RawDataItem(n, repo))
        self._repo = repo

    def add(self, name, data, input_variables = None, target_variables = None):
        """Add raw data to the repository

        Arguments:
            data_name {name of data} -- the name of the data added
            data {pandas datatable} -- the data as pndas datatable
        
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
        setattr(self, name, RawDataItem(path, self._repo, obj))

    def add_from_numpy_file(self, name, filename_X, x_names, filename_Y=None, y_names = None):
        path = name
        X = load(filename_X)
        Y = None
        if filename_Y is not None:
            Y = load(filename_Y)
        raw_data =  repo_objects.RawData(X, x_names, Y, y_names, repo_info = {RepoInfoKey.NAME: path})
        v = self._repo.add(raw_data, 'data ' + path + ' added to repository' , category = MLObjectType.RAW_DATA)
        obj = self._repo.get(path, version=v, full_object = False)
        setattr(self, name, RawDataItem(path, self._repo, obj))

class TrainingDataCollection(RepoObjectItem):

    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]
    
    def __init__(self, repo):
        super(TrainingDataCollection, self).__init__('training_data', repo)
        
        names = repo.get_names(MLObjectType.TRAINING_DATA)
        for n in names:
            setattr(self, TrainingDataCollection.__get_name_from_path(n), RepoObjectItem(n, repo))
        self._repo = repo

    def add(self, name, raw_data, start_index=0, 
        end_index=None, raw_data_version='last'):
        #path = 'training_data/' + name
        data_set = repo_objects.DataSet(raw_data, start_index, end_index, 
                raw_data_version, repo_info = {RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TRAINING_DATA})
        v = self._repo.add(data_set)
        tmp = self._repo.get(name, version=v)
        item = RepoObjectItem(name, self._repo, tmp)
        setattr(self, name, item)

class TestDataCollection(RepoObjectItem):
    @staticmethod
    def __get_name_from_path(path):
        return path.split('/')[-1]
    
    def __init__(self, repo):
        super(TestDataCollection, self).__init__('test_data', repo)
        names = repo.get_names(MLObjectType.TEST_DATA)
        for n in names:
            setattr(self, TestDataCollection.__get_name_from_path(n), RepoObjectItem(n,repo))
        self._repo = repo

    def add(self, name, raw_data, start_index=0, 
        end_index=None, raw_data_version='last'):
        data_set = repo_objects.DataSet(raw_data, start_index, end_index, 
                raw_data_version, repo_info = {RepoInfoKey.NAME: name, RepoInfoKey.CATEGORY: MLObjectType.TEST_DATA})
        v = self._repo.add(data_set)
        tmp = self._repo.get(name, version=v)
        item = RepoObjectItem(name, self._repo, tmp)
        setattr(self, name, item)

class MeasureItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(MeasureItem, self).__init__(name, ml_repo, repo_obj) 

class JobItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(JobItem, self).__init__(name, ml_repo, repo_obj) 

class MeasureCollection(RepoObjectItem):
    def __init__(self, name, ml_repo, model_name):
        super(MeasureCollection, self).__init__('measures', ml_repo)
        names = ml_repo.get_names(MLObjectType.MEASURE)
        for n in names:
            path = n.split('/')[2:]
            items = [None] * len(path)
            for i in range(len(items)-1):
                items[i] = RepoObjectItem(path[i], None)
            items[-1] = MeasureItem(n, ml_repo)
            self._set(path, items)
            #items[-2] = MeasuresOnDataItem

class TestCollection(RepoObjectItem):
    def __init__(self, name, ml_repo, model_name):
        super(TestCollection, self).__init__('tests', ml_repo)
        names = ml_repo.get_names(MLObjectType.TEST)
        for n in names:
            path = n.split('/')[2:]
            items = [None] * len(path)
            for i in range(len(items)-1):
                items[i] = RepoObjectItem(path[i], ml_repo)
            items[-1] = RepoObjectItem(n, ml_repo)
            self._set(path, items)

class JobCollection(RepoObjectItem):
    def __init__(self, name, ml_repo, model_name):
        super(JobCollection, self).__init__('jobs', ml_repo)
        names = ml_repo.get_names(MLObjectType.JOB)
        for n in names:
            if model_name in n:
                path = n.split('/')
                path = path[path.index('jobs')+1:]
                items = [None] * len(path)
                for i in range(len(items)-1):
                    items[i] = RepoObjectItem(path[i], None)
                items[-1] = JobItem(n, ml_repo)
                self._set(path, items)

class ModelItem(RepoObjectItem):
    def __init__(self, name, ml_repo, repo_obj = None):
        super(ModelItem,self).__init__(name, ml_repo, repo_obj)
        self.model = RepoObjectItem(name + '/model', ml_repo)
        self.eval = RepoObjectItem(name + '/eval', ml_repo)
        self.model_param = RepoObjectItem(name + '/model_param', ml_repo)
        self.tests = TestCollection(name + '/tests', ml_repo, name)
        self.measures = MeasureCollection(name+ '/measure', ml_repo, name)
        self.jobs = JobCollection(name+'/jobs', ml_repo, name)
        if ml_repo._object_exists(name+'/training_stat'):
            self.training_statistic = RepoObjectItem(name+'/training_stat', ml_repo)
        if ml_repo._object_exists(name+'/training_param'):
            self.training_param = RepoObjectItem(name + '/training_param', ml_repo)

        
        #self.param = RepoObjectItem(name)

    def set_label(self, label_name, version = repo_store.RepoStore.LAST_VERSION, message=''):
        self._repo.set_label(label_name, self._name+ '/model', version, message)


class LabelCollection(RepoObjectItem):
    def __init__(self, repo):
        super(LabelCollection,self).__init__(None, repo)
        names = repo.get_names(MLObjectType.LABEL)
        for n in names:
            #label = ml_repo.get()
            setattr(self, n, RepoObjectItem(n, repo))
        
class ModelCollection(RepoObjectItem):
    @staticmethod
    def __get_name_from_path(name):
        return name

    def __init__(self, repo):
        super(ModelCollection,self).__init__('models', repo)
        names = repo.get_names(MLObjectType.MODEL)
        for n in names:
            setattr(self, ModelCollection.__get_name_from_path(n), ModelItem(n,repo))
        self.labels = LabelCollection(repo)
        self._repo = repo

    def add(self, name):
        setattr(self, name, ModelItem(name,self._repo))
#endregion

class MLTree:

    @staticmethod
    def add_tree(ml_repo):
        """Adds a tree to a repository

        Args:
            ml_repo (MLRepo): the repository the tre is added
        """
        setattr(ml_repo, 'tree', MLTree(ml_repo))
        ml_repo._add_triggers.append(ml_repo.tree.reload)
        #setattr(ml_repo, 'raw_data', RawDataCollection(ml_repo))
        #setattr(ml_repo, 'training_data', TrainingDataCollection(ml_repo))
        #setattr(ml_repo, 'test_data', TestDataCollection(ml_repo))
        #setattr(ml_repo, 'models', ModelCollection(ml_repo))
        #setattr(ml_repo, 'reload_tree', MLTree._reload)

    def __create(self):
        self.raw_data = RawDataCollection(self.__ml_repo)
        self.training_data = TrainingDataCollection(self.__ml_repo)
        self.test_data = TestDataCollection(self.__ml_repo)
        self.models = ModelCollection(self.__ml_repo)

    def __init__(self, ml_repo):
        self.__ml_repo = ml_repo
        self.__create()

    def reload(self, **kwargs):
        self.__create()  # todo make this more efficient by just updating collections and items which are affected by this

    def modifications(self):
        result = {}
        result.update(self.raw_data.modifications())
        result.update(self.training_data.modifications())
        result.update(self.test_data.modifications())
        # result.update(self.models.modifications())
        return result

class ModelCompare:
    @staticmethod
    def get_model_differences(ml_repo:MLRepo, model1, version1, version2, data, model2 = None, n_points = 1, y_coordname = None):
        if model2 is None:
            model2 = model1
        if isinstance(data,str):
            data_sets = [data]
        else:
            data_sets = data
        eval_data_model_1 = str(NamingConventions.EvalData(model=model1, data=data))
        eval_data_1 = ml_repo.get(eval_data_model_1, version=None, modifier_versions={model1: version1}, full_object = True)
        eval_data_model_2 = str(NamingConventions.EvalData(model=model2, data=data))
        eval_data_2 = ml_repo.get(eval_data_model_2, version=None, modifier_versions={model2: version2}, full_object = True)
        if y_coordname is None:
            y_coord = 0
        else:
            y_coord = eval_data_1.y_coord_names.index(y_coordname)
        diff = eval_data_1[:,y_coord] - eval_data_2[:,y_coord]
        
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# the followng imports are needed to hash parameters
import hashlib
import json
class ModelAnalyzer:
    def __init__(self, ml_repo):
        self._ml_repo = ml_repo
        self._decision_tree = None
        self.result = {}

    @staticmethod        
    def _compute_local_model_coeffs(x_data, model_eval_function, model, y_coordinate,  n_samples, factor):
        rnd_shape = (n_samples,) + x_data.shape[1:]
        np.random.seed(42)
        rand_values = (2.0-2.0*factor)*np.random.random_sample(rnd_shape)+factor
        y = model_eval_function.create()(model, x_data)
        local_model_coeff = np.empty(x_data.shape)
        mse = np.empty((x_data.shape[0],))
        training_data_x = np.empty(rnd_shape)
        for i in range(x_data.shape[0]):
            for j in range(rand_values.shape[0]):
                training_data_x[j,:] = np.multiply(rand_values[j,:], x_data[i,:])
            training_data_y = model_eval_function.create()(model, training_data_x)[:,y_coordinate]
            reg = LinearRegression().fit(rand_values, training_data_y)
            prediction = reg.predict(rand_values)
            mse[i] = mean_squared_error(training_data_y, prediction)
            local_model_coeff[i,:] = reg.coef_

        return local_model_coeff, mse

    @staticmethod
    def _get_tree_figures(decision_tree, data, mse):
        #first extract tree leafs for statistics (statistics only on leafs)
        leaf_nodes = {}
        num_elements_per_node = [0]*decision_tree.tree_.node_count
        for i in range(len(decision_tree.tree_.children_left)):
            if decision_tree.tree_.children_left[i] == decision_tree.tree_.children_right[i]:
                leaf_nodes[i] = {'mse_max':-1.0, 'mse_min': 20000000000.0, 'mse_mean': 0.0}
        #depp = np.unique(decision_tree.tree_.apply(x.astype(np.float32)))
        tmp = decision_tree.apply(data)

        for i in range(len(tmp)):
            num_elements_per_node[tmp[i]] += 1
        for i in range(mse.shape[0]):
            leaf_nodes[tmp[i]]['mse_max'] = max(leaf_nodes[tmp[i]]['mse_max'], mse[i])
            leaf_nodes[tmp[i]]['mse_min'] = min(leaf_nodes[tmp[i]]['mse_min'], mse[i])
            leaf_nodes[tmp[i]]['mse_mean'] +=  mse[i]
        for k,v in leaf_nodes.items():
            v['num_data_points'] = num_elements_per_node[k]
            v['mse_mean'] /= float(num_elements_per_node[k])
            v['model_coefficients'] = decision_tree.tree_.value[k][:,0].tolist()
        return leaf_nodes, tmp

    
    @staticmethod
    def _create_result(name, model, data, param, result_data, big_data = None):
        result = repo_objects.Result(result_data, big_data)
        result.repo_info.name = name
        param_hash = hashlib.md5(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()
        result.repo_info.modification_info = {model.repo_info.name: model.repo_info.version, 
                    data.repo_info.name: data.repo_info.version, 'param_hash': param_hash}
        return result


    def analyze_local_model(self, model, data_str, n_samples, version = RepoStore.LAST_VERSION, data_version = RepoStore.LAST_VERSION, 
                    y_coordinate=None, start_index = 0, end_index= 100, full_object = True, factor=0.1, 
                    max_depth = 4):
        if y_coordinate is None:
            y_coordinate = 0
        if isinstance(y_coordinate, str):
            raise NotImplementedError()
        param = {'max_depth': max_depth, 'factor': factor, 'n_samples': n_samples, 
            'y_coordinate': y_coordinate, 'start_index': start_index, 'end_index': end_index}
        param_hash = hashlib.md5(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()

        # check if results of analysis are already stored in the repo
        model_ = self._ml_repo.get(model, version, full_object = False)
        data_ = self._ml_repo.get(data_str, data_version, full_object=False)
        result_name = 'analyzer_local_model_' + model_.repo_info.name + '_' + data_.repo_info.name
        result = self._ml_repo.get(result_name, None, 
                modifier_versions={model_.repo_info.name: model_.repo_info.version, data_.repo_info.name: data_.repo_info.version, 'param_hash': param_hash},
                throw_error_not_exist=False, full_object= full_object)
        if result != []:
            if isinstance(result,list):
                return result[0]
            else:
                return result

        model_definition_name = model.split('/')[0]
        model = self._ml_repo.get(model, version, full_object = True)
        model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO][model_definition_name]
        model_definition = self._ml_repo.get(model_definition_name, model_def_version)
        data = self._ml_repo.get(data_str, data_version, full_object=True)
        eval_func = self._ml_repo.get(model_definition.eval_function, RepoStore.LAST_VERSION)
        
            
        data_ = data.x_data[start_index:end_index, :]
        local_model_coeff, mse = ModelAnalyzer._compute_local_model_coeffs(data_, eval_func, model, y_coordinate, n_samples, factor)
        self._decision_tree = DecisionTreeRegressor(max_depth=max_depth)
        self._decision_tree.fit(data_, local_model_coeff)
        
        self.result['node_statistics'], datapoint_to_leaf_node_idx = ModelAnalyzer._get_tree_figures(self._decision_tree, data_, mse)
        # store result in repo
        self.result['parameter'] = param
        self.result['data'] = data_str
        self.result['model'] = model
        self.result['x_coord_names'] = data.x_coord_names
        result = ModelAnalyzer._create_result(result_name, 
            model, data, param, self.result, {'local_model_coeff':local_model_coeff, 'data_to_leaf_index':datapoint_to_leaf_node_idx, 'mse': mse })
        self._ml_repo.add(result)
        return result

    @staticmethod
    def _compute_ice(x_data, model_eval_function, model, 
                        direction, y_coordinate, n_steps = 100, scale = True):
        """Independent conditional expectation plot
        
        Args:
            x_data ([type]): [description]
            model_eval_function ([type]): [description]
            model ([type]): [description]
            direction ([type]): [description]
            n_steps (int, optional): Defaults to 100. [description]
        
        Returns:
            [type]: [description]
        """

        steps = [-1.0 + 2.0*float(x)/float(n_steps-1) for x in range(n_steps) ]
        # compute input for evaluation
        shape = (x_data.shape[0], len(steps),) #x_data.shape[1:]
        _x_data = np.empty(shape= (len(steps), ) +  x_data.shape[1:]) 
        result = np.empty(shape)
        eval_f = model_eval_function.create()
        for i in range(x_data.shape[0]):
            for j in range(len(steps)):
                _x_data[j] = x_data[i] + steps[j]*direction
            y = eval_f(model, _x_data)[:,y_coordinate]
            if scale:
                denom = max(np.linalg.norm(y),1e-10)
                result[i] = y / denom
            else:
                result[i] = y
        return result, steps
    
    def analyze_ice(self, model,  data, direction, version = RepoStore.LAST_VERSION, data_version = RepoStore.LAST_VERSION, 
                    y_coordinate=None, start_index = 0, end_index= 100, full_object = True, n_steps = 20, 
                    n_clusters=20, scale=True, random_state=42, percentile = 90):
        if y_coordinate is None:
            y_coordinate = 0
        if isinstance(y_coordinate, str):
            raise NotImplementedError()
        
        direction_tmp = [x for x in direction] # transform numpy array to list to make it json serializable
        param = {
            'y_coordinate': y_coordinate, 'start_index': start_index, 'end_index': end_index, 
            'n_steps': n_steps,
            'direction': direction_tmp,
            'n_clusters': n_clusters, 
            'scale' : scale, 'random_state': random_state,
            'percentile': percentile}
        param_hash = hashlib.md5(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()
        
        # check if results of analysis are already stored in the repo
        model_ = self._ml_repo.get(model, version, full_object = False)
        data_ = self._ml_repo.get(data, data_version, full_object=False)
        result_name = 'analyzer_ice_' + model_.repo_info.name + '_' + data_.repo_info.name
        result = self._ml_repo.get(result_name, None, 
                modifier_versions={model_.repo_info.name: model_.repo_info.version, 
                                data_.repo_info.name: data_.repo_info.version, 
                                'param_hash': param_hash},
                throw_error_not_exist=False, full_object= full_object)
        if result != []:
            if isinstance(result,list):
                return result[0]
            else:
                return result

        model_definition_name = model.split('/')[0]
        model = self._ml_repo.get(model, version, full_object = True)
        model_def_version = model.repo_info[RepoInfoKey.MODIFICATION_INFO][model_definition_name]
        model_definition = self._ml_repo.get(model_definition_name, model_def_version)
        data = self._ml_repo.get(data, data_version, full_object=True)
        
        eval_func = self._ml_repo.get(model_definition.eval_function, RepoStore.LAST_VERSION)
                    
        data_ = data.x_data[start_index:end_index, :]
        
        x, steps =  ModelAnalyzer._compute_ice(data_, eval_func, model, direction, y_coordinate, n_steps)
        big_obj = {}
        big_obj['ice'] = x
        big_obj['steps'] = np.array(steps)
        # now apply a clustering algorithm to search for good representations of all ice results
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=42)
        labels =  k_means.fit_predict(x)
        big_obj['labels'] = labels
        big_obj['cluster_centers'] = k_means.cluster_centers_
        # now store for each data point the distance to the respectiv cluster center
        tmp = k_means.transform(x)
        distance_to_center = np.empty((x.shape[0],))

        for i in range(x.shape[0]):
            distance_to_center[i] = tmp[i,labels[i]]
        big_obj['distance_to_center'] = distance_to_center
        #perc = np.percentile(distance_to_center, percentile)
        #percentile_ice = np.extract(distance_to_center > perc, )
        #for i in range(distance_to_center.shape[0]):
        #    if distance_to_center[i] > perc:
        #        percentile_ice.append(distance_to_center[i])
        #big_obj['percentiles'] = percentile_ice
        result = ModelAnalyzer._create_result(result_name, model, data, param, {'param': param, 'data': data.repo_info.name, 'model': model.repo_info.name}, big_obj)
        self._ml_repo.add(result)
        return result

    

