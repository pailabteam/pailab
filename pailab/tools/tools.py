import warnings
import logging
import collections
import numpy as np
from deepdiff import DeepDiff
from pailab.ml_repo.repo import MLObjectType, MLRepo, NamingConventions
from pailab.ml_repo.repo_objects import RepoInfoKey, DataSet  # pylint: disable=E0401
from pailab.ml_repo.repo_store import RepoStore  # pylint: disable=E0401
import pailab.ml_repo.repo_store as repo_store
import pailab.ml_repo.repo_objects as repo_objects
logger = logging.getLogger(__name__)

#region collections and items

   
def get_test_summary(repo:MLRepo):
    """Return test summary for all labeled models and all latest models
    
    Args:
        repo (MLRepo): repo
    """
    pass



# the followng imports are needed to hash parameters
import hashlib
import json
import functools

#region caching

class _CachedResults(repo_objects.RepoObject):
    """Object to store results of functions for caching reasons in the repo.
    
    """

    def __init__(self,repo_info, x):
        """Object to store results of functions for caching reasons in the repo.
        
        Args:
            repo_info (dict or RepoInfo object): repo info
            x (objects or tuple of objects): return values of a function to be cached
        """

        super(_CachedResults, self).__init__(repo_info)
        if isinstance(x,tuple):
            self._tuple = []
            for i in range(len(x)):
                self._add_object(x[i], i)
        else:
            if hasattr(x, 'repo_info'):
                self.repo_x = x
            else:
                self.x = x

    def _add_object(self, obj, counter):
        # repo objects are added separately to the repo
        if hasattr(obj, 'repo_info'):
            name = 'repo_obj_' + str(counter)
            setattr(self, name, '')
        else:
            name = 'obj_' + str(counter)
        setattr(self, name, obj)
        self._tuple.append(name)
        if isinstance(obj, np.ndarray):
            self.repo_info[RepoInfoKey.BIG_OBJECTS].append(name)
        
    def _add_to_repo(self, ml_repo):
        if hasattr(self, '_tuple'):
            for name in self._tuple:
                if name.startswith('repo'):
                    obj = getattr(self, name)
                    ml_repo.add(obj)
                    delattr(self, name)
                    setattr(self, name +'_version', obj.repo_info.version)
                    setattr(self, name + '_name', obj.repo_info.name)
        if hasattr(self,'repo_x'):
            ml_repo.add(self.repo_x)
            delattr('repo_x')
            setattr(self,'repo_x_name', self.repo_x.repo_info.name)
            setattr(self,'repo_x_version', self.repo_x.repo_info.version)
        ml_repo.add(self)
            
    def _fill_from_repo(self, ml_repo):
        if hasattr(self,'repo_x_name'):
            obj = ml_repo.get(self.repo_x_name, self.repo_x_version)
            setattr(self, 'repo_x', obj)
            
        if hasattr(self, '_tuple'):
            for name in self._tuple:
                if name.startswith('repo'):
                    obj = ml_repo.get(getattr(self, name + '_name'), getattr(self, name + '_version'))
                    setattr(self, name, obj)
            
    def _get(self):
        if hasattr(self,'_tuple'):
            tmp = [getattr(self, name) for name in self._tuple]
            return tuple(tmp)
        if hasattr(self,'repo_x'):
            return self.repo_x
        return self.x


def cache_f(f, ml_repo, *args, **kwargs):
    """Cache results of a function in the repo.

    Caches results of a function in the repo. Here, it uses a hashkey generated as MD5 sum from the function
    arguments that are not RepoObjects (serializing them as json). For RepoObjects it uses their version number to search if the function 
    has already been executed with the given parameters. It stores the results in the repo as a CachedResult object where RepoObjects 
    contained in the CachedResults are stored separately in the repo.
    
    Args:
        f (function): function to be cached
        ml_repo (MLRepo): repository used to store the results
    
    Returns:
        the results of function f
    """

    tmp = []
    modification_info = {}
    for arg in args:
        if hasattr(arg, 'repo_info'):
            modification_info[arg.repo_info.name] = arg.repo_info.version
        else:
            tmp.append(arg)
    param = {}
    param['args'] = tmp
    param['kwargs'] ={}
    for k,v in kwargs.items():
        if hasattr(v, 'repo_info'):
            modification_info[v.repo_info.name: v.repo_info.version]
        else:
            param['kwargs'][k] = v
            
    param_hash = hashlib.md5(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()
    modification_info['param_hash'] = param_hash
    hash_result_name = f.__name__
    result = ml_repo.get(hash_result_name, version = None, modifier_versions=modification_info, full_object=True, throw_error_not_exist = False)
    if result == []:
        x = f(*args, **kwargs)
        hash_results =  _CachedResults({RepoInfoKey.NAME: hash_result_name, RepoInfoKey.MODIFICATION_INFO: modification_info, RepoInfoKey.CATEGORY: MLObjectType.CACHED_VALUE}, x)
        hash_results._add_to_repo(ml_repo)
        return x
    else:
        result._fill_from_repo(ml_repo)
        return result._get()
    
def ml_cache(f):
    @functools.wraps(f)
    def wrapper(*argv, **kwargs):
        if 'cache' in kwargs.keys():
            ml_repo = kwargs['cache']
            if ml_repo is None:
                return f(*argv, **kwargs)    
            del kwargs['cache']
            return cache_f(f, ml_repo, *argv, **kwargs)
        else:
            return f(*argv, **kwargs)
    return wrapper

#endregion



def get_model_measure_list(ml_repo, measure, data, data_version = RepoStore.LAST_VERSION):
    """[summary]
    
    Args:
        ml_repo ([type]): [description]
        data ([type]): [description]
    """
    def _get_model_info(mod_info):
        for k,v in mod_info.items():
            if k.endswith('/model'):
                return k,v

    tmp = ml_repo.get_names(MLObjectType.MEASURE)
    result = []
    missing_measure = False
    for n in tmp:
        if data in n and measure in n:
            candidates = ml_repo.get(n, version = None, modifier_versions = {data: data_version}, throw_error_not_exist=False)
            if not isinstance(candidates, list):
                candidates = [candidates]
            if candidates == []:
                logger.warning('Missing measure ' + n + ' for data version: ' + data_version)
                missing_measure = True
            for c in candidates:
                model_name, model_version = _get_model_info(c.repo_info.modification_info)
                result.append({'model': model_name, 'version': model_version, measure + ', ' + data: c.value})
    warnings.warn('There were measures missing for the given data version, see log for details.')
    return result
       
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

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
        logger.debug('Create md5 hash for parameters.')
        param_hash = hashlib.md5(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()
        logger.debug('Add md5 hash for parameters to modification_info of results.')
        result.repo_info.modification_info = {model.repo_info.name: model.repo_info.version, 
                    data.repo_info.name: data.repo_info.version, 'param_hash': param_hash}
        return result


    def analyze_local_model(self, model, data_str, n_samples, version = RepoStore.LAST_VERSION, data_version = RepoStore.LAST_VERSION, 
                    y_coordinate=None, start_index = 0, end_index= 100, full_object = True, factor=0.1, 
                    max_depth = 4):
        logger.info('Start analyzing local model.')
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
        model_name = model
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
        self.result['model'] = model_name
        self.result['x_coord_names'] = data.x_coord_names
        result = ModelAnalyzer._create_result(result_name, 
            model, data, param, self.result, {'local_model_coeff':local_model_coeff, 'data_to_leaf_index':datapoint_to_leaf_node_idx, 'mse': mse })
        self._ml_repo.add(result)
        return result

   
    

