import logging
from graphviz import Digraph

from numpy import load
from deepdiff import DeepDiff
from pailab.ml_repo.repo import MLObjectType, MLRepo, NamingConventions
from pailab.ml_repo.repo_objects import RepoInfoKey  # pylint: disable=E0401
import pailab.ml_repo.repo_store as repo_store
import pailab.ml_repo.repo_objects as repo_objects
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, name):
        self.name = name
        self.depends_on = []
        self.modifies = []

    def add_dependence_on(self, node):
        if not node.name in self.depends_on:
            self.depends_on.append(node.name)

    def add_modifies(self, node):
        if not node.name in self.modifies:
            self.modifies.append(node.name)


class DependencyGraph:
    def __init__(self):
        self._nodes = {}

    def _add_node_if_not_exist(self, node):
        if not node in self._nodes:
            self._nodes[node] = Node(node)

    def add_dependency(self, node1, node2):
        self._add_node_if_not_exist(node1)
        self._add_node_if_not_exist(node2)
        self._nodes[node1].add_dependence_on(self._nodes[node2])
        self._nodes[node2].add_modifies(self._nodes[node1])

    def add_node(self, name):
        n = Node(name)
        self._nodes[name] = n

    def del_node(self, name):
        node = self._nodes[name]
        for n in node.modifies:
            self._nodes[n].depends_on.remove(name)
        for n in node.depends_on:
            self._nodes[n].modifies.remove(name)
        del self._nodes[name]


def get_dependency_graph(ml_repo, node='', category_to_color=None, ignore_categories=None):
    """Returns graph visualization from graphviz of th dependences around an object.

    This method creates a graph showing the dependencies between repo objects w.r.t. a special node or, if no node is specified, to all
    objects. To use this method, you must have installed graphviz.

    
    Args:
        ml_repo (MLRepo): repository
        node (str, optional): Name of object whose dependenies will be drawn. Defaults to '' which means that all dependencies are considered.
        category_to_color (map, optional): Map defining for each MLObject category the color used in the graph. Defaults to None which leads to a default color map.
        ignore_categories (iterable of M;ObjectType, optional): All MLObjectTypes defined over the iterable are ignored in the graph. Defaults to None.
       
    Returns:
        graphviz.Digraph: The resultign dependency structure.
    """
    def get_remaining_nodes(remaining_nodes, node, dependency_graph, modifiers=True, depends_on=True):
        remaining_nodes.append(node)
        tmp = dependency_graph._nodes[node]
        if modifiers:
            for n in tmp.modifies:
                if n not in remaining_nodes:
                    get_remaining_nodes(
                        remaining_nodes, n, dependency_graph, modifiers=True, depends_on=False)
        if depends_on:
            for n in tmp.depends_on:
                if n not in remaining_nodes:
                    get_remaining_nodes(
                        remaining_nodes, n, dependency_graph, modifiers=False, depends_on=True)

    def get_default_category_to_color():
        category_to_color = {}
        for k in MLObjectType:
            category_to_color[k] = 'white'
        category_to_color[MLObjectType.CALIBRATED_MODEL] = 'wheat'
        category_to_color[MLObjectType.MODEL] = 'wheat'
        category_to_color[MLObjectType.EVAL_DATA] = 'palegreen2'
        category_to_color[MLObjectType.TRAINING_DATA] = 'palegreen2'
        category_to_color[MLObjectType.TEST_DATA] = 'palegreen2'
        category_to_color[MLObjectType.RAW_DATA] = 'palegreen2'
        category_to_color[MLObjectType.TRAINING_FUNCTION] = 'lightgrey'
        category_to_color[MLObjectType.MODEL_EVAL_FUNCTION] = 'lightgrey'
        return category_to_color

    if ignore_categories is None:
        ignore_categories = [
            MLObjectType.LABEL, MLObjectType.COMMIT_INFO, MLObjectType.MAPPING, MLObjectType.JOB]
    dependency_graph = DependencyGraph()
    name_to_category = {}
    for k in MLObjectType:
        if k in ignore_categories:
            continue
        names = ml_repo.get_names(k.value)
        for n in names:
            name_to_category[n] = k
            obj = ml_repo.get(n)
            for l, v in obj.repo_info.modification_info.items():
                obj2 = ml_repo.get(l)
                dependency_graph.add_dependency(n, l)

    if node is not '':  # remove all nodes that are not connected to
        remaining_nodes = []
        all_nodes = [n for n in dependency_graph._nodes.keys()]
        get_remaining_nodes(remaining_nodes, node, dependency_graph)
        for k in all_nodes:
            if k not in remaining_nodes:
                dependency_graph.del_node(k)
    if category_to_color is None:
        category_to_color = get_default_category_to_color()

    dgr = Digraph(comment='dependency graph')
    for key in dependency_graph._nodes.keys():
        label_name = key+"\\n"+str(name_to_category[key].value)
        if key == node:
            dgr.node(key, label=label_name, color='red',
                     fillcolor='red', style='filled')
        else:
            dgr.node(key, label=label_name,
                     fillcolor=category_to_color[name_to_category[key]], style='filled')
    for key, node in dependency_graph._nodes.items():
        for d in node.modifies:
            dgr.edge(key, d)
    return dgr
