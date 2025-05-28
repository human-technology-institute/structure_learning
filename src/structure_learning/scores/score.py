"""

"""
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from structure_learning.data_structures import Graph
from structure_learning.utils.graph_utils import node_label_to_index, find_parents

class Score(ABC):
    """
    Base class for graph scores for structure learning using MCMC.
    Inheriting classes must implement the following methods:
        compute() -> dict
        compute_node() -> dict
    """
    def __init__(self, data : pd.DataFrame, graph : Union[np.array, Graph],\
                 to_string : str, is_log_space = True):
        """
        Initialises the Score abstract class.
        All classes that inherit from this class must implement the compute method.

        Parameters:
            data (pd.DataFrame):            dataset
            graph (nx.DiGraph, optional):   graph structure.
                                            Defaults to None. The graph must be a DAG.
        """
        self._data = data
        self._node_labels = list(data.columns)
        self.graph = graph if isinstance(graph, Graph) else Graph(incidence=graph, nodes=self._node_labels)
        self._node_label_to_index = {} 
        if self.graph.incidence is not None:
            self._node_label_to_index = self.graph._node_to_index_dict()
        self._is_log_space = is_log_space
        self._to_string = to_string

    # abstract method to be implemented by subclasses
    @abstractmethod
    def compute(self):
        """
        Implements a score function (e.g. BGe, Marginal Likelihood, etc)
        """
        pass

    def compute_node(self, node : str):
        """
        Implements a score function (e.g. BGe, Marginal Likelihood, etc) for a specific node
        """
        parentnodes = [i for i in self.graph.find_parents(node)]
        return self.compute_node_with_edges(node, parentnodes)

    @abstractmethod
    def compute_node_with_edges(self, node : str, parents: list = None):
        """
        Implements a score function (e.g. BGe, Marginal Likelihood, etc) for a specific node and parents
        """
        pass

    def __str__(self):
        """
        Return a string representation of this score instance
        """
        return self._to_string

    @property
    def to_string(self):
        return self._to_string

    @to_string.setter
    def to_string(self, string):
        self._to_string = string

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d

    @property
    def is_log_space(self):
        return self._is_log_space

    @is_log_space.setter
    def is_log_space(self, log_space: bool):
        self._is_log_space = log_space

    @property
    def node_labels(self):
        return self._node_labels

    @property
    def node_label_to_index(self):
        """
        Return mapping of node labels to indices
        """
        if self._node_label_to_index is None or len(self._node_label_to_index)==0:
            self.graph._update_node_index()
            self._node_label_to_index = self.graph._node_to_index_dict
        return self._node_label_to_index
