"""

"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from structure_learning.utils.graph_utils import node_label_to_index, find_parents

class Score(ABC):
    """
    Base class for graph scores for structure learning using MCMC.
    Inheriting classes must implement the following methods:
        compute() -> dict
        compute_node() -> dict
    """
    def __init__(self, data : pd.DataFrame, incidence : np.array,\
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
        self._adj_matrix = incidence
        self._node_label_to_index = node_label_to_index(self._node_labels)
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
        node_indx = self.node_label_to_index[node]
        parentnodes = [self.node_labels[i] for i in find_parents(self.incidence, node_indx)]
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
    def adj_matrix(self):
        """
        Adjacency matrix. adj_matrix, incidence and graph all refer to the same numpy.ndarray.
        """
        return self._adj_matrix

    @adj_matrix.setter
    def adj_matrix(self, mat):
        """
        Set adjacency matrix for scoring.
        """
        self._adj_matrix = mat

    @property
    def incidence(self):
        """
        Adjacency matrix. adj_matrix, incidence and graph all refer to the same numpy.ndarray.
        """
        return self._adj_matrix

    @incidence.setter
    def incidence(self, mat):
        """
        Set adjacency matrix for scoring.
        """
        self._adj_matrix = mat

    @property
    def graph(self):
        """
        Adjacency matrix. adj_matrix, incidence and graph all refer to the same numpy.ndarray.
        """
        return self._adj_matrix

    @graph.setter
    def graph(self, mat):
        """
        Set adjacency matrix for scoring.
        """
        self._adj_matrix = mat

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
        return self._node_label_to_index
