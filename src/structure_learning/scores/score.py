"""

"""
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
from structure_learning.data_structures import Graph
from structure_learning.data import Data

class Score(ABC):
    """
    Base class for graph scores for structure learning using MCMC.
    Inheriting classes must implement the following methods:
        compute() -> dict
        compute_node() -> dict
    """
    def __init__(self, data : Union[Data, pd.DataFrame]):
        """
        Initialises the Score abstract class.
        All classes that inherit from this class must implement the compute method.

        Parameters:
            data (pd.DataFrame):            dataset
            graph (nx.DiGraph, optional):   graph structure.
                                            Defaults to None. The graph must be a DAG.
        """
        self._data = data if isinstance(data, Data) else Data(values=data)
        self._node_labels = list(data.columns)

    # abstract method to be implemented by subclasses
    @abstractmethod
    def compute(self, graph: Graph):
        """
        Implements a score function (e.g. BGe, Marginal Likelihood, etc)
        """
        pass

    def compute_node(self, graph: Graph, node: str):
        """
        Implements a score function (e.g. BGe, Marginal Likelihood, etc) for a specific node
        """
        parentnodes = [i for i in graph.find_parents(node)]
        return self.compute_node_with_edges(node, parentnodes, graph._node_to_index_dict)

    @abstractmethod
    def compute_node_with_edges(self, node : str, parents: list = None, node_index_map: dict = None):
        """
        Implements a score function (e.g. BGe, Marginal Likelihood, etc) for a specific node and parents
        """
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d

    @property
    def node_labels(self):
        return self._node_labels
