"""

"""
from abc import ABC, abstractmethod
from typing import Union
import networkx as nx
import numpy as np

class StructureLearningProposal(ABC):
    """
    Base class for proposal classes for structure learning using MCMC.
    All inheriting classes must implement the following methods:
        propose_DAG() -> graph : numpy.ndarray, operation : str
        prob_Gcurr_Gprop_f() -> float
        prob_Gcurr_Gprop_f() -> float
    """
    def __init__(self, graph : Union[np.ndarray, nx.DiGraph], blacklist = None, whitelist = None):
        """
        Initialise StructureLearningProposal instance.

        Parameters:
            graph (networkx.DiGraph): graph
            blacklist (numpy.ndarray): mask for edges to ignore in the proposal
            whitelist (numpy.ndarray): mask for edges to include in the proposal
        """
        self._G_curr = graph.copy()
        self._G_prop = None

        self.node_labels = graph.nodes() if isinstance(graph, nx.DiGraph) else [str(i) for i in range(graph.shape[1])]
        self.num_nodes = len(self.node_labels)

        if blacklist is None:
            self._blacklist = np.zeros((self.num_nodes, self.num_nodes))
        else:
            self._blacklist = blacklist

        if whitelist is None:
            self._whitelist = np.zeros((self.num_nodes, self.num_nodes))
        else:
            self._whitelist = whitelist

        self._operation = None

    @abstractmethod
    def propose_DAG(self):
        """
        Propose a DAG
        """
        pass

    @abstractmethod
    def prob_Gcurr_Gprop_f(self):
        """
        Compute transition probability Q(G_curr|G_prop)
        """
        pass

    @abstractmethod
    def prob_Gprop_Gcurr_f(self):
        """
        Compute transition probability Q(G_prop|G_curr)
        """
        pass

    @property
    def current_graph(self):
        return self._G_curr

    @current_graph.setter
    def current_graph(self, graph):
        """
        Set current graph.

        Parameter:
            graph (numpy.ndarray): current graph
        """
        self._G_curr = graph

    @property
    def proposed_graph(self):
        return self._G_prop

    @proposed_graph.setter
    def proposed_graph(self, graph):
        """
        Set proposed graph.

        Parameter:
            graph (numpy.ndarray): proposed graph
        """
        self._G_prop = graph

    @property
    def whitelist(self):
        return self._whitelist

    @whitelist.setter
    def whitelist(self, arcs):
        """
        Set whitelist edges.

        Parameter:
            arcs (numpy.ndarray): whitelist edges
        """
        self._whitelist = arcs

    @property
    def blacklist(self):
        return self._blacklist

    @blacklist.setter
    def blacklist(self, arcs):
        """
        Set whitelist edges.

        Parameter:
            arcs (numpy.ndarray): blacklist edges
        """
        self._blacklist = arcs

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, op):
        """
        Set operation for proposal.

        Parameter:
            op (str): operation
        """
        self._operation = op
