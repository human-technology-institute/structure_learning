"""

"""
from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
import numpy as np
from mcmc.scores import Score

class MCMC(ABC):
    """
    Base class for MCMC.
    Inheriting classes must implement the following methods:
        run()
    """
    def __init__(self, data, initial_graph, max_iter, score_object, proposal_object):
        """
        Initialise MCMC object.

        Parameters:
            data (pandas.DataFrame): data
            initial_graph (numpy.ndarray): initial graph
            max_iter (int): number of iterations to run
            score_object (Score): a Score object
            proposal_object (Proposal): a Proposal object
        """
        self._data  = data
        self._node_labels = list(data.columns)
        self._num_nodes = len(self._node_labels)

        self._proposal_object = proposal_object
        self._score_object = score_object
        self._max_iter = max_iter

        if initial_graph is None:
            self._initial_graph = np.zeros((self._num_nodes, self._num_nodes))
        else:
            self._initial_graph = initial_graph

    @abstractmethod
    def run(self):
        """
        Run MCMC simulation.
        """
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d : pd.DataFrame):
        """
        Set data for MCMC simulation.

        Parameter:
            d (pandas.DataFrame): data
        """
        self._data = d

    @property
    def score_object(self):
        return self._score_object

    @score_object.setter
    def score_object(self, score : Score):
        """
        Set score object to use for MCMC simulation.

        Parameter:
            score (mcmc.scores.Score): score object
        """
        self._score_object = score

    @property
    def proposal_object(self):
        return self._proposal_object

    @proposal_object.setter
    def proposal_object(self, proposal):
        """
        Set proposal object to use for MCMC simulation.

        Parameter:
            proposal (mcmc.proposals.StructureLearningProposal): proposal object
        """
        self._proposal_object = proposal

    @property
    def initial_graph(self):
        return self._initial_graph

    @initial_graph.setter
    def initial_graph(self, graph : np.ndarray):
        """
        Set graph to use for MCMC simulation.

        Parameter:
            graph (numpy.npdarray): initial graph
        """
        self._initial_graph = graph

    @property
    def num_nodes(self):
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, n_nodes : int):
        """
        Set number of nodes.

        Parameter:
            n_nodes (int): number of nodes
        """
        self._num_nodes = n_nodes

    @property
    def node_labels(self):
        return self._node_labels

    @node_labels.setter
    def node_labels(self, labels : list):
        """
        Set node labels.

        Parameter:
            labels (list (str)): node labels
        """
        self._node_labels = labels

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, it: int):
        """
        Set number of iterations for MCMC simulation.

        Parameter:
            it (int): number of MCMC iterations
        """
        self._max_iter = it
