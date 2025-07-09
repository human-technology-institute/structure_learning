"""
This module implements the BGe (Bayesian Gaussian Equivalent) score for evaluating Bayesian networks.

The BGe score is used to compute the marginal likelihood of a Bayesian network given data. It supports operations
such as computing the score for the entire graph, individual nodes, and edges. The implementation includes
parameters for regularization and scoring.

Classes:
    BGeScore: Implements the BGe score computation.
"""

from typing import Union
import pandas as pd
import numpy as np
from scipy.special import loggamma as lgamma
from structure_learning.scores import Score
from structure_learning.data_structures import Graph
from structure_learning.data import Data

class BGeScore(Score):
    """
    BGe (Bayesian Gaussian Equivalent) Score
    """
    def __init__(self, data : Union[Data, pd.DataFrame]):
        """
        Initialise BGe instance.

        Parameters:
            data (Data | pandas.DataFrame): data
        """
        super().__init__(data)

        self._num_cols = data.shape[1] # number of variables
        self._num_obvs = data.shape[0] # number of observations
        self._mu0 = np.zeros(self._num_cols)

        # Scoring parameters.
        self._am = 1
        self._aw = self._num_cols + self._am + 1
        T0scale = self._am * (self._aw - self._num_cols - 1) / (self._am + 1)
        self._T0 = T0scale * np.eye(self._num_cols)
        self._TN = (
            self._T0 + (self._num_obvs - 1) * np.cov(data.values.T) + ((self._am * self._num_obvs) / (self._am + self._num_obvs))
            * np.outer(
                (self._mu0 - np.mean(data.values, axis=0)), (self._mu0 - np.mean(data.values, axis=0))
            )
        )

        self._awpN = self._aw + self._num_obvs
        self._constscorefact = - (self._num_obvs / 2) * np.log(np.pi) + 0.5 * np.log(self._am / (self._am + self._num_obvs))
        self._scoreconstvec = np.zeros(self._num_cols)
        for i in range(self._num_cols):
            awp = self._aw - self._num_cols + i + 1
            self._scoreconstvec[i] = (
                self._constscorefact
                - lgamma(awp / 2)
                + lgamma((awp + self._num_obvs) / 2)
                + (awp + i) / 2 * np.log(T0scale)
            )

        self._t = T0scale
        self._parameters = {}
        self._reg_coefficients = {}

    def compute(self, graph: Graph):
        """
        Compute the BGE for the data

        Returns:
            (dict): score and parameters
        """
        if Graph.has_cycle(graph):
            return {'score': -np.inf}
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node

        # Loop through each node in the graph
        for node in self.node_labels:

            log_ml_node = self.compute_node(graph, node)['score']

            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'parents': graph.find_parents(node)
            }

            total_log_ml += log_ml_node

        # save the parameters
        self._parameters = parameters

        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score

    def compute_node_with_edges(self, node : str, parents: list, node_index_map: dict):
        """
        Compute the BGE for edge(s)

        Parameter:
            node (str): node label
            parents (list (str)): node labels of parent nodes

        Returns:
            (dict): score and parameters
        """
        parameters = {}  # Dictionary to store the parameters for each node
        node_indx = node_index_map[node]
        parentnodes = [node_index_map[p] for p in parents] # get index of parents labels
        num_parents = len(parentnodes) # number of parents

        awpNd2 = (self._awpN - self._num_cols + num_parents + 1) / 2

        A = self._TN[node_indx, node_indx]

        if num_parents == 0:  # just a single term if no parents
            corescore = self._scoreconstvec[num_parents] - awpNd2 * np.log(A)
        else:
            D = self._TN[np.ix_(parentnodes, parentnodes)]
            choltemp = np.linalg.cholesky(D)
            logdetD = 2 * np.sum(np.log(np.diag(choltemp)))

            B = self._TN[np.ix_([node_indx], parentnodes)]
            logdetpart2 = np.log( A - np.sum(np.linalg.solve(choltemp, B.T)**2) )
            corescore = self._scoreconstvec[num_parents] - awpNd2 * logdetpart2 - logdetD / 2

        # Save the parameters for the node
        parameters[node] = {
            'parents': parentnodes
        }
        self._parameters = parameters

        score = {
            'score': corescore,
            'parameters': parameters
        }

        return score

    @property
    def am(self):
        return self._am

    @am.setter
    def am(self, n):
        self._am = n

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        self._parameters = params

    @property
    def reg_coefficients(self):
        return self._reg_coefficients

    @reg_coefficients.setter
    def reg_coefficients(self, coefficients):
        self._reg_coefficients = coefficients
