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
from scipy.linalg import solve_triangular

from structure_learning.scores import Score
from structure_learning.data_structures import Graph
from structure_learning.data import Data


class BGeScore(Score):
    """
    BGe (Bayesian Gaussian Equivalent) Score
    """

    def __init__(self, data: Union[Data, pd.DataFrame]):
        """
        Initialise BGe instance.

        Parameters:
            data (Data | pandas.DataFrame): data
        """
        super().__init__(data)

        self._num_cols = data.shape[1]  # number of variables
        self._num_obvs = data.shape[0]  # number of observations
        self._mu0 = np.zeros(self._num_cols)

        # Prior parameters.
        self._am = 1
        self._aw = self._num_cols + self._am + 1
        T0scale = self._am * (self._aw - self._num_cols - 1) / (self._am + 1)

        self._T0 = T0scale * np.eye(self._num_cols)

        # Summary statistics for the data.
        self._S = (self._num_obvs - 1) * np.cov(data.values.T)
        self._xbar = np.mean(data.values, axis=0)

        self._awpN = self._aw + self._num_obvs
        self._amN = self._am + self._num_obvs
        self._TN = (
                self._T0 + self._S + ((self._am * self._num_obvs) / (self._amN))
                * np.outer(
            (self._mu0 - self._xbar), (self._mu0 - self._xbar)
        )
        )
        self._muN = (self._am * self._mu0 + self._num_obvs * self._xbar) / (self._amN)

        # Helpers for computing the score.
        self._constscorefact = - (self._num_obvs / 2) * np.log(np.pi) + 0.5 * np.log(
            self._am / (self._am + self._num_obvs))
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

            node_res = self.compute_node(graph, node)
            log_ml_node = node_res['score']

            parameters = node_res['parameters'] # includes"node_idx", 'parents', "posterior"

            # Save the parameters for the node
            parameters[node] = {
                'score': log_ml_node,
                'parents': parameters['parents'] # graph.find_parents(node)
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

    def compute_node_with_edges(self, node: str, parents: list,
                                node_index_map: dict,
                                compute_full_posterior: bool = False
                                ):
        """
        Compute the BGE for edge(s)

        Parameter:
            node (str): node label
            parents (list (str)): node labels of parent nodes
            node_index_map (dict): mapping of node labels to indices
            compute_full_posterior (bool): whether to compute the full posterior distribution

        Returns:
            (dict): score and parameters
             If full_posterior=True, also return the quantities required  for the full posterior distribution of the regression
    coefficients.
        """
        parameters = {}  # Dictionary to store the parameters for each node
        node_indx = node_index_map[node]
        parentnodes = [node_index_map[p] for p in parents]  # get index of parents labels
        num_parents = len(parentnodes)  # number of parents

        # Effective local posterior shape - this is the shape of the posterior distribution for the node given its parents
        awpNd2 = (self._awpN - self._num_cols + num_parents + 1) / 2

        A = self._TN[node_indx, node_indx]

        posterior = None

        if num_parents == 0:  # just a single term if no parents
            corescore = self._scoreconstvec[num_parents] - awpNd2 * np.log(A)

            logdetD = 0.

            conditional_T = A

            if compute_full_posterior:
                # Compute the posterior NIW distribution parameters - inverse gamma dist for  variance, and normal for beta and intcept
                posterior = {
                    'sigma2_shape': awpNd2,
                    'sigma2_scale': conditional_T / 2.0,
                    # beta | sigma^2 parameters
                    'beta_mean': np.zeros(0),
                    'chol_parent': None
                }
        else:
            B = self._TN[np.ix_([node_indx], parentnodes)]
            D = self._TN[np.ix_(parentnodes, parentnodes)]

            # Inverting parent scatter matrix using Cholesky decomposition for numerical stability
            cholD = np.linalg.cholesky(D)
            logdetD = 2. * np.sum(np.log(np.diag(cholD)))

            # T_N[P, node_indx]

            # Computing
            chol_solve_B = solve_triangular(cholD, B.T, lower=True)

            # conditional_T is the Schur complement of the parent block in of node i
            # Effectively, this is the residual scatter after accounting for the parents.
            conditional_T = A - np.sum(chol_solve_B ** 2)

            corescore = (
                    self._scoreconstvec[num_parents]
                    - awpNd2 * np.log(conditional_T)
                    - 0.5 * logdetD
            )

            if compute_full_posterior:
                # Solve for beta_mena using Cholesky decomposition
                beta_mean = solve_triangular(cholD.T, chol_solve_B, lower=False).flatten()

                posterior = {
                    "sigma2_shape": awpNd2,
                    "sigma2_scale": conditional_T / 2.0,

                    "beta_mean": beta_mean,
                    "chol_parent_T": cholD,

                }

        # Save the parameters for the node
        parameters = {
            "node_idx": node_indx,
            'parents': parentnodes,
            "posterior": posterior
        }

        score = {
            'score': corescore,
            'parameters': parameters
        }

        return score

    def posterior_mode(self, node: str, parents: list,
                                  node_index_map: dict, node_parameters: dict = None):
        """
        Compute the posterior mode of the regression coefficients for a given node and its parents.
        Use pre-computed node_parameters if available, otherwise compute them.
        """
        param = node_parameters
        if param is None:
            param = self.compute_node_with_edges(node, parents, node_index_map, compute_full_posterior=True)[
                'parameters']

        beta_map = param['posterior']['beta_mean']
        sigma2_map = (param['posterior']["sigma2_scale"] / (param['posterior']["sigma2_shape"] + 1.0)
                      # mode of IG distribution
                      )
        intercept_map = self._muN[param['node_idx']] - beta_map @ self._muN[param['parents']]

        return {"beta": beta_map, "sigma2": sigma2_map, "intercept": intercept_map}

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
