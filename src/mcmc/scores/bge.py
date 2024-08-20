"""

"""
import pandas as pd
import numpy as np
from scipy.special import loggamma as lgamma
from mcmc.scores import Score
from mcmc.utils.graph_utils import find_parents

class BGeScore(Score):
    """
    BGe (Bayesian Gaussian Equivalent) Score
    """
    def __init__(self, data : pd.DataFrame, incidence : np.ndarray, is_log_space = True):
        """
        Initialise BGe instance.

        Parameters:
            data (pandas.DataFrame): data
            incidence (numpy.nparray): graph adjacency matrix
        """
        super().__init__(data, incidence, "BGe Score", is_log_space)

        self._num_cols = data.shape[1] # number of variables
        self._num_obvs = data.shape[0] # number of observations
        self._mu0 = np.zeros(self._num_cols)

        # Scoring parameters.
        self._am = 1
        self._aw = self._num_cols + self._am + 1
        T0scale = self._am * (self._aw - self._num_cols - 1) / (self._am + 1)
        self._T0 = T0scale * np.eye(self._num_cols)
        self._TN = (
            self._T0 + (self._num_obvs - 1) * np.cov(data.T) + ((self._am * self._num_obvs) / (self._am + self._num_obvs))
            * np.outer(
                (self._mu0 - np.mean(data, axis=0)), (self._mu0 - np.mean(data, axis=0))
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

        self._is_log_space = is_log_space
        self._t = T0scale
        self._parameters = {}
        self._reg_coefficients = {}

    def compute(self):
        """
        Compute the BGE for the data

        Returns:
            (dict): score and parameters
        """
        return self.compute_BGe_with_graph()

    def compute_node(self, node : str):
        """
        Compute the BGE for a node

        Parameter:
            node (str): node label

        Returns:
            (dict): score and parameters
        """
        return self.compute_BGe_with_node( node )

    def compute_BGe_with_graph(self):
        """
        Compute the BGE for the data

        Returns:
            (dict): score and parameters
        """
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node

        # Loop through each node in the graph
        for node in self.node_labels:

            log_ml_node = self.compute_BGe_with_node( node )['score']

            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'parents': find_parents(self.incidence, self.node_label_to_index[node])
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

    def compute_BGe_with_node(self, node : str):
        """
        Compute the BGE for a node

        Parameter:
            node (str): node label

        Returns:
            (dict): score and parameters
        """
        parameters = {}  # Dictionary to store the parameters for each node

        node_indx = self.node_label_to_index[node]
        parentnodes = find_parents(self.incidence, node_indx)
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

    def compute_local(self, node : str, parents: list):
        """
        Compute the BGE for edge(s)

        Parameter:
            node (str): node label
            parents (list (str)): node labels of parent nodes

        Returns:
            (dict): score and parameters
        """
        parameters = {}  # Dictionary to store the parameters for each node

        node_indx = self.node_label_to_index[node]
        parentnodes = [self.node_label_to_index[i]  for i in parents] # get index of parents labels
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

class WeightedBGeScore(BGeScore):
    """
    Weighted BGe Score
    """
    def __init__(self, data : pd.DataFrame, incidence : np.ndarray, penalty_coefficient = 0.01, is_log_space = True):
        """
        Initialise WeightedBGeScore object

        Parameters:
            data (pandas.DataFrame): data
            incidence (numpy.ndarray): graph adjacency matrix
            penalty_coefficient (float): penalty coefficient (default=0.01)
        """
        self.penalty_coefficient = penalty_coefficient
        super().__init__(data, incidence, is_log_space)

    def calculate_complexity_penalty(self):
        """
        Calculate the complexity penalty based on the number of edges in the graph.

        Returns:
            (float): complexity penalty
        """
        num_edges = np.sum(self.incidence > 0)  # Assuming the incidence matrix is binary
        complexity_penalty = self.penalty_coefficient * num_edges
        return complexity_penalty

    def calculate_node_complexity_penalty(self, num_parents):
        """
        Calculate the complexity penalty based on the number of parents a node has.

        Returns:
            (float): complexity penalty
        """
        complexity_penalty = self.penalty_coefficient * num_parents
        return complexity_penalty

    def compute_BGe_with_graph(self):
        """
        Compute BGe score

        Returns:
            (dict): score and parameters
        """
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node

        # Loop through each node in the graph
        for node in self.node_labels:

            log_ml_node = self.compute_BGe_with_node( node )['score']

            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'parents': find_parents(self.incidence, self.node_label_to_index[node])
            }

            total_log_ml += log_ml_node

        # calculate and subtract the complexity penalty
        complexity_penalty = self.calculate_complexity_penalty()
        total_log_ml -= complexity_penalty

        # save the parameters
        self._parameters = parameters

        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score

    def compute_BGe_with_node(self, node : str):
        """
        Compute BGe score for a node

        Returns:
            (dict): score and parameters
        """
        parameters = {}  # Dictionary to store the parameters for each node

        node_indx = self.node_label_to_index[node]
        parentnodes = find_parents(self.incidence, node_indx)
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

        # Calculate and subtract the node-specific complexity penalty
        complexity_penalty = self.calculate_node_complexity_penalty(num_parents)
        corescore -= complexity_penalty

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
