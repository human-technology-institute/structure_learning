"""

"""
import pandas as pd
import numpy as np
from scipy.special import gammaln
from scipy.stats import invgamma
from scipy.stats import multivariate_normal
from structure_learning.scores import Score
from structure_learning.utils.graph_utils import find_parents

class MarginalLogLikelihood(Score):
    """
    Marginal Log Likelihood Score.
    """
    def __init__(self, data : pd.DataFrame, incidence : np.ndarray = None, \
                 a : float = 1, b : float = 1, is_log_space = True):
        """
        Initialise MarginalLogLikelihood instance.

        Parameters:
            data (pandas.DataFrame): data
            incidence (np.ndarray): graph adjacency matrix (default=None)
            a (float): a_0 parameter for sigma^2 (default=1)
            b (float): b_0 parameter for sigma^2 (default=1)
            is_log_space (bool): (default=True)
        """
        super().__init__(data, incidence, "Log Marginal Likelihood", is_log_space=is_log_space)

        self.a = a
        self.b = b
        self.parameters = {}
        self.reg_coefficients = {}
        self._to_string = "Marginal Log Likelihood"

    # Compute the marginal likelihood for the data
    def compute(self):
        """
        Compute the marginal likelihood for the data.

        Returns:
            (dict): score and parameters
        """
        return self.compute_marginal_log_likelihood_with_graph() if self.incidence is not None \
            else self.compute_marginal_log_likelihood_from_data()

    def compute_node_with_edges(self, node: str, parents: list):
        """
        Compute the marginal likelihood for a node.

        Parameter:
            node (str): node label

        Returns:
            (dict): score and parameters
        """
        n, _ = self.data.shape

        # For sigma^2
        a0 = self.a
        b0 = self.b

        parameters = {} # Dictionary to store the parameters for each node

        num_parents = len(parents) # number of parents

        # Extract the data for the node
        y = self.data[node].values

        # If the node has parents
        if num_parents > 0:
            # Extract the data for the node's parents
            X = self.data.iloc[:,parents].values
            X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        else:
            # For root nodes, X is just an intercept term
            X = np.ones((len(y), 1))

        # Setting up the Priors for beta
        p_node = X.shape[1] # Number of predictors for this node + intercept
        Lambda0 = np.eye(p_node)*0.1	# Prior precision matrix
        m0 = np.zeros(p_node)	# Prior mean vector

        # Bayesian Linear Regression
        # Compute the posterior precision matrix Lambda_n for beta
        Lambda_n = Lambda0 + X.T @ X

        # Compute the posterior mean m_n for beta
        # if the matrix is singular, then this will give an error
        # so shall we use regularization techniques to avoid this?
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        #beta_hat = np.linalg.inv(X.T @ X + regularizer * np.eye(X.shape[1])) @ X.T @ y
        m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)

        # Compute a_n and b_n for sigma^2
        a_n = a0 + n/2
        b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)

        # Compute the Marginal Likelihood for this node and add to total
        log_ml_node = ( - (len(y)/2) * np.log(2*np.pi)
                        + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1])
                        + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0) )

        # Save the parameters for the node
        parameters = {
            'score' : log_ml_node,
            'Lambda_n': Lambda_n,
            'm_n': m_n,
            'a_n': a_n,
            'b_n': b_n
        }

        # compute the regression coefficients based on the parameters
        # self.reg_coefficients = self.recover_regr_paramns(parameters)

        # save the parameters
        self.parameters = parameters

        # Return the total marginal likelihood and the parameters
        score = {
            'score': log_ml_node,
            'parameters': parameters
        }
        return score

    # Compute the marginal likelihood conditioned on a graph structure
    ####################################################################
    def compute_marginal_log_likelihood_with_graph(self):
        """
        Compute the marginal likelihood for a graph.

        Returns:
            (dict): score and parameters
        """
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node

        # Loop through each node in the graph
        for node in self.node_labels:

            score_node = self.compute_node( node )

            log_ml_node = score_node['score']

            # Save the parameters for the node
            parameters[node] = score_node['parameters']

            total_log_ml += log_ml_node

        # compute the regression coefficients based on the parameters
        # self.reg_coefficients = self.recover_regr_paramns(parameters)

        # save the parameters
        self.parameters = parameters

        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score

    def compute_marginal_log_likelihood_from_data(self):
        """
        Compute the marginal likelihood directly from data.

        Returns:
            (dict): score and parameters
        """
        n, _ = self.data.shape

        # For sigma^2
        a0 = self.a
        b0 = self.b

        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each variable

        # Loop through each variable in the data
        for var in self.data.columns:

            # Extract the data for the variable
            y = self.data[var].values

            # Since we're not using a graph, we'll assume a simple model with just an intercept term for each variable
            X = np.ones((len(y), 1))

            # Setting up the Priors for beta
            p_var = X.shape[1]
            Lambda0 = np.eye(p_var) * 0.1
            m0 = np.zeros(p_var)

            # Bayesian Linear Regression
            Lambda_n = Lambda0 + X.T @ X

            # Compute the posterior mean m_n for beta
            m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ np.linalg.inv(X.T @ X) @ X.T @ y + Lambda0 @ m0)

            # Compute a_n and b_n for sigma^2
            a_n = a0 + n / 2
            b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)

            # Compute the Marginal Likelihood for this variable and add to total
            log_ml_var = ( - (len(y)/2) * np.log(2 * np.pi)
                        + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1])
                        + a0 * np.log(b0) - a_n * np.log(b_n)
                        + gammaln(a_n) - gammaln(a0))

            # Save the parameters for the variable
            parameters[var] = {
                'score' : total_log_ml,
                'Lambda_n': Lambda_n,
                'm_n': m_n,
                'a_n': a_n,
                'b_n': b_n,
            }

            total_log_ml += log_ml_var

        # compute the regression coefficients based on the parameters
        #self.reg_coefficients = self.recover_regr_paramns(parameters)

        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score

    # Recover the regression coefficients beta and sigma^2 from the parameters a_n and b_n, lambda_n and mu_n
    ####################################################################
    def recover_regr_params(self, parameters : dict):
        """
        Given the parameters of the marginal likelihood function a_n, b_n, lambda_n and mu_n,
        returns the regression coefficients beta and the regression sigma^2

        Parameters:
            parameters (dict): parameters obtained from compute() or compute_node()

        Returns:
            (dict): sampled sigma^2 and beta for each node
        """

        sampled_values = {}
        for node, param in parameters.items():
            # Sample sigma^2 from the inverse gamma distribution
            sigma2 = invgamma.rvs(a=param['a_n'], scale=  param['b_n'])

            # Sample beta from the multivariate normal distribution
            cov_matrix = sigma2 * np.linalg.inv(param['Lambda_n'])
            beta = multivariate_normal.rvs(mean=param['m_n'], cov=cov_matrix)

            sampled_values[node] = {
                'beta': beta,
                'sigma2': sigma2
            }

        self.reg_coefficients = sampled_values
        return sampled_values

    @property
    def to_string(self):
        return self._to_string

    def __str__(self):
        return self._to_string
