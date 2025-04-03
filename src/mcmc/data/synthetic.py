## Simulate data
## https://github.com/ermongroup/BCD-Nets/blob/main/dag_utils.py

import pandas as pd
import numpy as np
import scipy.stats as stats
import networkx as nx
from scipy.stats import multivariate_normal
from mcmc.utils.partition_utils import find_parent_nodes, remove_outgoing_edges

class SyntheticDataset(object):
    """
    Generate synthetic data from graphs.

    """
    def __init__(
        self,
        num_nodes: float,
        num_obs: float,
        node_labels: list,
        degree: float,
        graph_type: str="erdos-renyi",
        noise_scale: float=1.0, true_dag = None):

        """
        Initialise SyntheticDataset instance
        """
        self.num_nodes = num_nodes
        self.num_obs = num_obs
        self.node_labels = node_labels
        self.degree = degree
        self.graph_type = graph_type
        self.noise_scale = noise_scale
        self.w_range = (0.5, 2.5)
        self.true_dag = true_dag

        self._setup()

    def _setup(self):
        """
        Initial setup. Simulates random dag (if not given) and data.
        """
        self.W, _, self.P = SyntheticDataset.simulate_random_dag(
            self.num_nodes,
            self.degree,
            self.graph_type,
            self.w_range
        )

        if self.true_dag is None:
            self.data = pd.DataFrame(SyntheticDataset.simulate_data(
                self.W,
                self.num_obs,
                self.noise_scale
            ), columns=self.node_labels)
        else:
            self.data, self.W = SyntheticDataset.simulate_data_from_dag(
                self.true_dag,
                self.num_obs,
                self.num_nodes,
                self.node_labels,
                self.w_range,
                self.noise_scale
            )

        self.adj_mat = pd.DataFrame(np.where(self.W!=0, 1, 0), columns=self.node_labels)

    @staticmethod
    def simulate_data_from_dag(DAG: np.ndarray, num_obs, num_nodes, node_labels, w_range, noise_scale):
        """
        Simulate samples from ground truth DAG.

        Parameters:
            DAG (numpy.ndarray): ground truth DAG
            num_obs (int): number of observations to simulate
            num_nodes (int): number of nodes
            node_labels (list (str)): node labels
            w_range (2-tuple (float)): weight range +/- (low, high)
            noise_scale (float): scale parameter of noise distribution

        Returns:
            (numpy.ndarray): [n,d] sample matrix
        """

        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[num_nodes, num_nodes])
        U[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1.0

        W = (DAG != 0).astype(float) * U
        W_mat = np.eye(num_nodes) + W

        sigmas = np.ones((num_nodes,)) * noise_scale # Assuming equal variances

        # Generate the diagonal conditional variance matrix, diagonal values indicate sigma^2_j
        D_mat = np.eye(num_nodes) * sigmas

        # Covariance matrix
        sigma = np.linalg.pinv(W_mat.T) @ D_mat @ np.linalg.pinv(W_mat)

        data = pd.DataFrame(multivariate_normal.rvs(cov=sigma, size=num_obs), columns=node_labels)
        # Generate data from MVN distribution
        return data, W

    # def simulate_categorical_data_from_dag(DAG: np.ndarray, num_obs: int, node_labels: list, binary: bool = False, max_states = 5):

    #     n_nodes = DAG.shape[1]
    #     data = np.zeros((num_obs, n_nodes))

    #     n_states = [2]*n_nodes
    #     if not binary:
    #         n_states = np.random.choice(list(range(2, max_states+1)), size=(n_nodes,))

    #     incidence = DAG.copy()
    #     ordering = []
    #     for i in range(n_nodes):
    #         parents = find_parent_nodes(incidence)
    #         incidence = remove_outgoing_edges(incidence, parents)
    #         ordering.extend(parents)

    #     for node in ordering:
    #         if not np.any(DAG[:,node]): # no parent
    #             data[:, node] = np.random.choice(n_states[node], size=num_obs)
    #         else:
    #             states = np.arange(n_states[node])





    @staticmethod
    def simulate_random_dag(d, degree, graph_type, w_range):
        """Simulate random DAG with some expected degree.

        Parameters:
            d (int): number of nodes
            degree (int): expected node degree, in + out
            graph_type (str): {erdos-renyi, barabasi-albert, full}
            w_range (2-tuple (float)): weight range +/- (low, high)

        Returns:
            (numpy.ndarray): weighted DAG
            None
            (numpy.ndarray): permutation matrix
        """
        if graph_type == "erdos-renyi":
            prob = float(degree) / (d - 1)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        elif graph_type == "barabasi-albert":
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == "full":  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError("Unknown graph type")
        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B_perm != 0).astype(float) * U

        # At the moment the generative process is P.T @ lower @ P, we want
        # it to be P' @ upper @ P'.T.
        # We can return W.T, so we are saying W.T = P'.T @ lower @ P.
        # We can then return P.T, as we have
        # (P.T).T @ lower @ P.T = W.T

        return W.T, None, P.T

    @staticmethod
    def simulate_gaussian_dag(d, w_std):
        """Simulate dense DAG adjacency matrix

        Parameters:
            d (int): number of nodes
            w_std (float): weight std

        Returns:
            (numpy.ndarray): weighted DAG
            None
            (numpy.ndarray): permutation matrix
            (numpy.ndarray): lower triangular matrix
        """
        lower_entries = np.random.normal(loc=0.0, scale=w_std, size=(d * (d - 1) // 2))
        L = np.zeros((d, d))
        # We want the ground-truth W.T to be generated from PLP^\top
        # This is since we encode W.T as PLP^\top in the approach.
        L[np.tril_indices(d, -1)] = lower_entries
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        W = (P @ L @ P.T).T
        return W, None, P, L

    @staticmethod
    def simulate_data_V1(W, n, noise_scale=1.0, sigmas=None):
        """Simulate samples from SEM with specified type of noise.

        Parameters:
            W (numpy.ndarray): weigthed DAG
            n (int): number of samples
            noise_scale (float): scale parameter of noise distribution in linear SEM
            sigmas (numpy.ndarray): noise vector

        Returns:
            (numpy.ndarray) [n,d] sample matrix
        """
        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d], dtype=np.float64)

        if sigmas is None:
            sigmas = np.ones((d,)) * noise_scale

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            eta = X[:, parents].dot(W[parents, j])

            X[:, j] = eta + np.random.normal(scale=sigmas[j], size=n)

        return X

    @staticmethod
    def simulate_data(W, n, noise_scale=1.0, sigmas=None):
        """Simulate samples from SEM with specified type of noise.

        Parameters:
            W (numpy.ndarray): weigthed DAG
            n (int): number of samples
            noise_scale (float): scale parameter of noise distribution in linear SEM
            sigmas (numpy.ndarray): noise vector

        Returns:
            (numpy.ndarray) [n,d] sample matrix
        """
        d = W.shape[0]
        X = np.zeros([n, d], dtype=np.float64)

        if sigmas is None:
            sigmas = np.ones((d,)) * noise_scale # Assuming equal variances

        # Generate the diagonal conditional variance matrix, diagonal values indicate sigma^2_j
        D_mat = np.eye(d) * sigmas

        W_mat = W + np.eye(d)

        # Covariance matrix
        sigma = np.linalg.pinv(W_mat.T) @ D_mat @ np.linalg.pinv(W_mat)

        X = stats.multivariate_normal.rvs(cov=sigma, size=n)

        return X
