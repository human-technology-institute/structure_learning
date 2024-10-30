## Simulate Internally Standardised Synthethic Data
## Standardisation done via process described in https://arxiv.org/pdf/2406.11601 

import pandas as pd
import numpy as np
import scipy.stats as stats
import networkx as nx
from scipy.stats import multivariate_normal
from mcmc.utils.partition_utils import find_parent_nodes, remove_outgoing_edges
from collections import deque

class Standardised_Synthetic_Data(object):
    """
    Generate standardised synthetic data from graphs.
    """
    def __init__(
        self,
        num_nodes: int,
        num_obs: int,
        node_labels: list,
        degree: float,
        graph_type: str = "erdos-renyi",
        noise_scale: float = 1.0,
        true_dag = None):
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
        self.W, _, self.P = self.simulate_random_dag(
            self.num_nodes,
            self.degree,
            self.graph_type,
            self.w_range
        )

        if self.true_dag is None:
            self.data = pd.DataFrame(self.simulate_data(
                self.W,
                self.num_obs
            ), columns=self.node_labels)
        else:
            self.data, self.W = self.simulate_data_from_dag(
                self.true_dag,
                self.num_obs,
                self.num_nodes,
                self.node_labels,
                self.w_range,
                self.noise_scale
            )

        self.adj_mat = pd.DataFrame(np.where(self.W != 0, 1, 0), columns=self.node_labels)

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
    def topological_sort(adj_matrix):
        """
        Sort the random dag that has been generated topologically. 
        i.e. Source Nodes first and absorbing nodes last
        """   
        # Number of nodes in the DAG
        num_nodes = adj_matrix.shape[0]

        # Calculating number of edges point to a node
        in_degree = np.zeros(num_nodes, dtype=int) # Init List
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0: # Seeing if node i is a parent of node j
                    in_degree[j] += 1 # If node i is indeed a parent, increase 'in-degree' of node j by 1
        
        queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
        top_order = []
        
        while queue:
            node = queue.popleft()
            top_order.append(node)
            for j in range(num_nodes):
                if adj_matrix[node, j] != 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
    
        return top_order

    @staticmethod
    def simulate_data(adj_matrix, num_obs):
        """Simulate data from the weighted DAG."""
        num_nodes = adj_matrix.shape[0]
        data = np.zeros((num_obs, num_nodes))
        
        # Get topological order of nodes
        top_order = Standardised_Synthetic_Data.topological_sort(adj_matrix)
        
        for node in top_order:
            parents = [i for i in range(num_nodes) if adj_matrix[i, node] != 0]
            if not parents:
                # If no parents, generate data independently
                data[:, node] = np.random.randn(num_obs)
            else:
                # Manually extract and combine parent columns
                parent_values_list = []
                for parent in parents:
                    # Extract Parent Values
                    parent_values = data[:, parent].reshape(-1, 1)
                    # Standardising each parent node
                    parent_values_standardised = (parent_values - np.mean(parent_values)) / np.std(parent_values)
                    # Append to list
                    parent_values_list.append(parent_values_standardised)
                
                # Combine all parent columns into a single array
                parent_values_combined = np.hstack(parent_values_list)

                # Generate Values for Intercept and Noise
                noise = np.random.uniform(0, 10, size = num_obs)
                intercept = np.random.uniform(0, 10, size = num_obs)
                
                # Generate data based on weighted sum of parent values + noise
                weights = adj_matrix[parents, node]
                data[:, node] = intercept + np.dot(parent_values_combined, weights) + noise
        
        return data
