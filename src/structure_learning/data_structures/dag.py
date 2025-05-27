import itertools
from typing import Union, List, Tuple, Type, TypeVar
import networkx as nx
import numpy as np
from .graph import Graph

D = TypeVar['DAG']
class DAG(Graph):
    
    def __init__(self, incidence = None, nodes = None):
        super().__init__(incidence, nodes)
        if self.has_cycle(self.incidence):
            raise Exception('Cycle found in adjacency matrix')

    def compute_ancestor_matrix(self):
        """
        Compute ancestor matrix from adjacency matrix.

        Parameter:
            adj_matrix (numpy.ndarray): graph adjancency matrix

        Returns:
            (numpy.ndarray): ancestor matrix
        """
        adj_matrix = self.incidence
        num_nodes = adj_matrix.shape[0]

        # Initialize the ancestor matrix as the adjacency matrix
        ancestor_matrix = np.copy(adj_matrix)

        # Compute powers of the adjacency matrix and update the ancestor matrix
        power_matrix = np.copy(adj_matrix)
        for _ in range(num_nodes - 1):
            power_matrix = np.dot(power_matrix, adj_matrix)
            # If a path exists (i.e., value > 0), set it to 1
            power_matrix[power_matrix > 0] = 1
            ancestor_matrix = np.logical_or(ancestor_matrix, power_matrix).astype(int)

            res = ancestor_matrix.tolist()
            res = np.array(res)
            res = res.T

        return res
    
    @classmethod
    def generate_random(cls, nodes: Union[List[str], Tuple[str]], prob=0.5, seed=None):
        """
        Generate a random DAG, represented by a lower triangular adjacency matrix.

        Parameters:
            nodes (list): node labels
            prob (float): edge probability
            seed (int): seed for numpy RNG

        Returns:
            (numpy.ndarray): adjacency matrix
        """
        N = len(nodes)
        rng = np.random.default_rng(seed=seed) if seed is not None else np.random
        adjmat = np.zeros((N, N))
        adjmat[np.tril_indices_from(adjmat, k=-1)] = rng.binomial(1, prob, size=int(N * (N - 1) / 2))
        perm = rng.permutation(N)
        adjmat[:, perm] = adjmat
        adjmat[perm, :] = adjmat
        return DAG(incidence=adjmat, nodes=nodes)
    
    @classmethod
    def count_dags(cls, n: int) -> int:
        """
        Count all possible DAGs. Given $n$ nodes, the possible number of DAGs
        that can be built is given by
            a(n) = \sum_{k=1}^{n} (-1)^{(k+1)} \binom{n}{k} 2^{k(n-k)} a(n-k)

        Parameter:
            n (int): number of nodes

        Returns:
            (int): number of possible directed graphs
        """
        if n == 0:
            return 1

        total = 0
        for k in range(1, n + 1):
            total += (-1)**(k+1) * comb(n, k) * (2**(k*(n-k))) * cls.count_dags(n-k)
        return total
    
    @classmethod
    def generate_all_dags(cls, n_nodes: int, node_labels: List[str] = None) -> List[Type[D]]:
        """
        Generate a dictionary of all unique DAGs with N nodes.

        Args:
            n_nodes (int): number of nodes in the graph
            node_labels (list): list of node labels. Each label is a string.

        Returns:
            dict: dictionary of all unique DAGs with N nodes.
        """

        # generate node labels for N nodes
        if node_labels is None:
            node_labels = [f"X{i}" for i in range(n_nodes)]

        # Dictionary to store all unique DAGs
        base_dag_lst = []

        # Generate all possible directed edges among the nodes
        all_possible_edges = list(itertools.permutations(node_labels, 2))

        # Iterate over all possible adjacency matrices
        # Iterate over the subsets of all possible edges to form directed graphs
        for r in range(len(all_possible_edges)+1):
            for subset in itertools.combinations(all_possible_edges, r):

                # Initialize an NxN matrix filled with zeros
                adj_matrix = np.zeros((n_nodes, n_nodes))

                # Set entries corresponding to the edges in the current subset to 1
                for edge in subset:
                    source, target = edge
                    adj_matrix[node_labels.index(source)][node_labels.index(target)] = 1

                if not cls.has_cycle(adj_matrix):

                    base_dag_lst.append(Graph(incidence=adj_matrix, nodes=node_labels))

        return base_dag_lst