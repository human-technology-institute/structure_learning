import itertools
from typing import Union, List, Tuple
import networkx as nx
import numpy as np
from .graph import Graph

class DAG(Graph):
    
    def __init__(self, incidence = None, nodes = None):
        super().__init__(incidence, nodes)
        if self.has_cycle(self.incidence):
            raise Exception('Cycle found in adjacency matrix')
        
    @classmethod
    def generate_all_dags_from_ordering(nodes : list):
        """
        Generate all DAGs from a topological ordering of nodes.

        Parameter:
            nodes (list): nodes in topological order

        Returns:
            A generator of DAGs
        """
        # Generate all permutations of edges based on the given topological order
        all_edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
        unique_graphs = set()

        for edges in itertools.chain.from_iterable(itertools.combinations(all_edges, r) for r in range(len(all_edges)+1)):
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            # Check for cycles, since we only want DAGs
            if not nx.is_directed_acyclic_graph(G):
                continue

            # Generate a sorted adjacency matrix as a tuple and check if it's already in our set
            adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
            matrix_tuple = tuple(map(tuple, adj_matrix))
            if matrix_tuple not in unique_graphs:
                unique_graphs.add(matrix_tuple)
                yield G

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