"""
This module defines the DAG class (Directed Acyclic Graph).

Classes:
    DAG: Inherits from Graph and provides methods for DAG manipulation and generation.
"""

import itertools
from math import comb
from typing import Union, List, Tuple, Type, TypeVar
import pandas as pd
import networkx as nx
import numpy as np
import graphical_models as gm
from .graph import Graph
from .cpdag import CPDAG

D = TypeVar('DAG')
class DAG(Graph):
    """
    Represents a Directed Acyclic Graph (DAG).

    Attributes:
        incidence (numpy.ndarray): Adjacency matrix of the graph.
        nodes (list): List of node labels.
    """

    def __init__(self, incidence=None, nodes=None):
        """
        Initialize a DAG instance.

        Parameters:
            incidence (numpy.ndarray): Adjacency matrix of the graph.
            nodes (list): List of node labels.

        Raises:
            Exception: If the adjacency matrix contains cycles.
        """
        super().__init__(incidence, nodes)
        if self.has_cycle(self.incidence):
            raise Exception('Cycle found in adjacency matrix')

    def to_cpdag(self):
        """
        Convert the DAG to a CPDAG.

        Returns:
            CPDAG: Completed Partially Directed Acyclic Graph.
        """
        DAG_gm = gm.DAG.from_amat(self.incidence)
        return CPDAG(incidence=DAG_gm.cpdag().to_amat()[0], nodes=self.nodes)

    @classmethod
    def compute_ancestor_matrix(cls, adj_matrix=None):
        """
        Compute the ancestor matrix from an adjacency matrix.

        Parameters:
            adj_matrix (numpy.ndarray, optional): Adjacency matrix of the graph.

        Returns:
            numpy.ndarray: Ancestor matrix.
        """
        adj_matrix = cls.incidence if adj_matrix is None else adj_matrix
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
    def generate_random(cls, nodes, prob=0.5, seed=None):
        """
        Generate a random DAG represented by a lower triangular adjacency matrix.

        Parameters:
            nodes (list): List of node labels.
            prob (float): Edge probability.
            seed (int, optional): Seed for random number generation.

        Returns:
            DAG: Randomly generated DAG.
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
        Count all possible DAGs for a given number of nodes.

        Parameters:
            n (int): Number of nodes.

        Returns:
            int: Number of possible DAGs.
        """
        if n == 0:
            return 1

        total = 0
        for k in range(1, n + 1):
            total += (-1)**(k+1) * comb(n, k) * (2**(k*(n-k))) * cls.count_dags(n-k)
        return total

    @classmethod
    def generate_all_dags(cls, n_nodes, node_labels=None):
        """
        Generate all unique DAGs with a specified number of nodes.

        Parameters:
            n_nodes (int): Number of nodes in the graph.
            node_labels (list, optional): List of node labels.

        Returns:
            list: List of all unique DAGs.
        """
        if node_labels is None:
            node_labels = [f"X{i}" for i in range(n_nodes)]

        base_dag_lst = []
        all_possible_edges = list(itertools.permutations(node_labels, 2))

        for r in range(len(all_possible_edges) + 1):
            for subset in itertools.combinations(all_possible_edges, r):
                adj_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
                for edge in subset:
                    source, target = edge
                    adj_matrix[node_labels.index(source)][node_labels.index(target)] = True

                if not cls.has_cycle(adj_matrix):
                    base_dag_lst.append(DAG(incidence=adj_matrix, nodes=node_labels))

        return base_dag_lst
