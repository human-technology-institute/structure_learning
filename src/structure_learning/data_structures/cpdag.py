"""
This module defines the CPDAG class, which represents a Completed Partially Directed Acyclic Graph.

Classes:
    CPDAG: Inherits from Graph and provides methods for DAG enumeration and comparison.
"""

import numpy as np
import pandas as pd
import networkx as nx
from .graph import Graph 

class CPDAG(Graph):
    """
    Represents a Completed Partially Directed Acyclic Graph (CPDAG).

    Attributes:
        undirected_edges (list): List of undirected edges in the graph.
        _undirected_edges_idx (list): List of indices of undirected edges.
        _n_dags (int): Number of DAGs represented by the CPDAG.
        dags (list): List of DAGs represented by the CPDAG.
    """

    def __init__(self, incidence=None, nodes=None, weights=None):
        """
        Initialize a CPDAG instance.

        Parameters:
            incidence (numpy.ndarray): Adjacency matrix of the graph.
            nodes (list): List of node labels.
        """
        super().__init__(incidence, nodes, weights)
        self.undirected_edges = list()
        self._undirected_edges_idx = list()
        rows, cols = np.nonzero(np.tril(incidence & incidence.T, 0))
        for r, c in zip(rows, cols):
            self._undirected_edges_idx.append((r, c))
            self.undirected_edges.append((nodes[r], nodes[c]))
        self._n_dags = None
        self.dags = None

    @classmethod
    def from_dag(cls, dag, blocklist=None):
        """
        Create a CPDAG from a DAG.

        Parameters:
            dag (DAG): Directed Acyclic Graph.

        Returns:
            CPDAG: Completed Partially Directed Acyclic Graph.
        """
        return dag.to_cpdag(blocklist=blocklist)
    
    def contains(self, dag, blocklist=None):
        """
        Check if a DAG is represented by the CPDAG. 
        Parameters:
            dag (DAG): Directed Acyclic Graph.  
            blocklist (list): List of edges to block.
        Returns:
            bool: True if the DAG is represented by the CPDAG, False otherwise.
        """
        return dag.to_cpdag(blocklist=blocklist) == self

    def __contains__(self, dag):
        """
        Check if a DAG is represented by the CPDAG.

        Parameters:
            dag (DAG): Directed Acyclic Graph.

        Returns:
            bool: True if the DAG is represented by the CPDAG, False otherwise.
        """
        cpdag = dag.to_cpdag()
        return cpdag == self

    def __eq__(self, other):
        """
        Check if two CPDAGs are equal.

        Parameters:
            other (CPDAG): Another CPDAG instance.

        Returns:
            bool: True if the CPDAGs are equal, False otherwise.
        """
        return (self.incidence == other.incidence).all()

    def __len__(self):
        import cliquepicking as cp
        nodes = {node:idx for idx, node in enumerate(self.nodes)}
        edges = [(nodes[u], nodes[v]) for u, v in self.edges]
        return cp.mec_size(edges)
    
    def enumerate_dags(self):
        import cliquepicking as cp
        from .dag import DAG 
        nodes = {node:idx for idx, node in enumerate(self.nodes)}
        edges = [(nodes[u], nodes[v]) for u, v in self.edges]

        dags = cp.mec_list_dags(edges)
        for dag in dags:
            incidence = np.zeros((len(self.nodes), len(self.nodes)), dtype=bool)
            for u, v in dag:
                incidence[u, v] = True
            g = DAG(incidence=incidence, nodes=self.nodes)
            yield g

    def plot(self, filename=None, text=None, data: pd.DataFrame=None):

        if data is None:
            return super().plot(filename=filename, text=text)
        else:
            dags = [dag for dag in self.enumerate_dags()]
            weights, colors = dags[0].fit(data=data)
            return super().plot(filename=filename, text=text, edge_colors=colors, edge_weights=weights)
        