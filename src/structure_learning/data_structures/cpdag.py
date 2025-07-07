"""
This module defines the CPDAG class, which represents a Completed Partially Directed Acyclic Graph.

Classes:
    CPDAG: Inherits from Graph and provides methods for DAG enumeration and comparison.
"""

import numpy as np
import pandas as pd
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

    def __init__(self, incidence=None, nodes=None):
        """
        Initialize a CPDAG instance.

        Parameters:
            incidence (numpy.ndarray): Adjacency matrix of the graph.
            nodes (list): List of node labels.
        """
        super().__init__(incidence, nodes)
        self.undirected_edges = list()
        self._undirected_edges_idx = list()
        rows, cols = np.nonzero(np.tril(incidence & incidence.T, 0))
        for r, c in zip(rows, cols):
            self._undirected_edges_idx.append((r, c))
            self.undirected_edges.append((nodes[r], nodes[c]))
        self._n_dags = None
        self.dags = None

    @classmethod
    def from_dag(cls, dag):
        """
        Create a CPDAG from a DAG.

        Parameters:
            dag (DAG): Directed Acyclic Graph.

        Returns:
            CPDAG: Completed Partially Directed Acyclic Graph.
        """
        return dag.to_cpdag()

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

    def enumerate_dags(self, generate=True):
        """
        Enumerate all DAGs represented by the CPDAG.

        Parameters:
            generate (bool): Whether to generate DAG instances.

        Yields:
            Graph: DAG instances represented by the CPDAG.
        """
        from .dag import DAG 
        dim = len(self.undirected_edges)
        if dim == 0:
            self._n_dags = 1
            g = DAG(incidence=self.incidence, nodes=self.nodes)
            self.dags = [g]
            yield g
        else:
            if self.dags is None:
                self._n_dags = 0
                self.dags = []
                fstr = "{0:0" + str(dim) + "b}"
                for i in range(2**dim):
                    binstr = fstr.format(i)
                    incidence = self.incidence.copy()
                    for j, flip in enumerate(binstr):
                        r, c = self._undirected_edges_idx[j]
                        if flip == "1":
                            incidence[r, c] = False
                        else:
                            incidence[c, r] = False
                    if not Graph.has_cycle(incidence):
                        self._n_dags += 1
                        if generate:
                            g = DAG(incidence=incidence, nodes=self.nodes)
                            self.dags.append(g)
                            yield g

    def __len__(self):
        """
        Get the number of DAGs represented by the CPDAG.

        Returns:
            int: Number of DAGs.
        """
        [g for g in self.enumerate_dags(generate=False)]
        return self._n_dags

    def plot(self, filename=None, text=None, data: pd.DataFrame=None):

        if data is None:
            return super().plot(filename=filename, text=text)
        else:
            [dag for dag in self.enumerate_dags(generate=True)]
            weights, colors = self.dags[0].fit(data=data)
            return super().plot(filename=filename, text=text, edge_colors=colors, edge_weights=weights)