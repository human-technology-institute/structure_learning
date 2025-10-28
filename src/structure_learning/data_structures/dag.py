"""
This module defines the DAG class (Directed Acyclic Graph).

Classes:
    DAG: Inherits from Graph and provides methods for DAG manipulation and generation.
"""

import itertools
from math import comb
from typing import TypeVar
import networkx as nx
import numpy as np
import pandas as pd
from .graph import Graph

D = TypeVar('DAG')
class DAG(Graph):
    """
    Represents a Directed Acyclic Graph (DAG).

    Attributes:
        incidence (numpy.ndarray): Adjacency matrix of the graph.
        nodes (list): List of node labels.
    """

    def __init__(self, incidence=None, nodes=None, weights=None):
        """
        Initialize a DAG instance.

        Parameters:
            incidence (numpy.ndarray): Adjacency matrix of the graph.
            nodes (list): List of node labels.

        Raises:
            Exception: If the adjacency matrix contains cycles.
        """
        super().__init__(incidence, nodes, weights)
        if self.has_cycle(self.incidence):
            raise Exception('Cycle found in adjacency matrix')
        
    def fit(self, data: pd.DataFrame):  
        from pgmpy.models import LinearGaussianBayesianNetwork
        colors = {}
        model = LinearGaussianBayesianNetwork(list(self.edges))
        model.fit(data.values)
        weights = {(var, cpd.variable):cpd.beta[idx+1] for cpd in model.get_cpds() for idx,var in enumerate(cpd.evidence)}
        for edge,weight in weights.items():
            colors[(edge[0], edge[1])] = "#FE5600" if weight < 0 else "#5984FF"
        return weights, colors
        
    def plot(self, filename=None, text=None, data: pd.DataFrame=None):
        
        weights, colors = None, None
        if data is not None:
            weights, colors = self.fit(data)
        return super().plot(filename=filename, text=text, edge_colors=colors, edge_weights=weights)
    
    def to_cpdag(self, blocklist: np.ndarray = None, verbose=False):
        """
        Replace with arcs those edges whose orientations can be determined by Meek rules:
        =====
        See Koller & Friedman, Algorithm 3.5

        Adapted from pgmpy (https://github.com/uhlerlab/graphical_models/blob/main/graphical_models/classes/dags/pdag.py)
        Parameters:
            verbose (bool): If True, print detailed information about the process.
        Returns:
            CPDAG: Completed Partially Directed Acyclic Graph.
        """
        from .cpdag import CPDAG
        PROTECTED = 'P'  # indicates that some configuration definitely exists to protect the edge
        UNDECIDED = 'U'  # indicates that some configuration exists that could protect the edge
        NOT_PROTECTED = 'N'  # indicates no possible configuration that could protect the edge

        vstructures = [(self.nodes[i],self.nodes[j]) for i,j,k in self.v_structures()]
        vstructures += [(self.nodes[k],self.nodes[j]) for i,j,k in self.v_structures()]
        vstructures = set(vstructures)
        blocked_edges = {(self.nodes[r],self.nodes[c]) for r,c in zip(*np.nonzero(blocklist)) if self.incidence[c,r]} if blocklist is not None else set()
        edges1 = {(i, j) for i, j in self.edges if (i, j) not in vstructures and (j, i) not in vstructures and (i,j) not in blocked_edges and (j,i) not in blocked_edges}
        undecided_arcs = edges1 | {(j, i) for i, j in edges1}
        arc_flags = {arc: PROTECTED for arc in vstructures}
        arc_flags.update({(j,i): PROTECTED for i,j in blocked_edges})
        arc_flags.update({arc: UNDECIDED for arc in undecided_arcs})

        if verbose:
            print(f'Initial undecided arcs: {(undecided_arcs)}')
            print(f'Initial v-structures: {(vstructures)}')
            print(f'Initial blocked edges: {(blocked_edges)}')
            print(f'Initial flags: {(arc_flags)}')

        incidence = self.incidence.copy()
        incidence[incidence.T] = True
        for i,j,k in self.v_structures():
            incidence[j, k] = False
            incidence[j, i] = False
        if blocklist is not None:
            for i, j in zip(*np.nonzero(blocklist)):
                incidence[i, j] = False
        
        def has_edge(incidence, i, j, undirected=False):
            """
            Check if there is an edge between nodes i and j.
            If undirected is True, check for both directions.
            """
            i,j = self._node_to_index((i, j))
            if undirected:
                return incidence[i,j] or incidence[j,i]
            else:
                return incidence[i,j]
            
        while undecided_arcs:
            for arc in undecided_arcs:
                i, j = arc
                flag = NOT_PROTECTED

                # check configuration (a) -- causal chain
                s = ''
                for k in self.find_parents(i):
                    if not has_edge(incidence, k, j, undirected=True):
                        if arc_flags[(k, i)] == PROTECTED:
                            flag = PROTECTED
                            s = f': {k}->{i}-{j}'
                            break
                        else:
                            flag = UNDECIDED
                if verbose: print(f'{arc} marked {flag} by (a){s}')

                # check configuration (b) -- acyclicity
                s = ''
                if flag != PROTECTED:
                    for k in self.find_parents(j):
                        if i in self.find_parents(k):
                            if arc_flags[(i, k)] == PROTECTED and arc_flags[(k, j)] == PROTECTED:
                                flag = PROTECTED
                                s = f': {k}->{j}-{i}->{k}'
                                break
                            else:
                                flag = UNDECIDED
                    if verbose: print(f'{arc} marked {flag} by (b){s}')

                # check configuration (d)
                s = ''
                if flag != PROTECTED:
                    for k1, k2 in itertools.combinations(self.find_parents(j), 2):
                        if has_edge(incidence, i, k1) and has_edge(incidence,  i, k2) and not has_edge(incidence, k1, k2, undirected=True):
                            if arc_flags[(k1, j)] == PROTECTED and arc_flags[(k2, j)] == PROTECTED:
                                flag = PROTECTED
                                s = f': {i}-{k1}->{j}<-{k2}-{i}'
                                break
                            else:
                                flag = UNDECIDED
                    if verbose: print(f'{arc} marked {flag} by (c){s}')

                arc_flags[arc] = flag

            if all(arc_flags[arc] == NOT_PROTECTED for arc in undecided_arcs): break

            undecided_arcs_copy = undecided_arcs.copy()
            for arc in undecided_arcs.copy():
                if arc_flags[arc] == PROTECTED:
                    if verbose: print(f'Orienting {arc} as arc')
                    undecided_arcs.discard(arc)
                    undecided_arcs.discard((arc[1], arc[0]))
                    # self._replace_edge_with_arc(arc)
                    node1, node2 = self._node_to_index(arc)
                    incidence[node1, node2] = self.incidence[node1, node2]
                    incidence[node2, node1] = self.incidence[node2, node1]
            if undecided_arcs == undecided_arcs_copy:
                if verbose: print('No more arcs can be oriented, but undecided arcs remain.')
                break

        return CPDAG(incidence=incidence, nodes=self.nodes)

    @classmethod
    def compute_ancestor_matrix(cls, adj_matrix):
        """
        Compute the ancestor matrix from an adjacency matrix.

        Parameters:
            adj_matrix (numpy.ndarray, optional): Adjacency matrix of the graph.

        Returns:
            numpy.ndarray: Ancestor matrix.
        """
        if adj_matrix is None:
            raise Exception("Adjacency matrix not provided")
        
        if not isinstance(adj_matrix, np.ndarray):
            adj_matrix = adj_matrix.incidence
        
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

    def topological_sort(self):
        """
        Perform topological sorting of the nodes in the DAG.

        Returns:
            List: Nodes sorted in topological order.

        """

        graph = nx.DiGraph(self.incidence)
        sorted_nodes = list(nx.topological_sort(graph))
        return [self.nodes[node] for node in sorted_nodes]
