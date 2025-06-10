"""
This module implements the BDeuScore class, which calculates the Bayesian Dirichlet equivalent uniform (BDeu) score for a given graph structure.

The BDeu score is used in Bayesian network structure learning to evaluate the fit of a graph to a dataset. It considers the conditional probabilities of nodes given their parents and incorporates a prior distribution controlled by the alpha parameter.

Classes:
    BDeuScore: Computes the BDeu score for a graph structure based on the provided data.
"""

from typing import Union
from functools import cache
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import gammaln
from structure_learning.scores import Score
from structure_learning.data import Data
from structure_learning.data_structures import Graph

class BDeuScore(Score):
    """
    BDeu Score
    """
    def __init__(self, data : Union[Data,pd.DataFrame], alpha=10):
        """
        Initialise BDeuScore instance.

        Parameters:
            data (pandas.DataFrame): data
        """
        super().__init__(data)
        self.alpha = alpha
        self.states = {node: self.data.values[node].unique() for node in list(self.data.columns)}

    # todo: for larger datasets, this function is very slow.
    def compute(self, graph: Graph):
        """
        Compute BDeu score.

        Parameter:
            graph (Graph):      Graph object
        Returns:
            (float): BDeu score
        """
        if Graph.has_cycle(graph):
            return {'score': -np.inf}
        BDeu_score = 0.0

        parameters = {}
        for node in graph.nodes:
            node_score = self.compute_node(graph, node)
            parameters[node] = node_score['parameters'][node]
            BDeu_score += node_score['score']

        score = {
            'score': BDeu_score,
            'parameters': parameters
        }
        return score

    def compute_node_with_edges(self, node: str, parents: list, node_index_map: dict=None):
        """
        Adapted from https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/estimators/StructureScore.py
        """

        var_states = len(self.states[node])
        parents = list(parents)
        state_counts = self._state_counts(node, tuple(parents))

        num_parents_states = np.prod([len(self.states[parent]) for parent in parents])

        counts = np.asarray(state_counts)
        # counts size is different because reindex=False is dropping columns.
        counts_size = num_parents_states * var_states

        alpha = self.alpha / num_parents_states
        beta = self.alpha / counts_size
        # Compute log(gamma(counts + beta))
        log_gamma_counts = gammaln(counts + beta)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        log_gamma_conds = gammaln(log_gamma_conds + alpha)

        # Adjustment because of missing 0 columns when using reindex=False for computing state_counts to save memory.
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1])
            * var_states
            * gammaln(beta)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + num_parents_states * gammaln(alpha)
            - counts_size * gammaln(beta)
        )

        parameters = {}
        parameters[node] = {'parents': parents, 'score': score}

        score = {
            'score': score,
            'parameters': parameters
        }
        return score

    @cache
    def _state_counts(self, node: str, parents: tuple):
        parents = list(parents)
        return self.data.values.groupby([node] + parents).size().unstack(parents).fillna(0) if parents else self.data.values[node].value_counts().reindex(self.states[node]).to_frame()
