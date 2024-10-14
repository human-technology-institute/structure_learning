from typing import Union
from functools import cache
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import gammaln
from mcmc.scores import Score
from mcmc.utils.graph_utils import find_parents

class BDeuScore(Score):
    """
    BDeu Score
    """
    def __init__(self, data : pd.DataFrame, incidence : Union[np.ndarray, nx.DiGraph] = None, alpha=10):
        """
        Initialise BDeuScore instance.

        Parameters:
            data (pandas.DataFrame): data
            graph (numpy.ndarray | networkx.DiGraph): graph
        """
        super().__init__(data, incidence, "BDeu Score")
        self.incidence = incidence
        if incidence is not None and not isinstance(incidence, np.ndarray):
            self.incidence = nx.from_numpy_array(incidence, create_using=nx.DiGraph)
        self.alpha = 10.0
        self.states = {node: self.data[node].unique() for node in list(self.data.columns)}

    # todo: for larger datasets, this function is very slow.
    def compute(self):
        """
        Compute BDeu score.

        Returns:
            (float): BDeu score
        """
        BDeu_score = 0.0

        parameters = {}
        for node in self.node_labels:
            node_score = self.compute_node(node)
            parameters[node] = node_score['parameters'][node]
            BDeu_score += node_score['score']

        score = {
            'score': BDeu_score,
            'parameters': parameters
        }
        return score

    def compute_node_with_edges(self, node: str, parents: list):
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
        return self.data.groupby([node] + parents).size().unstack(parents).fillna(0) if parents else self.data[node].value_counts().reindex(self.states[node]).to_frame()
