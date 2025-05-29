"""

"""
from typing import Union
import pandas as pd
import numpy as np
from scipy.stats import gamma
import networkx as nx
from structure_learning.scores import Score
from structure_learning.data_structures import Graph

class BDScore(Score):
    """
    BD Score
    """
    def __init__(self, data: pd.DataFrame, alpha: float = 1.0):
        """
        Initialise BDScore instance.

        Parameters:
            data (pandas.DataFrame): data
            graph (numpy.ndarray | networkx.DiGraph): graph
        """
        super().__init__(data)
        self.alpha = alpha

    def compute(self, graph: Graph):
        """
        Compute BD score.

        Parameter:
            graph (Graph):      Graph object
        Returns:
            (float): BD score
        """
        if Graph.has_cycle(graph):
            return {'score': -np.inf}
        score = 1  # Initialize with 1 for multiplication
        for node in graph.nodes:
            parents = list(graph.find_parents(node))

            # Derive alpha based on the structure provided
            r_i = 2  # Binary nodes
            q_i_tilde = len(self.data.groupby(parents)) if parents else 1
            alpha_ijk = self.alpha / (r_i * q_i_tilde)

            if not parents:
                # Compute score for nodes without parents
                n_1 = np.sum(self.data[node])
                n_0 = len(self.data) - n_1
                score *= self._compute_term(n_1, n_0, alpha_ijk)
            else:
                # Compute score for nodes with parents
                for _, subset_data in self.data.groupby(parents):
                    n_1 = np.sum(subset_data[node])
                    n_0 = len(subset_data) - n_1
                    score *= self._compute_term(n_1, n_0, alpha_ijk)
        return score

    def _compute_term(self, n_1, n_0, alpha_ijk):
        """
        Compute the score contribution for a single term
        """
        term = (gamma(2 * alpha_ijk) / gamma(2 * alpha_ijk + n_1 + n_0) *
                gamma(alpha_ijk + n_1) / gamma(alpha_ijk) *
                gamma(alpha_ijk + n_0) / gamma(alpha_ijk))
        return term
