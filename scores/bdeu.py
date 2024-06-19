from typing import Union
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import gamma
from mcmc.scores import Score

class BDeuScore(Score):
    """
    BDeu Score
    """
    def __init__(self, data : pd.DataFrame, graph : Union[np.ndarray, nx.DiGraph] = None):
        """
        Initialise BDeuScore instance.

        Parameters:
            data (pandas.DataFrame): data
            graph (numpy.ndarray | networkx.DiGraph): graph
        """
        super().__init__(data, graph, "BDeu Score")
        self.graph = graph
        if isinstance(graph, np.ndarray):
            self.graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)

    # todo: for larger datasets, this function is very slow.
    def compute(self, alpha: float = 1.0):
        """
        Compute BDeu score.

        Returns:
            (float): BDeu score
        """
        BDeu_score = 0.0
        graph = self.graph
        data = self.data

        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            r = data[parents].drop_duplicates().shape[0] if parents else 1
            q = data[node].nunique()

            # Calculate alpha values
            alpha_j = alpha / r
            alpha_ij = alpha / (r * q)

            # Compute counts
            if parents:
                counts = data.groupby(parents + [node]).size().reset_index(name='counts')
                parent_counts = data.groupby(parents).size().reset_index(name='parent_counts')
                counts = counts.merge(parent_counts, on=parents)
            else:
                counts = data[node].value_counts().reset_index()
                counts.columns = [node, 'counts']
                counts['parent_counts'] = len(data)

            for _, row in counts.iterrows():
                BDeu_score += (gamma(alpha_ij + row['counts']) - gamma(alpha_ij) + gamma(alpha_j) - gamma(alpha_j + row['parent_counts']))

        return BDeu_score
