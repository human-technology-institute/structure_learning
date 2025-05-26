"""

"""
from typing import Union
import time
import random
import numpy as np
import scipy
import pandas as pd
from structure_learning.utils.graph_utils import collect_node_scores, generate_key_from_adj_matrix
from structure_learning.proposals import StructureLearningProposal, GraphProposal
from structure_learning.scores import Score, BGeScore, BDeuScore
from structure_learning.mcmc import MCMC

class StructureMCMC(MCMC):
    """
    Implementation of Structure MCMC.
    """
    def __init__(self, data: pd.DataFrame = None, initial_graph : np.ndarray = None, max_iter : int = 30000,
                 score_object : Union[str, Score] = None, proposal_object : StructureLearningProposal = None, pc_init = True,
                 blacklist: np.ndarray = None, whitelist: np.ndarray = None, seed: int = 32, sparse=True):
        """
        Initilialise Structure MCMC instance.

        Parameters:
            data (pd.DataFrame):                            Dataset. Optional if score_object is given.
            initial_graph (numpy.ndarray | None):           Initial graph for the MCMC simulation.
                                                            If None, simulation starts with a random graph or a graph
                                                            constructed from PC algorithm.
            max_iter (int):                                 The number of MCMC iterations to run. Default: 30000.
            score_object (Score):                           A score object implementing compute(). If None, BGeScore is used
                                                            (data must be provided). Default: None.
            proposal_object (StructureLearningProposal):    A proposal object. If None, a GraphProposal instance is used.
                                                            Default: None.
            pc_init (bool):                                 If True and initial_graph is not given, PC algorithm will be used
                                                            to generate initial graph.
            blacklist (numpy.ndarray):                      Mask for edges to ignore in the proposal
            whitelist (numpy.ndarray):                      Mask for edges to include in the proposal
        """

        super().__init__(data=data, initial_state=initial_graph, max_iter=max_iter, score_object=score_object,
                         proposal_object='graph' if proposal_object is None else proposal_object, pc_init=pc_init,
                         blacklist=blacklist, whitelist=whitelist, seed=seed)

        self._to_string = f"Structure_MCMC_n_{self.num_nodes}_iter_{self.max_iter}"
        self.sparse = sparse
        self.score_object.incidence = self.proposal_object.current_state
        state_score = self.score_object.compute()
        self.scores = collect_node_scores(state_score)
        result = {'current_state': self.proposal_object.current_state if not sparse else scipy.sparse.csr_matrix(self.proposal_object.current_state), 'proposed_state': None, 'score_current': state_score['score'],
                  'operation': 'initial', 'accepted': False, 'acceptance_prob': 0, 'score_proposed': state_score['score']}
        self.update_results(0, result)

        self._rng = np.random.default_rng(seed=seed)

    def step(self):
        """
        Perform one MCMC iteration

        Returns:
            (dict): information on one MCMC iteration
        """
        proposed_state, operation = self.proposal_object.propose()
        nodes_to_rescore = [self.node_labels[node] for node in self.proposal_object.get_nodes_to_rescore()]

        current_state = self.proposal_object.current_state
        current_state_score = sum(list(self.scores.values()))

        if operation != StructureLearningProposal.STAY_STILL:

            self.score_object.incidence = proposed_state
            scores_copy = self.scores.copy()

            for node in nodes_to_rescore:
                scores_copy[node] = self.score_object.compute_node(node)['score']
            proposed_state_score = sum(list(scores_copy.values()))

            acceptance_prob = self.proposal_object.compute_acceptance_ratio(current_state_score, proposed_state_score)
            u =  np.log(self._rng.uniform(0,1))
            is_accepted = u < acceptance_prob
            if is_accepted:
                self.proposal_object.accept()
                self.n_accepted += 1
                self.scores = scores_copy
        else:
            is_accepted = False
            acceptance_prob = proposed_state_score = 0

        if self.sparse:
            current_state = scipy.sparse.csr_matrix(current_state)
            proposed_state = scipy.sparse.csr_matrix(proposed_state)
        return {'graph': generate_key_from_adj_matrix(current_state), 'current_state': current_state, 'proposed_state': generate_key_from_adj_matrix(proposed_state), 'score_current': current_state_score,
                'operation': operation, 'accepted': is_accepted, 'acceptance_prob': acceptance_prob, 'score_proposed': proposed_state_score, 'timestamp': time.time() - self._start_time}
