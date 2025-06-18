"""
This module implements the StructureMCMC class, which is a specialized Markov Chain Monte Carlo (MCMC) sampler for structure learning in graphical models.

The StructureMCMC class extends the base MCMC class and provides functionality for sampling graph structures based on a given dataset, scoring function, and proposal mechanism. It supports initialization with a predefined graph or a graph generated using the PC algorithm, and allows for the inclusion of edge constraints through blacklists and whitelists.

Classes:
    StructureMCMC: A class for performing Structure MCMC simulations.

Dependencies:
    - numpy
    - pandas
    - structure_learning.proposals
    - structure_learning.data_structures
    - structure_learning.scores
    - structure_learning.samplers
"""

from typing import Union
import time
import numpy as np
import pandas as pd
from structure_learning.proposals import StructureLearningProposal, GraphProposal
from structure_learning.data_structures import DAG
from structure_learning.scores import Score
from structure_learning.samplers import MCMC

class StructureMCMC(MCMC):
    """
    Implementation of Structure MCMC.
    """
    def __init__(self, data: pd.DataFrame = None, initial_graph : np.ndarray = None, max_iter : int = 30000,
                 score_object : Union[str, Score] = None, proposal_object : StructureLearningProposal = None, pc_init = True,
                 blacklist: np.ndarray = None, whitelist: np.ndarray = None, seed: int = None, sparse=True,
                 result_type: str = 'distribution', graph_type='dag'):
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
                         blacklist=blacklist, whitelist=whitelist, seed=seed, result_type=result_type, graph_type=graph_type)

        self._to_string = f"Structure_MCMC_n_{self.num_nodes}_iter_{self.max_iter}"
        self.sparse = sparse

        if proposal_object is None:
            if self.initial_state is None:
                self.initial_state = self._pc_state if pc_init else DAG.generate_random(self.node_labels, 0.5, seed)
                if whitelist is not None:
                    self.initial_state.incidence[whitelist > 0] = True
                if blacklist is not None:
                    self.initial_state.incidence[blacklist > 0] = False
            proposal_object = GraphProposal(initial_state=self.initial_state, blacklist=blacklist, whitelist=whitelist, seed=seed)
        elif not isinstance(proposal_object, StructureLearningProposal):
            raise Exception('Unsupported proposal', proposal_object)
        
        self.proposal_object = proposal_object
        state_score = self.score_object.compute(self.proposal_object.current_state)
        self.scores = {node: info['score'] for node, info in state_score['parameters'].items()}
        result = {'graph': self.proposal_object.current_state, 'current_state': self.proposal_object.current_state, 'proposed_state': None, 'score_current': state_score['score'],
                  'operation': 'initial', 'accepted': False, 'acceptance_prob': 0, 'score_proposed': state_score['score'], 'timestamp': 0}
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

            self.score_object.graph = proposed_state
            scores_copy = self.scores.copy()

            for node in nodes_to_rescore:
                scores_copy[node] = self.score_object.compute_node(proposed_state, node)['score']
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

        return {'graph': current_state, 'current_state': current_state, 'proposed_state': proposed_state, 'score_current': current_state_score,
                'operation': operation, 'accepted': is_accepted, 'acceptance_prob': acceptance_prob, 'score_proposed': proposed_state_score, 'timestamp': time.time() - self._start_time}
