"""

"""
import time
from copy import deepcopy
from typing import Union
import numpy as np
import pandas as pd
import networkx as nx
from mcmc.utils.graph_utils import *
from mcmc.utils.partition_utils import *
from mcmc.utils.score_utils import *
from mcmc.proposals import StructureLearningProposal
from mcmc.proposals import PartitionProposal
from mcmc.data_structures import OrderedPartition
from mcmc.scores import Score, BGeScore, BDeuScore
from mcmc.mcmc import MCMC

class PartitionMCMC(MCMC):
    """
    Implementation of Partition MCMC.
    """
    def __init__(self, data : pd.DataFrame = None, initial_state : Union[OrderedPartition, np.ndarray] = None, max_iter : int = 30000,
                 proposal_object : StructureLearningProposal = None, score_object : Union[str, Score] = None,
                 pc_init: bool = True, blacklist = None, whitelist = None, searchspace = None, plus1: bool = False, seed : int  = 32):
        """
        Initilialise Partition MCMC instance.

        Parameters:
            data (pd.DataFrame):                                    Dataset. Optional if score_object is given.
            initial_state (numpy.ndarray | OrderedPartition):       Initial graph/partition for the MCMC simulation.
                                                                    If None, simulation starts with a random graph or a graph
                                                                    constructed from PC algorithm.
            max_iter (int):                                         The number of MCMC iterations to run. Default: 30000.
            score_object (Score):                                   A score object implementing compute(). If None, BGeScore is used
                                                                    (data must be provided). Default: None.
            proposal_object (StructureLearningProposal):            A proposal object. If None, a GraphProposal instance is used.
                                                                    Default: None.
            pc_init (bool):                                         If True and initial_graph is not given, PC algorithm will be used
                                                                    to generate initial graph (start DAG).
            blacklist (numpy.ndarray):                              Mask for edges to ignore in the proposal
            whitelist (numpy.ndarray):                              Mask for edges to include in the proposal
            searchspace (str | numpy.ndarray):                      Graph search space. "FULL" | "PC" | np.ndarray | None. If none, full search space is used.
            plus1 (bool):                                           Use plus1 neighborhood
        """
        if isinstance(initial_state, np.ndarray):
            initial_state = build_partition(initial_state, list(score_object.data.columns if score_object else data.columns))

        super().__init__(data=data, initial_state=initial_state, max_iter=max_iter, proposal_object=proposal_object,
                         score_object=score_object, pc_init=(pc_init or searchspace=="PC"), blacklist=blacklist, whitelist=whitelist, plus1=plus1, seed=seed)
        self._to_string = f"Partition_MCMC_n_{self.num_nodes}_iter_{self.max_iter}"

        self._node_label_to_idx = node_label_to_index(self.node_labels)

        self._max_parents = self.num_nodes - 1

        if isinstance(searchspace, str):
            if searchspace == "FULL":
                searchspace = np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)
            elif searchspace == "PC":
                searchspace = self.pc_graph
            else:
                raise Exception("Unsupported search space")
        print(searchspace)

        self.parent_table = list_possible_parents(self._max_parents, self.node_labels, whitelist=whitelist,
                                                  blacklist=blacklist, plus1=plus1, searchspace=searchspace)
        self.score_table = score_possible_parents(self.parent_table, self.node_labels, self.score_object)
        self._rng = np.random.default_rng(seed=seed)

        # compute the scores of the current partition
        current_state = self.proposal_object.current_state
        party_curr, permy_curr, posy_curr = convert_partition_to_party_permy_posy(current_state)
        self.scores = partition_score(self.node_labels, self.node_labels, self.parent_table, self.score_table, permy_curr, party_curr, posy_curr)
        current_state_score = sum(self.scores['total_scores'].values())

        # Sample a DAG from the initial partition
        sample = sample_score(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)
        G, DAG_score = sample['incidence'], sample['logscore']

        result = {'graph': G, 'DAG_score': DAG_score, 'partition': current_state, 'party': party_curr, 'permy': permy_curr, 'posy': posy_curr, 'operation': 'initial', 'accepted' : False,
                  'score_current' : current_state_score, 'score_proposed' : -1, 'acceptance_prob' : -1, 'proposed_state': None}
        self.current_step = result
        self.update_results(0, result)

    def step(self):
        """
        Perform one MCMC iteration

        Returns:
            (dict): information on one MCMC iteration
        """
        current_state = self.current_step['partition']
        current_state_score = self.current_step['score_current']
        party_curr, permy_curr, posy_curr = self.current_step['party'], self.current_step['permy'], self.current_step['posy']

        proposed_state, operation = self.proposal_object.propose()

        nodes_to_rescore = self.proposal_object.get_nodes_to_rescore()

        if operation == StructureLearningProposal.STAY_STILL:
            sample = sample_score(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)
            G, DAG_score = sample['incidence'], sample['logscore']
            result = self.current_step.copy()
            result['DAG_score'] = DAG_score
            result['operation'] = operation
            result['accepted'] = False
            result['graph'] = G
        else:
            party_prop, permy_prop, posy_prop = convert_partition_to_party_permy_posy(proposed_state)
            scores_copy = deepcopy(self.scores)
            rescore = partition_score(list(nodes_to_rescore), self.node_labels, self.parent_table, self.score_table, permy_prop, party_prop, posy_prop)
            for key in scores_copy:
                scores_copy[key].update(rescore[key])
            proposed_state_score = sum(scores_copy['total_scores'].values())

            u = np.log(self._rng.uniform(0, 1))
            acceptance_prob = self.proposal_object.compute_acceptance_ratio(current_state_score, proposed_state_score)
            is_accepted = u < acceptance_prob

            if is_accepted:
                self.proposal_object.accept()
                self.n_accepted += 1
                self.scores = scores_copy
                current_state_score = proposed_state_score
                party_curr, permy_curr, posy_curr = party_prop, permy_prop, posy_prop

            sample = sample_score(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)
            G, DAG_score = sample['incidence'], sample['logscore']

            result = {'graph': G, 'DAG_score': DAG_score, 'partition': self.proposal_object.current_state, 'party': party_curr, 'permy': permy_curr, 'posy': posy_curr,
                      'operation': operation, 'accepted' : is_accepted, 'score_current' : current_state_score, 'score_proposed' : proposed_state_score,
                      'acceptance_prob' : acceptance_prob, 'proposed_state': proposed_state}
        self.current_step = result
        return result
