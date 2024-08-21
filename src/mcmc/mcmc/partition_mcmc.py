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
    def __init__(self, data : pd.DataFrame = None, initial_partition : OrderedPartition = None, max_iter : int = 30000,
                 proposal_object : StructureLearningProposal = None, score_object : Union[str, Score] = None,
                 pc_init: bool = True, blacklist = None, whitelist = None, plus1: bool = False):
        """
        Initilialise Partition MCMC instance.

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
            plus1 (bool):                                   Use plus1 neighborhood
        """
        super().__init__(data=data, initial_state=initial_partition, max_iter=max_iter, proposal_object=proposal_object,
                         score_object=score_object, pc_init=pc_init, blacklist=blacklist, whitelist=whitelist, plus1=plus1)
        self._to_string = f"Structure_MCMC_n_{self.num_nodes}_iter_{self.max_iter}"

        self._node_label_to_idx = node_label_to_index(self.node_labels)

        self._max_parents = self.num_nodes - 1

        self.parent_table = list_possible_parents(self._max_parents, self.node_labels, whitelist=whitelist,
                                                  blacklist=blacklist, plus1=plus1, init_cpdag=self.cpdag)
        self.score_table = score_possible_parents(self.parent_table, self.node_labels, self.score_object)

        # compute the scores of the current partition
        current_state = self.proposal_object.current_state
        party_curr, permy_curr, posy_curr = convert_partition_to_party_permy_posy(current_state)
        self.scores = partition_score(self.node_labels, self.node_labels, self.parent_table, self.score_table, permy_curr, party_curr, posy_curr)
        current_state_score = sum(self.scores['total_scores'].values())

        # Sample a DAG from the initial partition
        G = sample_score(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)['incidence']

        result = {'graph': G, 'partition': current_state, 'party': party_curr, 'permy': permy_curr, 'posy': posy_curr, 'operation': 'initial', 'accepted' : False,
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

        while True:
            proposed_state, operation = self.proposal_object.propose()
            if self.is_valid_partition(proposed_state):
                break
            print('Reproposing...')

        nodes_to_rescore = self.proposal_object.get_nodes_to_rescore()

        if operation == StructureLearningProposal.STAY_STILL:
            G = sample_score(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)['incidence']
            result = self.current_step.copy()
            result['operation'] = operation
            result['accepted'] = False
            result['graph'] = G
        else:
            proposed_state = proposed_state
            party_prop, permy_prop, posy_prop = convert_partition_to_party_permy_posy(proposed_state)
            scores_copy = deepcopy(self.scores)
            rescore = partition_score(list(nodes_to_rescore), self.node_labels, self.parent_table, self.score_table, permy_prop, party_prop, posy_prop)
            for key in scores_copy:
                scores_copy[key].update(rescore[key])
            proposed_state_score = sum(scores_copy['total_scores'].values())

            u = np.log(np.random.uniform(0, 1))
            acceptance_prob = self.proposal_object.compute_acceptance_ratio(current_state_score, proposed_state_score)
            is_accepted = u < acceptance_prob

            if is_accepted:
                self.proposal_object.accept()
                self.n_accepted += 1
                self.scores = scores_copy
                current_state_score = proposed_state_score
                party_curr, permy_curr, posy_curr = party_prop, permy_prop, posy_prop

            G = sample_score(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)['incidence']

            result = {'graph': G, 'partition': self.proposal_object.current_state, 'party': party_curr, 'permy': permy_curr, 'posy': posy_curr,
                      'operation': operation, 'accepted' : is_accepted, 'score_current' : current_state_score, 'score_proposed' : proposed_state_score,
                      'acceptance_prob' : acceptance_prob, 'proposed_state': proposed_state}
        self.current_step = result
        return result

    def is_valid_partition(self, partition: OrderedPartition):
        """
        Check if partition adheres to blacklist and whitelist

        Parameter:
            (partition): partition to check

        Returns:
            (bool): partition validity
        """
        node_labels = np.array(self.node_labels)

        if self.whitelist is None and self.blacklist is None:
            return True

        if self.whitelist is not None: # check if partition adheres to node ordering based on required edges
            for node in self._node_label_to_idx:
                idx = self._node_label_to_idx[node]
                part_idx = partition.find_node(node)

                required_parents = node_labels[self.whitelist[:,idx]==1]

                for parent in required_parents:
                    part_idx_parent = partition.find_node(parent)
                    if part_idx >= part_idx_parent:
                        return False

        if self.blacklist is not None: # check if partition adheres to node ordering based on banned edges
            for node in self._node_label_to_idx:
                idx = self._node_label_to_idx[node]
                part_idx = partition.find_node(node)

                banned_parents = node_labels[self.blacklist[:,idx]==1]

                for parent in banned_parents:
                    part_idx_parent = partition.find_node(parent)

                    if part_idx == part_idx_parent+1: # parent node appears in partition immediately to the left
                        if partition.partitions[part_idx_parent].size == 1:
                            return False

        return True
