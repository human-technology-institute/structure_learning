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

    """
    def __init__(self, init_part : Union[OrderedPartition, np.ndarray] = None, max_iter : int = 30000, proposal_object : StructureLearningProposal = None, score_object : Union[str, Score] = None, data : pd.DataFrame = None, pc_init: bool = True, blacklist = None, whitelist = None, plus1: bool = False):

        self.to_string = "Partition MCMC"

        if score_object is None:
            if data is None:
                raise Exception("Data must be provided")
            else:
                score_object = BGeScore(data=data, incidence=None)
        elif type(score_object) == str:
            if score_object.lower() == 'bge':
                if data is None:
                    raise Exception("Data must be provided")
                else:
                    score_object = BGeScore(data=data, incidence=None)
            elif score_object.lower() in ['bde', 'bdeu']:
                if data is None:
                    raise Exception("Data must be provided")
                else:
                    score_object = BDeuScore(data=data, incidence=None)
            else:
                raise Exception(f"Unsupported score {score_object}")

        cpdag = None
        if proposal_object is None:
            if init_part is None:
                n_nodes = len(score_object.data.columns)
                if pc_init:
                    print('Running PC algorithm')
                    initial_graph, cpdag = initial_graph_pc(score_object.data, True)
                else: # start with random
                    initial_graph = generate_DAG(n_nodes, 0.5)
                init_part = build_partition(incidence=initial_graph if not plus1 else np.zeros((n_nodes, n_nodes)), node_labels=list(score_object.data.columns))
            elif isinstance(init_part, np.ndarray):
                init_part = build_partition(init_part, list(score_object.data.columns))
            proposal_object = PartitionProposal(ordered_partition=init_part, whitelist=whitelist, blacklist=blacklist)

        super().__init__(score_object.data, None, max_iter, score_object, proposal_object)

        self.init_part = self.proposal_object.ordered_partition
        self.blacklist = blacklist
        self.whitelist = whitelist

        self._node_label_to_idx = node_label_to_index(self.node_labels)

        self._max_parents = self.num_nodes - 1

        self.parent_table = list_possible_parents(self._max_parents, self.node_labels, whitelist=whitelist, blacklist=blacklist, plus1=plus1, init_cpdag=cpdag)
        self.score_table = score_possible_parents(self.parent_table, self.node_labels, self.score_object)
        self.mcmc_res = {}

    def update_MCMC_res( self, iteration, graph, partition, party, permy, posy,  operation, isAccepted, score_P_curr, score_P_prop, acceptance_prob ):

        self.mcmc_res[iteration] = {}
        self.mcmc_res[iteration] =  {"graph": graph,
                                    "partition": partition,
                                    "party": party,
                                    "permy": permy,
                                    "posy": posy,

                                    "operation": operation,
                                    "accepted" : isAccepted,

                                    "score_P_curr" : score_P_curr,
                                    "score_P_prop" : score_P_prop,
                                    "acceptance_prob" : acceptance_prob}

    def run( self ):
        acceptance_prob = -1
        ACCEPT = 0
        iter = 0
        mcmc_res = {}

        # start with the current partition
        P_curr = self.init_part.copy()
        chosen_move = "initial"

        # compute the scores of the current partition
        party_curr, permy_curr, posy_curr = convert_partition_to_party_permy_posy( P_curr )
        all_scores_curr = partition_score(self.node_labels, self.node_labels, self.parent_table, self.score_table, permy_curr, party_curr, posy_curr )
        score_P_curr = sum(all_scores_curr['total_scores'].values())

        self.proposal_object.set_ordered_partition( P_curr )
        neigh_size_curr, _  = self.proposal_object.compute_neighborhoods()  # get the size of the neighborhood of the current partition
        nbh_join_curr = self.proposal_object.get_nbh_join_existing()        # get the number of ways to join partitions from the current partition
        nbh_create_curr = self.proposal_object.get_nbh_create_new()         # get the number of ways to create new partitions from the current partition

        # Sample a DAG from the initial partition
        G = sample_score(self.num_nodes, self.node_labels, all_scores_curr , self.parent_table, self._node_label_to_idx)['incidence']
        self.update_MCMC_res(0, G, P_curr, party_curr, permy_curr, posy_curr, "MCMC Start", -1, score_P_curr, -1, -1 ) # save the results

        t = time.time()

        for _ in range( self.max_iter):

            is_accepted = 0
            acceptance_prob = 0

            t = time.time()
            self.proposal_object = PartitionProposal( P_curr )
            while True:
                P_prop = self.proposal_object.propose_partition()
                if self.is_valid_partition(P_prop):
                    break
                print('Reproposing...')

            self.proposal_object.set_ordered_partition( P_prop )
            chosen_move =  self.proposal_object.get_chosen_move()
            nodes_to_rescore = self.proposal_object.get_to_rescore()

            t = time.time()
            neigh_size_prop, _  = self.proposal_object.compute_neighborhoods()  # get the size of the neighborhood of the proposed partition
            nbh_join_prop = self.proposal_object.get_nbh_join_existing()        # get the number of ways to join partitions from the proposed partition
            nbh_create_prop = self.proposal_object.get_nbh_create_new()         # get the number of ways to create new partitions from the proposed partition

            if chosen_move == "stay_still":
                self.update_MCMC_res(iter, G, P_curr, party_curr, permy_curr, posy_curr,  chosen_move, 0, score_P_curr, score_P_curr, acceptance_prob )
                iter += 1
                continue

            party_prop, permy_prop, posy_prop = convert_partition_to_party_permy_posy( P_prop )

            t = time.time()
            rescored_scores_prop_dict = partition_score(list( nodes_to_rescore), self.node_labels, self.parent_table, self.score_table, permy_prop, party_prop, posy_prop )
            all_scores_prop = deepcopy(all_scores_curr)
            for key in all_scores_prop:
                all_scores_prop[key].update( rescored_scores_prop_dict[key])

            score_P_prop = sum(all_scores_prop['total_scores'].values())

            # sample alpha uniformly from [0,1]
            alpha = np.log( np.random.uniform(0, 1) )
            acceptance_prob = self.log_acceptance_ratio( chosen_move, score_P_curr, score_P_prop, neigh_size_curr, neigh_size_prop, nbh_join_prop, nbh_join_curr, nbh_create_prop, nbh_create_curr )

            try:
                acceptance_prob = acceptance_prob[0]
            except Exception:
                acceptance_prob

            if alpha < acceptance_prob:

                ACCEPT += 1
                P_curr = P_prop  # accept the proposal
                self.proposal_object.set_ordered_partition( P_prop )

                party_curr = party_prop
                permy_curr = permy_prop
                posy_curr = posy_prop
                all_scores_curr = all_scores_prop
                score_P_curr = score_P_prop
                neigh_size_curr = neigh_size_prop
                nbh_join_curr =  nbh_join_prop
                nbh_create_curr = nbh_create_prop
                is_accepted = 1
            iter += 1

            t = time.time()
            G = sample_score(self.num_nodes, self.node_labels, all_scores_curr , self.parent_table, self._node_label_to_idx)['incidence']
            self.update_MCMC_res(iter, G, P_curr, party_curr, permy_curr, posy_curr,  chosen_move, is_accepted, score_P_curr, score_P_prop, acceptance_prob )

        return self.mcmc_res, np.round(ACCEPT / self.max_iter,4)

    def log_acceptance_ratio( self, chosen_move, score_P_curr, score_P_prop, neigh_size_curr, neigh_size_prop, nbh_join_prop, nbh_join_curr, nbh_create_prop, nbh_create_curr ):

        if PartitionProposal.SWAP_ADJACENT == chosen_move or PartitionProposal.SWAP_GLOBAL == chosen_move:
            return self.log_acceptance_ratio_swap( score_P_curr, score_P_prop )

        if PartitionProposal.SPLIT_PARTITIONS == chosen_move or PartitionProposal.MERGE_PARTITIONS == chosen_move:

            return self.log_acceptance_ratio_join_merge( neigh_size_curr, neigh_size_prop, score_P_curr, score_P_prop )

        if PartitionProposal.MOVE_NODE_TO_EXISTING_PARTITION == chosen_move or PartitionProposal.MOVE_NODE_TO_NEW_PARTITION == chosen_move:
            return self.log_acceptance_ratio_new_existing( nbh_join_prop, nbh_join_curr, nbh_create_prop, nbh_create_curr, score_P_curr, score_P_prop )
        print(chosen_move)

    def log_acceptance_ratio_swap( self, score_P_curr, score_P_prop ):
        return min(0, score_P_prop - score_P_curr)

    def log_acceptance_ratio_join_merge( self,neigh_size_curr, neigh_size_prop, score_P_curr, score_P_prop ):
        return min(0, score_P_prop - score_P_curr + np.log(neigh_size_curr) - np.log(neigh_size_prop))

    def log_acceptance_ratio_new_existing( self, nbh_join_prop, nbh_join_curr, nbh_create_prop, nbh_create_curr, score_P_curr, score_P_prop ):

        numerator =  score_P_prop + np.log(nbh_join_curr + nbh_create_curr)
        denominator = score_P_curr + np.log(nbh_join_prop + nbh_create_prop)

        return min(0, numerator - denominator )

    def to_string(self):
        return self.to_string

    def get_mcmc_res_graphs(self, results):
        mcmc_graph_lst = []
        for i in results:
            mcmc_graph_lst.append( results[i]['graph'] )
        return mcmc_graph_lst

    def is_valid_partition(self, partition: OrderedPartition):

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
