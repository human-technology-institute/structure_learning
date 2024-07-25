"""

"""
import time
from copy import deepcopy
import numpy as np
from mcmc.utils.graph_utils import *
from mcmc.utils.partition_utils import *
from mcmc.utils.score_utils import *
from mcmc.proposals import StructureLearningProposal
from mcmc.proposals import PartitionProposal
from mcmc.data_structures import OrderedPartition
from mcmc.scores import Score
from mcmc.mcmc import MCMC

class PartitionMCMC(MCMC):
    """

    """
    def __init__(self, init_part : OrderedPartition, max_iter : int, proposal_object : StructureLearningProposal, score_object : Score):
        #super().__init__(init_part, proposal_object, score_object)

        self.to_string = "Partition MCMC"
        super().__init__(score_object.data, None, max_iter, score_object, proposal_object)
        self.init_part = init_part

        self._node_label_to_idx = node_label_to_index(self.node_labels)

        self._max_parents = self.num_nodes - 1

        self.parent_table = list_possible_parents( self._max_parents, self.node_labels)
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
        print('Initialise: ', '{:.5f}'.format(time.time()-t))

        for _ in range( self.max_iter):

            is_accepted = 0
            acceptance_prob = 0

            # Propose a new partition
            # print("Current Partition: ", P_curr.__str__())

            t = time.time()
            self.proposal_object = PartitionProposal( P_curr )
            # print("Current Partition: ", P_curr.__str__())
            P_prop = self.proposal_object.propose_partition()
            # print("Proposed Partition: ", P_prop.__str__())
            # print('Rescoring: ', self.proposal_object.to_rescore)
            self.proposal_object.set_ordered_partition( P_prop )
            chosen_move =  self.proposal_object.get_chosen_move()
            nodes_to_rescore = self.proposal_object.get_to_rescore()
            # print('Propose: ', '{:.5f}'.format(time.time()-t))
            # print("Proposed Partition: ", P_prop.__str__())

            t = time.time()
            neigh_size_prop, _  = self.proposal_object.compute_neighborhoods()  # get the size of the neighborhood of the proposed partition
            nbh_join_prop = self.proposal_object.get_nbh_join_existing()        # get the number of ways to join partitions from the proposed partition
            nbh_create_prop = self.proposal_object.get_nbh_create_new()         # get the number of ways to create new partitions from the proposed partition
            # print('Compute neighborhood: ', '{:.5f}'.format(time.time()-t))

            if chosen_move == "stay_still":
                self.update_MCMC_res(iter, G, P_curr, party_curr, permy_curr, posy_curr,  chosen_move, 0, score_P_curr, score_P_curr, acceptance_prob )
                iter += 1
                continue

            # Get the party, permy and posy of the proposed partition
            # self.proposal_object.set_ordered_partition( P_prop )
            party_prop, permy_prop, posy_prop = convert_partition_to_party_permy_posy( P_prop )

            t = time.time()
            rescored_scores_prop_dict = partition_score(list( nodes_to_rescore), self.node_labels, self.parent_table, self.score_table, permy_prop, party_prop, posy_prop )

            # rescored_scores_prop_dict2 = partition_score(self.node_labels, self.node_labels, self.parent_table, self.score_table, permy_prop, party_prop, posy_prop )
            #rescored_scores_prop_indx = rescored_scores_prop_dict['total_scores']
            #rescored_curr = [ all_scores_curr['total_scores'][i] for i in   list( rescored_scores_prop_indx.keys()   ) ]
            #score_P_prop = (score_P_curr - sum(rescored_curr) + sum( list(rescored_scores_prop_dict['total_scores'].values())))

            all_scores_prop = deepcopy(all_scores_curr)
            for key in all_scores_prop:
                all_scores_prop[key].update( rescored_scores_prop_dict[key])

            score_P_prop = sum(all_scores_prop['total_scores'].values())
            # print('Score: ', '{:.5f}'.format(time.time()-t))

            # sample alpha uniformly from [0,1]
            alpha = np.log( np.random.uniform(0, 1) )
            #print("Score P curr", score_P_curr)
            acceptance_prob = self.log_acceptance_ratio( chosen_move, score_P_curr, score_P_prop, neigh_size_curr, neigh_size_prop, nbh_join_prop, nbh_join_curr, nbh_create_prop, nbh_create_curr )
            # print('Score: ', score_P_prop, score_P_curr, acceptance_prob, alpha)

            try:
                acceptance_prob = acceptance_prob[0]
            except Exception:
                acceptance_prob

            # print("acceptance_prob: ", acceptance_prob)
            # print("\n")
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
            # print('Sample: ', '{:.5f}'.format(time.time()-t))
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
