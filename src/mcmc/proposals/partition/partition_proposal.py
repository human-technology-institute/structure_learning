import numpy as np
import math
from mcmc.utils.partition_utils import *
from mcmc.data_structures.partition import *

class PartitionProposal():

    SWAP_ADJACENT = "swap_adjacent"
    SWAP_GLOBAL = "swap_global"
    SPLIT_OR_MERGE = "split_or_merge"
    MERGE_PARTITIONS = "merge_partitions"
    SPLIT_PARTITIONS = "split_partitions"
    MOVE_NODE_TO_NEW_OR_EXISTING = "move_node_to_new_or_existing"
    MOVE_NODE_TO_EXISTING_PARTITION = "move_node_to_existing_partition"
    MOVE_NODE_TO_NEW_PARTITION = "move_node_to_new_partition"
    STAY_STILL = "stay_still"

    possible_moves = [SWAP_ADJACENT, SWAP_GLOBAL, SPLIT_OR_MERGE, MOVE_NODE_TO_NEW_OR_EXISTING, STAY_STILL]

    def __init__(self, ordered_partition: OrderedPartition ):

        self.update_partition(ordered_partition)
        self.chosen_move = None
        self.to_rescore = set()
        self.num_moves = len(self.possible_moves)
        self.nbh_join_existing = None
        self.nbh_create_new = None
        self.move_probs = self.calculate_move_probs()

    def update_partition(self, new_partition):
        self.ordered_partition = new_partition
        self.num_part = len(self.ordered_partition.partitions)
        self.nbh_size, self.nbh = self.compute_neighborhoods()
        self.nodes = self.ordered_partition.get_all_nodes()
        self.num_nodes = len(self.nodes)

    def compute_neighborhoods( self ):
        """
        Compute the number of neighborhoods of an ordered partiion by using the equation
        sum_{i=1}^{m} ( sum_{c=1}^{k_i-1} ( comb( k_i, c) ) ) + m - 1
        """
        comb_lst = [sum(math.comb(part_i.size, c) for c in range(1, part_i.size)) for part_i in self.ordered_partition.partitions]
        return np.sum(comb_lst) + self.num_part - 1, comb_lst

    def propose_partition(self):

        # reset the nodes to rescore
        self.set_to_rescore( set() )
        while True:
            chosen_move = np.random.choice(self.possible_moves, size=1, p=self.move_probs)[0]
            if not("swap" in chosen_move and self.ordered_partition.size < 2):
                break

        if chosen_move == self.SWAP_ADJACENT:
            return self.swap_adjacent()
        elif chosen_move == self.SWAP_GLOBAL:
            return self.swap_global()
        elif chosen_move == self.SPLIT_OR_MERGE:
            return self.split_or_merge_move()
        elif chosen_move == self.MOVE_NODE_TO_NEW_OR_EXISTING:
            return self.move_node_to_existing_or_new_partition()
        elif chosen_move == self.STAY_STILL:
            self.chosen_move = self.STAY_STILL
            return self.ordered_partition.copy()

    def calculate_move_probs(self):

        # Choose the probability of the different moves
        prob1start = 40 / 100
        prob1 = prob1start * 100

        if self.num_nodes > 3:
            prob1 = round(6 * prob1 * self.num_nodes / (self.num_nodes**2 + 10 * self.num_nodes - 24))
        prob1 /= 100

        prob2start = 99 / 100 - prob1start
        prob2 = prob2start * 100
        if self.num_nodes > 3:
            prob2 = round(6 * prob2 * self.num_nodes / (self.num_nodes**2 + 10 * self.num_nodes - 24))
        prob2 /= 100

        move_probs = np.array([prob1, prob1start - prob1, prob2start - prob2, prob2, 0.01])
        move_probs /= move_probs.sum()  # Normalization

        return move_probs

    def swap_adjacent(self):

        assert self.ordered_partition.size >= 2
        self.chosen_move = self.SWAP_ADJACENT
        new_ordered_partition = self.ordered_partition.copy()

        # choose random partition element
        party, _, _ = convert_partition_to_party_permy_posy(new_ordered_partition)
        p = np.array(possible_permutations_neighbors(sum(party), party))
        part_idx = np.random.choice(new_ordered_partition.size - 1, p=p/p.sum())
        # choose a node from partition at part_idx
        node_left = np.random.choice(list(new_ordered_partition.partitions[part_idx].nodes))
        # choose a node from partition at part_idx+1
        node_right = np.random.choice(list(new_ordered_partition.partitions[part_idx+1].nodes))

        # swap
        new_ordered_partition.partitions[part_idx].remove_single_node(node_left)
        new_ordered_partition.partitions[part_idx].add_single_node(node_right)

        new_ordered_partition.partitions[part_idx+1].remove_single_node(node_right)
        new_ordered_partition.partitions[part_idx+1].add_single_node(node_left)

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(new_ordered_partition.get_all_nodes_from_right(part_idx+1))
        self.to_rescore = self.to_rescore.union({node_right})

        new_ordered_partition.update_IDs()

        return new_ordered_partition

    def swap_global(self ):

        assert self.ordered_partition.size >= 2
        self.chosen_move = self.SWAP_GLOBAL
        new_ordered_partition = self.ordered_partition.copy()

        # choose random partition elements
        party, _, _ = convert_partition_to_party_permy_posy(new_ordered_partition)
        p = np.array(possible_permutations(sum(party), party))
        part_idx_1 = np.random.choice(new_ordered_partition.size, p=p/p.sum())
        part_idx_2 = np.random.choice(list(set(range(new_ordered_partition.size)).difference({part_idx_1})))

        left_part_idx, right_part_idx = sorted([part_idx_1, part_idx_2])

        # choose a node from partition at left_part_idx
        node_left = np.random.choice(list(new_ordered_partition.partitions[left_part_idx].nodes))
        # choose a node from partition at right_part_idx
        node_right = np.random.choice(list(new_ordered_partition.partitions[right_part_idx].nodes))

        # swap
        new_ordered_partition.partitions[left_part_idx].remove_single_node(node_left)
        new_ordered_partition.partitions[left_part_idx].add_single_node(node_right)

        new_ordered_partition.partitions[right_part_idx].remove_single_node(node_right)
        new_ordered_partition.partitions[right_part_idx].add_single_node(node_left)

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(new_ordered_partition.get_all_nodes_from_right(left_part_idx))

        new_ordered_partition.update_IDs()

        return new_ordered_partition

    def split_or_merge_move(self):

        party, _, _ = convert_partition_to_party_permy_posy(self.ordered_partition)
        p = np.array(possible_splits_joins(sum(party), party))
        p = p if p.sum() else p+1
        node = np.random.choice(list(self.ordered_partition.all_nodes), p=p/p.sum())
        idx = self.ordered_partition.find_node(node)

        can_split = self.ordered_partition.partitions[idx].size >= 2
        if can_split and np.random.choice([0,1]):
            return self.split_move(idx)
        else:
            return self.join_move(idx)

    def split_move(self, idx : int):

        self.chosen_move = self.SPLIT_PARTITIONS
        new_ordered_partition = self.ordered_partition.copy()
        assert new_ordered_partition.partitions[idx].size >= 2

        # randomly select which nodes at partition idx will be at the new left partition
        n_nodes_left = np.random.randint(1, new_ordered_partition.partitions[idx].size)
        nodes_involved = list(new_ordered_partition.partitions[idx].nodes)
        nodes_left = np.random.choice(nodes_involved, replace=False, size=n_nodes_left)

        # split
        new_ordered_partition.partitions[idx].remove_nodes(nodes_left)
        new_ordered_partition.insert_partition(idx, set(nodes_left))

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(new_ordered_partition.get_all_nodes_from_right(idx))

        # remove empty partition if exists
        new_ordered_partition.remove_empty_partitions()

        new_ordered_partition.update_IDs()

        return new_ordered_partition

    def join_move(self, idx : int):

        assert self.ordered_partition.size >= 2
        self.chosen_move = self.MERGE_PARTITIONS
        new_ordered_partition = self.ordered_partition.copy()

        adj = np.random.choice([-1,1])
        if idx == 0:
            adj = 1
        if idx == new_ordered_partition.size-1:
            adj = -1

        left_part_idx, right_part_idx = sorted([idx, idx+adj])

        # join partitions at idx and idx+1
        nodes_right = new_ordered_partition.partitions[right_part_idx].nodes
        new_ordered_partition.partitions[right_part_idx].remove_nodes(nodes_right)
        nodes_left = new_ordered_partition.partitions[left_part_idx].nodes
        new_ordered_partition.partitions[left_part_idx].add_nodes(nodes_right)

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(new_ordered_partition.get_all_nodes_from_right(left_part_idx))

        # remove empty partition if exists
        new_ordered_partition.remove_empty_partitions()

        new_ordered_partition.update_IDs()

        return new_ordered_partition

    def node_to_existing_partition(self):

        assert self.ordered_partition.size >= 2
        self.chosen_move = self.MOVE_NODE_TO_EXISTING_PARTITION
        new_ordered_partition = self.ordered_partition.copy()

        # sample a node
        nodes = new_ordered_partition.get_all_nodes()
        party, _, posy = convert_partition_to_party_permy_posy(new_ordered_partition)
        p = np.array(calculate_join_possibilities(len(nodes), party, posy))
        node_to_move = np.random.choice(list(nodes), p=p/p.sum())

        # get partition where the node belongs to
        current_partition_idx = new_ordered_partition.find_node(node_to_move)

        # select a different partition to move to
        target_partition_idx = np.random.choice(list(set(range(new_ordered_partition.size)).difference({current_partition_idx})))

        # move
        new_ordered_partition.partitions[current_partition_idx].remove_single_node(node_to_move)
        new_ordered_partition.partitions[target_partition_idx].add_single_node(node_to_move)

        # mark nodes to rescore
        min_idx, max_idx = sorted([current_partition_idx, target_partition_idx])
        self.to_rescore = self.to_rescore.union(new_ordered_partition.get_all_nodes_from_right(min_idx))
        self.to_rescore = self.to_rescore.union({node_to_move})

        # remove empty partition if exists
        new_ordered_partition.remove_empty_partitions()

        new_ordered_partition.update_IDs()

        return new_ordered_partition

    def node_to_new_partition(self):

        self.chosen_move = self.MOVE_NODE_TO_NEW_PARTITION
        new_ordered_partition = self.ordered_partition.copy()

        # sample a node
        nodes = new_ordered_partition.get_all_nodes()
        party, _, posy = convert_partition_to_party_permy_posy(new_ordered_partition)
        p = np.array(calculate_partition_transitions(len(nodes), party, posy))
        node_to_move = np.random.choice(list(nodes), p=p/p.sum())

        # get partition where the node belongs to
        current_partition_idx = new_ordered_partition.find_node(node_to_move)

        # sample where to go
        target_partition_idx = np.random.choice(p[list(nodes).index(node_to_move)])
        if new_ordered_partition.partitions[current_partition_idx].size == 1:
            if target_partition_idx >= current_partition_idx:
                target_partition_idx += 1
                if current_partition_idx < new_ordered_partition.size-1 and new_ordered_partition.partitions[current_partition_idx+1].size == 1:
                    target_partition_idx += 1

        # remove node from current partition
        new_ordered_partition.partitions[current_partition_idx].remove_single_node(node_to_move)

        # insert new partition containing the node to target index
        new_ordered_partition.insert_partition(target_partition_idx,  {node_to_move})

        # mark nodes to rescore
        min_idx, max_idx = sorted([current_partition_idx, target_partition_idx])
        self.to_rescore = self.to_rescore.union(new_ordered_partition.get_all_nodes_from_right(min_idx))
        self.to_rescore = self.to_rescore.union({node_to_move})

        # remove empty partition if exists
        new_ordered_partition.remove_empty_partitions()

        new_ordered_partition.update_IDs()

        return new_ordered_partition

    def move_node_to_existing_or_new_partition(self):

        party, _, posy = convert_partition_to_party_permy_posy( self.ordered_partition )
        num_nodes = sum(party)

        self.nbh_join_existing = sum(calculate_join_possibilities(num_nodes, party, posy ))
        self.nbh_create_new = sum(calculate_partition_transitions(num_nodes, party, posy ))

        prob_join_existing = self.nbh_join_existing / (self.nbh_join_existing + self.nbh_create_new)
        prob_create_new = self.nbh_create_new / (self.nbh_join_existing + self.nbh_create_new)

        if np.random.choice([0,1], size=1, p=[prob_join_existing, prob_create_new])[0] == 0:
            return self.node_to_existing_partition()
        else:
            return self.node_to_new_partition()

    def get_nbh_join_existing(self):
        return self.nbh_join_existing

    def set_nbh_join_existing(self, nbh_join_existing):
        self.nbh_join_existing = nbh_join_existing

    def get_nbh_create_new(self):
        return self.nbh_create_new

    def set_nbh_create_new(self, nbh_create_new):
        self.nbh_create_new = nbh_create_new

    def get_to_rescore(self):
        # return self.ordered_partition.all_nodes
        return self.to_rescore

    def set_to_rescore(self, to_rescore):
        self.to_rescore = to_rescore

    def get_chosen_move(self):
        return self.chosen_move

    def set_chosen_move(self, move):
        self.chosen_move = move

    def get_ordered_partition(self):
        return self.ordered_partition

    def set_ordered_partition(self, new_partition):
        self.ordered_partition = new_partition
        self.num_part = len(self.ordered_partition.partitions)
        self.nbh_size, self.nbh = self.compute_neighborhoods()

        party, _, posy = convert_partition_to_party_permy_posy( self.ordered_partition )
        num_nodes = sum(party)

        self.nbh_join_existing = sum(calculate_join_possibilities(num_nodes, party, posy ))
        self.nbh_create_new = sum(calculate_partition_transitions(num_nodes, party, posy ))

    def get_num_partitions(self):
        return self.num_part

    def set_num_partitions(self, num_part):
        self.num_part = num_part

    def get_nbh_size(self):
        return self.nbh_size

    def set_nbh_size(self, nbh_size):
        self.nbh_size = nbh_size

    def get_nbh(self):
        return self.nbh

    def set_nbh(self, nbh):
        self.nbh = nbh
