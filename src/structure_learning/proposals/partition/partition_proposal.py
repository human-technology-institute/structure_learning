import numpy as np
import math
from structure_learning.data_structures.partition import *
from structure_learning.proposals import StructureLearningProposal

class PartitionProposal(StructureLearningProposal):

    SWAP_ADJACENT = "swap_adjacent"
    SWAP_GLOBAL = "swap_global"
    SPLIT_OR_MERGE = "split_or_merge"
    MERGE_PARTITIONS = "merge_partitions"
    SPLIT_PARTITIONS = "split_partitions"
    MOVE_NODE_TO_NEW_OR_EXISTING = "move_node_to_new_or_existing"
    MOVE_NODE_TO_EXISTING_PARTITION = "move_node_to_existing_partition"
    MOVE_NODE_TO_NEW_PARTITION = "move_node_to_new_partition"
    STAY_STILL = StructureLearningProposal.STAY_STILL

    operations = [SWAP_ADJACENT, SWAP_GLOBAL, SPLIT_OR_MERGE, MOVE_NODE_TO_NEW_OR_EXISTING, STAY_STILL]

    def __init__(self, initial_state: OrderedPartition, whitelist: np.ndarray = None, blacklist: np.ndarray = None, seed = 32):

        super().__init__(initial_state, blacklist, whitelist, seed) # initialize the parent class
        self.current_state = initial_state
        self._update()
        self.operation = None
        self.to_rescore = set()
        self.num_moves = len(self.operations)
        self.nbh_join_existing = None
        self.nbh_create_new = None
        self.move_probs = self._calculate_move_probs()
        self.proposed_state = None

    def propose(self):
        # reset the nodes to rescore
        self.to_rescore = set()
        while True:
            operation = self._rng.choice(self.operations, size=1, p=self.move_probs)[0]
            if not("swap" in operation and self.current_state.size < 2):
                break

        self.proposed_state = self.current_state.copy()
        if operation == self.SWAP_ADJACENT:
            self._swap_adjacent()
        elif operation == self.SWAP_GLOBAL:
            self._swap_global()
        elif operation == self.SPLIT_OR_MERGE:
            self._split_or_merge_move()
        elif operation == self.MOVE_NODE_TO_NEW_OR_EXISTING:
            self._move_node_to_existing_or_new_partition()
        else:
            self.operation = self.STAY_STILL
        return self.proposed_state, self.operation

    def compute_acceptance_ratio(self, current_state_score, proposed_state_score):
        if self.SWAP_ADJACENT == self.operation or self.SWAP_GLOBAL == self.operation:
            self._Q_current_proposed = 1
            self._Q_proposed_current = 1
        elif self.SPLIT_PARTITIONS == self.operation or self.MERGE_PARTITIONS == self.operation:
            self._Q_current_proposed = self.compute_neighborhoods(self.current_state)[0]
            self._Q_proposed_current = self.compute_neighborhoods(self.proposed_state)[0]
        elif self.MOVE_NODE_TO_EXISTING_PARTITION == self.operation or self.MOVE_NODE_TO_NEW_PARTITION == self.operation:
            self._Q_current_proposed = sum(self._compute_neighborhoods_new_existing_partition(self.current_state))
            self._Q_proposed_current = sum(self._compute_neighborhoods_new_existing_partition(self.proposed_state))
        else:
            raise Exception("Invalid operation ", self.operation)

        numerator = proposed_state_score + np.log(self._Q_current_proposed)
        denominator = current_state_score + np.log(self._Q_proposed_current)

        try:
            acceptance_ratio = numerator - denominator
        except Exception as e:
            print(e)
            acceptance_ratio = -np.inf

        return min(0, acceptance_ratio)

    def get_nodes_to_rescore(self):
        return self.to_rescore

    def accept(self):
        self.current_state = self.proposed_state
        self._update()

    def _update(self):
        self.num_part = len(self.current_state.partitions)
        self.nbh_size, self.nbh = self.compute_neighborhoods(self.current_state)
        self.nodes = self.current_state.get_all_nodes()
        self.num_nodes = len(self.nodes)
        self.nbh_join_existing, self.nbh_create_new = self._compute_neighborhoods_new_existing_partition(self.current_state)

    def compute_neighborhoods(self, state):
        """
        Compute the number of neighborhoods of an ordered partiion by using the equation
        sum_{i=1}^{m} ( sum_{c=1}^{k_i-1} ( comb( k_i, c) ) ) + m - 1
        """
        comb_lst = [sum(math.comb(part_i.size, c) for c in range(1, part_i.size)) for part_i in state.partitions]
        return np.sum(comb_lst) + self.num_part - 1, comb_lst

    def _calculate_move_probs(self):

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

    def _swap_adjacent(self):

        assert self.current_state.size >= 2
        self.operation = self.SWAP_ADJACENT

        # choose random partition element
        party, _, _ = self.proposed_state.to_party_permy_posy()
        p = np.array(self._possible_permutations_neighbors(sum(party), party))
        part_idx = self._rng.choice(self.proposed_state.size - 1, p=p/p.sum())
        # choose a node from partition at part_idx
        node_left = self._rng.choice(list(self.proposed_state.partitions[part_idx].nodes))
        # choose a node from partition at part_idx+1
        node_right = self._rng.choice(list(self.proposed_state.partitions[part_idx+1].nodes))

        # swap
        self.proposed_state.partitions[part_idx].remove_single_node(node_left)
        self.proposed_state.partitions[part_idx].add_single_node(node_right)

        self.proposed_state.partitions[part_idx+1].remove_single_node(node_right)
        self.proposed_state.partitions[part_idx+1].add_single_node(node_left)

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(self.proposed_state.get_all_nodes_from_right(part_idx+1))
        self.to_rescore = self.to_rescore.union({node_right})

        self.proposed_state.update_IDs()

        return self.proposed_state

    def _swap_global(self ):

        assert self.current_state.size >= 2
        self.operation = self.SWAP_GLOBAL

        # choose random partition elements
        party, _, _ = self.proposed_state.to_party_permy_posy()
        p = np.array(self._possible_permutations(sum(party), party))
        part_idx_1 = self._rng.choice(self.proposed_state.size, p=p/p.sum())
        part_idx_2 = self._rng.choice(list(set(range(self.proposed_state.size)).difference({part_idx_1})))

        left_part_idx, right_part_idx = sorted([part_idx_1, part_idx_2])

        # choose a node from partition at left_part_idx
        node_left = self._rng.choice(list(self.proposed_state.partitions[left_part_idx].nodes))
        # choose a node from partition at right_part_idx
        node_right = self._rng.choice(list(self.proposed_state.partitions[right_part_idx].nodes))

        # swap
        self.proposed_state.partitions[left_part_idx].remove_single_node(node_left)
        self.proposed_state.partitions[left_part_idx].add_single_node(node_right)

        self.proposed_state.partitions[right_part_idx].remove_single_node(node_right)
        self.proposed_state.partitions[right_part_idx].add_single_node(node_left)

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(self.proposed_state.get_all_nodes_from_right(left_part_idx))

        self.proposed_state.update_IDs()

        return self.proposed_state

    def _split_or_merge_move(self):

        party, _, _ = self.proposed_state.to_party_permy_posy()
        p = np.array(self._possible_splits_joins(sum(party), party))
        p = p if p.sum() else p+1
        node = self._rng.choice(list(self.current_state.all_nodes), p=p/p.sum())
        idx = self.current_state.find_node(node)

        can_split = self.current_state.partitions[idx].size >= 2
        can_merge = self.current_state.size > 1

        choice = self._rng.choice([0,1])
        if can_split and choice==0:
            return self._split_move(idx)
        elif can_merge and choice==1:
            return self._join_move(idx)
        else:
            self.operation = self.STAY_STILL
            return self.current_state.copy()

    def _split_move(self, idx : int):

        self.operation = self.SPLIT_PARTITIONS
        assert self.proposed_state.partitions[idx].size >= 2

        # randomly select which nodes at partition idx will be at the new left partition
        n_nodes_left = self._rng.integers(1, self.proposed_state.partitions[idx].size)
        nodes_involved = list(self.proposed_state.partitions[idx].nodes)
        nodes_left = self._rng.choice(nodes_involved, replace=False, size=n_nodes_left)

        # split
        self.proposed_state.partitions[idx].remove_nodes(nodes_left)
        self.proposed_state.insert_partition(idx, set(nodes_left))

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(self.proposed_state.get_all_nodes_from_right(idx))

        # remove empty partition if exists
        self.proposed_state.remove_empty_partitions()

        self.proposed_state.update_IDs()

        return self.proposed_state

    def _join_move(self, idx : int):

        assert self.current_state.size >= 2
        self.operation = self.MERGE_PARTITIONS

        adj = self._rng.choice([-1,1])
        if idx == 0:
            adj = 1
        if idx == self.proposed_state.size-1:
            adj = -1

        left_part_idx, right_part_idx = sorted([idx, idx+adj])

        # join partitions at idx and idx+1
        nodes_right = self.proposed_state.partitions[right_part_idx].nodes
        self.proposed_state.partitions[right_part_idx].remove_nodes(nodes_right)
        nodes_left = self.proposed_state.partitions[left_part_idx].nodes
        self.proposed_state.partitions[left_part_idx].add_nodes(nodes_right)

        # mark nodes to rescore
        self.to_rescore = self.to_rescore.union(self.proposed_state.get_all_nodes_from_right(left_part_idx))

        # remove empty partition if exists
        self.proposed_state.remove_empty_partitions()

        self.proposed_state.update_IDs()

        return self.proposed_state

    def _node_to_existing_partition(self):

        assert self.current_state.size >= 2
        self.operation = self.MOVE_NODE_TO_EXISTING_PARTITION

        # sample a node
        nodes = self.proposed_state.get_all_nodes()
        party, _, posy = self.proposed_state.to_party_permy_posy()
        p = np.array(self._calculate_join_possibilities(len(nodes), party, posy))
        node_to_move = self._rng.choice(list(nodes), p=p/p.sum())

        # get partition where the node belongs to
        current_partition_idx = self.proposed_state.find_node(node_to_move)

        # select a different partition to move to
        target_partition_idx = self._rng.choice(list(set(range(self.proposed_state.size)).difference({current_partition_idx})))

        # move
        self.proposed_state.partitions[current_partition_idx].remove_single_node(node_to_move)
        self.proposed_state.partitions[target_partition_idx].add_single_node(node_to_move)

        # mark nodes to rescore
        min_idx, max_idx = sorted([current_partition_idx, target_partition_idx])
        self.to_rescore = self.to_rescore.union(self.proposed_state.get_all_nodes_from_right(min_idx))
        self.to_rescore = self.to_rescore.union({node_to_move})

        # remove empty partition if exists
        self.proposed_state.remove_empty_partitions()

        self.proposed_state.update_IDs()

        return self.proposed_state

    def _node_to_new_partition(self):

        self.operation = self.MOVE_NODE_TO_NEW_PARTITION

        # sample a node
        nodes = self.proposed_state.get_all_nodes()
        party, _, posy = self.proposed_state.to_party_permy_posy()
        p = np.array(self._calculate_partition_transitions(len(nodes), party, posy))
        node_to_move = self._rng.choice(list(nodes), p=p/p.sum())

        # get partition where the node belongs to
        current_partition_idx = self.proposed_state.find_node(node_to_move)

        # sample where to go
        target_partition_idx = self._rng.choice(p[list(nodes).index(node_to_move)])
        if self.proposed_state.partitions[current_partition_idx].size == 1:
            if target_partition_idx >= current_partition_idx:
                target_partition_idx += 1
                if current_partition_idx < self.proposed_state.size-1 and self.proposed_state.partitions[current_partition_idx+1].size == 1:
                    target_partition_idx += 1

        # remove node from current partition
        self.proposed_state.partitions[current_partition_idx].remove_single_node(node_to_move)

        # insert new partition containing the node to target index
        self.proposed_state.insert_partition(target_partition_idx,  {node_to_move})

        # mark nodes to rescore
        min_idx, max_idx = sorted([current_partition_idx, target_partition_idx])
        self.to_rescore = self.to_rescore.union(self.proposed_state.get_all_nodes_from_right(min_idx))
        self.to_rescore = self.to_rescore.union({node_to_move})

        # remove empty partition if exists
        self.proposed_state.remove_empty_partitions()

        self.proposed_state.update_IDs()

        return self.proposed_state

    def _move_node_to_existing_or_new_partition(self):

        self.nbh_join_existing, self.nbh_create_new = self._compute_neighborhoods_new_existing_partition(self.current_state)

        prob_join_existing = self.nbh_join_existing / (self.nbh_join_existing + self.nbh_create_new)
        prob_create_new = self.nbh_create_new / (self.nbh_join_existing + self.nbh_create_new)

        if self._rng.choice([0,1], size=1, p=[prob_join_existing, prob_create_new])[0] == 0:
            return self._node_to_existing_partition()
        else:
            return self._node_to_new_partition()

    def _compute_neighborhoods_new_existing_partition(self, state):
        party, _, posy = state.to_party_permy_posy()
        num_nodes = sum(party)

        existing = sum(self._calculate_join_possibilities(num_nodes, party, posy ))
        new = sum(self._calculate_partition_transitions(num_nodes, party, posy ))

        return existing, new

    def _calculate_join_possibilities(self, n, party, posy):
        m = len(party)
        join_possibs = [0] * n
        for k in range(n):
            join_possibs[k] = m - 1
            node_element = posy[k]
            if party[node_element] == 1:  # Nodes in a partition element of size 1
                if node_element < m - 1:
                    if party[node_element + 1] == 1:  # And if the next partition element is also size 1
                        join_possibs[k] = m - 2  # We only allow them to jump to the left to count the swap only once
        return join_possibs
    
    def _calculate_partition_transitions(self, n, party, posy):
        m = len(party)
        hole_possibs = [0] * n
        for k in range(n):
            node_element = posy[k]
            if party[node_element] == 1:  # Nodes in a partition element of size 1 cannot move to the neighbouring holes
                hole_possibs[k] = m - 1
                if node_element < m - 1:
                    if party[node_element + 1] == 1:  # And if the next partition element is also size 1
                        hole_possibs[k] = m - 2  # We only allow them to jump to the left to count the swap only once
            elif party[node_element] == 2:  # Nodes in a partition element of size 2 cannot move to the hole on the left
                hole_possibs[k] = m  # Since this would count the same splitting twice
            else:
                hole_possibs[k] = m + 1
        return hole_possibs
    
    def _possible_splits_joins(self, n, party):
        partypossibs = [0] * n
        kbot = 0
        m = len(party)
        for j in range(m):
            elemsize = party[j]
            ktop = kbot + elemsize - 1
            for k in range(kbot, ktop):
                partypossibs[k] = math.comb(elemsize, k-kbot+1)
            kbot = ktop + 1
        return partypossibs

    def _possible_permutations_neighbors(self, n, party):
        m = len(party)
        possibs = [0] * (m-1)
        if m > 1:
            for i in range(m-1):
                possibs[i] = party[i]*party[i+1]
        return possibs
    
    def _possible_permutations(self, n, party):
        m = len(party)
        possibs = [0] * (m)
        if m > 1:
            remainder = n
            for i in range(m):
                remainder = remainder - party[i]
                possibs[i] = party[i]*remainder
        return possibs