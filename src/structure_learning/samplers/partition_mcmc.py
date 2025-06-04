"""

"""
import time
import random
from copy import deepcopy
from typing import Union
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch.distributions.categorical import Categorical
from structure_learning.proposals import StructureLearningProposal
from structure_learning.proposals import PartitionProposal
from structure_learning.data_structures import OrderedPartition, Graph, DAG
from structure_learning.scores import Score, BGeScore, BDeuScore
from structure_learning.samplers import MCMC

class PartitionMCMC(MCMC):
    """
    Implementation of Partition MCMC.
    """
    def __init__(self, data : pd.DataFrame = None, initial_state : Union[OrderedPartition, np.ndarray] = None, max_iter : int = 30000,
                 proposal_object : StructureLearningProposal = None, score_object : Union[str, Score] = None,
                 pc_init: bool = True, blacklist = None, whitelist = None, searchspace = None, plus1: bool = False, seed : int  = 32, 
                 result_type='iterations', graph_type='dag', concise=True):
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
            blacklist (numpy.ndarray):                              Mask for edges to ignore in the proposal
            whitelist (numpy.ndarray):                              Mask for edges to include in the proposal
            searchspace (str | numpy.ndarray):                      Graph search space. "FULL" | "PC" | np.ndarray | None. If none, full search space is used.
            plus1 (bool):                                           Use plus1 neighborhood
        """
        # elif isinstance(initial_state, Graph):
        #     initial_state = OrderedPartition.from_graph(initial_state, list(score_object.data.columns if score_object else data.columns))
        # elif not isinstance(initial_state, OrderedPartition):
        #     raise Exception("Initial state must be of type Graph, numpy array, or an OrderedPartition")

        super().__init__(data=data, initial_state=initial_state, max_iter=max_iter, proposal_object=proposal_object,
                         score_object=score_object, pc_init=(searchspace=="PC"), blacklist=blacklist, whitelist=whitelist, plus1=plus1, seed=seed, result_type=result_type, graph_type=graph_type)
        self._to_string = f"Partition_MCMC_n_{self.num_nodes}_iter_{self.max_iter}"
        self.concise = concise

        if self.proposal_object is None:
            if self.initial_state is None:
                self.initial_state = DAG.generate_random(self.node_labels, 0.5, seed)
                if searchspace=="PC":
                    self.initial_state = self._pc_state.incidence.astype(int) - self.initial_state.incidence.astype(int) 
                    self.initial_state[self.initial_state < 0] = 0
                    self.initial_state = self.initial_state.astype(bool)
                if whitelist is not None:
                    self.initial_state[whitelist > 0] = True
                if blacklist is not None:
                    self.initial_state[blacklist > 0] = False
            if isinstance(self.initial_state, np.ndarray):
                self.initial_state = OrderedPartition.from_numpy(self.initial_state, list(self.score_object.data.columns if self.score_object else data.columns))
            elif isinstance(self.initial_state, Graph):
                self.initial_state = OrderedPartition.from_graph(self.initial_state)
            else:
                raise Exception("Unsupported initial state type")
            proposal_object = PartitionProposal(self.initial_state, whitelist=whitelist, blacklist=blacklist, seed=seed)
        elif not isinstance(proposal_object, StructureLearningProposal):
            raise Exception('Unsupported proposal', proposal_object)

        self.proposal_object = proposal_object

        self._max_parents = self.num_nodes - 1

        if isinstance(searchspace, str):
            if searchspace == "FULL":
                searchspace = (np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)).astype(bool)
            elif searchspace == "PC":
                searchspace = self.pc_graph.incidence
            else:
                raise Exception("Unsupported search space")

        self.parent_table = self._list_possible_parents(self._max_parents, self.node_labels, whitelist=whitelist,
                                                  blacklist=blacklist, plus1=plus1, searchspace=searchspace)
        self.score_table = self._score_possible_parents(self.parent_table, self.node_labels, self.score_object)
        self._rng = np.random.default_rng(seed=seed)
        random.seed(42)
        torch.random.manual_seed(seed)

        # compute the scores of the current partition
        current_state = self.proposal_object.current_state
        party_curr, permy_curr, posy_curr = current_state.to_party_permy_posy()
        self.scores = self._partition_score(self.node_labels, self.node_labels, self.parent_table, self.score_table, permy_curr, party_curr, posy_curr)
        self.current_state_score = sum(self.scores['total_scores'].values())
        self._start_time = time.time()

        # Sample a DAG from the initial partition
        sample = self._sample_from_partition(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)
        G, DAG_score = DAG(incidence=sample['incidence'], nodes=self.node_labels), sample['logscore']
        result = {
            'graph': G, 'score_current': DAG_score, 'operation': 'initial', 'accepted' : False, 'pscore_current' : self.current_state_score, 'pscore_proposed' : -1,
            'acceptance_prob' : -1, 'proposed_state': None, 'score_proposed': -1, 'timestamp': time.time() - self._start_time
        }

        if not self.concise:
            result.update({'current_partition': self.proposal_object.current_state, 'party': party_curr, 'permy': permy_curr, 'posy': posy_curr, 'proposed_partition': None})
                        
        self.current_step = result
        self.update_results(0, result)

    def step(self):
        """
        Perform one MCMC iteration

        Returns:
            (dict): information on one MCMC iteration
        """
        while True:
            proposed_state, operation = self.proposal_object.propose()
            nodes_to_rescore = self.proposal_object.get_nodes_to_rescore()
            party_prop, permy_prop, posy_prop = proposed_state.to_party_permy_posy()
            scores_copy = deepcopy(self.scores)
            rescore = self._partition_score(list(nodes_to_rescore), self.node_labels, self.parent_table, self.score_table, permy_prop, party_prop, posy_prop)
            for key in scores_copy:
                scores_copy[key].update(rescore[key])
            proposed_state_score = sum(scores_copy['total_scores'].values())
            if operation == StructureLearningProposal.STAY_STILL or not (np.isinf(proposed_state_score) or np.isnan(proposed_state_score)):
                break

        if operation == StructureLearningProposal.STAY_STILL:
            sample = self._sample_from_partition(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)
            G, DAG_score = DAG(incidence=sample['incidence'], nodes=self.node_labels), sample['logscore']
            result = self.current_step.copy()
            result['score_current'] = DAG_score
            result['operation'] = operation
            result['accepted'] = False
            result['graph'] = G
            result['timestamp'] = time.time() - self._start_time
        else:
            proposed_sample = self._sample_from_partition(self.num_nodes, self.node_labels, scores_copy, self.parent_table, self._node_label_to_idx)
            proposed_G, proposed_DAG_score = DAG(incidence=proposed_sample['incidence'], nodes=self.node_labels), proposed_sample['logscore']

            u = np.log(self._rng.uniform(0, 1))
            acceptance_prob = self.proposal_object.compute_acceptance_ratio(self.current_state_score, proposed_state_score)
            is_accepted = u < acceptance_prob

            if is_accepted:
                self.proposal_object.accept()
                self.n_accepted += 1
                self.scores = scores_copy
                self.current_state_score = proposed_state_score
                party_curr, permy_curr, posy_curr = party_prop, permy_prop, posy_prop

            sample = self._sample_from_partition(self.num_nodes, self.node_labels, self.scores, self.parent_table, self._node_label_to_idx)
            G, DAG_score = DAG(incidence=sample['incidence'], nodes=self.node_labels), sample['logscore']

            result = {
                'graph': G, 'score_current': DAG_score, 'operation': operation, 'accepted' : is_accepted, 'pscore_current' : self.current_state_score, 'pscore_proposed' : proposed_state_score,
                'acceptance_prob' : acceptance_prob, 'proposed_state': proposed_G, 'score_proposed': proposed_DAG_score, 'timestamp': time.time() - self._start_time
            }

            if not self.concise:
                result.update({'current_partition': self.proposal_object.current_state, 'party': party_curr, 'permy': permy_curr, 'posy': posy_curr, 'proposed_partition': proposed_state})
                
        self.current_step = result
        return result
    
    def __is_valid_partition__(self, partition: OrderedPartition, searchspace: np.ndarray):
        party, permy, posy = partition.to_party_permy_posy()
        for nodeidx in range(len(searchspace)):
            pos_node = posy.find(nodeidx)
            perm_node = permy[pos_node]
            parentidx = np.argwhere(searchspace[:,nodeidx])
            for parent in parentidx:
                pos_parent = posy.find(parent)
                perm_parent = permy[pos_parent]



    def _sample_from_partition(self, n, node_labels, scores, parenttable, node_label_to_idx):
        """
        Sample DAG from partition
        """
        incidence = np.zeros((n, n))
        sample_score = 0
        for i in range(n):
            node = node_labels[i]
            try:
                cat = Categorical(logits=torch.from_numpy(scores['all_possible_scores_node'][node] - scores['total_scores'][node])) # using torch here so we don't have to normalize and exponentiate the logits (np.random.choice)
                k = cat.sample().numpy()
                parent_row = parenttable[i][scores['allowed_rows'][node][k],:]# Filter out NaN values. np.isnan can only be applied to numeric values.
                parent_set = [node_label_to_idx[parent_row[j]] for j in range(len(parent_row)) if not (isinstance(parent_row[j], float) and np.isnan(parent_row[j]))]
                incidence[parent_set,i] = 1
                sample_score += scores['all_possible_scores_node'][node][k]
            except Exception as e:
                sample_score += -np.inf
        dag = {}
        dag['incidence'] = incidence
        dag['logscore'] = sample_score
        return dag
    
    def _partition_score(self, score_nodes, node_labels, parenttable, scoretable, permy, party, posy):
        """
        Score partition
        """
        n = len(score_nodes)
        partition_scores = {node:0 for node in score_nodes}
        all_scores = {}
        allowed_score_rows = {}
        m = len(party)

        tablesize = parenttable[0].shape[1]

        for node in score_nodes:
            i = node_labels.index(node)
            position = permy.index(node)
            partyelement = posy[position]

            if partyelement == 0: # no parents are allowed
                partition_scores[node] = scoretable[i][0]
                all_scores[node] = partition_scores[node]
                allowed_score_rows[node] = np.array([0])

            else:
                bannednodes = set([permy[k] for k in range(len(permy)) if posy[k] >= partyelement])
                requirednodes = set([permy[k] for k in range(len(permy)) if posy[k] == partyelement-1])
                allowedrows = set(list(range(1, parenttable[i].shape[0])))

                for j in range(parenttable[i].shape[1]):
                    bannedrows = [row for row in allowedrows if parenttable[i][row,j] in bannednodes]
                    allowedrows = allowedrows - set(bannedrows)

                notrequiredrows = allowedrows.copy()
                for j in range(parenttable[i].shape[1]):
                    requiredrows = [row for row in notrequiredrows if parenttable[i][row,j] in requirednodes]
                    notrequiredrows = notrequiredrows - set(requiredrows)

                allowedrows = list(allowedrows - notrequiredrows)

                all_scores[node] = scoretable[i][allowedrows, 0] if len(allowedrows) > 0 else np.array([-np.inf])
                allowed_score_rows[node] = allowedrows
                maxallowed = max(all_scores[node]) if len(allowedrows) > 0 else -np.inf
                try:
                    partition_scores[node] = (maxallowed + np.log(np.sum(np.exp(all_scores[node] - maxallowed)))) if not np.isinf(maxallowed) else -np.inf
                except:
                    partition_scores[node] = -np.inf
            if isinstance(partition_scores[node], np.ndarray):
                partition_scores[node] = partition_scores[node][0]

        scores = {}
        scores['all_possible_scores_node'] = all_scores
        scores['allowed_rows'] = allowed_score_rows
        scores['total_scores'] = partition_scores

        return scores
    
    def _score_possible_parents(self, parent_table, node_labels, score_object):
        """
        This function scores all the possible parents
        """
        n = len(node_labels)
        listy = [None] * n

        for idx, node_label in enumerate(node_labels):
            score_temp = self._table_dag_score(parent_table[idx], node_label, score_object)
            listy[idx] = np.array(score_temp).reshape(-1, 1)  # Assuming score_temp is a list

        return listy
    
    def _table_dag_score(self, parent_rows, node, score_object):
        """
        Scoring rows are the parent sets
        """
        nrows = parent_rows.shape[0]
        p_local = np.zeros(nrows)

        for i in range(nrows):
            parent_nodes = self.__filter_nans(parent_rows[i])
            p_local[i] = score_object.compute_node_with_edges(node, parent_nodes, self._node_label_to_idx)['score']

        return p_local
    
    def __filter_nans(self, row):
        """
        Filter out nans from an iterable object of floats.

        Parameters:
            row (iterable): list of floats and nans

        Returns:
            (iterable): the input with all the nans removed
        """
        return [x for x in row if not isinstance(x, float) or not np.isnan(x)]
    
    def _list_possible_parents(self, max_parents, elements, whitelist=None, blacklist=None, plus1=False, searchspace=None):
        """
        Generate a matrix with all the possible parents of a given node
        up to the maximum number of parents.

        Parameters:
            max_parents (int): maximum number of parents
            elements (list): nodes

        Returns:
            (list): all possible parents
        """
        listy = [None] * len(elements)

        if not isinstance(elements, np.ndarray):
            elements = np.array(elements)

        for i, element in enumerate(elements):
            remaining_elements = [e for e in elements if e != element] # all nodes except self

            possible_parent_nodes = remaining_elements if searchspace is None else elements[searchspace[:,i]] # possible parents

            remaining_elements = (set(remaining_elements) - set(possible_parent_nodes)) if plus1 else [] # possible plus 1

            # get required parent nodes
            required_parents = tuple(elements[whitelist[:, i]==1]) if whitelist is not None else []
            n_required = len(required_parents)

            # get banned parent nodes
            banned_parents = elements[blacklist[:, i]==1] if blacklist is not None else []

            possible_parent_nodes = set(possible_parent_nodes) - set(required_parents) - set(banned_parents) # remove required nodes and blacklisted nodes from set of possible parents
            remaining_elements = set(remaining_elements) - set(required_parents) - set(banned_parents) # remove required nodes and  blacklisted nodes from possible plus 1

            # Initialize an empty list to store tuples of possible parents
            matrix_of_parents_list = []

            # Adding the "empty" tuple first, filled with np.nan
            if n_required == 0:
                matrix_of_parents_list.append(tuple([np.nan] * max_parents))

            for r in range(1, len(possible_parent_nodes) + 1):
                possible_parents = list(combinations(possible_parent_nodes, r))
                possible_parents_plus1 = list(combinations(remaining_elements, 1)) + [[np.nan]]

                # Fill the remaining spaces with np.nan if necessary
                if len(possible_parents_plus1) > 1:
                    possible_parents = [list(required_parents) + list(pp) + list(remaining_element) + [np.nan,] * (max_parents - r - n_required - len(remaining_element)) for pp in possible_parents for remaining_element in possible_parents_plus1]
                else:
                    possible_parents = [tuple(required_parents) + tuple(pp) + (np.nan,) * (max_parents - r - n_required) for pp in possible_parents]

                matrix_of_parents_list.extend(possible_parents)

            # Convert the list of tuples into a NumPy array with dtype='object'
            matrix_of_parents = np.array(matrix_of_parents_list, dtype='object')
            listy[i] = matrix_of_parents

        return listy