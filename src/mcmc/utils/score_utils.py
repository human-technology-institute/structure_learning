"""

"""
from itertools import combinations
import numpy as np
import torch
from torch.distributions.categorical import Categorical

# This function produces
def list_possible_parents(max_parents, elements, whitelist=None, blacklist=None):
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

    print(type(elements))

    for i, element in enumerate(elements):
        remaining_elements = [e for e in elements if e != element]

        # get required parent nodes
        required_parents = tuple(elements[whitelist[:, i]==1]) if whitelist is not None else []
        n_required = len(required_parents)

        # get banned parent nodes
        banned_parents = elements[blacklist[:, i]==1] if blacklist is not None else []

        remaining_elements = set(remaining_elements) - set(required_parents) - set(banned_parents)

        # Initialize an empty list to store tuples of possible parents
        matrix_of_parents_list = []

        # Adding the "empty" tuple first, filled with np.nan
        if n_required == 0:
            matrix_of_parents_list.append(tuple([np.nan] * max_parents))

        for r in range(1, max_parents + 1 - n_required):
            possible_parents = list(combinations(remaining_elements, r))
            # Fill the remaining spaces with np.nan if necessary
            possible_parents = [tuple(required_parents) + tuple(pp) + (np.nan,) * (max_parents - r - n_required) for pp in possible_parents]
            matrix_of_parents_list.extend(possible_parents)

        # Convert the list of tuples into a NumPy array with dtype='object'
        matrix_of_parents = np.array(matrix_of_parents_list, dtype='object')
        listy[i] = matrix_of_parents

    return listy

def filter_nans(row):
    """
    Filter out nans from an iterable object of floats.

    Parameters:
        row (iterable): list of floats and nans

    Returns:
        (iterable): the input with all the nans removed
    """
    return [x for x in row if not isinstance(x, float) or not np.isnan(x)]

# Scoring rows are the parent sets
def table_dag_score(parent_rows, node, score_object):
    """

    """
    nrows = parent_rows.shape[0]
    p_local = np.zeros(nrows)

    for i in range(nrows):
        parent_nodes = filter_nans(parent_rows[i])
        p_local[i] = score_object.compute_BGe_with_edge(node,parent_nodes)['score']

    return p_local

# This function scores all the possible parents
def score_possible_parents(parent_table, node_labels, score_object):
    """

    """
    n = len(node_labels)
    listy = [None] * n

    for idx, node_label in enumerate(node_labels):
        score_temp = table_dag_score(parent_table[idx], node_label, score_object)
        listy[idx] = np.array(score_temp).reshape(-1, 1)  # Assuming score_temp is a list

    return listy

def partition_score(score_nodes, node_labels, parenttable, scoretable, permy, party, posy):
    """

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
                partition_scores[node] = maxallowed + np.log(np.sum(np.exp(all_scores[node] - maxallowed)))
            except:
                partition_scores[node] = -np.inf
        if isinstance(partition_scores[node], np.ndarray):
            partition_scores[node] = partition_scores[node][0]

    scores = {}
    scores['all_possible_scores_node'] = all_scores
    scores['allowed_rows'] = allowed_score_rows
    scores['total_scores'] = partition_scores

    return scores

def sample_score(n, node_labels, scores, parenttable, node_label_to_idx):
    """

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
            print(e)
            print('Possible inf')
            sample_score += -np.inf
    DAG = {}
    DAG['incidence'] = incidence
    DAG['logscore'] = sample_score
    return DAG
