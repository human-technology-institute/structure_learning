"""

"""
from itertools import combinations
import numpy as np

# This function produces
def list_possible_parents(max_parents, elements):
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

    for i, element in enumerate(elements):
        remaining_elements = [e for e in elements if e != element]

        # Initialize an empty list to store tuples of possible parents
        matrix_of_parents_list = []

        # Adding the "empty" tuple first, filled with np.nan
        matrix_of_parents_list.append(tuple([np.nan] * max_parents))

        for r in range(1, max_parents + 1):
            possible_parents = list(combinations(remaining_elements, r))
            # Fill the remaining spaces with np.nan if necessary
            possible_parents = [pp + (np.nan,) * (max_parents - r) for pp in possible_parents]
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

def partition_score(scorenodes, parenttable, scoretable, permy, party, posy):
    """

    """
    n = len(scorenodes)

    partition_scores = {}

    all_scores = {}  # Use lists to hold multiple scores
    allowed_score_rows = {}
    allowed_rows = {}

    max_parents = parenttable[0].shape[1]

    # Convert permy to indices for easier comparison
    permy_indices = {node: index for index, node in enumerate(sorted(permy))}

    for i, node in enumerate(scorenodes):

        position = permy.index(node)
        partyelement = posy[position]

        i = permy_indices[node]

        if partyelement == 0:
            #print("Enter here")
            partition_scores[i] = scoretable[i][0]


            all_scores[i] = partition_scores[i]
            allowed_score_rows[i] = np.array([0])

            # if partition_scores[i] is of type array, then convert it to a scalar
            if isinstance(partition_scores[i], np.ndarray):
                partition_scores[i] = partition_scores[i][0]

        else:
            banned_nodes = [permy[i] for i in range(len(permy)) if posy[i] >= partyelement]
            required_nodes = [permy[i] for i in range(len(permy)) if posy[i] == partyelement - 1]
            allowed_rows = list(range(1, parenttable[i].shape[0] ))

            for j in range(max_parents):

                # Find banned rows: rows where the element in the j-th column is in bannednodes
                banned_rows = [row for row in allowed_rows if parenttable[i][row, j] in banned_nodes]

                # Update allowedrows by removing bannedrows
                # Convert bannedrows to a set for efficient removal
                banned_rows_set = set(banned_rows)

                if len(banned_rows_set) > 0:
                    allowed_rows = [row for row in allowed_rows if row not in banned_rows_set]

            not_required_rows = allowed_rows.copy()
            for j in range(max_parents):
                required_rows = [row for row in not_required_rows if parenttable[i][row, j] in required_nodes]
                if len(required_rows) > 0:
                    not_required_rows = [row for row in not_required_rows if row not in required_rows]

            allowed_rows = list(set(allowed_rows) - set(not_required_rows))

            # Check if allowed_rows is empty
            if len(allowed_rows) == 0:
                # CATARINA ADDED THIS SO IT WOULD NOT FREEZE
                # Handle the case where there are no allowed rows (e.g., set to a low default value or other logic)
                max_allowed = -np.inf  # Example default value
                all_scores[i] = np.array([max_allowed])
                print("\n[WARNING] No allowed rows for node ", node, " with parent set ",
                      required_nodes, " and banned set ", banned_nodes, "\n")
            else:
                all_scores[i] = scoretable[i][allowed_rows, 0]
                max_allowed = np.max(all_scores[i])

            allowed_score_rows[i] = np.array(allowed_rows)

            partition_scores[i] = max_allowed + np.log(np.sum(np.exp(all_scores[i] - max_allowed)))

            # if partition_scores[i] is of type array, then convert it to a scalar
            if isinstance(partition_scores[i], np.ndarray):
                partition_scores[i] = partition_scores[i][0]

    scores = {
        'all_possible_scores_node': all_scores,
        'allowed_rows': allowed_score_rows,
        'total_scores': partition_scores
    }
    return scores

def sample_score(n, scores, parenttable, node_label_to_idx):
    """

    """
    # Initialize the adjacency matrix
    incidence = np.zeros((n, n))
    sampled_score = 0

    # Loop over each node
    for i in list(scores['all_possible_scores_node'].keys()):

        score_length = len(scores['all_possible_scores_node'][i])

        print(scores['all_possible_scores_node'][i], scores['total_scores'][i],
              np.exp(scores['all_possible_scores_node'][i] - scores['total_scores'][i]))
        probabilities = np.exp(scores['all_possible_scores_node'][i] - scores['total_scores'][i])
        probabilities /= np.sum(probabilities)  # Normalize to sum to 1

        k = np.random.choice(score_length, 1, p=probabilities)[0]

        parent_row = parenttable[i][scores['allowed_rows'][i][k], :]

        # Filter out NaN values. np.isnan can only be applied to numeric values.
        parent_set = [node_label_to_idx[parent_row[j]] for j in range(len(parent_row))
                      if not (isinstance(parent_row[j], float) and np.isnan(parent_row[j]))]

        # Fill in elements of the adjacency matrix
        incidence[parent_set, i] = 1
        # Add the score
        sampled_score += scores['all_possible_scores_node'][i][k]

    # Return the DAG as a dictionary
    return {'incidence': incidence, 'logscore': sampled_score}
