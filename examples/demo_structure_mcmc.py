import sys
sys.path.append('../src') # Add local path to import custom modules from the 'src' directory

# This file is a demo which shows a very simple implementation
# of structure MCMC for a known Bayesian Network, whose parameters
# get drawn randomly – or can also be fixed.
# The file is intended purely for instructional purposes
# to familiarise students with the concepts and more transparent handling
# of objects and functions.

# Import necessary packages standard libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

# Import BGEScore class from structure-learning library
from mcmc.scores import BGeScore

# Basic Directed Acyclic Graph (DAG) class using adjacency matrix
# Note that for simplicity in this example file
# the DAG is solely represented by the structure, not the parameters of the BN
# since we "integrate them out" through the use of the BGeScore, which
# essentially returns the score for the graph considering all parameter
# configurations.
class DAG:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)   # initialise empty adjacency matrix (np.zeros)

    def add_edge(self, i, j):
        self.adj_matrix[i, j] = 1

    def remove_edge(self, i, j):
        self.adj_matrix[i, j] = 0

    def reverse_edge(self, i, j):
        self.adj_matrix[i, j] = 0
        self.adj_matrix[j, i] = 1

    def has_edge(self, i, j):
        return self.adj_matrix[i, j] == 1

    def is_acyclic(self):
        graph = nx.DiGraph(self.adj_matrix)
        return nx.is_directed_acyclic_graph(graph)

    def copy(self):
        new_dag = DAG(self.num_nodes)
        new_dag.adj_matrix = self.adj_matrix.copy()
        return new_dag

    def plot(self):
        graph = nx.DiGraph(self.adj_matrix)
        nx.draw_networkx(graph, arrows=True)
        plt.show()

# Main MCMC function
# This function is meant to be a transparent
# (not necessarilly efficient) version of structure MCMC
# for ease of understanding and debugging.
# It keeps tracks of all iterations, including
# those iterations that resulted in non-compliant dags (unsuccesful iterations).
# It returns:
# - the best dag structure
# - scores: a list of all scores for succesful iterations
# - posterior: The samples that make up the posterior over graphs,
# i.e. an array of graphs, each with its sampled frequency and 
# corresponding score.
# It also reports the number of un succesfull iterations.
def structure_mcmc(data_df, num_iterations=1000):
    num_nodes = data_df.shape[1]
    current_dag = DAG(num_nodes)
    current_score = compute_bge_score_for_dag(current_dag, data_df)

    best_dag = current_dag.copy()
    best_score = current_score

    # List of scores to inspect its dynamics
    scores = []
    successful_iterations = 0
    unsuccessful_iterations = 0

    # Posterior dictionary: key = DAG representation (tuple of tuples), value = {score, count}
    posterior = {}

    # Count the initial state.
    dag_repr = tuple(map(tuple, current_dag.adj_matrix))
    posterior[dag_repr] = {'score': current_score, 'count': 1}
    scores.append(current_score)

    # Unsucessful iterations are disregarded, i.e. the num of iterations only
    # represents the sucessful number of iterations.
    while successful_iterations < num_iterations:
        proposal_dag = current_dag.copy()
        # Randomly choose a move
        move_type = np.random.choice(['add', 'remove', 'reverse'])
        i, j = np.random.choice(num_nodes, size=2, replace=False)

        if move_type == 'add' and not proposal_dag.has_edge(i, j):
            proposal_dag.add_edge(i, j)
        elif move_type == 'remove' and proposal_dag.has_edge(i, j):
            proposal_dag.remove_edge(i, j)
        elif move_type == 'reverse' and proposal_dag.has_edge(i, j):
            proposal_dag.reverse_edge(i, j)
        else:
            unsuccessful_iterations += 1
            continue  # Invalid move; do not count

        if not proposal_dag.is_acyclic():
            unsuccessful_iterations += 1
            continue  # Reject non-acyclic proposals

        # Evaluate new graph and accept/reject
        # Valid proposal: count as a successful iteration.
        proposal_score = compute_bge_score_for_dag(proposal_dag, data_df)
        acceptance_prob = min(1, np.exp(proposal_score - current_score))

        if np.random.rand() < acceptance_prob:
            current_dag = proposal_dag
            current_score = proposal_score
            if current_score > best_score:
                best_dag = current_dag.copy()
                best_score = current_score
                
        scores.append(current_score)
        successful_iterations += 1

        # Update posterior dictionary for the current DAG.
        dag_repr = tuple(map(tuple, current_dag.adj_matrix))
        if dag_repr in posterior:
            posterior[dag_repr]['count'] += 1
        else:
            posterior[dag_repr] = {'score': current_score, 'count': 1}

    # Compute relative frequency for each unique DAG.
    for key in posterior:
        posterior[key]['rel_freq'] = posterior[key]['count'] / successful_iterations

    return best_dag, scores, posterior, unsuccessful_iterations


# Wrapper to compute BGe score for a DAG
def compute_bge_score_for_dag(dag, data_df):
    incidence = dag.adj_matrix
    score_obj = BGeScore(data_df, incidence, is_log_space=True)
    score = score_obj.compute()['score']
    return score


# Enumerate all unique DAGs for the given data using only off-diagonal entries
def enumerate_dags_and_scores(data_df):
    num_nodes = data_df.shape[1]
    all_dags = []
    scores = []

    off_diag_indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    for edges in product([0, 1], repeat=len(off_diag_indices)):
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for index, (i, j) in enumerate(off_diag_indices):
            adj_matrix[i, j] = edges[index]
        # Only keep acyclic graphs
        dag_candidate = nx.DiGraph(adj_matrix)
        if nx.is_directed_acyclic_graph(dag_candidate):
            dag = DAG(num_nodes)
            dag.adj_matrix = adj_matrix
            score = compute_bge_score_for_dag(dag, data_df)
            all_dags.append(dag)
            scores.append(score)
    return all_dags, scores

# Generate synthetic data given a DAG structure with Gaussian noise
# This follows the heirarchical/topological order of the DAG nodes
def simulate_gaussian_dag(dag, n_samples=500, noise_scale=0.1):
    num_nodes = dag.shape[0]
    data = np.zeros((n_samples, num_nodes))
    ordered_nodes = list(nx.topological_sort(nx.DiGraph(dag)))

    for node in ordered_nodes:
        parents = np.where(dag[:, node])[0]
        if len(parents) == 0:
            data[:, node] = np.random.normal(0, 1, size=n_samples)
        else:
            parent_data = data[:, parents]
            weights = np.random.uniform(0.5, 2, size=len(parents))
            data[:, node] = parent_data @ weights + np.random.normal(0, noise_scale, size=n_samples)
    return data

# --- Demo execution ---

# Define the true DAG: A -> B <- C
true_dag_adj = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 1, 0]
])

# Simulate dataset
np.random.seed(67)
data_np = simulate_gaussian_dag(true_dag_adj, n_samples=1300, noise_scale=7)

data_df = pd.DataFrame(data_np, columns=['A', 'B', 'C'])

# Enumerate all DAGs and compute their BGe scores (true distribution).
all_dags, dag_scores = enumerate_dags_and_scores(data_df)
exp_scores = np.exp(np.array(dag_scores) - np.max(dag_scores))
norm_const = np.sum(exp_scores)
true_probs = exp_scores / norm_const

true_posterior = {}
for dag, score, prob in zip(all_dags, dag_scores, true_probs):
    dag_repr = tuple(map(tuple, dag.adj_matrix))
    true_posterior[dag_repr] = {'score': score, 'true_prob': prob}

# Plot the true posterior distribution from enumeration.
sorted_indices = np.argsort(dag_scores)[::-1]
sorted_true_probs = np.array(true_probs)[sorted_indices]
sorted_true_dags = [all_dags[i] for i in sorted_indices]

# Set bar colours: red if it matches the true DAG, blue otherwise.
bar_colors_true = [
    'red' if np.array_equal(dag.adj_matrix, true_dag_adj) else 'blue'
    for dag in sorted_true_dags
]

plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_true_probs)), sorted_true_probs, color=bar_colors_true)
plt.xlabel('DAG Index (sorted by true posterior probability)')
plt.ylabel('True Posterior Probability')
plt.title('True Posterior (Enumeration) of DAGs')
plt.show()

# Run MCMC.
best_dag, scores, mcmc_posterior, unsuccessful_iters = structure_mcmc(data_df, num_iterations=1000)
print("Unsuccessful iterations:", unsuccessful_iters)

# Plot the MCMC posterior distribution.
# Combine keys from mcmc_posterior and true_posterior.
all_keys = set(list(true_posterior.keys()) + list(mcmc_posterior.keys()))
# Sort keys by true posterior probability (if available) descending.
all_keys_sorted = sorted(all_keys, key=lambda k: true_posterior[k]['true_prob'] if k in true_posterior else 0, reverse=True)
mcmc_probs_list = [mcmc_posterior[k]['rel_freq'] if k in mcmc_posterior else 0 for k in all_keys_sorted]
true_probs_list = [true_posterior[k]['true_prob'] if k in true_posterior else 0 for k in all_keys_sorted]

# Identify index of the true DAG.
true_dag_repr = tuple(map(tuple, true_dag_adj))
true_index = all_keys_sorted.index(true_dag_repr) if true_dag_repr in all_keys_sorted else None

x = np.arange(len(all_keys_sorted))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, true_probs_list, width, label='True Posterior')
plt.bar(x + width/2, mcmc_probs_list, width, label='MCMC Posterior')
if true_index is not None:
    plt.bar(x[true_index] - width/2, true_probs_list[true_index], width, edgecolor='red', fill=False, linewidth=2)
    plt.bar(x[true_index] + width/2, mcmc_probs_list[true_index], width, edgecolor='red', fill=False, linewidth=2)
plt.xlabel('Unique DAG (sorted)')
plt.ylabel('Probability / Relative Frequency')
plt.title('Comparison of True Distribution and MCMC Posterior')
plt.legend()
plt.xticks(x, ['DAG {}'.format(i+1) for i in range(len(all_keys_sorted))], rotation=45)
plt.tight_layout()
plt.show()

print("Finished")
