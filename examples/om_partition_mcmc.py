# Imports
import os
import datetime
import numpy as np

from mcmc.scores.bge import BGeScore
from mcmc.proposals.graph.graph_proposal import GraphProposal
from mcmc.data.synthetic import SyntheticDataset

from mcmc.mcmc.partition_mcmc import PartitionMCMC

import pickle

import pandas as pd

from mcmc.inference.posterior import *

import time
import csv

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from tqdm.notebook import trange
from tkinter import Tk, filedialog


def KL_MCMC(MCMC_dict, true_dist, n, logscore=False):
    """
    MCMC_dict: Dictionary with the visited graph and counts
    true_dist: the true distribution
    n: number of iterates
    logscore: If true dist is log(P)
    """

    KL = 0
    for k, mcmc_count in MCMC_dict.items():
        true_log_p = true_dist[k]
        if not logscore:
            # if true_dist is in log space
            true_log_p = np.log(true_log_p)
        p_approx = mcmc_count / n
        KL += p_approx * (np.log(p_approx) - true_log_p)
    return KL


# %%
def KL_OM(unique_graphs, unique_scores, true_dist, logscore=False):
    """
    unique_graphs: A list with the visited graph
    unique_scores: A list with scores of visited graph
    true_dist: the true distribution
    logscore: T if true dist is given as log(p)
    """

    KL = 0

    # Prevents overflow during exponentiation by substracting
    max_score = max(unique_scores)

    log_raw_p_approx_vec = np.asarray(unique_scores)
    log_p_approx_vec = log_raw_p_approx_vec - max_score
    p_approx_vec = np.exp(log_p_approx_vec)
    norm_factor = np.sum(p_approx_vec)
    log_p_approx_vec -= np.log(norm_factor)

    for k, log_p_approx in zip(unique_graphs, log_p_approx_vec):
        true_log_p = true_dist[k]
        if not logscore:
            # if true_dist is in log space
            true_log_p = np.log(true_log_p)

        KL += np.exp(log_p_approx) * (log_p_approx - true_log_p)

    return KL


# %%

def KL_comparison_OM_MCMC(mcmc_results, ground_truth_distribution, logscore=False):
    # log score if groundtruth is log(p)
    graph_mcmc_count = {}
    unique_graph_id = []  # needed to handle overflow/underflow in computing normalization factor
    unique_accepted_graph_id = []  # accepted only
    unique_scores = []  # needed to handle overflows
    unique_accepted_scores = []  # needed to handle overflows

    KL_MCMC_t = []
    KL_OM_t = []
    KL_OM_accepted_only_t = []

    max_mcmc_iter = len(mcmc_results)
    for i in range(max_mcmc_iter):
        res = mcmc_results[i]
        if "G_prop" in res.keys() and res['G_prop'] is not None:
            graph_prop = generate_key_from_adj_matrix(res['G_prop'])
            score_prop = res['G_score_prop']['score']
        else:
            graph_prop = None

        graph_id = generate_key_from_adj_matrix(res['graph'])  # Accepted graph/state
        score = res['G_score_curr']['score']  # Accepted score
        # Score is log of the true score

        graph_mcmc_count[graph_id] = graph_mcmc_count[graph_id] + 1 if graph_id in graph_mcmc_count else 1

        # OM update
        if not (graph_id in unique_graph_id):
            # When a proposal is accepted and was never added before
            unique_graph_id.append(graph_id)
            unique_scores.append(score)
        elif graph_prop is not None and not (graph_prop in unique_graph_id):
            # Looking at proposal
            unique_graph_id.append(graph_prop)
            unique_scores.append(score_prop)

        # OM update - accepted graphs
        if not (graph_id in unique_accepted_graph_id):
            # When a proposal is accepted and was never added before
            unique_accepted_graph_id.append(graph_id)
            unique_accepted_scores.append(score)

        KL_OM_t.append(KL_OM(unique_graph_id, unique_scores, ground_truth_distribution, logscore=logscore))
        KL_OM_accepted_only_t.append(KL_OM(unique_accepted_graph_id, unique_accepted_scores, ground_truth_distribution, logscore=logscore))
        KL_MCMC_t.append(KL_MCMC(graph_mcmc_count, ground_truth_distribution, i + 1, logscore=logscore))

        # print(f"{len(unique_accepted_graph_id)} - {len(unique_accepted_graph_id)}")

    return KL_MCMC_t, KL_OM_t, KL_OM_accepted_only_t


print("Reusing data from Structure MCMC")


def select_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select CSV Metadata File",
        filetypes=[("CSV files", "metadata*.csv")]
    )
    return file_path


# Request the user to select an Excel file
file_path = select_file()

directory = os.path.dirname(file_path)

metadata_df = pd.read_csv(file_path, header=0)

print(metadata_df)

n_exp = int(metadata_df.loc[0, 'Num_experiments'])
num_nodes = int(metadata_df.loc[0, 'num_nodes'])
node_labels = [f"X{i + 1}" for i in range(num_nodes)]
noise_scale = float(metadata_df.loc[0, 'noise_scale'])
score_type = BGeScore
mcmc_iter = int(metadata_df.loc[0, 'mcmc_iter'])
degree = int(metadata_df.loc[0, 'dag_sparse_degree'])  # erdos-renyi sparsity - equivalent to 0.75
graph_type = metadata_df.loc[0, 'graph_type']

DIRPATH = os.path.abspath("")
MAINRESPATH = os.path.join(DIRPATH, 'om_results')
os.makedirs(MAINRESPATH, exist_ok=True)

RESPATH = os.path.join(MAINRESPATH, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(RESPATH, exist_ok=True)

# Save metadata
metadata_dict = {"Num_experiments": n_exp,
                 "num_nodes": num_nodes,
                 "noise_scale": noise_scale,
                 "score_type": score_type.__name__,
                 "MCMC_start_point": 'random',
                 "mcmc_iter": mcmc_iter,
                 "graph_type": graph_type,
                 "dag_sparse_degree": degree,
                 "MCMC_type": "partition"

                 }

MCMC_results_dict = {}
OM_results_dict = {}
OM_accepted_results_dict = {}

with open(os.path.join(RESPATH, 'metadata.csv'), 'w', newline="") as csvfile:
    w = csv.DictWriter(csvfile, metadata_dict.keys())
    w.writeheader()
    w.writerow(metadata_dict)

t = trange(n_exp, desc='Bar desc', leave=True)
for exp_i in t:
    print(f"Experiment: {exp_i}")
    # Generate true dag
    t.set_description("%i: Load true DAG" % exp_i)
    t.refresh()
    time.sleep(0.01)
    print(f"{exp_i}: Generate true DAG")

    # Load dag from Structure MCMC experiment
    true_dag = np.loadtxt(os.path.join(directory, f"true_DAG_{exp_i}.csv"), delimiter=',')
    print(true_dag)

    # Save to results folder
    np.savetxt(os.path.join(RESPATH, f"true_DAG_{exp_i}.csv"), true_dag, delimiter=',', fmt='%d')

    assert not (has_cycle(true_dag)), "not a DAG"

    # Generate data based on the true DAG
    t.set_description("%i: Load data" % exp_i)
    t.refresh()
    time.sleep(0.01)
    print(f"{exp_i}: Generate data")

    data = pd.read_csv(os.path.join(directory, f"data_{exp_i}.csv"), delimiter=',',
                       names=[f"X{i + 1}" for i in range(num_nodes)])

    #np.loadtxt(os.path.join(directory, f"data_{exp_i}.csv"), delimiter=',')
    # save data to file
    np.savetxt(os.path.join(RESPATH, f"data_{exp_i}.csv"), data, delimiter=',')

    # Load ground truth distribution from MCMC folder
    t.set_description("%i: Generate ground truth" % exp_i)
    t.refresh()
    time.sleep(0.01)
    print(f"{exp_i}: Load ground truth")
    with open(os.path.join(directory, f'true_distribution_results_{exp_i}.pkl'), 'rb') as file:
        log_true_distr = pickle.load(file)

    with open(os.path.join(RESPATH, f'true_distribution_results_{exp_i}.pkl'), 'wb') as f:
        pickle.dump(log_true_distr, f)

    # MCMC
    t.set_description("%i: MCMC" % exp_i)
    t.refresh()
    time.sleep(0.5)
    print(f"{exp_i}: MCMC")
    mcmc_obj = PartitionMCMC(max_iter=mcmc_iter, data=data, score_object='bge')
    mcmc_res, accept_rate = mcmc_obj.run()

    t.set_description("%i: KL comparison" % exp_i)
    t.refresh()
    time.sleep(0.01)
    KL_MCMC_res, KL_OM_res, KL_OM_accepted_res = KL_comparison_OM_MCMC(mcmc_res, log_true_distr, logscore=True)

    MCMC_results_dict.update({f"KL_MCMC_{exp_i}": KL_MCMC_res})
    OM_results_dict.update({f"KL_OM_{exp_i}": KL_OM_res})
    OM_accepted_results_dict.update({f"KL_OM_accepted_{exp_i}": KL_OM_accepted_res})

    df = pd.DataFrame(MCMC_results_dict)
    df.to_csv(os.path.join(RESPATH, 'MCMC_KL_results.csv'), index=True)
    df = pd.DataFrame(OM_results_dict)
    df.to_csv(os.path.join(RESPATH, 'OM_KL_results.csv'), index=True)
    df = pd.DataFrame(OM_accepted_results_dict)
    df.to_csv(os.path.join(RESPATH, 'OM_KL_accepted_only_results.csv'), index=True)

    # For debugging purposes
    graph_list = mcmc_obj.get_mcmc_res_graphs(mcmc_res)
    # score_list = mcmc_obj.get_mcmc_res_scores(mcmc_res)

    with open(os.path.join(RESPATH, f'MCMC_results_{exp_i}.pkl'), 'wb') as f:
        pickle.dump(graph_list, f)

# Plotting
df_MCMC = pd.read_csv(os.path.join(RESPATH, 'MCMC_KL_results.csv'), index_col=0)
df_OM = pd.read_csv(os.path.join(RESPATH, 'OM_KL_results.csv'), index_col=0)
quantiles = [0.05, 0.95]

MCMC_mean = df_MCMC.mean(axis=1)
MCMC_std = df_MCMC.std(axis=1)

MCMC_quantiles = df_MCMC.apply(lambda row: row.quantile(quantiles), axis=1)

OM_mean = df_OM.mean(axis=1)
OM_std = df_OM.std(axis=1)
OM_quantiles = df_OM.apply(lambda row: row.quantile(quantiles), axis=1)

plt.plot(MCMC_mean)
plt.plot(MCMC_mean, color='blue', label='MCMC')
plt.fill_between(MCMC_mean.index, MCMC_quantiles[quantiles[0]], MCMC_quantiles[quantiles[1]], color='blue', alpha=0.2)
plt.plot(OM_mean, color='red', label='OM')
plt.fill_between(OM_mean.index, OM_quantiles[quantiles[0]], OM_quantiles[quantiles[1]], color='red', alpha=0.2)

plt.xlabel('#Iteration')
plt.ylabel('KL')
plt.yscale('log')
plt.title(f'KL divergence with respect to ground truth distribution: #Exp: {n_exp}')
plt.legend()

plt.savefig(os.path.join(RESPATH, 'MCMC_OM_Comparison.png'))
