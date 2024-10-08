# Imports
import argparse
import os
import datetime
import numpy as np

from mcmc.scores.bge import BGeScore
from mcmc.proposals.graph.graph_proposal import GraphProposal
from mcmc.data.synthetic import SyntheticDataset

from mcmc.mcmc.structure_mcmc import StructureMCMC

import pickle

import pandas as pd

from mcmc.inference.posterior import *

import time
import csv

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

from tqdm.notebook import trange
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def JS_MCMC(MCMC_dict, true_dist_weights, n):
    """
    MCMC_dict: Dictionary with the visited graph and counts
    true_dist: the wights of the true distribution
    n: number of iterates
    """

    mcmc_dist_p = [v / n for v in MCMC_dict.values()]

    return jensenshannon(mcmc_dist_p, true_dist_weights)


def W_MCMC(MCMC_dict, true_dist, n, logscore=False):
    """
    MCMC_dict: Dictionary with the visited graph and counts
    true_dist: the true distribution
    n: number of iterates
    logscore: If true dist is log(P)
    """

    true_dist_support = [k for k in true_dist.keys()]
    mcmc_dist_support = [k for k in MCMC_dict.keys()]
    true_dist_weights = [np.exp(v) if logscore else v for v in true_dist.values()]
    mcmc_dist_weights = [v / n for v in MCMC_dict.values()]

    W = float(wasserstein_distance(true_dist_support, mcmc_dist_support,
                                   u_weights=true_dist_weights,
                                   v_weights=mcmc_dist_weights))
    return W


# %%
def W_OM(unique_graphs, unique_scores, true_dist, logscore=False):
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

    true_dist_support = [k for k in true_dist.keys()]
    true_dist_weights = [np.exp(v) if logscore else v for v in true_dist.values()]

    W = float(wasserstein_distance(true_dist_support, unique_scores,
                                   u_weights=true_dist_weights,
                                   v_weights=np.exp(log_p_approx_vec)))
    return W


def JS_OM(om_dict, true_dist_weights):
    """

    """

    # Prevents overflow during exponentiation by substracting
    log_raw_p_approx_vec = np.asarray([float(v) for v in om_dict.values()])
    max_score = np.max(log_raw_p_approx_vec)

    log_p_approx_vec = log_raw_p_approx_vec - max_score
    p_approx_vec = np.exp(log_p_approx_vec)
    norm_factor = np.sum(p_approx_vec)
    p_approx_vec /= norm_factor

    # true_dist_weights = [np.exp(v) if logscore else v for v in true_dist.values()]
    return jensenshannon(p_approx_vec, true_dist_weights)


# %%

def JS_comparison_OM_MCMC(mcmc_results, ground_truth_distribution, JS_step=50, logscore=False):
    # log score if groundtruth is log(p)
    true_dist_weights = [np.exp(v) if logscore else v for v in ground_truth_distribution.values()]

    graph_mcmc_count = {k: 0 for k in ground_truth_distribution.keys()}
    graph_om_score = {k: -np.inf for k in ground_truth_distribution.keys()}
    graph_om_accepted_score = {k: -np.inf for k in ground_truth_distribution.keys()}

    sample_num = []
    JS_MCMC_t = []
    JS_OM_t = []
    JS_OM_accepted_only_t = []

    max_mcmc_iter = len(mcmc_results)
    for i in range(max_mcmc_iter):
        res = mcmc_results[i]
        if "Gprop" in res.keys():
            graph_prop = generate_key_from_adj_matrix(res['Gprop'])
            score_prop = res['score_Gprop']
        else:
            graph_prop = None

        graph_id = generate_key_from_adj_matrix(res['graph'])  # Accepted graph/state
        score = res['score']  # Accepted score
        # Score is log of the true score

        graph_mcmc_count[graph_id] += 1

        graph_om_score[graph_id] = score
        # OM update
        if graph_prop is not None:
            # Looking at proposal
            graph_om_score[graph_prop] = score_prop

        # OM update - accepted graphs
        graph_om_accepted_score[graph_id] = score

        if i % JS_step == 0 or i == max_mcmc_iter:
            sample_num.append(i)
            JS_OM_t.append(JS_OM(graph_om_score, true_dist_weights))
            JS_OM_accepted_only_t.append(
                JS_OM(graph_om_accepted_score, true_dist_weights))
            JS_MCMC_t.append(JS_MCMC(graph_mcmc_count, true_dist_weights, i + 1))

        # print(f"{len(unique_accepted_graph_id)} - {len(unique_accepted_graph_id)}")

    return sample_num, JS_MCMC_t, JS_OM_t, JS_OM_accepted_only_t


n_exp = 50
num_nodes = 5
node_labels = [f"X{i + 1}" for i in range(num_nodes)]
noise_scale = 1.
score_type = BGeScore
mcmc_iter = 1000
num_obsv = 200
graph_type = "Structure MCMC"

DIRPATH = os.path.abspath("")
MAINRESPATH = os.path.join(DIRPATH, 'om_results')
os.makedirs(MAINRESPATH, exist_ok=True)

# Save metadata
metadata_dict = {"Num_experiments": n_exp,
                 "num_nodes": num_nodes,
                 "noise_scale": noise_scale,
                 "score_type": score_type.__name__,
                 "MCMC_start_point": 'random',
                 "mcmc_iter": mcmc_iter,
                 "graph_type": graph_type,
                 "MCMC_type": "Structure"
                 }

smcmc_Res_folder = r"C:\Users\161342\code\structure_learning\examples\om_results\structure_MCMC"
smcmc_folders = ["2024-09-24_00-10-31", "2024-09-24_23-51-35", "2024-09-25_23-34-13"]

for folder in smcmc_folders:
    directory = os.path.join(smcmc_Res_folder, folder)
    RESPATH = os.path.join(MAINRESPATH, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(RESPATH, exist_ok=True)

    metadata_df = pd.read_csv(os.path.join(directory, 'metadata.csv'), header=0)

    for col in metadata_df.columns:
        metadata_dict[col] = metadata_df.loc[0, col]

    with open(os.path.join(RESPATH, 'metadata.csv'), 'w', newline="") as csvfile:
        w = csv.DictWriter(csvfile, metadata_dict.keys())
        w.writeheader()
        w.writerow(metadata_dict)

    MCMC_JS_results_dict = {}
    OM_JS_results_dict = {}
    OM_JS_accepted_results_dict = {}

    for exp_i in range(n_exp):
        print(f"Experiment: {exp_i}")

        # Load dag from Structure MCMC experiment
        true_dag = np.loadtxt(os.path.join(directory, f"true_DAG_{exp_i}.csv"), delimiter=',')
        print(true_dag)

        # Save to results folder
        np.savetxt(os.path.join(RESPATH, f"true_DAG_{exp_i}.csv"), true_dag, delimiter=',', fmt='%d')

        assert not (has_cycle(true_dag)), "not a DAG"

        # Generate data based on the true DAG
        print(f"{exp_i}: Generate data")

        data = pd.read_csv(os.path.join(directory, f"data_{exp_i}.csv"), delimiter=',',
                           names=[f"X{i + 1}" for i in range(num_nodes)])

        # np.loadtxt(os.path.join(directory, f"data_{exp_i}.csv"), delimiter=',')
        # save data to file
        np.savetxt(os.path.join(RESPATH, f"data_{exp_i}.csv"), data, delimiter=',')

        # Load ground truth distribution from MCMC folder
        print(f"{exp_i}: Load ground truth")
        with open(os.path.join(directory, f'true_distribution_results_{exp_i}.pkl'), 'rb') as file:
            log_true_distr = pickle.load(file)

        with open(os.path.join(RESPATH, f'true_distribution_results_{exp_i}.pkl'), 'wb') as f:
            pickle.dump(log_true_distr, f)

        # MCMC
        print(f"{exp_i}: MCMC")
        if metadata_dict['MCMC_start_point'] == 'random':
            # static function does not change sdg
            sdj_random_g = SyntheticDataset(num_nodes=metadata_dict['num_nodes'], num_obs=len(data),
                                            node_labels=node_labels,
                                            degree=metadata_dict['dag_sparse_degree'],
                                            noise_scale=metadata_dict['noise_scale'],
                                            graph_type=metadata_dict['graph_type'])
            initial_graph = sdj_random_g.adj_mat.values
        else:
            raise ValueError

        np.savetxt(os.path.join(RESPATH, f"MCMC_initial_graph_{exp_i}.csv"), initial_graph, delimiter=',', fmt='%d')
        proposal_object = GraphProposal(initial_graph, whitelist=None, blacklist=None)
        score_object = score_type(data=data, incidence=initial_graph)

        #metadata_dict['mcmc_iter']
        mcmc_obj = StructureMCMC(initial_graph, 1000, proposal_object, score_object)
        mcmc_res, accept_rate = mcmc_obj.run()

        sample_num, JS_MCMC_res, JS_OM_res, JS_OM_accepted_res = JS_comparison_OM_MCMC(mcmc_res, log_true_distr, logscore=True)
        if len(MCMC_JS_results_dict)==0:
            MCMC_JS_results_dict.update({f"sample_num": sample_num})
            OM_JS_results_dict.update({f"sample_num": sample_num})
            OM_JS_accepted_results_dict.update({f"sample_num": sample_num})

        MCMC_JS_results_dict.update({f"W_MCMC_{exp_i}": JS_MCMC_res})
        OM_JS_results_dict.update({f"W_OM_{exp_i}": JS_OM_res})
        OM_JS_accepted_results_dict.update({f"KL_OM_accepted_{exp_i}": JS_OM_accepted_res})

        df = pd.DataFrame(MCMC_JS_results_dict)
        df.to_csv(os.path.join(RESPATH, 'MCMC_JS_results.csv'), index=True)
        df = pd.DataFrame(OM_JS_results_dict)
        df.to_csv(os.path.join(RESPATH, 'OM_JS_results.csv'), index=True)
        df = pd.DataFrame(OM_JS_accepted_results_dict)
        df.to_csv(os.path.join(RESPATH, 'OM_JS_accepted_only_results.csv'), index=True)

        # For debugging purposes
        graph_list = mcmc_obj.get_mcmc_res_graphs(mcmc_res)
        score_list = mcmc_obj.get_mcmc_res_scores(mcmc_res)

        with open(os.path.join(RESPATH, f'MCMC_results_{exp_i}.pkl'), 'wb') as f:
            pickle.dump(graph_list, f)

        with open(os.path.join(RESPATH, f'score_results_{exp_i}.pkl'), 'wb') as f:
            pickle.dump(score_list, f)

        with open(os.path.join(RESPATH, f'mcmc_results_{exp_i}.pkl'), 'wb') as f:
            pickle.dump(mcmc_res, f)

    # Plotting
    df_MCMC = pd.read_csv(os.path.join(RESPATH, 'MCMC_W_results.csv'), index_col=0)
    df_OM = pd.read_csv(os.path.join(RESPATH, 'OM_W_results.csv'), index_col=0)
    quantiles = [0.05, 0.95]

    MCMC_mean = df_MCMC.mean(axis=1)
    MCMC_std = df_MCMC.std(axis=1)

    MCMC_quantiles = df_MCMC.apply(lambda row: row.quantile(quantiles), axis=1)

    OM_mean = df_OM.mean(axis=1)
    OM_std = df_OM.std(axis=1)
    OM_quantiles = df_OM.apply(lambda row: row.quantile(quantiles), axis=1)

    plt.plot(MCMC_mean)
    plt.plot(MCMC_mean, color='blue', label='MCMC')
    plt.fill_between(MCMC_mean.index, MCMC_quantiles[quantiles[0]], MCMC_quantiles[quantiles[1]], color='blue',
                     alpha=0.2)
    plt.plot(OM_mean, color='red', label='OM')
    plt.fill_between(OM_mean.index, OM_quantiles[quantiles[0]], OM_quantiles[quantiles[1]], color='red', alpha=0.2)

    plt.xlabel('#Iteration')
    plt.ylabel('KL')
    plt.yscale('log')
    plt.title(f'KL divergence with respect to ground truth distribution: #Exp: {n_exp}')
    plt.legend()

    plt.savefig(os.path.join(RESPATH, 'MCMC_OM_Comparison.png'))
