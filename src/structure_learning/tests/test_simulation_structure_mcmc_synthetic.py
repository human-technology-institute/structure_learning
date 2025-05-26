"""
Test Structure MCMC on synthetic data
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
from structure_learning.data import SyntheticDataset
from structure_learning.mcmc import StructureMCMC
from structure_learning.utils.graph_utils import convert_adj_mat_to_graph, generate_key_from_adj_matrix
from structure_learning.proposals import GraphProposal
from structure_learning.scores import BGeScore

def simulate(n_restarts=1, n_nodes=3, node_degree=2, n_observations=200, n_iterations=300000, \
            save_results=False, results_dir='.', seed=46):
    """
    Run simulations using Structure MCMC on synthetic data

    Parameters:
        n_restarts (int):       Number of independent simulations to run.
                                Each run composes a different synthetic dataset.
        n_nodes (int):          Number of nodes in the generated graphs.
        node_degree (int):      Expected node degree of generated graphs.
        n_iterations (int):     Number of MCMC iterations to run.
        save_results (bool):    If True, save results to file.
        results_dir (str):      Directory or path in which to save the results.
                                Ignored if save_results is False.
    """
    # generate timestamp for this run
    now = datetime.now().strftime('%Y%m%d')

    # for reproducibility
    np.random.seed(seed)

    # create results dir if needed
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    node_labels = [chr(ord('a') + i) for i in range(n_nodes)]

    for r_idx in range(n_restarts):
        synthetic_data = SyntheticDataset(num_nodes=n_nodes, num_obs=n_observations, \
                                          node_labels=node_labels, degree=node_degree)
        initial_graph = pd.DataFrame(np.random.choice([0,1], size=(n_nodes, n_nodes))*np.tri(n_nodes, n_nodes, -1))
        G = convert_adj_mat_to_graph(initial_graph)
        score = BGeScore(data=synthetic_data, incidence=initial_graph.values)
        proposal = GraphProposal(G)
        M = StructureMCMC(data=synthetic_data, initial_graph=initial_graph.values, max_iter=n_iterations, proposal_object=proposal, score_object=score)
        mcmc_results, acceptance = M.run()
        graphs = M.get_graphs(mcmc_results)

        key = generate_key_from_adj_matrix(synthetic_data.adj_mat.values)
        keys = set([generate_key_from_adj_matrix(g) for g in graphs])

        if save_results:
            np.savez_compressed(f"{results_dir}/{now}_{r_idx}.npz", \
                                adj_mat=synthetic_data.adj_mat,
                                data=synthetic_data.data, mcmc_results=mcmc_results)

        yield mcmc_results, acceptance, synthetic_data, key in keys

if __name__ == "__main__":
    simulate(save_results=True)
