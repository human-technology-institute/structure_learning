"""

"""
import pickle
import numpy as np
from scipy.stats import entropy
from sklearn import metrics
from structure_learning.utils.graph_utils import update_graph_frequencies

def kl_divergence(P : dict, Q : dict, epsilon: float = 1e-15):
    """
    Computes the KL divergence between two distributions.
    Requires that the two distributions have the same length.
    """
    # Ensure that the distributions have the same length
    # if the dist1 and dist2 do not have the same length, return an error and exit
    len_diff = np.abs(len(P) - len(Q))
    if len_diff > 0:
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1

    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(list(P.values())) + epsilon
    q = np.array(list(Q.values())) + epsilon

    return entropy(p, q)

def jensen_shannon_divergence(P : dict, Q : dict, epsilon: float = 1e-15):
    """
    Compute the jensen_shannon_divergence between two distributions.
    Requires that the two distributions have the same length.
    """
    # Ensure the distributions have the same length
    len_diff = np.abs(len(P) - len(Q))
    if len_diff > 0:
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1

    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(list(P.values())) + epsilon
    q = np.array(list(Q.values())) + epsilon

    # Normalize the distributions to ensure they are proper probability distributions
    p /= p.sum()
    q /= q.sum()

    # Compute M
    m = 0.5 * (p + q)

    # Compute the Jensen-Shannon divergence
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))

    return jsd


def mean_squared_error(P : dict, Q : dict):
    """
    Compute mean squared error.
    """
    P = np.array(list(P.values())).astype(float)
    Q = np.array(list(Q.values())).astype(float)

    return np.mean((P - Q)**2)

def mean_absolute_error(P : dict, Q : dict):
    """
    Compute mean absolute error.
    """
    P = np.array(list(P.values())).astype(float)
    Q = np.array(list(Q.values())).astype(float)

    return np.mean(np.abs(P - Q))

def expected_shd_DAGFlows( chain_list : list, true_DAG : np.ndarray):
    """
    Compute the Expected Structural Hamming Distance.

    This function computes the Expected SHD between a posterior approximation
    given as a collection of samples from the posterior, and the ground-truth
    graph used in the original data generation process.
    Code from:
    https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/utils/metrics.py

    Parameters:
        chain_list (list (numpy.ndarray)):  Posterior approximation.
                                            The array must have size `(B, N, N)`, where `B` is the
                                            number of sample graphs from the posterior approximation,
                                            and `N` is the number of variables in the graphs.
        true_DAG (numpy.ndarray):           Adjacency matrix of the ground-truth graph.
                                            The array must have size `(N, N)`,
                                            where `N` is the number of variables in the graph.

    Returns:
        (float):                            The Expected SHD.
    """

    posterior = np.array(list(chain_list))

    # Compute the pairwise differences
    diff = np.abs(posterior - np.expand_dims(true_DAG, axis=0))
    diff = diff + diff.transpose((0, 2, 1))

    # Ignore double edges
    diff = np.minimum(diff, 1)
    shds = np.sum(diff, axis=(1, 2)) / 2

    return np.mean(shds)


def pairwise_structural_hamming_distance(x, y):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
    This means, edge reversals do not double count, and that getting an undirected edge wrong only counts 1

    Parameters:
        x (numpy.ndarray): batch of adjacency matrices  [N, d, d]
        y (numpy.ndarray): batch of adjacency matrices  [M, d, d]

    Returns:
        (numpy.ndarray): matrix of shape ``[N, M]``  where elt ``i,j`` is  SHD(``x[i]``, ``y[j]``)
    """

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    assert(x.ndim == 3 and y.ndim == 3)

    # via computing pairwise differences
    pw_diff = np.abs(np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0))
    pw_diff = pw_diff + pw_diff.transpose((0, 1, 3, 2))

    # ignore double edges
    pw_diff = np.where(pw_diff > 1, 1, pw_diff)
    shd = np.sum(pw_diff, axis=(2, 3)) / 2

    return shd

def expected_edges_DAGFlows(mcmc_graph_list):
    """
    Compute the expected number of edges.

    This function computes the expected number of edges in graphs sampled from
    the posterior approximation.
    Code from:
    https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/utils/metrics.py

    Parameters:
        mcmc_graph_list (list (numpy.ndarray)):  Posterior approximation.
                                            The array must have size `(B, N, N)`, where `B` is the
                                            number of sample graphs from the posterior approximation,
                                            and `N` is the number of variables in the graphs.

    Returns:
        (float): The expected number of edges.
    """
    num_edges = np.sum(mcmc_graph_list, axis=(1, 2))
    return np.mean(num_edges)

def threshold_metrics_DAGFlows(mcmc_graph_list, true_DAG):
    """
    Compute threshold metrics (e.g. AUROC, Precision, Recall, etc...).
    Code from:
    https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/utils/metrics.py

     Parameters:
        mcmc_graph_list (list (numpy.ndarray)):  Posterior approximation.
                                            The array must have size `(B, N, N)`, where `B` is the
                                            number of sample graphs from the posterior approximation,
                                            and `N` is the number of variables in the graphs.
        true_DAG (numpy.ndarray):           Adjacency matrix of the ground-truth graph.
                                            The array must have size `(N, N)`,
                                            where `N` is the number of variables in the graph.

    Returns
    -------
        (dict): threshold metrics.
    """
    # Expected marginal edge features
    p_edge = np.mean(mcmc_graph_list, axis=0)
    p_edge_flat = p_edge.reshape(-1)

    gt_flat = true_DAG.reshape(-1)

    # Threshold metrics
    fpr, tpr, _ = metrics.roc_curve(gt_flat, p_edge_flat)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt_flat, p_edge_flat)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt_flat, p_edge_flat)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'prc_auc': prc_auc,
        'ave_prec': ave_prec,
    }

def compute_divergence(self, interval = 100, results_output_path = None):
    """

    """
    upper_bound = len(self.mcmc_graph_list) // interval

    # create an empty disctionary with all graph frequencies from the true posterior set to zero
    # this index structure will be used to update the graph frequencies as the MCMC chain progresses
    true_posterior_index = {key: 0 for key in self.true_posterior.keys()}

    results = {}
    results['JSD'] = []
    results['MSE'] = []
    results['MAE'] = []

    for i in range(0, interval + 1):
        mcmc_chain_sample_j = self.mcmc_graph_list[0: i * upper_bound]
        approx_posterior_j = update_graph_frequencies(mcmc_chain_sample_j, true_posterior_index)

        results['JSD'].append(self.jensen_shannon_divergence(approx_distr_sample=approx_posterior_j))
        results['MSE'].append(self.mean_squared_error(approx_distr_sample=approx_posterior_j))
        results['MAE'].append(self.mean_absolute_error(approx_distr_sample=approx_posterior_j))

    # save results to file
    if results_output_path is not None:
        self.save_results(results, results_output_path)

def save_results(self, results, filename):
    """
    Pickle results
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
