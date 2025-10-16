"""
This module provides various metrics for evaluating distributions and graphs.

It includes implementations of metrics such as KL Divergence, Jensen-Shannon Divergence,
Mean Squared Error, Mean Absolute Error, Expected Structural Hamming Distance, and RHat.
These metrics are used to compare distributions or evaluate the similarity between graphs.

Classes:
    Metric: Abstract base class for metrics.
    KLD: Computes the KL Divergence between two distributions.
    JSD: Computes the Jensen-Shannon Divergence between two distributions.
    MSE: Computes the Mean Squared Error between two distributions.
    MAE: Computes the Mean Absolute Error between two distributions.
    SHD: Computes the Expected Structural Hamming Distance between graphs.
    RHat: Computes the RHat metric for multiple distributions.

Functions:
    entropy: Computes the entropy between two distributions.
    kl_divergence: Computes the KL divergence between two distributions.
    jensen_shannon_divergence: Computes the Jensen-Shannon divergence between two distributions.
    mean_squared_error: Computes the mean squared error between two distributions.
    mean_absolute_error: Computes the mean absolute error between two distributions.
    expected_shd: Computes the Expected Structural Hamming Distance.
    rhat: Computes the RHat metric for the provided distributions and measurements.
"""

from abc import abstractmethod
from typing import Union, List
from functools import reduce
import tqdm
import numpy as np
from structure_learning.data_structures import DAG, Graph
from structure_learning.distributions import Distribution

class Metric:
    """
    Abstract base class for metrics.

    This class serves as a blueprint for all metric implementations. It defines
    the structure and behavior that all derived metric classes must follow.
    """

    @abstractmethod
    def compute(self, **kwargs):
        pass

def entropy(P, Q):
    return np.sum(P * (np.log(P) - np.log(Q)))

def kl_divergence(P : list, Q : list, epsilon: float = 1e-10):
    """
    Computes the KL divergence between two distributions.
    Requires that the two distributions have the same length and keys.
    """
    # Ensure that the distributions have the same length
    if len(P) != len(Q):
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1

    # Convert the distributions to lists (ensuring consistent order)
    p = np.maximum(np.array(P), epsilon)
    q = np.maximum(np.array(Q), epsilon)

    return entropy(p, q)

class KLD(Metric):

    """
    Computes the KL Divergence (KLD) between two distributions.

    This class implements the KLD metric, which measures the difference between
    two probability distributions. It inherits from the Metric base class and
    provides a concrete implementation of the compute method.
    """

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()).union(set(dist2.particles.keys()))

        dist1.normalise()
        dist2.normalise()

        P = [dist1.particles[key]['p'] if key in dist1 else 0. for key in keys]
        Q = [dist2.particles[key]['p'] if key in dist2 else 0. for key in keys]

        return kl_divergence(P, Q)

def jensen_shannon_divergence(P : list, Q : list, epsilon: float = 1e-10):
    """
    Compute the jensen_shannon_divergence between two distributions.
    Requires that the two distributions have the same length.
    """
    # Ensure the distributions have the same length
    if len(P) != len(Q):
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1

    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(P)
    q = np.array(Q)
    p[p<epsilon] = epsilon
    q[q<epsilon] = epsilon

    # Normalize the distributions to ensure they are proper probability distributions
    p /= p.sum()
    q /= q.sum()

    # Compute M
    m = 0.5 * (p + q)

    # Compute the Jensen-Shannon divergence
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))

    return jsd

class JSD(Metric):
    """
    Computes the Jensen-Shannon Divergence (JSD) between two distributions.

    This class implements the JSD metric, which measures the similarity between
    two probability distributions. It inherits from the Metric base class and
    provides a concrete implementation of the compute method.
    """

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()).union(set(dist2.particles.keys()))

        dist1.normalise()
        dist2.normalise()

        P = [dist1.particles[key]['p'] if key in dist1 else 0. for key in keys]
        Q = [dist2.particles[key]['p'] if key in dist2 else 0. for key in keys]

        return jensen_shannon_divergence(P, Q)

def mean_squared_error(P : list, Q : list):
    """
    Compute mean squared error.
    """
    # Ensure the distributions have the same length
    if len(P) != len(Q):
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return np.nan
    P = np.array(P).astype(float)
    Q = np.array(Q).astype(float)

    return np.mean((P - Q)**2)

class MSE(Metric):
    """
    Computes the Mean Squared Error (MSE) between two distributions.

    This class implements the MSE metric, which measures the average squared difference
    between two probability distributions. It inherits from the Metric base class and
    provides a concrete implementation of the compute method.
    """

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()).union(set(dist2.particles.keys()))

        dist1.normalise()
        dist2.normalise()

        P = [dist1.particles[key]['p'] if key in dist1 else 0 for key in keys]
        Q = [dist2.particles[key]['p'] if key in dist2 else 0 for key in keys]

        return mean_squared_error(P, Q)

def mean_absolute_error(P : list, Q : list):
    """
    Compute mean absolute error.
    """
    # Ensure the distributions have the same length
    if len(P) != len(Q):
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return np.nan
    P = np.array(P).astype(float)
    Q = np.array(Q).astype(float)

    return np.mean(np.abs(P - Q))

class MAE(Metric):
    """
    Computes the Mean Absolute Error (MAE) between two distributions.

    This class implements the MAE metric, which measures the average absolute difference
    between two probability distributions. It inherits from the Metric base class and
    provides a concrete implementation of the compute method.
    """

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()).union(set(dist2.particles.keys()))

        dist1.normalise()
        dist2.normalise()

        P = [dist1.particles[key]['p'] if key in dist1 else 0 for key in keys]
        Q = [dist2.particles[key]['p'] if key in dist2 else 0 for key in keys]

        return mean_absolute_error(P, Q)

def expected_shd(chain_list : List[np.ndarray], true_DAG : np.ndarray, p: List = None):
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

    posterior = np.array(list(chain_list)).astype(int)

    # Compute the pairwise differences
    diff = np.abs(posterior - np.expand_dims(true_DAG.astype(int), axis=0))
    diff = diff + diff.transpose((0, 2, 1))

    # Ignore double edges
    diff = np.minimum(diff, 1)
    shds = np.sum(diff, axis=(1, 2)) / 2

    return np.mean(shds) if p is None else np.mean(shds*np.array(p))

class SHD(Metric):
    """
    Computes the Expected Structural Hamming Distance (SHD) between graphs.

    This class implements the SHD metric, which measures the expected structural
    difference between a posterior approximation and a ground-truth graph. It inherits
    from the Metric base class and provides a concrete implementation of the compute method.
    """

    def compute(self, dags: Union[Distribution, Graph], true_graph: Graph):
        p = None
        if isinstance(dags, DAG):
            dags = [dags.incidence]
        else:
            try:
                p = dags.prop('p')
            except:
                dags.normalise()
                p = dags.prop('p')
            dags = [Graph.from_key(key=dag, nodes=true_graph.nodes).incidence for dag in dags.particles]

        return expected_shd(dags, true_graph.incidence, p)

def rhat(P: list, th: list):
    """
    Compute rhat metric for the provided distributions and measurement.

    Parameters:
        P (list):       distributions (must be normalised)
        th (list):      scores/measurements
    """
    M = len(P)
    theta_m = np.array([np.sum(np.array(p)*np.array(t)) for p,t in zip(P,th)])
    theta = np.mean(theta_m)
    s_m = [np.sum(np.array(p)*(np.array(t)**2)) for p,t in zip(P,th)] - theta_m**2
    B = (1/(M-1))*np.sum((theta_m - theta)**2)
    W = np.mean(s_m)
    v = W + B
    Rhat = np.sqrt(v/W)
    return Rhat

class RHat(Metric):
    """
    Computes the RHat metric for multiple distributions.

    This class implements the RHat metric, which is used to assess the convergence
    of multiple distributions. It inherits from the Metric base class and provides
    a concrete implementation of the compute method.
    """
    def compute(self, dists: List[Distribution], prop='logp'):
        if len(dists) < 2:
            raise Exception("There must be at least 2 distributions.")
        keys = set(dists[0].particles.keys())
        dists[0].normalise()
        for dist in dists[1:]:
            keys = keys.union(dist.particles.keys())
            dist.normalise()

        P = [[dist.particles[key]['p'] if key in dist else 0 for key in keys] for dist in dists]
        th = [[dist.particles[key][prop] if key in dist else 0 for key in keys] for dist in dists]

        return rhat(P, th)

def marginal_edge_probabilities(dist: Distribution):
    def fn(arg1, arg2):
        term1 = arg1
        if isinstance(arg1, tuple):
            p = arg1[1]['p']
            term1 = np.array([p if x=='1' else 0 for x in arg1[0]])

        if isinstance(arg2, tuple):
            p = arg2[1]['p']
            term2 = np.array([p if x=='1' else 0 for x in arg2[0]])

        return term1 + term2

    from functools import reduce
    mep = reduce(fn, tqdm.tqdm(dist.particles.items()))
    mep = np.concatenate((mep, [0]))
    n = int(np.sqrt(len(mep)))
    return mep.reshape(n,n+1)[:n, :n]

class MEP(Metric):
        
    def compute(self, dist: Distribution):
        return marginal_edge_probabilities(dist)

def marginal_ancestor_probabilities(dist: Distribution):
    def fn(acc, args):
        k, v = args
        n = len(k.split())
        dag = DAG.from_key(key=k, nodes=list(range(n)))
        return acc + DAG.compute_ancestor_matrix(dag.incidence).T*v['p']

    return reduce(fn, dist.particles.items(), 0)

class MarginalAncestorProbabilities(Metric):
        
    def compute(self, dist: Distribution):
        return marginal_ancestor_probabilities(dist)

def rhat_edge(dists: List[Distribution]):
    mep = map(marginal_edge_probabilities, dists)

    means = np.array(mep)
    var = means - means**2

    B = np.var(means, axis=0)
    W = np.mean(var, axis=0)

    v = W + B
    R_hat = np.sqrt(v/W)
    return R_hat

class RHat_edge(Metric):
    
    def compute(self, dists: Distribution):
        return rhat_edge(dists)

_metrics = {
    'kl': KLD,
    'kld': KLD,
    'KLD': KLD,
    'js': JSD,
    'jsd': JSD,
    'JSD': JSD, 
    'mse': MSE,
    'MSE': MSE,
    'mae': MAE,
    'MAE': MAE,
    'shd': SHD,
    'SHD': SHD,
    'rhat': RHat,
    'RHat': RHat,
    'marginal_edge_probabilities': MEP,
    'MEP': MEP,
    'marginal_ancestor_probabilities': MarginalAncestorProbabilities,
    'MarginalAncestorProbabilities': MarginalAncestorProbabilities,
    'rhat_edge': RHat_edge,
    'RHat_edge': RHat_edge
}

def get_metric(metric: str):
    if metric not in _metrics:
        raise Exception('Metric not found,', metric)
    return _metrics[metric]()