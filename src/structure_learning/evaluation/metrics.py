from abc import abstractmethod
from typing import Union, List
import numpy as np
from structure_learning.data_structures import DAG
from structure_learning.distributions import Distribution

class Metric:
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, dist1: Distribution, dist2: Distribution):
        pass

def entropy(P, Q):
    return np.sum(P * (np.log(P) - np.log(Q)))

def kl_divergence(P : list, Q : list, epsilon: float = 1e-15):
    """
    Computes the KL divergence between two distributions.
    Requires that the two distributions have the same length and keys.
    """
    # Ensure that the distributions have the same length
    if len(P) != len(Q):
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1

    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(P) + epsilon
    q = np.array(Q) + epsilon

    return entropy(P, Q)

class KLD(Metric):

    def __init__(self):
        super().__init__()

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()) & set(dist2.particles.keys())

        dist1.normalise()
        dist2.normalise()

        P = [dist1.particles[key]['p'] if key in dist1 else 0. for key in keys]
        Q = [dist2.particles[key]['p'] if key in dist2 else 0. for key in keys]

        return kl_divergence(P, Q)

def jensen_shannon_divergence(P : list, Q : list, epsilon: float = 1e-15):
    """
    Compute the jensen_shannon_divergence between two distributions.
    Requires that the two distributions have the same length.
    """
    # Ensure the distributions have the same length
    if len(P) != len(Q):
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1

    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(P) + epsilon
    q = np.array(Q) + epsilon

    # Normalize the distributions to ensure they are proper probability distributions
    p /= p.sum()
    q /= q.sum()

    # Compute M
    m = 0.5 * (p + q)

    # Compute the Jensen-Shannon divergence
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))

    return jsd

class JSD(Metric):

    def __init__(self):
        super().__init__()

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()) & set(dist2.particles.keys())

        dist1.normalise()
        dist2.normalise()

        P = [dist1.particles[key]['p'] if key in dist1 else 0 for key in keys]
        Q = [dist2.particles[key]['p'] if key in dist2 else 0 for key in keys]

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

    def __init__(self):
        super().__init__()

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()) & set(dist2.particles.keys())

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

    def __init__(self):
        super().__init__()

    def compute(self, dist1: Distribution, dist2: Distribution):
        
        keys = set(dist1.particles.keys()) & set(dist2.particles.keys())

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

    def __init__(self):
        super().__init__()

    def compute(self, dags: Union[Distribution, DAG], true_DAG: DAG):
        p = None
        if isinstance(dags, DAG):
            dags = [dags.incidence]
        else:
            try:
                p = dags.prop('p')
            except:
                dags.normalise()
                p = dags.prop('p')
            dags = [DAG.from_key(key=dag, nodes=true_DAG.nodes).incidence for dag in dags.particles]

        return expected_shd(dags, true_DAG.incidence, p)