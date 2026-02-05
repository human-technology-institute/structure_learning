"""
This module provides methods for causal inference and effect estimation using Bayesian approaches.

Classes:
    CausalEffects:
        A class for performing causal inference and estimating effects using directed acyclic graphs (DAGs) and observational data.

Functions:
    gibbs_linear(X, y, n_iter=2000, burn_in=500):
        Performs Bayesian linear regression with unknown variance via Gibbs sampling.

    gibbs_probit(X, y, n_iter=2000, burn_in=500):
        Implements the Albert-Chib Gibbs sampler for probit regression.

    estimate_hybrid_dag(adj_matrix, data, domains, n_iter=2000, burn_in=500):
        Estimates posterior samples of parameters for each node in a DAG.

    normalise_data(data, domains):
        Standardises continuous columns to mean=0, sd=1 while leaving binary columns unchanged.

    denormalise_linear_sample(beta_norm, child_idx, parent_idxs, mus, sds):
        Converts normalised linear beta samples to their original scale.

    denormalise_probit_sample(beta_norm, child_idx, parent_idxs, mus, sds):
        Converts normalised probit beta samples to their original latent scale.

    simulate_do_effects(adj_matrix, intervention, est_params, domains, data, do_value=1.0, multiply=False):
        Simulates do-intervention effects on raw data, injecting noise at each step.

"""

from typing import Union, List
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt
import seaborn as sns
import sumu
from structure_learning.data_structures import DAG
from structure_learning.data import Data
from structure_learning.distributions import MCMCDistribution

# --- Gibbs samplers for parameter estimation --- #
def gibbs_linear(X, y, n_iter=2000, burn_in=500):
    """Bayesian linear regression with unknown variance via Gibbs."""
    N, D = X.shape
    beta_samples = np.zeros((n_iter-burn_in, D))
    sigma2_samples = np.zeros(n_iter-burn_in)
    
    # initial values & priors
    beta = np.zeros(D)
    sigma2 = 1.0
    invV0 = np.eye(D) * 1e-6  # weak prior precision
    a0, b0 = 1.0, 1.0         # Inv-Gamma(a0, b0)
    
    for t in range(n_iter):
        # sample beta | sigma2, y
        Vn = np.linalg.inv(invV0 + X.T @ X / sigma2)
        mun = Vn @ (X.T @ y / sigma2)
        beta = np.random.multivariate_normal(mun, Vn)
        
        # sample sigma2 | beta, y
        resid = y - X @ beta
        an = a0 + N/2
        bn = b0 + 0.5 * np.sum(resid**2)
        sigma2 = 1 / np.random.gamma(an, 1/bn)
        
        if t >= burn_in:
            idx = t - burn_in
            beta_samples[idx] = beta
            sigma2_samples[idx] = sigma2
    
    return beta_samples, sigma2_samples

def gibbs_probit(X, y, n_iter=2000, burn_in=500):
    """Albert-Chib Gibbs sampler for probit regression."""
    N, D = X.shape
    beta_samples = np.zeros((n_iter-burn_in, D))
    beta = np.zeros(D)
    
    for t in range(n_iter):
        # 1) Sample latent z
        mu = X @ beta
        # define truncation bounds
        a = np.where(y==1, 0 - mu, -np.inf - mu)
        b = np.where(y==1, np.inf - mu, 0 - mu)
        z = truncnorm.rvs(a, b, loc=mu, scale=1)
        
        # 2) Sample beta | z
        V_post = np.linalg.inv(X.T @ X + np.eye(D))
        mu_post = V_post @ (X.T @ z)
        beta = np.random.multivariate_normal(mu_post, V_post)
        
        if t >= burn_in:
            idx = t - burn_in
            beta_samples[idx] = beta
    
    return beta_samples

def estimate_hybrid_dag(adj_matrix, data, domains, n_iter=5000, burn_in=2000):
    """
    Given:
      - adj_matrix: (n x n) adjacency (0/1) of a DAG,
      - data:      (N x n) observations,
      - domains:   length-n list, 'continuous' or 'binary'
    Returns posterior samples of parameters for each node.
    """
    params = []
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = [adj_matrix]
    for m in adj_matrix:
        G = nx.DiGraph(m)
        N, n = data.shape
        param_samples = {}
        
        for j in range(n): # Iterating over each node
            parents = list(G.predecessors(j))
            # design matrix: intercept + parent columns
            Xj = np.column_stack([np.ones(N)] + [data[:, p] for p in parents])
            yj = data[:, j]
            
            if domains[j] == 'continuous':
                beta_samps, sigma2_samps = gibbs_linear(Xj, yj, n_iter, burn_in)
                param_samples[j] = {
                    'type': 'linear',
                    'beta': beta_samps,
                    'sigma2': sigma2_samps
                }
            else:
                beta_samps = gibbs_probit(Xj, yj, n_iter, burn_in)
                param_samples[j] = {
                    'type': 'probit',
                    'beta': beta_samps
                }
        params.append(param_samples)
    
    return params

def normalise_data(data, domains):
    """
    Standardise continuous columns to mean=0, sd=1; leave binaries unchanged.
    Returns:
      data_norm : np.ndarray
      mus       : dict (column means)
      sds       : dict (column s.d.)
    """
    data = np.asarray(data, dtype=float)
    N, n = data.shape
    mus, sds = {}, {}
    data_norm = data.copy()

    for j, dom in enumerate(domains):
        if dom == 'continuous':
            mu = data[:, j].mean()
            sd = data[:, j].std(ddof=1)
            mus[j], sds[j] = mu, sd
            data_norm[:, j] = (data[:, j] - mu) / sd
        else:
            # binary: leave as-is, mean=0, sd=1
            mus[j], sds[j] = 0.0, 1.0
            data_norm[:, j] = data[:, j]

    return data_norm, mus, sds

def denormalise_linear_sample(beta_norm, child_idx, parent_idxs, mus, sds):
    """
    Given one normalised linear beta sample [intercept, slopes...],
    return (intercept_orig, slopes_orig_list).
    """
    # slopes on original scale
    slopes_orig = [
        beta_norm[i+1] * (sds[child_idx] / sds[p])
        for i, p in enumerate(parent_idxs)
    ]
    # intercept on original scale
    intercept_orig = (
        mus[child_idx]
        + beta_norm[0] * sds[child_idx]
        - sum(slopes_orig[i] * mus[parent_idxs[i]] for i in range(len(parent_idxs)))
    )
    return intercept_orig, slopes_orig


def denormalise_probit_sample(beta_norm, child_idx, parent_idxs, mus, sds):
    """
    Given one normalised probit beta sample [intercept, slopes...],
    return (intercept_orig, slopes_orig_list) on latent scale.
    """
    slopes_orig = [
        beta_norm[i+1] / sds[p]
        for i, p in enumerate(parent_idxs)
    ]
    intercept_orig = (
        beta_norm[0]
        - sum(beta_norm[i+1] * mus[parent_idxs[i]] / sds[parent_idxs[i]]
              for i in range(len(parent_idxs)))
    )
    return intercept_orig, slopes_orig

def simulate_do_effects(adj_matrix, intervention, est_params, domains, data, do_value=1.0, multiply=False, sds = None, tol=1e-8):
    """
    Perform do-intervention simulations on standardised data, injecting noise at each step.
    - Continuous variables are on z-scale (mean 0, sd 1).
    - Binary variables are 0/1.
    
    est_params[j] should contain:
      - 'beta_mean': array [intercept, slopes...]
      - optional 'sigma2_mean' for continuous nodes
    do_value: if multiply=False, adds do_value to Xi; if multiply=True, multiplies Xi by do_value
    multiply: boolean flag to apply do_value as multiplier instead of additive shift

    Continuous effects are rescaled back to original units;
    binary effects are left as probability differences.
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = [adj_matrix]
        est_params = [est_params]
    effects = []

    for idx,m in enumerate(adj_matrix):
        G = nx.DiGraph(m)
        topo = list(nx.topological_sort(G))
        n = m.shape[0]
        N = data.shape[0]

        any_node = 0
        T = est_params[idx][any_node]['beta'].shape[0]

        baseline_means = data.mean(axis=0)

        for t in range(T):
            effect_matrix = np.zeros((n, n))

            for i in range(n):
                if i not in intervention:
                    continue

                data_do = data.copy()
                
                # Distinct values of variable i
                vals = np.unique(np.round(data[:, i], 8))

                # Two-level (binary-like) variable: set to low or high level 
                if isinstance(do_value,str) and vals.size == 2:  
                    low_val, high_val = vals.min(), vals.max() 
                    mode = do_value.lower() 
                    if mode == 'high': 
                        # do(X_i = high) 
                        data_do[:, i] = high_val 
                    elif mode == 'low': 
                        # do(X_i = low) 
                        data_do[:, i] = low_val
                    else: 
                        raise ValueError("Invalid do_value for binary-like variable. Use 'high' or 'low'.")
                    
                # Continuous variable: Numeric intervention (shift or scale)
                else:
                    if multiply:
                        data_do[:, i] = data_do[:, i] * do_value
                    else:
                        data_do[:, i] = data_do[:, i] + do_value/sds[i]

                # propagate through children with noise
                for j in topo:
                    if j == i:
                        continue
                    parents = list(G.predecessors(j))

                    beta = est_params[idx][j]['beta'][t, :]
                    X = data_do[:, parents] if parents else np.zeros((N, 0))
                    mu = beta[0] + (X @ beta[1:])

                    if domains[j] == 'continuous':
                        # add Gaussian noise with estimated sigma
                        sigma2_draws = est_params[idx][j].get('sigma2', None)
                        if sigma2_draws is None:
                            sigma = 1.0
                        else:
                            sigma = float(np.sqrt(sigma2_draws[t]))
                        data_do[:, j] = mu + np.random.normal(scale=sigma, size=N)
                    else:
                        # sample latent z ~ N(mu,1) and threshold
                        z = np.random.normal(loc=mu, scale=1.0, size=N)
                        data_do[:, j] = (z > 0).astype(int)

                delta = data_do.mean(axis=0) - baseline_means 
                effect_matrix[i, :] = delta                
                # Rescaling effects if sds provided:
                if sds is not None:
                    for j in range(n):
                        if domains[j] == 'continuous':
                            effect_matrix[i, j] *= sds[j]
            effects.append(effect_matrix)
    
    return np.array(effects)
    
class CausalEffects:
    def __init__(self, graphs: Union[DAG, List[DAG], MCMCDistribution], data: Data):
        """
        Initialize the CausalEffects object with a graph and data.

        Parameters:
            graph (DAG): The directed acyclic graph representing the causal structure.
            data (Data): The observational data.
        """
        self.graphs = graphs
        self.data = data
        self.domains = [data.variable_types[v] for v in data.columns]

        # Standardise internally (continuous only; binary left 0/1)
        self.data_norm, self.mus, self.sds = normalise_data(self.data.values, self.domains)


    def beeps(self, edges: List[tuple] = None, plot: bool = False):
        """
        Compute pairwise causal effects using the BEEPS algorithm.

        Returns:
            List[np.ndarray]: A list of pairwise causal effect matrices for each graph.
        """
        if self.graphs is None:
            raise ValueError("No graph provided for causal effects computation.")
        graphs = self.graphs if isinstance(self.graphs, list) else ([self.graphs] if isinstance(self.graphs, DAG) else [DAG.from_key(key=g, nodes=list(self.data.columns)) for g in self.graphs.particles])
        weights = 1. if not isinstance(self.graphs, MCMCDistribution) else self.graphs.prop('p')
        effects = sumu.beeps(dags=[g.incidence for g in graphs], data=self.data.values.values).sample_pairwise()
        node_to_index = {node:idx for idx,node in enumerate(self.data.columns)}
        if plot:
            if edges is None:
                edges = [(node1, node2) for node1 in self.data.columns for node2 in self.data.columns if node1 != node2]
            self.plot(effects, weights, edges)

        return effects, weights
    
    def plot(self, effects, weights, edges):
        effects = np.asarray(effects)
        weights = np.asarray(weights).reshape(-1)
        #effects_reshaped = pd.DataFrame(np.reshape(effects, (-1, np.prod(effects.shape[1:]))), columns=[(node1, node2) for node1 in self.data.columns for node2 in self.data.columns])
        #effects_reshaped['weights'] = weights.flatten()
        effects_reshaped = pd.DataFrame(effects.reshape(effects.shape[0], -1),
            columns=[(node1, node2) for node1 in self.data.columns for node2 in self.data.columns])
        effects_reshaped['weights'] = weights
        effects_melt = pd.melt(effects_reshaped, value_vars=edges, value_name='param', var_name='edge', id_vars='weights')
        effects_melt['edge_str'] = effects_melt['edge'].apply(lambda x: f"{x[0]} -> {x[1]}")
        sns.kdeplot(data=effects_melt, x='param', weights='weights', hue='edge', fill=True, common_norm=False)
        plt.xlabel('Causal Effect')
        plt.ylabel('Density')
        plt.title('Pairwise Causal Effects')
        #plt.legend([f"{edge[0]} -> {edge[1]}" for edge in edges])

    def do(self, intervention: List[Union[int, str]], do_value: float = 1.0, multiply: bool = False) -> np.ndarray:
        """
        Perform a do-intervention on the graph and data.

        Parameters:
            intervention (List[Union[int, str]]): The node indices or labels to intervene on.
            do_value (float): The value to set for the intervention.
            multiply (bool): If True, applies the intervention as a multiplier; otherwise, adds it.

        Returns:
            np.ndarray: The effect of the intervention on the data.
        """
        return self.simulate(intervention, do_value, multiply)

    def simulate(self, intervention: List[Union[int, str]], do_value: float = 1.0, multiply: bool = False, plot=False, edges=None) -> np.ndarray:
        """
        Perform do-intervention on the graph and data.

        Parameters:
            intervention (Union[int, str]): The node index or label to intervene on.
            do_value (float): The value to set for the intervention.
            multiply (bool): If True, applies the intervention as a multiplier; otherwise, adds it.

        Returns:
            np.ndarray: The effect of the intervention on the data.
        """
        est_params, adj_matrix, weights = self.estimate_effects()
        intervention_idx = [self.data.variables.index(i) for i in intervention] if len(intervention) > 0 and isinstance(intervention[0], str) else intervention
        effects = simulate_do_effects(adj_matrix, intervention_idx, est_params, self.domains, self.data_norm, do_value, multiply, self.sds)
        
        # infer T per DAG
        K = len(adj_matrix)
        T_list = [est_params[k][0]['beta'].shape[0] for k in range(K)]  # T_k

        if K == 1 and isinstance(self.graphs, DAG):
            T = T_list[0]
            weights_draws = np.ones(T, dtype=float) / T  # parameter-only uncertainty
        else:
            # mixed: repeat each p(G_k) equally across its T_k parameter draws
            weights_draws = np.concatenate([
                np.full(T_list[k], weights[k] / T_list[k], dtype=float)
                for k in range(K)
            ])
        
        if plot:
            if edges is None:
                edges = [(node1, node2) for node1 in intervention for node2 in self.data.columns if node1 != node2]
            self.plot(effects, weights_draws, edges)
        return effects, weights_draws
     
    def estimate_effects(self, n_iter=2000, burn_in=500):
        """
        Estimate the effects of interventions using Gibbs sampling.
        Parameters:
            n_iter (int): Number of iterations for Gibbs sampling.
            burn_in (int): Number of burn-in iterations to discard.
        Returns:
            dict: Estimated parameters for each node.
        """        
        if isinstance(self.graphs, DAG):
            adj_matrix = [self.graphs.incidence]
            weights = np.array([1.0], dtype=float) 
        elif isinstance(self.graphs, MCMCDistribution):
            adj_matrix = [DAG.from_key(key=g, nodes=list(self.data.columns)).incidence for g in self.graphs.particles]
            #adj_matrix = [DAG.from_key(key=g, nodes=list(self.data.columns)).incidence for g in sorted(self.graphs.particles.items(),key=lambda kv: kv[1]['p'],reverse=True)[:max_dags]]
            weights = np.asarray(self.graphs.prop('p'), dtype=float) #np.expand_dims(self.graphs.prop('p'), (1,2))
            #weights = np.expand_dims(sorted(self.graphs.prop('p'),reverse=True)[:max_dags], (1,2))
        else:
            adj_matrix = [g.incidence for g in self.graphs]
            weights = 1.
        data_values = self.data_norm
        return estimate_hybrid_dag(adj_matrix, data_values, self.domains, n_iter, burn_in), adj_matrix, weights
