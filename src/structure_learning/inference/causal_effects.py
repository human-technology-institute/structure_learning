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

from typing import Dict, Union, Optional, List, Tuple
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

def estimate_hybrid_dag(adj_matrix, data, domains, n_iter=10000, burn_in=5000):
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

def simulate_do_effects(adj_matrix, intervention, est_params, domains, data, do_value=1.0, sds = None, tol=1e-8):
    """
    Perform do-intervention simulations on standardised data, injecting noise at each step.
    - Continuous variables are on z-scale (mean 0, sd 1).
    - Binary variables are 0/1.
    
    est_params[idx][j] should contain:
    - 'beta': posterior samples of regression coefficients,
            shape (T, 1 + #parents) where column 0 is the intercept
    - 'sigma2': posterior samples of variance (continuous nodes only), shape (T,)

    Intervention
    - Continuous nodes: Additive shift intervention: do(X := X + do_value).
    - Binary nodes: set do(X = 0/1).

    Continuous effects are rescaled back to original units;
    binary effects are left as probability differences.


    FIX #1: baseline is generated from the fitted model (model-generated baseline), not from the observed data mean.

    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = [adj_matrix]
        est_params = [est_params]

    if sds is None:
        raise ValueError("sds must be provided (needed to convert original-unit shift to z-scale).")
 
    effects = []

    for idx,m in enumerate(adj_matrix):
        G = nx.DiGraph(m)
        topo = list(nx.topological_sort(G))
        n = m.shape[0]
        N = data.shape[0]

        any_node = 0
        T = est_params[idx][any_node]['beta'].shape[0]

        for t in range(T):
            # Baseline: simulate from fitted model with no intervention
            data_base = np.zeros((N, n), dtype=float)
            for j in topo:
                parents = list(G.predecessors(j))
                beta = est_params[idx][j]['beta'][t, :]
                Xp = data_base[:, parents] if parents else np.zeros((N, 0))
                mu = beta[0] + (Xp @ beta[1:])

                if domains[j] == 'continuous':
                    sigma2_draws = est_params[idx][j].get('sigma2', None)
                    sigma = 1.0 if sigma2_draws is None else float(np.sqrt(sigma2_draws[t]))
                    data_base[:, j] = mu + np.random.normal(scale=sigma, size=N)
                else:
                    z = np.random.normal(loc=mu, scale=1.0, size=N)
                    data_base[:, j] = (z > 0).astype(int)

            baseline_means = data_base.mean(axis=0)

            # Effects: Intervene on specified node
            effect_matrix = np.zeros((n, n), dtype=float)

            for i in range(n):
                if i not in intervention:
                    continue

                data_do = data_base.copy()

                # Intervention on node i
                # Binary variable: set to 0 or 1 as specified by do_value
                if domains[i] == "binary":
                    if do_value not in (0, 1):
                        raise ValueError(
                            f"For binary variable {i}, do_value must be 0 or 1."
                        )
                    data_do[:, i] = int(do_value)

                # Continuous variable: Numeric intervention (shift or scale)
                else:
                    data_do[:, i] = data_do[:, i] + do_value/sds[i]


                desc = nx.descendants(G, i)  # set of nodes affected downstream
                for j in topo:
                    if j == i or j not in desc:
                        continue
                    parents = list(G.predecessors(j))

                    beta = est_params[idx][j]['beta'][t, :]
                    Xp = data_do[:, parents] if parents else np.zeros((N, 0))
                    mu = beta[0] + (Xp @ beta[1:])

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
                for j in range(n):
                    if domains[j] == 'continuous':
                        effect_matrix[i, j] *= sds[j]
            effects.append(effect_matrix)
    
    return np.array(effects)

def simulate_do_effects_joint_diff(adj_matrix, do_map, est_params, domains, data, sds=None):
    """
    Joint intervention with possibly different shifts per node.
    Returns one effect vector per (DAG, parameter draw): shape (total_draws, n)

    do_map: dict {node_index: value}
      - continuous: value = additive shift in ORIGINAL units (do(X := X + value))
      - binary: value must be 0 or 1 (hard set)
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = [adj_matrix]
        est_params = [est_params]

    if sds is None:
        raise ValueError("sds must be provided (needed to convert original-unit shift to z-scale).")

    effects = []

    for idx, m in enumerate(adj_matrix):
        G = nx.DiGraph(m)
        topo = list(nx.topological_sort(G))
        n = m.shape[0]
        N = data.shape[0]

        T = est_params[idx][0]['beta'].shape[0]

        intervened_nodes = list(do_map.keys())

        # descendants of any intervened node are the only nodes that can change
        affected = set()
        for i in intervened_nodes:
            affected.update(nx.descendants(G, i))
        affected.difference_update(set(intervened_nodes))

        for t in range(T):
            # ---- (1) baseline from fitted model ----
            data_base = np.zeros((N, n), dtype=float)
            for j in topo:
                parents = list(G.predecessors(j))
                beta = est_params[idx][j]['beta'][t, :]
                Xp = data_base[:, parents] if parents else np.zeros((N, 0))
                mu = beta[0] + (Xp @ beta[1:])

                if domains[j] == "continuous":
                    sigma2_draws = est_params[idx][j].get("sigma2", None)
                    sigma = 1.0 if sigma2_draws is None else float(np.sqrt(sigma2_draws[t]))
                    data_base[:, j] = mu + np.random.normal(scale=sigma, size=N)
                else:
                    z = np.random.normal(loc=mu, scale=1.0, size=N)
                    data_base[:, j] = (z > 0).astype(int)

            base_means = data_base.mean(axis=0)

            # ---- (2) apply joint intervention on top of baseline ----
            data_do = data_base.copy()
            for i, val in do_map.items():
                if domains[i] == "binary":
                    if val not in (0, 1):
                        raise ValueError(f"Binary node {i}: intervention value must be 0 or 1.")
                    data_do[:, i] = int(val)
                else:
                    # shift in ORIGINAL units -> shift in z-scale
                    data_do[:, i] = data_do[:, i] + float(val) / sds[i]

            # ---- (3) resimulate ONLY affected descendants ----
            for j in topo:
                if j not in affected:
                    continue
                parents = list(G.predecessors(j))
                beta = est_params[idx][j]['beta'][t, :]
                Xp = data_do[:, parents] if parents else np.zeros((N, 0))
                mu = beta[0] + (Xp @ beta[1:])

                if domains[j] == "continuous":
                    sigma2_draws = est_params[idx][j].get("sigma2", None)
                    sigma = 1.0 if sigma2_draws is None else float(np.sqrt(sigma2_draws[t]))
                    data_do[:, j] = mu + np.random.normal(scale=sigma, size=N)
                else:
                    z = np.random.normal(loc=mu, scale=1.0, size=N)
                    data_do[:, j] = (z > 0).astype(int)

            # ---- (4) effect vector ----
            delta = data_do.mean(axis=0) - base_means

            # rescale continuous outcomes back to original units
            for j in range(n):
                if domains[j] == "continuous":
                    delta[j] *= sds[j]

            effects.append(delta)

    return np.asarray(effects)
    
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
    
    def plot(self, effects, weights, targets: Optional[List[Union[int, str]]] = None,kind: str = "kde",ci: Tuple[float, float] = (0.025, 0.975) ):

        """
        Plot vector intervention effects.
             effects.shape == (draws, n_nodes)
             weights.shape == (draws,)

        targets: list of node names or indices to display.
        kind: "kde" or "forest"
        ci: credible interval for forest plot (default 95%)
        """
        effects = np.asarray(effects)
        weights = np.asarray(weights).reshape(-1)

        if effects.ndim != 2:
            raise ValueError("This plot() expects vector effects with shape (draws, n_nodes).")

        if len(weights) != effects.shape[0]:
            raise ValueError(f"weights length {len(weights)} does not match number of draws {effects.shape[0]}")

        if targets is None or len(targets) == 0:
            raise ValueError("targets must be provided (or let simulate(plot=True) auto-select them).")

        var_names = list(self.data.columns)

        # Convert targets to names
        if isinstance(targets[0], int):
            target_names = [var_names[int(t)] for t in targets]
        else:
            missing = set(targets) - set(var_names)
            if missing:
                raise ValueError(f"Unknown target names: {missing}")
            target_names = list(targets)

        # Normalize weights (safe for plotting)
        weights = weights / weights.sum()

        if kind == "kde":
            df = pd.DataFrame(effects, columns=var_names)
            df["weights"] = weights
            df_m = df.melt(id_vars="weights",value_vars=target_names,var_name="variable",value_name="effect")
        
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data=df_m,x="effect",hue="variable",weights=df_m["weights"],fill=True,common_norm=False)
        
            plt.axvline(0, color="black", lw=1, alpha=0.6)
            plt.xlabel("Effect (intervention − baseline)")
            plt.ylabel("Density")
            plt.title("Posterior distributions of intervention effects")
            plt.tight_layout()
            plt.show()
            return

        if kind == "forest":
            def weighted_quantile(x, w, qs):
                x = np.asarray(x)
                w = np.asarray(w)
                order = np.argsort(x)
                x = x[order]
                w = w[order]
                cw = np.cumsum(w)
                cw = cw / cw[-1]
                return np.interp(qs, cw, x)

            rows = []
            for name in target_names:
                j = var_names.index(name)
                x = effects[:, j]
                mean = np.sum(weights * x)
                lo, hi = weighted_quantile(x, weights, [ci[0], ci[1]])
                rows.append((name, mean, lo, hi))

            df = pd.DataFrame(rows, columns=["variable", "mean", "lo", "hi"]).sort_values("mean")

            plt.figure(figsize=(7, max(3, 0.35 * len(df))))
            y = np.arange(len(df))
            plt.hlines(y, df["lo"], df["hi"], color="tab:blue", lw=2)
            plt.plot(df["mean"], y, "o", color="tab:blue")
            plt.axvline(0, color="black", lw=1, alpha=0.6)
            plt.yticks(y, df["variable"])
            plt.xlabel("Effect (posterior mean and credible interval)")
            plt.title("Intervention effects (forest plot)")
            plt.tight_layout()
            plt.show()
            return
        else:
            raise ValueError("kind must be 'kde' or 'forest'.")

    def do(self, do_map: Dict[Union[int, str], Union[int, float]])-> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single or joint do-intervention on the graph and data.

        Parameters:
            do_map (Dict[Union[int, str], Union[int, float]]): A dictionary mapping node names (str) or indices (int) to their intervention values.

        Returns:
            np.ndarray: The effect of the intervention on the data.
        """
        return self.simulate(do_map)

    #def simulate(self, intervention: List[Union[int, str]], do_value: float = 1.0, plot=False, edges=None) -> np.ndarray:
    
    def simulate(self, do_map: Dict[Union[int, str], Union[int, float]],
             plot: bool = False, targets: Optional[List[Union[int, str]]] = None,
             kind: str = "kde")-> Tuple[np.ndarray, np.ndarray]:

        """
        Perform (single or joint) intervention defined by do_map on the graph and data.

        Parameters:
            do_map: dict {node_name or node_index: value}
                 - continuous: value = additive SHIFT in original units (do(X := X + value))
                 - binary: value must be 0 or 1 (hard set)

        Returns:
            effects: (num_draws_total, n_nodes) vector per draw
            weights_draws: (num_draws_total,)

        """

        if do_map is None or len(do_map) == 0:
            raise ValueError("do_map must be a non-empty dict, e.g. {'A': 1.0} or {3: -0.5}.")

        est_params, adj_matrix, weights = self.estimate_effects()
        #intervention_idx = [self.data.variables.index(i) for i in intervention] if len(intervention) > 0 and isinstance(intervention[0], str) else intervention

        # Convert do_map keys to indices (allow names or indices)
        do_map_idx: Dict[int, Union[int, float]] = {}
        for k, v in do_map.items():
            idx = self.data.variables.index(k) if isinstance(k, str) else int(k)
            do_map_idx[idx] = v

        #effects = simulate_do_effects(adj_matrix, do_map_idx, est_params, self.domains, self.data_norm, self.sds)
        effects = simulate_do_effects_joint_diff(adj_matrix,do_map_idx,est_params,self.domains,self.data_norm,self.sds)

        
        # infer T per DAG
        K = len(adj_matrix)
        T_list = [est_params[k][0]['beta'].shape[0] for k in range(K)]  

        if K == 1 and isinstance(self.graphs, DAG):
            T = T_list[0]
            # parameter-only uncertainty
            weights_draws = np.ones(T, dtype=float) / T  
        else:
            # mixed: repeat each p(G_k) equally across its T parameter draws
            weights_draws = np.concatenate([
                np.full(T_list[k], weights[k] / T_list[k], dtype=float)
                for k in range(K)
            ])
        
        if plot:
            if targets is None or len(targets) == 0:
                raise ValueError("Provide targets (node names or indices) when plot=True.")

            self.plot(effects, weights_draws, targets, kind)
        return effects, weights_draws
     
    def estimate_effects(self, n_iter=10000, burn_in=5000):
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
            weights = np.asarray(self.graphs.prop('p'), dtype=float)
            #weights = np.expand_dims(sorted(self.graphs.prop('p'),reverse=True)[:max_dags], (1,2))
        else:
            adj_matrix = [g.incidence for g in self.graphs]
            weights = 1.
        data_values = self.data_norm
        return estimate_hybrid_dag(adj_matrix, data_values, self.domains, n_iter, burn_in), adj_matrix, weights
