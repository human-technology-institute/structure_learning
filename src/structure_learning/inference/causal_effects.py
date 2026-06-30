"""
This module provides methods for causal inference and effect estimation using Bayesian approaches.

Classes:
    CausalEffects:
        A class for performing causal inference and estimating effects using directed acyclic graphs (DAGs) and observational data.
"""

from typing import Dict, Union, Optional, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns
from structure_learning.data_structures import DAG
from structure_learning.data import Data
from structure_learning.distributions import MCMCDistribution
    
class CausalEffects:
    def __init__(self, graphs: Union[DAG, List[DAG], MCMCDistribution], data: Data, seed=None):
        """
        Initialize the CausalEffects object with a graph and data.

        Parameters:
            graph (DAG): The directed acyclic graph representing the causal structure.
            data (Data): The observational data.
        """
        if isinstance(graphs, DAG):
            self.weights = np.array([1.0], dtype=float) 
            graphs = [graphs.incidence]
        elif isinstance(graphs, MCMCDistribution):
            self.weights = np.asarray(graphs.prop('p'), dtype=float)
            graphs = [DAG.from_key(key=g, nodes=list(self.data.columns)).incidence for g in graphs.particles]
        else:
            graphs = [g.incidence for g in graphs]
            self.weights = np.array([1.0], dtype=float)
        
        self.graphs = graphs
        self.data = data
        self.domains = [data.variable_types[v] for v in data.columns]

        # Standardise internally (continuous only; binary left 0/1)
        self.data_norm = self.data.standardise()
        self.mus = [self.data_norm.mus.get(v, 0.0) for v in self.data.columns]
        self.sds = [self.data_norm.sds.get(v, 0.0) for v in self.data.columns]
        self.data_norm = self.data_norm.values.values

        self.rng = np.random.default_rng(seed) if seed is not None else np.random
    
    def plot(self, effects, weights, targets: Optional[List[Union[int, str]]] = None, kind: str = "kde", ci: Tuple[float, float] = (0.025, 0.975)):
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
    
    def simulate(self, do_map: Dict[Union[int, str], Union[int, float]] = None, intervention: str = None, do_value: float = 1.0,
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
            if intervention is None:
                raise ValueError("do_map (e.g. {'A': 1.0} or {3: -0.5}) or intervention must be provided.")
            else:
                do_map = {intervention: do_value}

        est_params = self.estimate_effects()

        # Convert do_map keys to indices (allow names or indices)
        do_map_idx: Dict[int, Union[int, float]] = {}
        for k, v in do_map.items():
            idx = self.data.variables.index(k) if isinstance(k, str) else int(k)
            do_map_idx[idx] = v

        effects = self.__simulate_do_effects_joint_diff__(self.graphs, do_map_idx, est_params, self.domains, self.data_norm, self.sds)

        # infer T per DAG
        K = len(self.graphs)
        T_list = [est_params[k][0]['beta'].shape[0] for k in range(K)]  

        if K == 1 and isinstance(self.graphs, DAG):
            T = T_list[0]
            # parameter-only uncertainty
            weights_draws = np.ones(T, dtype=float) / T  
        else:
            # mixed: repeat each p(G_k) equally across its T parameter draws
            weights_draws = np.concatenate([
                np.full(T_list[k], self.weights[k] / T_list[k], dtype=float)
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
        data_values = self.data_norm
        return self.__estimate_hybrid_dag__(self.graphs, data_values, self.domains, n_iter, burn_in)

    # --- Gibbs samplers for parameter estimation --- #
    def __gibbs_linear__(self, X, y, n_iter=2000, burn_in=500):
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
            beta = self.rng.multivariate_normal(mun, Vn)
            
            # sample sigma2 | beta, y
            resid = y - X @ beta
            an = a0 + N/2
            bn = b0 + 0.5 * np.sum(resid**2)
            sigma2 = 1 / self.rng.gamma(an, 1/bn)
            
            if t >= burn_in:
                idx = t - burn_in
                beta_samples[idx] = beta
                sigma2_samples[idx] = sigma2
        
        return beta_samples, sigma2_samples

    def __gibbs_probit__(self, X, y, n_iter=2000, burn_in=500):
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
            beta = self.rng.multivariate_normal(mu_post, V_post)
            
            if t >= burn_in:
                idx = t - burn_in
                beta_samples[idx] = beta
        
        return beta_samples

    def __estimate_hybrid_dag__(self, adj_matrix, data, domains, n_iter=10000, burn_in=5000):
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
                    beta_samps, sigma2_samps = self.__gibbs_linear__(Xj, yj, n_iter, burn_in)
                    param_samples[j] = {
                        'type': 'linear',
                        'beta': beta_samps,
                        'sigma2': sigma2_samps
                    }
                else:
                    beta_samps = self.__gibbs_probit__(Xj, yj, n_iter, burn_in)
                    param_samples[j] = {
                        'type': 'probit',
                        'beta': beta_samps
                    }
            params.append(param_samples)
        
        return params

    def __simulate_do_effects_joint_diff__(self, adj_matrix, do_map, est_params, domains, data, sds=None):
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
                        data_base[:, j] = mu + self.rng.normal(scale=sigma, size=N)
                    else:
                        z = self.rng.normal(loc=mu, scale=1.0, size=N)
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
                        data_do[:, j] = mu + self.rng.normal(scale=sigma, size=N)
                    else:
                        z = self.rng.normal(loc=mu, scale=1.0, size=N)
                        data_do[:, j] = (z > 0).astype(int)

                # ---- (4) effect vector ----
                delta = data_do.mean(axis=0) - base_means

                # rescale continuous outcomes back to original units
                for j in range(n):
                    if domains[j] == "continuous":
                        delta[j] *= sds[j]

                effects.append(delta)

        return np.asarray(effects)