"""
This module provides methods for causal inference and effect estimation using Bayesian approaches.

Classes:
    CausalEffects:
        A class for performing causal inference and estimating effects using directed acyclic graphs (DAGs) and observational data.
"""
from typing import Union, List
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
        self.data = data
              
        if isinstance(graphs, DAG):
            self.weights = np.array([1.0], dtype=float) 
            graphs = [graphs.incidence]
        elif isinstance(graphs, MCMCDistribution):
            self.weights = np.asarray(graphs.prop('p'), dtype=float)
            graphs = [DAG.from_key(key=g, nodes=list(self.data.columns)).incidence for g in graphs.particles]
        else:
            graphs = [g.incidence for g in graphs]
            self.weights = 1.
        self.graphs = graphs
        
        self.domains = [data.variable_types[v] for v in data.columns]

        # Standardise
        self.data_norm = self.data.standardise()
        self.mus = [self.data_norm.mus.get(v, 0.0) for v in self.data.columns]
        self.sds = [self.data_norm.sds.get(v, 0.0) for v in self.data.columns]
        self.data_norm = self.data_norm.values.values

        self.rng = np.random.default_rng(seed) if seed is not None else np.random
    
    def plot(self, effects, weights, edges):
        effects = np.asarray(effects)
        weights = np.asarray(weights).reshape(-1)
        effects_reshaped = pd.DataFrame(effects.reshape(effects.shape[0], -1),
            columns=[(node1, node2) for node1 in self.data.columns for node2 in self.data.columns])
        effects_reshaped['weights'] = weights
        effects_melt = pd.melt(effects_reshaped, value_vars=edges, value_name='param', var_name='edge', id_vars='weights')
        effects_melt['edge_str'] = effects_melt['edge'].apply(lambda x: f"{x[0]} -> {x[1]}")
        sns.kdeplot(data=effects_melt, x='param', weights='weights', hue='edge', fill=True, common_norm=False)
        plt.xlabel('Causal Effect')
        plt.ylabel('Density')
        plt.title('Pairwise Causal Effects')

    def simulate(self, intervention: List[Union[int, str]], do_value: float = 1.0, plot=False, edges=None, n_iter=2000, burn_in=500):
        """
        Perform do-intervention on the graph and data.

        Parameters:
            intervention (Union[int, str]): The node index or label to intervene on.
            do_value (float): The value to set for the intervention.
            plot (bool): Whether to plot the distribution of effects.
            edges (List[Tuple[str, str]], optional): List of edges to plot effects for. If None, plots all edges.
            n_iter (int): Number of iterations for Gibbs sampling.
            burn_in (int): Number of burn-in iterations to discard.
             
        """
        est_params = self.estimate_parameters(n_iter=n_iter, burn_in=burn_in)
        intervention_idx = [self.data.variables.index(i) for i in intervention] if len(intervention) > 0 and isinstance(intervention[0], str) else intervention
        effects = []

        for idx,m in enumerate(self.graphs):
            G = nx.DiGraph(m)
            topo = list(nx.topological_sort(G))
            n = m.shape[0]
            N = self.data_norm.shape[0]

            any_node = 0
            T = est_params[idx][any_node]['beta'].shape[0]

            baseline_means = self.data_norm.mean(axis=0)

            for t in range(T):
                effect_matrix = np.zeros((n, n))

                for i in range(n):
                    if i not in intervention_idx:
                        continue

                    data_do = self.data_norm.copy()

                    # Two-level (binary-like) variable: set to low or high level 
                    if self.domains[i] == Data.BINARY_TYPE:
                        if do_value not in (0, 1):
                            raise ValueError(
                                f"For binary variable {i}, do_value must be 0 or 1."
                            )
                        data_do[:, i] = int(do_value)

                    else:
                        data_do[:, i] = data_do[:, i] + do_value/self.sds[i]

                    # propagate through children with noise
                    for j in topo:
                        if j == i:
                            continue
                        parents = list(G.predecessors(j))

                        beta = est_params[idx][j]['beta'][t, :]
                        X = data_do[:, parents] if parents else np.zeros((N, 0))
                        mu = beta[0] + (X @ beta[1:])

                        if self.domains[j] == Data.CONTINUOUS_TYPE:
                            # add Gaussian noise with estimated sigma
                            sigma2_draws = est_params[idx][j].get('sigma2', None)
                            if sigma2_draws is None:
                                sigma = 1.0
                            else:
                                sigma = float(np.sqrt(sigma2_draws[t]))
                            data_do[:, j] = mu + self.rng.normal(scale=sigma, size=N)
                        else:
                            # sample latent z ~ N(mu,1) and threshold
                            z = self.rng.normal(loc=mu, scale=1.0, size=N)
                            data_do[:, j] = (z > 0).astype(int)

                    delta = data_do.mean(axis=0) - baseline_means 
                    effect_matrix[i, :] = delta                
                    # Rescaling effects if sds provided:
                    if self.sds is not None:
                        for j in range(n):
                            if self.domains[j] == Data.CONTINUOUS_TYPE:
                                effect_matrix[i, j] *= self.sds[j]
                effects.append(effect_matrix)
        
        effects = np.array(effects)
        
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
            if edges is None:
                edges = [(node1, node2) for node1 in intervention for node2 in self.data.columns if node1 != node2]
            self.plot(effects, weights_draws, edges)
        return effects, weights_draws
     
    def estimate_parameters(self, n_iter=2000, burn_in=500):
        """
        Estimate the effects of interventions using Gibbs sampling.
        Parameters:
            n_iter (int): Number of iterations for Gibbs sampling.
            burn_in (int): Number of burn-in iterations to discard.

        Returns:
            param_samples (list): A list of dictionaries containing parameter samples for each graph.
        """
        params = []
        for m in self.graphs:
            G = nx.DiGraph(m)
            N, n = self.data_norm.shape
            param_samples = {}
            
            for j in range(n): # Iterating over each node
                parents = list(G.predecessors(j))
                # design matrix: intercept + parent columns
                Xj = np.column_stack([np.ones(N)] + [self.data_norm[:, p] for p in parents])
                yj = self.data_norm[:, j]
                
                if self.domains[j] == Data.CONTINUOUS_TYPE:
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
        _, D = X.shape
        beta_samples = np.zeros((n_iter-burn_in, D))
        beta = np.zeros(D)
        
        for t in range(n_iter):
            # 1) Sample latent z
            mu = X @ beta
            # define truncation bounds
            a = np.where(y==1, 0 - mu, -np.inf - mu)
            b = np.where(y==1, np.inf - mu, 0 - mu)
            z = truncnorm.rvs(a, b, loc=mu, scale=1, random_state=self.rng)
            
            # 2) Sample beta | z
            V_post = np.linalg.inv(X.T @ X + np.eye(D))
            mu_post = V_post @ (X.T @ z)
            beta = self.rng.multivariate_normal(mu_post, V_post)
            
            if t >= burn_in:
                idx = t - burn_in
                beta_samples[idx] = beta
        
        return beta_samples
