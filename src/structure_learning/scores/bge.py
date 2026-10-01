"""
This module implements the BGe (Bayesian Gaussian Equivalent) score for evaluating Bayesian networks.

The BGe score is used to compute the marginal likelihood of a Bayesian network given data. It supports operations
such as computing the score for the entire graph, individual nodes, and edges. The implementation includes
parameters for regularization and scoring.

Classes:
    BGeScore: Implements the BGe score computation.
"""

from typing import Union
import pandas as pd
import numpy as np
from scipy.special import gammaln
from scipy.linalg import solve_triangular
from scipy.stats import invgamma, t

from structure_learning.scores import Score
from structure_learning.data_structures import Graph
from structure_learning.data import Data

import matplotlib.pyplot as plt

class BGeScore(Score):
    """
    BGe (Bayesian Gaussian Equivalent) Score
    """

    def __init__(self, data: Union[Data, pd.DataFrame]):
        """
        Initialise BGe instance.

        Parameters:
            data (Data | pandas.DataFrame): data
        """
        super().__init__(data)

        self._num_cols = data.shape[1]  # number of variables
        self._num_obvs = data.shape[0]  # number of observations
        self._mu0 = np.zeros(self._num_cols)

        # Prior parameters.
        self._am = 1
        self._aw = self._num_cols + self._am + 1
        T0scale = self._am * (self._aw - self._num_cols - 1) / (self._am + 1)

        self._T0 = T0scale * np.eye(self._num_cols)

        # Summary statistics for the data.
        self._S = (self._num_obvs - 1) * np.cov(data.values.T)
        self._xbar = np.mean(data.values, axis=0)

        self._awpN = self._aw + self._num_obvs
        self._amN = self._am + self._num_obvs
        self._TN = (
                self._T0 + self._S + ((self._am * self._num_obvs) / (self._amN))
                * np.outer(
            (self._mu0 - self._xbar), (self._mu0 - self._xbar)
        )
        )
        self._muN = (self._am * self._mu0 + self._num_obvs * self._xbar) / (self._amN)

        # Helpers for computing the score.
        self._constscorefact = - (self._num_obvs / 2) * np.log(np.pi) + 0.5 * np.log(
            self._am / (self._am + self._num_obvs))
        self._scoreconstvec = np.zeros(self._num_cols)
        for i in range(self._num_cols):
            awp = self._aw - self._num_cols + i + 1
            self._scoreconstvec[i] = (
                    self._constscorefact
                    - gammaln(awp / 2)
                    + gammaln((awp + self._num_obvs) / 2)
                    + (awp + i) / 2 * np.log(T0scale)
            )

        self._t = T0scale
        self._parameters = {}
        self._reg_coefficients = {}

    def compute(self, graph: Graph, compute_full_posterior: bool = False):
        """
        Compute the BGE for the data

        compute_full_posterior (bool): whether to compute the full posterior distribution

        Returns:
            (dict): score and parameters
        """
        if Graph.has_cycle(graph):
            return {'score': -np.inf}

        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node

        # Loop through each node in the graph
        for node in self.node_labels:

            node_res = self.compute_node(graph, node, compute_full_posterior)
            log_ml_node = node_res['score']

            node_parameters = node_res['parameters'] # includes"node_idx", 'parents', "posterior"

            # Save the parameters for the node
            parameters[node] = {
                'score': log_ml_node,
                'node_idx': node_parameters['node_idx'],
                'parents': node_parameters['parents'],  # graph.find_parents(node)
                'node_label': node,
                'parent_label': node_parameters['parent_label'],
                'posterior': node_parameters['posterior']
            }

            total_log_ml += log_ml_node

        # save the parameters
        self._parameters = parameters

        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score

    def compute_node_with_edges(self, node: str, parents: list,
                                node_index_map: dict,
                                compute_full_posterior: bool = False
                                ):
        """
        Compute the BGE for edge(s)

        Parameter:
            node (str): node label
            parents (list (str)): node labels of parent nodes
            node_index_map (dict): mapping of node labels to indices
            compute_full_posterior (bool): whether to compute the full posterior distribution

        Returns:
            (dict): score and parameters
             If full_posterior=True, also return the quantities required  for the full posterior distribution of the regression
    coefficients.
        """
        parameters = {}  # Dictionary to store the parameters for each node
        node_indx = node_index_map[node]
        parentnodes = [node_index_map[p] for p in parents]  # get index of parents labels
        num_parents = len(parentnodes)  # number of parents

        # Effective local posterior shape - this is the shape of the posterior distribution for the node given its parents
        awpNd2 = (self._awpN - self._num_cols + num_parents + 1) / 2

        A = self._TN[node_indx, node_indx]

        posterior = None

        if num_parents == 0:  # just a single term if no parents
            corescore = self._scoreconstvec[num_parents] - awpNd2 * np.log(A)

            logdetD = 0.

            conditional_T = A

            if compute_full_posterior:
                # Compute the posterior NIW distribution parameters - inverse gamma dist for  variance, and normal for beta and intcept
                posterior = {
                    'sigma2_shape': awpNd2,
                    'sigma2_scale': conditional_T / 2.0,
                    # beta | sigma^2 parameters
                    "muN_node": self._muN.iloc[node_indx],
                    "muN_parents":np.zeros(0),
                    'beta_mean': np.zeros(0),
                    'chol_parent': None,
                    'kappa_N': self._amN
                }
        else:
            B = self._TN[np.ix_([node_indx], parentnodes)]
            D = self._TN[np.ix_(parentnodes, parentnodes)]

            # Inverting parent scatter matrix using Cholesky decomposition for numerical stability
            cholD = np.linalg.cholesky(D)
            logdetD = 2. * np.sum(np.log(np.diag(cholD)))

            # T_N[P, node_indx]

            # Computing
            chol_solve_B = solve_triangular(cholD, B.T, lower=True)

            # conditional_T is the Schur complement of the parent block in of node i
            # Effectively, this is the residual scatter after accounting for the parents.
            conditional_T = A - np.sum(chol_solve_B ** 2)

            corescore = (
                    self._scoreconstvec[num_parents]
                    - awpNd2 * np.log(conditional_T)
                    - 0.5 * logdetD
            )

            if compute_full_posterior:
                # Solve for beta_mena using Cholesky decomposition
                beta_mean = solve_triangular(cholD.T, chol_solve_B, lower=False).flatten()

                posterior = {
                    "sigma2_shape": awpNd2,
                    "sigma2_scale": conditional_T / 2.0,
                    "muN_node": self._muN.iloc[node_indx],
                    "muN_parents": self._muN.iloc[parentnodes],
                    "beta_mean": beta_mean,
                    "chol_parent_T": cholD,
                    'kappa_N': self._amN
                }

        # Save the parameters for the node
        parameters = {
            'node_label': node,
            'parent_label': parents,
            "node_idx": node_indx,
            'parents': parentnodes,
            "posterior": posterior
        }

        score = {
            'score': corescore,
            'parameters': parameters
        }

        return score

    def posterior_mode(self, node: str, graph: Graph = None,
                       node_parameters: dict = None):
        """
        Compute the posterior mode of the regression coefficients for a given node and its parents.
        Use pre-computed node_parameters if available, otherwise compute them.
        """
        param = node_parameters
        if param is None:
            param = self.compute_node(graph, node, compute_full_posterior=True)['parameters']

        beta_map = param['posterior']['beta_mean']
        sigma2_map = (param['posterior']["sigma2_scale"] / (param['posterior']["sigma2_shape"] + 1.0)
                      # mode of IG distribution
                      )
        intercept_map = param['posterior']["muN_node"] - beta_map @ param['posterior']['muN_parents']

        return {"beta": beta_map, "sigma2": sigma2_map, "intercept": intercept_map}

    def node_marginal_likelihood(self, posterior):
        """
        Reconstruct the local (node level) BGe log marginal likelihood from the
        node posterior parameters.

        """

        aN = posterior["sigma2_shape"]
        bN = posterior["sigma2_scale"]


        k = len(posterior["beta_mean"])
        N = self._num_obvs

        # Setting priors
        a0 = aN - N / 2.0

        # Recover T_{N,i|P}
        conditional_T = 2.0 * bN

        # log |T_{N,PP}|
        if k == 0:
            logdet_parent_T = 0.0
        else:
            cholD = posterior["chol_parent_T"]

            logdet_parent_T = (
                    2.0
                    * np.sum(np.log(np.diag(cholD)))
            )

        kappa0 = self._am
        kappaN = posterior["kappa_N"]

        # Assumes an isotropic BGe prior: T0 = self._t * I

        log_ml = (-N / 2.0 * np.log(np.pi) + 0.5 * np.log(kappa0 / kappaN) + gammaln(aN)
                - gammaln(a0) + (a0 + k / 2.0) * np.log(self._t) - aN * np.log(conditional_T)
                  - 0.5 * logdet_parent_T
        )

        return log_ml

    @staticmethod
    def sample_node_parameters(node_parameters, n_samples=1, rng=None):
        """
            Sample (m, beta, sigma2) from the local BGe posterior.
        """

        if rng is None:
            rng = np.random.default_rng()

        posterior = node_parameters["posterior"]
        num_parents = len(node_parameters['parents'])

        # Sampling sigma^2 - the residual variance
        sigma2 = invgamma.rvs(
            a=posterior["sigma2_shape"],
            scale=posterior["sigma2_scale"],
            size=n_samples,
            random_state=rng
        )

        # Sampling beta | sigma2
        beta_mean = posterior["beta_mean"]
        if num_parents == 0:

            beta = np.empty(0)

        else:

            z = rng.standard_normal(size=(num_parents, n_samples))

            # If D = L L^T:
            #
            # L^{-T} z ~ N(0, D^{-1})
            beta_noise = solve_triangular(
                posterior["chol_parent_T"].T,
                z,
                lower=False,
            ).T

            beta = (
                    beta_mean[None, :]
                    + np.sqrt(sigma2)[:, None] * beta_noise
            )

        # Intercept | bea, sigma^2
        m_mean = (posterior["muN_node"] - beta @ posterior["muN_parents"])

        m = rng.normal(loc=m_mean,
                       scale=np.sqrt(sigma2 / posterior["kappa_N"]),
                       )

        return {
            "intercept": m,
            "beta": beta,
            "sigma2": sigma2,
        }

    @staticmethod
    def sample_y(node_parameters, x_parents, rng=None):

        if rng is None:
            rng = np.random.default_rng()

        node_params = BGeScore.sample_node_parameters(node_parameters, rng=rng)

        mean = (
                node_params["intercept"]
                + node_params["beta"] @ np.atleast_2d(x_parents)
        )

        return rng.normal(
            loc=mean,
            scale=np.sqrt(node_params["sigma2"]),
        )

    @property
    def am(self):
        return self._am

    @am.setter
    def am(self, n):
        self._am = n

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        self._parameters = params

    @property
    def reg_coefficients(self):
        return self._reg_coefficients

    @reg_coefficients.setter
    def reg_coefficients(self, coefficients):
        self._reg_coefficients = coefficients

    def plot_parameter_posterior(
            self,
            inferred_params,
            true_beta=None,
            true_sigma2=None,
            true_intercept=None,
    ):
        """
        Plot marginal posterior distributions for a node.

        If the node has parents:
            plots the marginal Student-t posterior for each beta.

        If the node has no parents:
            plots the marginal Student-t posterior for the intercept.
        """

        posterior = inferred_params['posterior']
        parents = inferred_params['parent_label']
        node_label = inferred_params['node_label']
        node_idx = inferred_params['node_idx']
        parent_idx = inferred_params['parents']



        # ============================================================
        # 1. Inverse-Gamma coefficients
        # ============================================================

        kappa0 = self._am
        k = len(parents)

        a0 = (self._aw - self._num_cols + k + 1) / 2.0
        b0 = self._t / 2.0

        aN = posterior["sigma2_shape"]
        bN = posterior["sigma2_scale"]
        kappaN = self._am + self._num_obvs

        df = 2.0 * aN


        # sigma2 + intercept + k betas
        n_plots = k + 2

        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(5 * n_plots, 4)
        )

        if n_plots == 1:
            axes = [axes]

        ax = axes[0]

        # IG

        # Use joint range covering prior and posterior
        xmax = max(
            invgamma.ppf(0.995, a=a0, scale=b0),
            invgamma.ppf(0.995, a=aN, scale=bN),
        )

        x = np.linspace(
            max(1e-8, xmax / 10000),
            xmax,
            1000
        )

        ax.plot(
            x,
            invgamma.pdf(x, a=a0, scale=b0),
            label="Prior"
        )

        ax.plot(
            x,
            invgamma.pdf(x, a=aN, scale=bN),
            label="Posterior"
        )

        if true_sigma2 is not None:
            ax.axvline(
                true_sigma2,
                linestyle=":",
                label="True value"
            )

        ax.set_title(
            rf"{node_label}: $\sigma^2$"
        )
        ax.set_xlabel(r"$\sigma^2$")
        ax.set_ylabel("Density")
        ax.legend()

        # ============================================================
        # 2. Beta coefficients
        # ============================================================

        if k > 0:

            beta_mean = posterior["beta_mean"]
            cholD = posterior["chol_parent_T"]

            L_inv = solve_triangular(
                cholD,
                np.eye(k),
                lower=True
            )

            D_inv_diag = np.sum(
                L_inv ** 2,
                axis=0
            )


            # Beta prior

            beta_prior_scale = np.sqrt(
                (b0 / a0) / self._t
            )

            beta_prior_df = 2.0 * a0

            # beta posterior
            beta_post_scale = np.sqrt(
                (bN / aN) * D_inv_diag
            )

            beta_post_df = 2.0 * aN

            for j, parent in enumerate(parents):

                ax = axes[j + 1]

                # Joint plotting range
                prior_low = t.ppf(
                    0.001,
                    beta_prior_df,
                    loc=0,
                    scale=beta_prior_scale
                )

                prior_high = t.ppf(
                    0.999,
                    beta_prior_df,
                    loc=0,
                    scale=beta_prior_scale
                )

                post_low = t.ppf(
                    0.001,
                    beta_post_df,
                    loc=beta_mean[j],
                    scale=beta_post_scale[j]
                )

                post_high = t.ppf(
                    0.999,
                    beta_post_df,
                    loc=beta_mean[j],
                    scale=beta_post_scale[j]
                )

                xmin = min(prior_low, post_low)
                xmax = max(prior_high, post_high)

                x = np.linspace(
                    xmin,
                    xmax,
                    1000
                )

                # Prior
                ax.plot(
                    x,
                    t.pdf(
                        x,
                        beta_prior_df,
                        loc=0,
                        scale=beta_prior_scale
                    ),
                    label="Prior"
                )

                # Posterior
                ax.plot(
                    x,
                    t.pdf(
                        x,
                        beta_post_df,
                        loc=beta_mean[j],
                        scale=beta_post_scale[j]
                    ),
                    label="Posterior"
                )

                if true_beta is not None:
                    ax.axvline(
                        true_beta[j],
                        linestyle=":",
                        label="True value"
                    )

                ax.set_title(
                    rf"$\beta_{{{parent}\rightarrow {node_label}}}$"
                )

                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel("Density")
                ax.legend()

        # ============================================================
        # 3. Intercept
        # ============================================================

        ax = axes[-1]

        # ------------------------------------------------------------
        # Prior intercept
        # ------------------------------------------------------------

        # mu0 = 0 in your model
        intercept_prior_mean = self._mu0[node_idx]

        intercept_prior_scale = np.sqrt(
            b0 / (a0 * kappa0)
        )

        intercept_prior_df = 2.0 * a0

        # ------------------------------------------------------------
        # Posterior intercept
        # ------------------------------------------------------------

        if k == 0:

            # Root node
            intercept_post_mean = self._muN[node_idx]

            intercept_post_scale = np.sqrt(
                bN / (aN * kappaN)
            )

        else:

            beta_mean = posterior["beta_mean"]

            muP = self._muN[parent_idx]

            # E[m | data]
            intercept_post_mean = (
                    self._muN[node_idx]
                    - beta_mean @ muP
            )

            # --------------------------------------------------------
            # Marginal variance contribution from beta uncertainty.
            #
            # m | beta,sigma2 ~
            # N(mu_i - beta' mu_P, sigma2/kappaN)
            #
            # beta | sigma2 ~
            # N(beta_N, sigma2 D^-1)
            #
            # Therefore:
            #
            # m | sigma2 ~ N(
            #     mu_i - beta_N' mu_P,
            #     sigma2 * (
            #         1/kappaN
            #         + mu_P' D^-1 mu_P
            #     )
            # )
            # --------------------------------------------------------

            z = solve_triangular(
                cholD,
                muP,
                lower=True
            )

            mu_Dinv_mu = z @ z

            intercept_post_scale = np.sqrt(
                (bN / aN)
                * (
                        1.0 / kappaN
                        + mu_Dinv_mu
                )
            )

        intercept_post_df = 2.0 * aN

        # ------------------------------------------------------------
        # Plot range
        # ------------------------------------------------------------

        prior_low = t.ppf(
            0.001,
            intercept_prior_df,
            loc=intercept_prior_mean,
            scale=intercept_prior_scale
        )

        prior_high = t.ppf(
            0.999,
            intercept_prior_df,
            loc=intercept_prior_mean,
            scale=intercept_prior_scale
        )

        post_low = t.ppf(
            0.001,
            intercept_post_df,
            loc=intercept_post_mean,
            scale=intercept_post_scale
        )

        post_high = t.ppf(
            0.999,
            intercept_post_df,
            loc=intercept_post_mean,
            scale=intercept_post_scale
        )

        x = np.linspace(
            min(prior_low, post_low),
            max(prior_high, post_high),
            1000
        )

        ax.plot(
            x,
            t.pdf(
                x,
                intercept_prior_df,
                loc=intercept_prior_mean,
                scale=intercept_prior_scale
            ),
            label="Prior"
        )

        ax.plot(
            x,
            t.pdf(
                x,
                intercept_post_df,
                loc=intercept_post_mean,
                scale=intercept_post_scale
            ),
            label="Posterior"
        )

        if true_intercept is not None:
            ax.axvline(
                true_intercept,
                linestyle=":",
                label="True value"
            )

        ax.set_title(
            f"{node_label}: intercept"
        )

        ax.set_xlabel("Intercept")
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()
        plt.show()