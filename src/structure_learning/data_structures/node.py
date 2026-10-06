from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.linalg import solve_triangular
from scipy.stats import invgamma, t
from scipy.special import gammaln

import matplotlib.pyplot as plt

# Local imports
from structure_learning.data.data import Data


class AbstractNode(ABC):
    """Abstract base class for nodes in a structure learning tree.
    TODO: Clarify preprocessing step.


    ASSUMES that X amd Y are already preprocessed (e.g., scaled) if needed.
    This is done to avoid redundant preprocessing during inference
    Attributes:
        data (pd.DataFrame): The dataset containing both features and target variable.
        cols (list[str]): List of column names in the dataset.
        target_col (str): Name of the target variable column in the dataset
        prior (dict): Prior for model parameters.
        seed (int): Random seed for reproducibility. Relevant only when posterior are sampled.
    """

    def __init__(self,
                 data: Data,
                 parents: Sequence[Hashable],
                 target_col: Hashable,
                 prior: dict | None = None,
                 rng=None,
                 **kwargs):
        self._cols = parents
        self._target_col = target_col

        self._data = data[[target_col] + list(parents)].values

        self._n, self._d = self._data.shape
        self._prior = prior
        self._rng = rng

    @property
    def data(self):
        """Return the dataset as a pandas DataFrame."""
        return pd.DataFrame(self._data, columns=[self._target_col] + list(self._cols))

    @property
    def num_obs(self):
        """Number of observations in the dataset."""
        return self._n

    @property
    def d(self):
        """Dimension of the dataset (number of features + target)."""
        return self._d

    @property
    def num_parents(self):
        """Number of parent variables (features) in the dataset."""
        return self._d - 1  # Subtract 1 for the target variable

    @abstractmethod
    def fit(self, **kwargs):
        """Fit the model to the data or set up a sampling scheme"""
        raise NotImplementedError

    def predict(self, X_new: pd.DataFrame, N=1, x_scalers: list = None, inverse_scaler_hndl=None) -> np.ndarray:
        """Predict the target variable for new data points.
        Args:
            X_new (pd.DataFrame): New feature data for prediction. Should have the same number of columns as num_parents.
            N (int): Number of samples to draw from the posterior predictive distribution.
            x_scalers (list, optional): List of scalers for each feature in X_new. If provided, each feature column
                                        in X_new will be scaled using the corresponding scaler before prediction.
            inverse_scaler_hndl (callable, optional): Function to inverse transform the predictions back to the original scale.
        Returns:
            np.ndarray: Predicted values for the target variable. Shape will be (N, number of samples in X_new).
        """

        x = X_new[self._cols]

        # TODO: handle x_scalers if needed - might be needed if we use product from anothe rprediction

        predictions = self._predict(x, N=N)
        if inverse_scaler_hndl is not None:
            predictions = inverse_scaler_hndl(predictions)
        return predictions

    @abstractmethod
    def _predict(self, X_new: pd.DataFrame, N=1) -> np.ndarray:
        """Internal method to perform prediction. Should be implemented by subclasses."""
        raise NotImplementedError


class NIW_Node(AbstractNode):
    """Node implementing a Normal-Inverse-Wishart prior for Bayesian linear regression.

    The NIW distribution is defined jointly over

        [target, parents...]

    The supplied prior should therefore have dimension equal to
    1 + number of parents.

    A BGe-compatible node is obtained by supplying the local NIW
    prior induced by the global BGe prior.

    """

    def __init__(self,
                 data: Data,
                 parents: Sequence[Hashable],
                 target_col: Hashable,
                 prior: dict | None = None,
                 rng=None,
                 **kwargs
                 ):
        super().__init__(data, parents=parents, target_col=target_col, prior=prior, rng=rng, **kwargs)

        # Set default prior parameters if not provided

        if self._prior is None:
            # self._d is joint dimension: target + parents
            self._prior = {
                'mu_0': np.zeros(self.d),
                'kappa_0': 1.,
                'nu_0': self._d + 2.,
                'T_0': np.eye(self.d)
            }

        if self._rng is None:
            self._rng = np.random.default_rng()

        self._validate_prior()

        self.posterior_params = None

    def _validate_prior(self):
        """Check dimensions and validity of the NIW prior."""

        q = self.d

        mu_0 = np.asarray(self._prior["mu_0"])
        T_0 = np.asarray(self._prior["T_0"])

        if mu_0.shape != (q,):
            raise ValueError(
                f"mu_0 must have shape ({q},), "
                f"got {mu_0.shape}"
            )

        if T_0.shape != (q, q):
            raise ValueError(
                f"T_0 must have shape ({q}, {q}), "
                f"got {T_0.shape}"
            )

        if self._prior["kappa_0"] <= 0:
            raise ValueError("kappa_0 must be positive.")

        # IW_q requires nu > q - 1
        if self._prior["nu_0"] <= q - 1:
            raise ValueError(
                f"nu_0 must be > {q - 1}."
            )

        # Also verifies positive definiteness
        try:
            np.linalg.cholesky(T_0)
        except np.linalg.LinAlgError:
            raise ValueError("T_0 must be positive definite.")

    def fit(self, **kwargs):
        """
        Compute the NIW posterior.

        Prior
        -----
        Sigma ~ IW(nu_0, T_0)

        mu | Sigma ~ N(mu_0, Sigma / kappa_0)

        Posterior
        ---------
        Sigma | data ~ IW(nu_n, T_n)

        mu | Sigma, data ~ N(mu_n, Sigma / kappa_n)
        """

        Z = self.data.to_numpy()
        N = self.num_obs

        # ------------------------------------------------------
        # Sufficient statistics - #TODO: consider to move out so can be updated on the fly if we want to do online learning
        # ------------------------------------------------------

        z_bar = Z.mean(axis=0)

        Z_centered = Z - z_bar

        # Scatter matrix:
        S = Z_centered.T @ Z_centered

        # ------------------------------------------------------
        # Prior
        # ------------------------------------------------------

        mu_0 = np.asarray(self._prior["mu_0"])
        kappa_0 = self._prior["kappa_0"]
        nu_0 = self._prior["nu_0"]
        T_0 = np.asarray(self._prior["T_0"])

        # ------------------------------------------------------
        # NIW update
        # ------------------------------------------------------

        kappa_N = kappa_0 + N

        mu_N = (kappa_0 * mu_0 + N * z_bar) / kappa_N

        nu_N = nu_0 + N

        delta = z_bar - mu_0

        T_N = T_0 + S + (kappa_0 * N / kappa_N) * np.outer(delta, delta)

        # ============================================================
        # Convert NIW -> local regression posterior - to prevent repeated computation later on when we want to
        # sample from the posterior predictive distribution
        # ============================================================

        A = T_N[0, 0]

        if self.num_parents == 0:

            conditional_T = A

            beta_mean = np.empty(0)

            cholD = None

        else:

            B = T_N[0, 1:] # [target, parents] cross-covariance
            D = T_N[1:, 1:] # [parents, parents] scatter matrix

            # Inverting parent scatter matrix using Cholesky decomposition for numerical stability
            cholD = np.linalg.cholesky(D)

            # Solve for B using the Cholesky factorization
            chol_solve_B = solve_triangular(
                cholD,
                B,
                lower=True,
            )

            # conditional_T is the Schur complement of the parent block in of node i
            # Effectively, this is the residual scatter after accounting for the parents.
            conditional_T = (
                    A
                    - chol_solve_B @ chol_solve_B
            )

            beta_mean = solve_triangular(
                cholD.T,
                chol_solve_B,
                lower=False,
            )

        # Because q = target + all parents:
        #
        # a_N = (nu_N - q + k + 1) / 2
        #     = nu_N / 2

        sigma2_shape = nu_N / 2.0

        self._posterior = {
            # Keep NIW quantities
            "mu_N": mu_N,
            "kappa_N": kappa_N,
            "nu_N": nu_N,
            "T_N": T_N,

            # Local regression posterior
            "sigma2_shape": sigma2_shape,
            "sigma2_scale": conditional_T / 2.0,
            "beta_mean": beta_mean,
            "chol_parent_T": cholD,

            # Same naming/convention as your BGe code
            "muN_node": mu_N[0],
            "muN_parents": mu_N[1:],
        }

        return self

    def posterior_mode(self):
        """
            Compute the posterior mode of the regression parameters.
        """

        if self._posterior is None:
            raise RuntimeError("Call fit() first.")

        posterior = self._posterior

        beta_map = posterior["beta_mean"]

        sigma2_map = (
                posterior["sigma2_scale"]
                / (posterior["sigma2_shape"] + 1.0)
        )

        intercept_map = (
                posterior["muN_node"]
                - beta_map @ posterior["muN_parents"]
        )

        return {
            "beta": beta_map,
            "sigma2": sigma2_map,
            "intercept": intercept_map,
        }

    def sample_parameters(self, n_samples=1):

        if self._posterior is None:
            raise RuntimeError("Call fit() first.")

        posterior = self._posterior

        num_parents = self._d - 1

        # ============================================================
        # sigma²
        # ============================================================

        sigma2 = invgamma.rvs(
            a=posterior["sigma2_shape"],
            scale=posterior["sigma2_scale"],
            size=n_samples,
            random_state=self._rng,
        )

        # ============================================================
        # beta | sigma²
        # ============================================================

        if num_parents == 0:

            beta = np.empty(
                (n_samples, 0)
            )

        else:

            z = self._rng.standard_normal(
                size=(num_parents, n_samples)
            )

            beta_noise = solve_triangular(
                posterior["chol_parent_T"].T,
                z,
                lower=False,
            ).T

            beta = (
                    posterior["beta_mean"][None, :]
                    + np.sqrt(sigma2)[:, None]
                    * beta_noise
            )

        # ============================================================
        # intercept | beta, sigma²
        # ============================================================

        intercept_mean = (
                posterior["muN_node"]
                - beta @ posterior["muN_parents"]
        )

        intercept = self._rng.normal(
            loc=intercept_mean,
            scale=np.sqrt(
                sigma2 / posterior["kappa_N"]
            ),
        )

        return {
            "intercept": intercept,
            "beta": beta,
            "sigma2": sigma2,
        }

    def _predict(self,
                 X_new: pd.DataFrame,
                 N=1,
                 ) -> np.ndarray:

        """
            Predict the target variable for new data points using posterior predictive distribution.
        """

        X = X_new[self._cols].to_numpy()

        params = self.sample_parameters(
            n_samples=N
        )

        # Shape:
        # beta = (N, k)
        # X.T = (k, n_new)
        #
        # -> mean = (N, n_new)

        mean = params["intercept"][:, None] + params["beta"] @ X.T

        predictions = self._rng.normal(
            loc=mean,
            scale=np.sqrt(
                params["sigma2"]
            )[:, None],
        )

        return predictions

    from scipy.special import gammaln

    def marginal_likelihood(self):
        """
        Log marginal likelihood of this local conditional model
        under the NIW prior.

        Returns log p(y | X_parents).
        """

        if self._posterior is None:
            raise RuntimeError("Call fit() first.")

        posterior = self._posterior

        N = self.num_obs
        k = self.num_parents

        # --------------------------------------------------------
        # Posterior
        # --------------------------------------------------------

        aN = posterior["sigma2_shape"]
        conditional_TN = 2.0 * posterior["sigma2_scale"]

        if k == 0:
            logdet_DN = 0.0
        else:
            chol_DN = posterior["chol_parent_T"]

            logdet_DN = (
                    2.0 * np.sum(np.log(np.diag(chol_DN)))
            )

        # --------------------------------------------------------
        # Prior
        # --------------------------------------------------------

        T0 = np.asarray(self._prior["T_0"])

        # For the local joint NIW [target, parents]:
        a0 = self._prior["nu_0"] / 2.0

        A0 = T0[0, 0]

        if k == 0:

            conditional_T0 = A0
            logdet_D0 = 0.0

        else:

            B0 = T0[0, 1:]
            D0 = T0[1:, 1:]

            chol_D0 = np.linalg.cholesky(D0)

            z0 = solve_triangular(
                chol_D0,
                B0,
                lower=True,
            )

            conditional_T0 = A0 - z0 @ z0

            logdet_D0 = (
                    2.0 * np.sum(np.log(np.diag(chol_D0)))
            )

        # --------------------------------------------------------
        # Mean precision
        # --------------------------------------------------------

        kappa0 = self._prior["kappa_0"]
        kappaN = posterior["kappa_N"]

        # --------------------------------------------------------
        # log marginal likelihood
        # --------------------------------------------------------

        log_ml = (-N * np.log(np.pi) / 2.0  + 0.5 * np.log(kappa0 / kappaN) + gammaln(aN)
                - gammaln(a0) + a0 * np.log(conditional_T0) - aN * np.log(conditional_TN)
                + 0.5 * (logdet_D0 - logdet_DN)
                  )

        return log_ml

    def plot_parameter_posterior(
            self,
            true_beta=None,
            true_sigma2=None,
            true_intercept=None,
    ):
        """
        Plot prior and posterior marginal distributions for:

            sigma^2
            each beta
            intercept
        """

        if self._posterior is None:
            raise RuntimeError("Call fit() first.")

        posterior = self._posterior

        parents = self._cols
        node_label = self._target_col

        k = self._d - 1

        # ========================================================
        # PRIOR regression representation
        # ========================================================

        mu0 = np.asarray(self._prior["mu_0"])
        T0 = np.asarray(self._prior["T_0"])

        kappa0 = self._prior["kappa_0"]
        a0 = self._prior["nu_0"] / 2.0

        A0 = T0[0, 0]

        if k == 0:

            conditional_T0 = A0

            beta0 = np.empty(0)
            cholD0 = None
            D0_inv_diag = np.empty(0)

        else:

            B0 = T0[0, 1:]
            D0 = T0[1:, 1:]

            cholD0 = np.linalg.cholesky(D0)

            z0 = solve_triangular(
                cholD0,
                B0,
                lower=True,
            )

            conditional_T0 = A0 - z0 @ z0

            beta0 = solve_triangular(
                cholD0.T,
                z0,
                lower=False,
            )

            # diag(D0^-1)
            L0_inv = solve_triangular(
                cholD0,
                np.eye(k),
                lower=True,
            )

            D0_inv_diag = np.sum(
                L0_inv ** 2,
                axis=0,
            )

        b0 = conditional_T0 / 2.0

        # ========================================================
        # POSTERIOR
        # ========================================================

        aN = posterior["sigma2_shape"]
        bN = posterior["sigma2_scale"]

        betaN = posterior["beta_mean"]
        cholDN = posterior["chol_parent_T"]

        kappaN = posterior["kappa_N"]

        if k > 0:
            LN_inv = solve_triangular(
                cholDN,
                np.eye(k),
                lower=True,
            )

            DN_inv_diag = np.sum(
                LN_inv ** 2,
                axis=0,
            )

        # ========================================================
        # Figure
        # ========================================================

        # sigma² + k betas + intercept
        n_plots = k + 2

        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(5 * n_plots, 4),
        )

        axes = np.atleast_1d(axes)

        # ========================================================
        # sigma²
        # ========================================================

        ax = axes[0]

        xmax = max(
            invgamma.ppf(0.995, a=a0, scale=b0),
            invgamma.ppf(0.995, a=aN, scale=bN),
        )

        x = np.linspace(
            max(1e-8, xmax / 10000),
            xmax,
            1000,
        )

        ax.plot(
            x,
            invgamma.pdf(x, a=a0, scale=b0),
            label="Prior",
        )

        ax.plot(
            x,
            invgamma.pdf(x, a=aN, scale=bN),
            label="Posterior",
        )

        if true_sigma2 is not None:
            ax.axvline(
                true_sigma2,
                linestyle=":",
                label="True value",
            )

        ax.set_title(rf"{node_label}: $\sigma^2$")
        ax.set_xlabel(r"$\sigma^2$")
        ax.set_ylabel("Density")
        ax.legend()

        # ========================================================
        # beta
        # ========================================================

        for j, parent in enumerate(parents):

            ax = axes[j + 1]

            # Marginal Student-t
            prior_df = 2.0 * a0
            post_df = 2.0 * aN

            prior_scale = np.sqrt(
                (b0 / a0) * D0_inv_diag[j]
            )

            post_scale = np.sqrt(
                (bN / aN) * DN_inv_diag[j]
            )

            prior_low = t.ppf(
                0.001,
                df=prior_df,
                loc=beta0[j],
                scale=prior_scale,
            )

            prior_high = t.ppf(
                0.999,
                df=prior_df,
                loc=beta0[j],
                scale=prior_scale,
            )

            post_low = t.ppf(
                0.001,
                df=post_df,
                loc=betaN[j],
                scale=post_scale,
            )

            post_high = t.ppf(
                0.999,
                df=post_df,
                loc=betaN[j],
                scale=post_scale,
            )

            x = np.linspace(
                min(prior_low, post_low),
                max(prior_high, post_high),
                1000,
            )

            ax.plot(
                x,
                t.pdf(
                    x,
                    df=prior_df,
                    loc=beta0[j],
                    scale=prior_scale,
                ),
                label="Prior",
            )

            ax.plot(
                x,
                t.pdf(
                    x,
                    df=post_df,
                    loc=betaN[j],
                    scale=post_scale,
                ),
                label="Posterior",
            )

            if true_beta is not None:
                ax.axvline(
                    true_beta[j],
                    linestyle=":",
                    label="True value",
                )

            ax.set_title(
                rf"$\beta_{{{parent}\rightarrow {node_label}}}$"
            )
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel("Density")
            ax.legend()

        # ========================================================
        # Intercept
        # ========================================================

        ax = axes[-1]

        # Prior location
        intercept0 = (
                mu0[0]
                - beta0 @ mu0[1:]
        )

        # Posterior location
        interceptN = (
                posterior["muN_node"]
                - betaN @ posterior["muN_parents"]
        )

        if k == 0:

            prior_factor = 1.0 / kappa0
            post_factor = 1.0 / kappaN

        else:

            # mu_P' D^-1 mu_P
            z0 = solve_triangular(
                cholD0,
                mu0[1:],
                lower=True,
            )

            prior_factor = (
                    1.0 / kappa0
                    + z0 @ z0
            )

            zN = solve_triangular(
                cholDN,
                posterior["muN_parents"],
                lower=True,
            )

            post_factor = (
                    1.0 / kappaN
                    + zN @ zN
            )

        intercept_prior_scale = np.sqrt(
            (b0 / a0) * prior_factor
        )

        intercept_post_scale = np.sqrt(
            (bN / aN) * post_factor
        )

        prior_df = 2.0 * a0
        post_df = 2.0 * aN

        lo = min(
            t.ppf(
                0.001,
                prior_df,
                loc=intercept0,
                scale=intercept_prior_scale,
            ),
            t.ppf(
                0.001,
                post_df,
                loc=interceptN,
                scale=intercept_post_scale,
            ),
        )

        hi = max(
            t.ppf(
                0.999,
                prior_df,
                loc=intercept0,
                scale=intercept_prior_scale,
            ),
            t.ppf(
                0.999,
                post_df,
                loc=interceptN,
                scale=intercept_post_scale,
            ),
        )

        x = np.linspace(lo, hi, 1000)

        ax.plot(
            x,
            t.pdf(
                x,
                prior_df,
                loc=intercept0,
                scale=intercept_prior_scale,
            ),
            label="Prior",
        )

        ax.plot(
            x,
            t.pdf(
                x,
                post_df,
                loc=interceptN,
                scale=intercept_post_scale,
            ),
            label="Posterior",
        )

        if true_intercept is not None:
            ax.axvline(
                true_intercept,
                linestyle=":",
                label="True value",
            )

        ax.set_title(f"{node_label}: intercept")
        ax.set_xlabel("Intercept")
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()
        plt.show()


class NIW_BGe_GlobalPrior_Node(NIW_Node):
    """
    NIW node whose prior is induced from a global BGe prior.

    As with NIW node, the local joint distribution is over [target, parents...]

    All posterior fitting, parameter sampling, prediction,
    marginal likelihood calculation and plotting are inherited
    from NIW_Node.

    The global BGe prior is defined over all variables in `data`.
    For the target and its parents, the corresponding marginal
    NIW prior is extracted and passed to `NIW_Node`.

    Parameters
    ----------
    data : Data
        Dataset containing the target and parent variables. The column
        ordering is also used as the variable ordering of the global
        BGe prior.

    parents : Sequence[Hashable]
        Labels of the parent variables of `target_col`.

    target_col : Hashable
        Label of the target variable.

    global_prior : dict | None, optional
        Parameters of the global BGe prior. Expected keys are:

        ``mu_0`` : array-like, shape (p,)
            Prior mean vector over all p variables.

        ``a_mu`` : float
            Prior precision (effective sample size) for the mean.

        ``a_w`` : float
            Degrees-of-freedom parameter of the global Wishart /
            inverse-Wishart prior.

        ``T_0`` : array-like, shape (p, p)
            Positive-definite global prior scale matrix.

        The ordering of ``mu_0`` and ``T_0`` must match the column
        ordering of the entire (global) `data`.

        If None, the default BGe prior, based on the BiDAG package default, is used:

            mu_0 = 0
            a_mu = 1
            a_w = p + a_mu + 1

        and

            T_0 = t I_p

        where

            t = a_mu * (a_w - p - 1) / (a_mu + 1).

        With the default a_mu = 1, this gives

            a_w = p + 2
            t = 1/2
            T_0 = 0.5 I_p.

    rng : numpy.random.Generator | None, optional
        Random number generator used for posterior and predictive
        sampling. If None, a new default generator is created.
    """

    def __init__(self,
                 data: Data,
                 parents: Sequence[Hashable],
                 target_col: Hashable,
                 global_prior: dict | None = None,
                 rng=None,
                 **kwargs,
                 ):
        p = data.shape[1]

        if global_prior is None:
            a_mu = 1.0
            a_w = p + a_mu + 1.0

            T0_scale = (
                    a_mu
                    * (a_w - p - 1.0)
                    / (a_mu + 1.0)
            )

            global_prior = {
                "variables": list(data.values.columns),
                "mu_0": np.zeros(p),
                "a_mu": a_mu,
                "a_w": a_w,
                "T_0": T0_scale * np.eye(p),
            }

        prior = self._get_induced_prior(
            data=data,
            parents=parents,
            target_col=target_col,
            global_prior=global_prior,
        )

        self._global_prior = global_prior

        super().__init__(
            data=data,
            parents=parents,
            target_col=target_col,
            prior=prior,
            rng=rng,
            **kwargs,
        )

    @staticmethod
    def _get_induced_prior(
            data,
            parents,
            target_col,
            global_prior,
    ):
        """
        Construct the NIW prior for [target, parents...] induced
        by the global BGe prior.
        """

        global_variables = global_prior.get(
            "variables",
            list(data.values.columns),
        )

        p = len(global_variables)

        # Check that the prior and data describe the same global variables
        if set(global_variables) != set(data.values.columns):
            raise ValueError(
                "global_prior variables must match the variables in data."
            )

        local_variables = [target_col] + list(parents)
        global_variables = global_prior.get(
            "variables",
            list(data.values.columns),
        )
        # Indices relative to the global BGe variable ordering
        idx = [
            global_variables.index(var)
            for var in local_variables
        ]

        # Local joint dimension:
        # target + parents
        q = len(idx)

        mu_0 = np.asarray(global_prior["mu_0"])
        T_0 = np.asarray(global_prior["T_0"])

        # Check consistency
        if mu_0.shape != (p,):
            raise ValueError(...)

        if T_0.shape != (p, p):
            raise ValueError(...)


        # Marginalisation from global dimension p to local dimension q
        nu_0 = (
                global_prior["a_w"]
                - p
                + q
        )

        return {
            "mu_0": mu_0[idx].copy(),

            # NIW notation
            "kappa_0": global_prior["a_mu"],

            "nu_0": nu_0,

            "T_0": T_0[np.ix_(idx, idx)].copy(),
        }


class Gaussian_Maximum_Likilhood_Node(AbstractNode):
    """Node implementing a Gaussian maximum likelihood model for Bayesian linear regression."""

    # TODO: review after changing abstract class -> target is now col 0 of _data and data[1:] are the parents

    def fit(self, **kwargs):
        """Fit the model to the data by computing maximum likelihood estimates."""
        X_design = np.column_stack((np.ones(self._n), self._x.values))
        y = self._y.values.flatten()

        # Compute MLE estimates
        beta_mle = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y)
        residuals = y - X_design @ beta_mle
        sigma2_mle = (residuals.T @ residuals) / self._n

        self.mle_params = {
            'beta_mle': beta_mle,
            'sigma2_mle': sigma2_mle
        }

    def _predict(self, X_new: pd.DataFrame, N=1) -> np.ndarray:
        """Predict the target variable for new data points using MLE estimates."""
        n_new = X_new.shape[0]
        X_design_new = np.column_stack((np.ones(n_new), X_new[self._cols].values))

        beta_mle = self.mle_params['beta_mle']
        sigma2_mle = self.mle_params['sigma2_mle']

        predictions = np.zeros((N, n_new))

        for i in range(N):
            # Sample from normal distribution with MLE parameters
            predictions[i, :] = X_design_new @ beta_mle + np.random.normal(0, np.sqrt(sigma2_mle), n_new)

        return predictions


class Laplace_Approximate_Logistic_Node(AbstractNode):
    """Node implementing Bayesian logistic regression using Laplace approximation."""

    def __init__(self,
                 data: Data,
                 cols: list[str],
                 target_col: str,
                 prior: dict = None,
                 seed: int = None,
                 **kwargs):
        super().__init__(data, cols, target_col, prior, seed, **kwargs)

    def fit(self, **kwargs):
        """Fit the model to the data using Laplace approximation."""
        # TODO: Placeholder for fitting logic
        pass

    def _predict(self, X_new: pd.DataFrame, N=1) -> np.ndarray:
        """Predict the target variable for new data points using posterior predictive distribution."""
        # TODO: Placeholder for prediction logic
        return predictions


class Bayesian_Logistic_Node(AbstractNode):
    """Node implementing Bayesian logistic regression using MCMC sampling."""

    def __init__(self,
                 data: Data,
                 cols: list[str],
                 target_col: str,
                 prior: dict = None,
                 seed: int = None,
                 **kwargs):
        super().__init__(data, cols, target_col, prior, seed, **kwargs)

    def fit(self, **kwargs):
        """Fit the model to the data using MCMC sampling."""
        # Placeholder for fitting logic
        # TODO: set a chain - maybe with STAN
        raise NotImplementedError

    def _predict(self, X_new: pd.DataFrame, N=1) -> np.ndarray:
        """Predict the target variable for new data points using posterior predictive distribution."""

        # TODO: implement prediction logic based on MCMC samples
        return predictions
