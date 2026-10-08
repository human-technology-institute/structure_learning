from typing import Union

import numpy as np
from scipy.linalg import solve
import pandas as pd

from structure_learning.approximators import State
from structure_learning.approximators.structure_mcmc import StructureMCMC
from structure_learning.scores import Score
from structure_learning.priors import Prior
from structure_learning.scores import Score
from structure_learning.proposals import StructureLearningProposal, GraphProposal
from structure_learning.data_structures import DAG
from structure_learning.data import Data
from .pc import PC
from structure_learning.data_structures.node import NIW_BGe_GlobalPrior_Node


class GibbsMissingDataNIWSampler(StructureMCMC):
    """

    Experimental
    Currently, hijacking the StructureMCMC mechanism to handle missing data. In general, a gibbs step should have multiple samplers
    Current mechanism the core sampler is a standardMCMC class. The only change in this class is the step function that sample missing data

    """

    def __init__(self, data: pd.DataFrame = None, initial_state: np.ndarray = None, max_iter: int = 30000,
                 score_object: Union[str, Score] = None, proposal_object: StructureLearningProposal = None,
                 prior: Prior = None, pc_init=True, pc_significance_level=0.01, pc_ci_test='pearsonr',
                 blacklist: np.ndarray = None, whitelist: np.ndarray = None, seed: int = None, sparse=True,
                 result_type: str = 'distribution', graph_type='dag', burn_in: float = 0.1, verbose=True, **kwargs):

        # As we have missing data, we need to wrap the initialisation of the StructureMCMC class.
        self.original_data = data.__copy__()

        self.nan_indices = self.original_data.values.index[self.original_data.values.isna().any(axis=1)]

        # Step 1: Initialise graph and parameters
        if proposal_object is None or proposal_object == 'graph':
            if initial_state is None:
                if pc_init:
                    pc = PC(data=score_object.data, significance_level=pc_significance_level, ci_test=pc_ci_test)
                    initial_state, _ = pc.run()
                else:
                    initial_state = DAG.generate_random(self.node_labels, 0.5, seed)

                # Check compliance with blacklist and whiteliist
                if whitelist is not None:
                    self.initial_state.incidence[whitelist > 0] = True
                if blacklist is not None:
                    self.initial_state.incidence[blacklist > 0] = False
            elif isinstance(self.initial_state, np.ndarray):
                self.initial_state = DAG(nodes=self.node_labels, incidence=self.initial_state)
            proposal_object = GraphProposal(initial_state=self.initial_state, blacklist=blacklist, whitelist=whitelist,
                                            seed=seed)
        elif not isinstance(proposal_object, StructureLearningProposal):
            raise Exception('Unsupported proposal', proposal_object)

        # sample initial parameters given an initial graph from the prior
        self.param_state = self.sample_params(Data(data.values.head(0), variables=data.variables),
                                              # Using empty data to sample from the prior
                                              initial_state
                                              )

        # 2. Data handling - to use the StructureMCMC structure, Data must not have NaNs - sampling missing data given sampled params
        data_missing_resampled = self.sample_missing_data(data, self.nan_indices, self.param_state['mu'],
                                                          self.param_state['sigma'])

        #TODO: dump data in a file

        # Init of StructureMCMC instance
        score_object.data = data_missing_resampled

        #TODO: why StrctureMCMC is only pandas dataframe and not our DATA strcture.
        super().__init__(data_missing_resampled.values, initial_state, max_iter, score_object, proposal_object, prior, pc_init,
                         pc_significance_level, pc_ci_test, blacklist, whitelist, seed, sparse, result_type, graph_type,
                         burn_in, verbose, **kwargs)


    def step(self):

        smcmc_res = super().step()

        current_structure_state = smcmc_res['graph']

        self.param_state = self.sample_params(self.data,
                                              current_structure_state
                                              )


        data_missing_resampled = self.sample_missing_data(self.original_data, self.nan_indices, self.param_state['mu'],
                                                          self.param_state['sigma'])

        # TODO: Dump data into file
        self.score_object.data = data_missing_resampled

        return {**smcmc_res, **self.param_state}


    def sample_params(self,
                      data: Data,
                      graph: DAG,
                      ):
        """
        #TODO: this is not genenralizable as I assume NIW model
        Sample missing values from node-wise conditional models following
        the DAG's topological order.

        Existing observed values are never overwritten.

        Parameters
        ----------
        data : pd.DataFrame
            Data with columns corresponding to graph nodes.

        graph : nx.DiGraph
            Directed acyclic graph.

        Returns
        -------
        pd.DataFrame
            Copy of data with sampleable missing values filled.
        """
        if DAG.has_cycle(graph):
            # TODO: this might not be neccesary in the future if we can guarantee that the graph is a DAG.
            raise ValueError("graph must be a DAG")


        node_order = list(data.columns)
        p = len(data.columns)
        node_order = {node: i for i, node in enumerate(node_order)}

        sigma2 = np.zeros(p)  # Node residual noise
        intercept = np.zeros(p)
        B = np.zeros((len(node_order), len(node_order)))

        # build the covariance matrix using the BGeNode for each node in the graph
        for node, i in node_order.items():

            parents = list(graph.find_parents(node))
            columns_to_sample = parents + [node]

            # Only sampling missing data - TODO: missing data per node, should be captured in INIT - keep an empty list if no missing data so easy to continue. keep indices instead?
            missing = data[columns_to_sample].isna().any(axis=1)

            # ------------------------------------------------------------
            # Determine which missing observations can actually be sampled
            # ------------------------------------------------------------

            valid_node_data = Data(data.values[~missing], variables=data.variables)

            BGeNode = NIW_BGe_GlobalPrior_Node(
                data=valid_node_data,
                parents=parents,
                target_col=node,
                rng=rng,
            ).fit()

            params = BGeNode.sample_parameters()

            intercept[i] = np.asarray(params["intercept"]).item()
            sigma2[i] = np.asarray(params["sigma2"]).item()
            for parent, beta in zip(parents, params["beta"].flatten()):
                B[node_order[parent], i] = beta

        A = np.eye(p) - B.T

        A_inv = np.linalg.solve(A, np.eye(p))

        mu = A_inv @ np.ravel(intercept)

        D = np.diag(np.ravel(sigma2))

        sigma = A_inv @ D @ A_inv.T

        return {"mu": mu, "Ssigma": sigma}

    def sample_missing_data(self, data, nan_indices, mu, sigma, rng=None):
        """
        Jointly sample all missing values in one row at a time from

            X_missing | X_observed

        where X ~ N(mu, Sigma).

        row, mu and Sigma must use the same variable ordering.
        """

        if rng is None:
            rng = np.random.default_rng()

        sampled_data = data.values.copy()

        for row_idx in nan_indices:

            row = sampled_data.iloc[row_idx].copy()

            missing_mask = np.isnan(row)
            missing = np.flatnonzero(missing_mask)
            observed = np.flatnonzero(~missing_mask)

            # Nothing to impute
            if len(missing) == 0:
                continue

            # \[X_M \mid X_O=x_O \sim N(\mu_{M|O},\Sigma_{M|O})\]
            #
            mu_m = mu[missing]
            sigma_mm = sigma[np.ix_(missing, missing)]

            # If entire row is missing, sample from marginal
            if len(observed) == 0:
                row[missing] = rng.multivariate_normal(
                    mean=mu_m,
                    cov=sigma_mm,
                )
            else:
                mu_o = mu[observed]
                x_o = row[observed]
                Sigma_mo = sigma[np.ix_(missing, observed)]
                Sigma_oo = sigma[np.ix_(observed, observed)]

                # K = Sigma_mo @ inv(Sigma_oo)
                # Don't explicitly invert Sigma_oo
                K = solve(Sigma_oo, Sigma_mo.T, assume_a="pos").T

                conditional_mean = mu_m + K @ (x_o - mu_o)

                conditional_cov = sigma_mm - K @ Sigma_mo.T

                # Computational stability step to ensure symmetric cov
                conditional_cov = (conditional_cov + conditional_cov.T) / 2.

                row[missing] = rng.multivariate_normal(mean=conditional_mean,
                                                       cov=conditional_cov
                                                       )

            sampled_data.iloc[row_idx] = row

        return Data(sampled_data, variables=self.original_data.variables)
