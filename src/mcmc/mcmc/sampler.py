from abc import ABC, abstractmethod

import numpy as np

from mcmc.proposals.proposal import StructureLearningProposal
from mcmc.inference.prior import DiscretePrior

from mcmc.utils.graph_utils import generate_key_from_adj_matrix, generate_adj_matrix_from_key

# TODO: logging
# TODO: Convergence of sampler - e.g. R-hat
class Sampler(ABC):

    def __init__(self, model, proposal:StructureLearningProposal, prior:DiscretePrior, **kwargs):
        """

        Args:
            model: An abstract model, must have a method likelihood = f(theta, data). Model maintains the data.
            proposal:  An abstract proposal structure, return a theta_p to estimate given theta_c (current)
            prior: As abstract prior p(theta)
            **kwargs:
        """

        self.model = model
        self.proposal = proposal
        self.prior = prior
        self.Nsamples = 0

        self._chain_thetas = []
        self._chain_logpost = []

        # Model containers - to prevent multiple calculation of the same model - relevant for discrete models only
        # This assumes a UNIQUE score for a state (e.g., wouldn't work for graphs with missing data or latent nodes
        self.state_dict = {}
        self.counter_dict = {}


    @abstractmethod
    def step(self) -> dict:
        """
        Specific step function that executes the pre-process, analysis and post-process of individual samples
        Returns:

        """
        raise NotImplemented

    def run(self, theta0, Nsamples):
        """
        Run: cycles over all Nsamples starting at theta0.

        Args:
            Nsamples: Number of samples to draw
            theta0: Start sample

        Returns:
        """

        self._chain_thetas.append(theta0)
        self._chain_logpost.append(self.log_posterior(theta0))
        for i in range(Nsamples):
            res = self.step()
            self._chain_thetas.append(res['theta'])
            self._chain_logpost.append(res['log_posterior'])

            # Update
            state_key = generate_key_from_adj_matrix(res['theta'])
            self.state_dict[state_key] = res['log_posterior']

            if state_key in self.counter_dict.keys():
                self.counter_dict[state_key] += 1
            else:
                self.counter_dict[state_key] = 1

            self.Nsamples += 1

        return self.postprocess()

    def log_posterior(self, theta):
        # The model maintains the data, therefore only a function of theta
        return self.model.log_likelihood(theta) + self.prior.log_likelihood(theta)

    def current_state(self):
        return self._chain_thetas[-1], self._chain_logpost[-1]

    def postprocess(self):
        """
        this function allows for specific pre- and post-processing of the run (e.g. removal of burn in_
        Returns:

        """
        pass


#####################################################################################################################
# Brute force
# TODO

################################################################################################################
class MCMCSampler(Sampler):

    def __init__(self, model, proposal, prior, **kwargs):
        super().__init__(model, proposal, prior, **kwargs)

        self.burn_in = int(kwargs.get("burn_in", 0))  # the number of samples for burn_in
        self.Naccepts = 0
        self._chain_accept_reject = []

    def step(self):
        """
        _c = current
        _p = proposed

        """
        theta_c, logpost_c = self.current_state()
        proposed_dict = self.proposal.propose(theta_c)
        logqratio = proposed_dict["logqratio"]  # the ratio between q(theta_c -> theta_p)/q(theta_p -> theta_c)
        theta_p = proposed_dict["theta_prop"]
        if theta_p in self.state_dict.keys():  # To save compute time
            logpost_p = self.state_dict[theta_p]
        else:
            logpost_p = self.log_posterior(theta_p)

        mhratio = min(0, logpost_p - logpost_c + logqratio)
        if np.log(np.random.uniform()) < mhratio:
            theta, logpost = theta_p, logpost_p
            self.Naccepts += 1
            self._chain_accept_reject.append(True)
        else:
            theta, logpost = theta_c, logpost_c
            self._chain_accept_reject.append(False)
        self.proposal.adapt(theta)
        return {'theta': theta,
                'log_posterior': logpost,
                'theta_prop': theta_p,
                'logpost_prop': logpost_p,
                'logqratio': logqratio}

    def postprocess(self):
        return self._chain_thetas[self.burn_in:], self._chain_logpost[self.burn_in:]


#####################################################################################################################
# OPAD
class OPADSampler(Sampler):

    def __init__(self, model, proposal, prior, **kwargs):
        """

        Args:
            model:
            proposal:
            prior:
            **kwargs:  sampler: an un-instatiated object of type Sampler.
                       opad_plus_option: list of tuples or a dict, with the dict keys of additional samples states,
                       for example [(theta1, lop_p1), (theta2, lop_p2)].

        """

        super().__init__(model, proposal, prior, **kwargs)

        # OPAD uses another sampler to traverse the search space.
        # This sampler must be of class Sampler
        sampler = kwargs.get("sampler", None)
        assert isinstance(sampler, Sampler)
        self._sampler = sampler.__init__(model, proposal, prior, **kwargs)  # instatiating internal sampler

        self.opad_plus_option = int(kwargs.get("opad_plus", None))
        if isinstance(self.opad_plus_option, (list, tuple)):
            self.opad_plus_option = {item[0]: item[1] for item in self.opad_plus_option}
        elif not (self.opad_plus_option is None or isinstance(self.opad_plus_option, dict)):
            raise ValueError("opad_plus: must be either None, list of tuples or a dictionary")
        # self.opad_plus_option - converted to a dict

    def step(self):
        """
        """

        # Return sample from the internal sampler, e.g. MCMC
        res = self._sampler.step()

        # The main OPAD disctionary are handled by the super class which maintains State and counter of visited states

        # OPAD+ options - update states with additional samples states
        if self.opad_plus_option is not None:
            for key2state, key2value in self.opad_plus_option.items():

                state_key = generate_key_from_adj_matrix(res[key2state])

                self.state_dict[state_key] = res[key2value]
                if key2state in self.counter_dict.keys():
                    self.counter_dict[state_key] += 1
                else:
                    self.counter_dict[state_key] = 1

        return res

    def postprocess(self):
        return self.state_dict
