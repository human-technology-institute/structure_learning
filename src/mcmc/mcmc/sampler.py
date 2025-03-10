from abc import ABC, abstractmethod

import numpy as np


class Sampler(ABC):

    def __init__(self, model, proposal, prior, **kwargs):
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
            self.Nsamples += 1

        return self.postprocess()

    def log_posterior(self, theta):
        # The model maintains the data, therefore only a function of theta
        return self.model.log_likelihood(theta) + self.prior.log_likelihood(theta)

    def current_state(self):
        return self._chain_thetas[-1], self._chain_logpost[-1]

    def postprocess(self):
        """
        this function allows for specific pre and post processing of the run (e.g. removal of burn in_
        Returns:

        """
        pass


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
        logpost_p = self.log_posterior(theta_p)

        mhratio = min(0, logpost_p - logpost_c - logqratio)
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
#OPAD