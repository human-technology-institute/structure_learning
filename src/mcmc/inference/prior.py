from abc import ABC, abstractmethod

import numpy as np


# TODO: logging
# TODO: Convergence of sampler - e.g. R-hat
class DiscretePrior(ABC):

    def __init__(self, blacklist: np.ndarray, whitelist: np.ndarray, **kwargs):
        self.blacklist = blacklist.astype(bool)
        self.whitelist = whitelist.astype(bool)

    def log_likelihood(self, theta):
        if np.any(theta[self.blacklist] != 0):
            return -np.inf

        if not np.all(theta[self.whitelist] != 0):
            return -np.inf

        return self._log_likelihood(theta)

    @abstractmethod
    def _log_likelihood(self, theta):
        raise NotImplemented


class UniformPrior(DiscretePrior):

    def _log_likelihood(self, theta):
        return 0.
