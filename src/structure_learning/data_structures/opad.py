
from typing import Union, List, Tuple
import numpy as np
from structure_learning.data_structures.distribution import Distribution

class OPAD(Distribution):

    def __init__(self, particles: Union[List, Tuple], logp: Union[List, np.ndarray], theta: Union[List, np.ndarray]):
        super().__init__(particles, logp, theta)

        # compute normalization
        self.p, self.logZ, self.max_log_p = self.normalise(self.logp)

    @classmethod
    def normalise(cls, logp: Union[List, np.ndarray], return_constants=False):
        logp = np.array(logp)
        max_logp = max(logp)
        diff = np.exp(logp - max_logp)
        Z = diff.sum()
        p = diff/Z
        return p if not return_constants else (p, np.log(Z), max_logp)
