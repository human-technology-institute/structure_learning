
from typing import Union, List, Tuple
import numpy as np
from structure_learning.data_structures.distribution import Distribution

class OPAD(Distribution):
    """
    This class implements the OPAD re-weighing mechanism described in 
    """
    def __init__(self, particles: Union[List, Tuple], logp: Union[List, np.ndarray], theta: Union[List, np.ndarray]):
        super().__init__(particles, logp, theta)
        self.normalise()

    @classmethod
    def compute_normalisation(cls, logp: Union[List, np.ndarray], return_constants=False):
        """
        Compute the normalisation factor given the log scores.

        Parameters:
            logp (list | np.ndarray):   The log scores 
            return_constants (bool):    If True, also returns log(Z) and max score.

        Returns:
            (np.array):                          Normalised scores
            (float):                             Normalisation factor
            (np.array):                          Maximum score
        """
        logp = np.array(logp)
        max_logp = max(logp)
        diff = np.exp(logp - max_logp)
        Z = diff.sum()
        p = diff/Z
        return p if not return_constants else (p, np.log(Z), max_logp)
    
    def normalise(self):
        """
        Normalise the current set of particles in the distribution.
        """
        self.p, self.logZ, self.max_log_p = self.compute_normalisation(self.logp)
        for particle, _p in zip(self.particles, self.p):
            self.particles[particle]['p'] = _p
    
    def update(self, particle, iteration, data, normalise=True):
        """
        Add new particles to the distribution and renormalise.
        """
        super().update(particle, iteration, data)
        if normalise:
            self.normalise()

