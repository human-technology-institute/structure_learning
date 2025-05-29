
from typing import Union, List, Tuple
from copy import deepcopy
import numpy as np
from .distribution import Distribution

class OPAD(Distribution):
    """
    This class implements the OPAD re-weighing mechanism described in 
    """
    def __init__(self, particles: Union[List, Tuple], logp: Union[List, np.ndarray], theta: Union[List, np.ndarray]):
        super().__init__(particles, logp, theta)
        self.normalise()

    @classmethod
    def __new__(cls, obj):
        if isinstance(obj, Distribution):
            odist = OPAD()
            odist.particles = deepcopy(obj.particles)
            odist.normalise()
            return odist 
        else:
            raise Exception("Type not supported")
    
    def update(self, particle, iteration, data, normalise=True):
        """
        Add new particles to the distribution and renormalise.
        """
        super().update(particle, iteration, data)
        if normalise:
            self.normalise()

