from abc import abstractmethod
import numpy as np
from structure_learning.data_structures import DAG

class Prior:
    
    @abstractmethod
    def compute(self, dag: DAG):
        pass

class UniformPrior(Prior):

    def __init__(self, n=1, normalise=True):
        super().__init__()
        self.n = n
        self.normalise = normalise

    def compute(self, dag: DAG):
        return np.log(1/len(self)) if self.normalise else 0
    
    def __len__(self):
        return self.n
    
