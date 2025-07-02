import numpy as np
from structure_learning.data_structures import DAG

class Prior:
    pass

class UniformPrior(Prior):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def compute(self, dag: DAG, normalise=True):
        return np.log(1/len(self)) if normalise else 0
    
    def __len__(self):
        return self.n
    
