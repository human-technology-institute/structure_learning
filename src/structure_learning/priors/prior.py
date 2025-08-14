from abc import abstractmethod
from typing import Union
import numpy as np
from structure_learning.distributions import Distribution
from structure_learning.data_structures import DAG

class Prior:
    
    def __init__(self, blacklist: np.ndarray = None, whitelist: np.ndarray = None):
        self.blacklist = blacklist
        self.whitelist = whitelist

    def is_valid_dag(self, dag: DAG):
        valid = True
        if self.blacklist is not None:
            valid = not np.logical_and(dag, self.blacklist).any()

        if self.whitelist is not None:
            valid = dag[self.whitelist].all()

        return valid
        
    @abstractmethod
    def compute(self, dag: DAG):
        pass

class UniformPrior(Prior):

    def __init__(self, blacklist: np.ndarray = None, whitelist: np.ndarray = None, value=0):
        super().__init__(blacklist=blacklist, whitelist=whitelist)
        self.value = value

    def compute(self, dag: DAG):
        if not self.is_valid_dag(dag):
            return -np.inf
        
        return self.value
    
class CategoricalPrior(Prior):

    def __init__(self, dist: Union[dict,Distribution], blacklist: np.ndarray = None, whitelist: np.ndarray = None, default_value=0):
        super().__init__(blacklist=blacklist, whitelist=whitelist)
        self.dist = dist
        self.default_value = default_value

    def compute(self, dag: DAG):
        if not self.is_valid_dag(dag):
            return -np.inf
        
        dag_key = dag.to_key()
        if dag_key in self.dist:
            return self.dist.particles[dag_key]['logp'] if isinstance(self.dist, Distribution) else self.dist[dag_key]

        return self.default_value
        
class CategoricalEdgePrior(CategoricalPrior):

    def __init__(self, dist: Union[dict,Distribution], blacklist: np.ndarray = None, whitelist: np.ndarray = None, default_value=0):
        super().__init__(dist=dist, blacklist=blacklist, whitelist=whitelist, default_value=default_value)
        self.edges = list(dist.particles.keys() if isinstance(self.dist, Distribution) else dist.keys())
    
    def compute(self, dag: DAG):
        logp = -np.inf
        if not self.is_valid_dag(dag):
            return logp
        
        for edge in self.edges:
            if not isinstance(edge, tuple):
                raise Exception('Edges must be passed as tuples')
            
            if dag.has_edge(*edge):
                logp = max(logp, (self.dist.particles[edge]['logp'] if isinstance(self.dist, Distribution) else self.dist[edge]))

        if np.isinf(logp): # assign default value
            logp = self.default_value

        return logp
        
        