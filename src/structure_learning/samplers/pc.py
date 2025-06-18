from typing import Union
import pgmpy
import graphical_models as gm
import networkx as nx
import pandas as pd
from .sampler import Sampler
from structure_learning.data import Data
from structure_learning.data_structures.dag import DAG
from structure_learning.data_structures.cpdag import CPDAG

class PC(Sampler):
    
    def __init__(self, data: Union[Data,pd.DataFrame], significance_level=0.01, ci_test='pearsonr'):
        super().__init__(data)
        self.significance_level = significance_level
        self.ci_test = ci_test
        self.results = None

    def run(self):
        if self.results is None:
            self._estimator = pgmpy.estimators.PC(data=self.data.values)
            
            cpdag = self._estimator.estimate(ci_test=self.ci_test, significance_level=self.significance_level)

            dag = nx.to_numpy_array(cpdag.to_dag()).astype(bool)

            self.dag = DAG(incidence=dag, nodes=list(self.data.columns))
            self.cpdag = CPDAG(incidence=nx.to_numpy_array(cpdag).astype(bool), nodes=list(self.data.columns))

            self.results = {'DAG': self.dag, 'CPDAG': self.cpdag}
        return self.dag, self.cpdag
    
    def config(self):
        return {
            'significance_level': self.significance_level,
            'ci_test': self.ci_test
        }
        