import pgmpy
import graphical_models as gm
import networkx as nx
import pandas as pd
from structure_learning.data_structures.dag import DAG
from structure_learning.data_structures.cpdag import CPDAG

class PC:
    
    def __init__(self, data: pd.DataFrame, significance_level=0.01, ci_test='chi_square'):
        self.data = data
        self.significance_level = significance_level
        self.ci_test = ci_test
        self.results = None

    def run(self):
        return self.step()
    
    def step(self):
        if self.result is None:
            self._estimator = pgmpy.estimators.PC(data=self.data)
            
            cpdag = self._estimator.estimate(ci_test=self.ci_test, significance_level=self.significance_level)

            pdag = gm.PDAG.from_amat(cpdag)
            dag = pdag.to_dag().to_amat()

            self.dag = DAG(incidence=dag, nodes=list(self.data.columns))
            self.cpdag = CPDAG(incidence=cpdag, nodes=list(self.data.columns))

            self.results = {'DAG': self.dag, 'CPDAG': self.cpdag}
        return self.dag, self.cpdag
        