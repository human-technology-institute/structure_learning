import pandas as pd
from structure_learning.scores import Score
from structure_learning.data_structures import DAG

class Distribution:
    
    def __init__(self, particles, logp, theta=None):
        self.particles = particles
        self.logp = logp
        self.theta = theta

    def visualise(self):
        pass

    @classmethod
    def compute_distribution(cls, data: pd.DataFrame, score: Score, sort:bool = False):
        dags = DAG.generate_all_dags(len(data.columns), list(data.columns))

        logp = []
        scorer = Score(data=data, graph=None)
        for dag in dags:
            scorer.graph = dag
            logp.append(scorer.compute())

        return Distribution(particles=dags, logp=logp)
