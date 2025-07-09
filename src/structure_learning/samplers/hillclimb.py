from typing import Union
from collections import deque
import pgmpy
import networkx as nx
import numpy as np
import pandas as pd
from structure_learning.data import Data
from structure_learning.scores import Score, BGeScore
from structure_learning.data_structures.dag import DAG
from structure_learning.proposals import GraphProposal
from .sampler import Sampler

class HillClimb(Sampler):
    # adapted from https://pgmpy.org/_modules/pgmpy/estimators/HillClimbSearch.html#HillClimbSearch.estimate
    def __init__(self, data: Union[Data, pd.DataFrame], initial_state: DAG=None, score: Score=BGeScore, max_iter=100000, epsilon=1e-4, 
                 tabu_length=100, keep_particles=False, blacklist=None, whitelist=None, seed=None, probabilistic=False):
        super().__init__(data)
        self.initial_state = initial_state if initial_state is not None else DAG.generate_random(nodes=data.columns, seed=seed)
        self.scorer = score(data=data)
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tabu = deque([], maxlen=tabu_length)
        self.current_state = self.initial_state
        self.current_score = self.scorer.compute(self.initial_state)['score']
        if seed is None:
            seed = np.random.randint(1000)
        self.proposal = GraphProposal(initial_state=self.initial_state, blacklist=blacklist, whitelist=whitelist, seed=seed)
        self.probabilistic = probabilistic
        self.particles = None
        if keep_particles:
            self.particles = [initial_state]
        self.config_dict = dict(initial_state=self.initial_state, score=type(self.scorer).__name__, max_iter=max_iter, epsilon=epsilon, 
                 tabu_length=tabu_length, keep_particles=keep_particles, seed=seed, probabilistic=probabilistic)

    def run(self, increment=None):
        ctr = 0
        self.iterations = 0

        while True:
            if ctr >= increment:
                break
            
            if self.iterations > self.max_iter:
                break
            # get neighbors
            _, del_indx_mat, add_indx_mat, rev_indx_mat, _, _, _ = self.proposal._compute_nbhood(self.current_state.incidence)
            neighbors = []

            for mat, op in zip([del_indx_mat, add_indx_mat, rev_indx_mat], ['-', '+', '|']):
                idx = np.argwhere(mat == 1)
                for r,c in idx:
                    new_DAG = self.current_state.copy()
                    if op == '-':
                        new_DAG.incidence[r,c] = False
                    elif op == '+':
                        new_DAG.incidence[r,c] = True
                    else:
                        new_DAG.incidence[r,c] = False
                        new_DAG.incidence[c,r] = True
                    if (op, (r,c)) not in self.tabu:
                        score = self.scorer.compute(new_DAG)['score']
                        neighbors.append((score, new_DAG, (op, (r,c))))
                        
            # get next state
            if self.probabilistic:
                scores = [n[0] for n in neighbors]
                p = np.exp(scores - max(scores))
                p /= p.sum()
                idx = np.random.choice(len(p), p=p)
                score, next_state, operation = neighbors[idx]
            else:
                score, next_state, operation = max(neighbors, key=lambda x: x[0])
                if (score - self.current_score) > self.epsilon:
                    break
            
            self.tabu.append(operation)
            self.current_state = next_state

            if self.particles is not None:
                self.particles.append(self.current_state)

            ctr += 1
            self.iterations += 1

        return self.current_state

    def config(self):
        """
        Returns the configuration of the HillClimb algorithm.
        """
        return self.config_dict

