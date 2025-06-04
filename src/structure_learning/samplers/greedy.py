import sys
import time
from typing import Union
from copy import deepcopy
from collections import defaultdict, OrderedDict
import heapq
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from structure_learning.proposals import GraphProposal
from structure_learning.scores import BGeScore, BDeuScore
from structure_learning.data_structures import DAG
from structure_learning.data import Data

class GreedySearch:

    DETERMINISTIC_STRATEGY = 'deterministic'
    PROBABILISTIC_STRATEGY = 'probabilistic'
    PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY = 'probabilistic_partial'

    def __init__(self, data: Union[Data, pd.DataFrame], incidence: Union[np.ndarray, DAG], blacklist=None, whitelist=None, 
                 seed=32, n_particles=1000000, max_evaluations=10000, score_type='bdeu', 
                 strategy=DETERMINISTIC_STRATEGY, include_reversal=False, max_unexplored=2000, retain_size=1000):
        
        self.proposal = GraphProposal(initial_state=incidence, blacklist=blacklist, whitelist=whitelist, seed=seed)
        self.n_particles = n_particles
        self.particles = OrderedDict()
        self.data = data
        self.n_nodes = len(self.data.columns)
        self.initial_state = incidence
        self.unexplored = []
        self.unexplored_keys = []
        self.unexplored_scores = []
        self.unexplored_timestamp = []
        self.strategy = strategy
        self.include_reversal = include_reversal
        self.neighbour_count = defaultdict(int)
        self.neighbour_score = defaultdict(float)
        self.state_score = {}
        self.max_evaluations = max_evaluations
        self.max_unexplored = max_unexplored
        self.retain_size = retain_size
        self.scorer = BDeuScore(data=data, incidence=incidence) if score_type=='bdeu' else BGeScore(data=data, incidence=incidence)

    def get_state_to_explore(self):
        if self.strategy == self.DETERMINISTIC_STRATEGY:
            return heapq.heappop(self.unexplored)
        elif self.strategy == self.PROBABILISTIC_STRATEGY:
            log_scores = -1*np.array(self.unexplored_scores)
            Z = np.exp(log_scores - max(log_scores))
            p = Z/sum(Z)
            next_state_idx = np.random.choice(len(self.unexplored_keys), p=p)
            next_state = self.unexplored_keys.pop(next_state_idx)
            next_state_score = self.unexplored_scores.pop(next_state_idx)
            timestamp = self.unexplored_timestamp.pop(next_state_idx)
            return next_state_score, next_state, timestamp
        elif self.strategy == self.PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY:
            log_scores = -1*np.array([self.neighbour_score[n] for n in self.unexplored])
            Z = np.exp(log_scores - max(log_scores))
            p = Z/sum(Z)
            next_state_idx = np.random.choice(len(self.unexplored), p=p)
            next_state = self.unexplored.pop(next_state_idx)
            return self.state_score[next_state], next_state
        else:
            raise Exception('Unimplemented strategy', self.strategy)

    def add_neighbours(self, neighbours):
        if self.strategy in [self.DETERMINISTIC_STRATEGY]:
            for neighbour in neighbours:
                heapq.heappush(self.unexplored, neighbour)
        elif self.strategy == self.PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY:
            self.unexplored.extend(neighbours)
            if len(self.neighbour_score) > self.max_unexplored:
                sorted_list = sorted(self.neighbour_score, key=self.neighbour_score.get, reverse=False)[:self.retain_size]
                # cleanup
                self.state_score = {state:self.state_score[state] for state in sorted_list}
                self.neighbour_count = {state:self.neighbour_count[state] for state in sorted_list}
                self.neighbour_score = {state:self.neighbour_score[state] for state in sorted_list}
                self.unexplored = sorted_list
        elif self.strategy == self.PROBABILISTIC_STRATEGY:
            self.unexplored_keys.extend([n[1] for n in neighbours])
            self.unexplored_scores.extend([n[0] for n in neighbours])
            self.unexplored_timestamp.extend([n[2] for n in neighbours])
        else:
            raise Exception('Unimplemented strategy', self.strategy)

    def run(self):
        n_particles = self.n_particles
        max_evaluations = self.max_evaluations

        self.scorer.incidence = self.initial_state
        start_time = time.time()
        score = self.scorer.compute()['score']*-1#(-1 if self.strategy == GreedySearch.DETERMINISTIC_STRATEGY else 1)
         # multiply -1 to use min-heap if strategy is not probabilistic (requires fixed sized heap)
        current_state_key = self.initial_state.to_key()
        heapq.heappush(self.unexplored, (score.item(), current_state_key, 0) if self.strategy!=GreedySearch.PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY else current_state_key)
        self.unexplored_keys.append(current_state_key)
        self.unexplored_scores.append(score)
        self.unexplored_timestamp.append(0)
        self.state_score[current_state_key] = score
        self.neighbour_score[current_state_key] = score

        n_evaluations = 1
        iterations = 0
        while True:
            
            # stopping criteria
            if self.strategy == GreedySearch.DETERMINISTIC_STRATEGY:
                print(iterations, len(self.particles) + len(self.unexplored))
                if len(self.unexplored) == 0 or n_evaluations >= max_evaluations:
                    self.particles.update({key: {'score': -score, 'timestamp': ts} for score,key,ts in self.unexplored})
                    break

            if self.strategy == GreedySearch.PROBABILISTIC_STRATEGY:
                print(iterations, len(self.particles) + len(self.unexplored_scores))
                if len(self.unexplored_scores) == 0 or n_evaluations >= max_evaluations:
                    self.particles.update({state: {'score': -self.unexplored_scores[idx], 'timestamp': self.unexplored_timestamp[idx]} for idx,state in enumerate(self.unexplored_keys)})
                    break

            if self.strategy == GreedySearch.PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY:
                print(iterations, len(self.particles))
                if len(self.neighbour_score) == 0 or len(self.unexplored) == 0:
                    break

            if len(self.particles) >= max_evaluations:
                break
            
            # get next particle to explore
            state_to_explore = self.get_state_to_explore() 
            current_score, current_state_key =  state_to_explore[:2]
            current_state = DAG.from_key(current_state_key, list(self.data.columns))

            if current_state_key in self.particles:
                continue

            current_time = time.time()
            timestamp = (current_time - start_time) if len(state_to_explore) < 3 else state_to_explore[2]
            self.particles[current_state_key] = {'score': -current_score, 'timestamp': timestamp}
            self.neighbour_count.pop(current_state_key, None)
            self.neighbour_score.pop(current_state_key, None)
            self.state_score.pop(current_state_key, None)
            
            print(len(self.particles))
            l = (len(self.particles))
            if l % 1000 == 0:
                print(l, n_evaluations)

            # get neighbours
            _, del_indx_mat, add_indx_mat, rev_indx_mat, num_deletion, num_addition, num_reversal = self.proposal._compute_nbhood(current_state)
            neighbours = []

            # evaluate all neighbors from edge deletion
            idx = np.argwhere(del_indx_mat == 1)
            for r,c in idx:
                new_state = current_state.copy()
                new_state[r, c] = 0
                # score
                self.scorer.incidence = new_state
                score = self.scorer.compute()['score']*-1 # multiply -1 to use min-heap
                new_state_key = new_state.to_key()
                in_queue = new_state_key in self.neighbour_score
                self.state_score[new_state_key] = score
                if not in_queue:
                    self.neighbour_count[new_state_key] = 0
                self.neighbour_count[new_state_key] += 1
                self.neighbour_score[new_state_key] = current_score if self.neighbour_count[new_state_key]==1 else \
                    (self.neighbour_score[new_state_key]*(self.neighbour_count[new_state_key]-1) + current_score)/(self.neighbour_count[new_state_key])
                if new_state_key not in self.particles and not in_queue:
                    neighbours.append((score.item(), new_state_key, time.time() - start_time) if self.strategy!=GreedySearch.PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY else new_state_key)

            # evaluate all neighbors from edge addition
            idx = np.argwhere(add_indx_mat == 1)
            for r,c in idx:
                new_state = current_state.copy()
                new_state[r, c] = 1
                # score
                self.scorer.incidence = new_state
                score = self.scorer.compute()['score']*-1 # multiply -1 to use min-heap
                new_state_key = new_state.to_key()
                in_queue = new_state_key in self.neighbour_score
                self.state_score[new_state_key] = score
                if not in_queue:
                    self.neighbour_count[new_state_key] = 0
                self.neighbour_count[new_state_key] += 1
                self.neighbour_score[new_state_key] = current_score if self.neighbour_count[new_state_key]==1 else \
                    (self.neighbour_score[new_state_key]*(self.neighbour_count[new_state_key]-1) + current_score)/(self.neighbour_count[new_state_key])
                if new_state_key not in self.particles and not in_queue:
                    neighbours.append((score.item(), new_state_key, time.time() - start_time) if self.strategy!=GreedySearch.PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY else new_state_key)

            # evaluate all neighbors from edge reversal
            if self.include_reversal:
                idx = np.argwhere(rev_indx_mat == 1)
                for r,c in idx:
                    new_state = current_state.copy()
                    new_state[r, c] = 0
                    new_state[c, r] = 1
                    # score
                    self.scorer.incidence = new_state
                    score = self.scorer.compute()['score']*-1 # multiply -1 to use min-heap
                    new_state_key = new_state.to_key()
                    in_queue = new_state_key in self.neighbour_score
                    self.state_score[new_state_key] = score
                    if not in_queue:
                        self.neighbour_count[new_state_key] = 0
                    self.neighbour_count[new_state_key] += 1
                    self.neighbour_score[new_state_key] = current_score if self.neighbour_count[new_state_key]==1 else \
                        (self.neighbour_score[new_state_key]*(self.neighbour_count[new_state_key]-1) + current_score)/(self.neighbour_count[new_state_key])
                    if new_state_key not in self.particles and not in_queue:
                        neighbours.append((score.item(), new_state_key, time.time() - start_time) if self.strategy!=GreedySearch.PROBABILISTIC_PARTIAL_EXPLORATION_STRATEGY else new_state_key)

            # add neighbour(s) to unexplored list
            if len(neighbours) > 0:
                self.add_neighbours(neighbours)

            n_evaluations += len(neighbours) 
            iterations += 1

        return self.particles