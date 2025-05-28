from abc import abstractmethod
from typing import List, Iterable, Dict, Hashable, TypeVar, Type
from copy import deepcopy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from structure_learning.scores import Score
from structure_learning.data_structures import DAG

D = TypeVar('Distribution')
class Distribution:
    """
    Base class for distributions.
    """
    def __init__(self, particles: Iterable[Hashable] = [], logp: Iterable = [], theta: Dict=None):
        """
        Initialise distribution. Particles are stored internally as a dictionary to store their information. 

        Parameters:
            particles (Iterable):       List of particles to add in the distribution
            logp (Iterable):            Scores (log probabilities) of the particles
            theta (Dict):           Additional particles data
        """
        self.particles = {}

        for p,lp in zip(particles, logp):
            self.particles[p] = {'logp': lp}

        if theta is not None:
            for p,th_dict in zip(particles, theta):
                self.particles[p].update(th_dict)

    @property
    def logp(self):
        return self.prop('logp')
    
    def prop(self, name):
        """
        Return data about particles.

        Parameter:
            name:           Key name for the particle data
    
        Returns:
            (list)          particle data
        """
        return [v[name] for v in self.particles.values()]
    
    def __contains__(self, particle):
        """
        Checks if particle is in the distribution.
        """
        return particle in self.particles
    
    def update(self, particle, data):
        """
        Adds particle to distribution. If particle already exists, update its data.

        Parameter:
            particle (Hashable):    Particle to add
            data (dict):            Particle data
        """
        if particle in self:
            for k,v in data.items():
                self.particles[particle][k].append(v)
            self.particles[particle]['freq'] += 1
        else:
            self.particles[particle] = {}
            for k,v in data.items():
                self.particles[particle][k] = [v] if not isinstance(v, list) else v
            self.particles[particle]['freq'] = 1

    def hist(self, normalise=False):
        """
        Returns the histogram of particles.

        Parameters:
            normalise (bool):       If True, return normalised counts.
        """
        k, v = list(self.particles.keys()), self.prop('freq')
        return k, np.array(v)/sum(v)

    def plot(self, normalise=False):
        particles, count = self.hist(normalise=normalise)
        bars = plt.bar(particles, count)
        return bars, particles, count

    # arithmetic
    def __copy__(self):
        dclone = Distribution()
        dclone.particles = deepcopy(self.particles)
        return dclone

    def __add__(self, other: Type['D']) -> Type['D']:
        dsum = self.__copy__()
        for particle, data in other.particles.items():
            dsum.update(particle, data)
            dsum.particles[particle]['freq'] = data['freq'] + (0 if particle not in self else self.particles[particle]['freq'])
        return dsum

    def __sub__(self, other: Type['D']) -> Type['D']:
        dsub = Distribution()
        for particle, data in self.particles.items():
            if particle not in other:
                dsub.update(particle, data)
        return dsub
        
    # @classmethod
    # def compute_distribution(cls, data: pd.DataFrame, score: Score, sort:bool = False):
    #     dags = DAG.generate_all_dags(len(data.columns), list(data.columns))

    #     logp = []
    #     scorer = Score(data=data, graph=None)
    #     for dag in dags:
    #         scorer.graph = dag
    #         logp.append(scorer.compute())

    #     return Distribution(particles=dags, logp=logp)
    
class MCMCDistribution(Distribution):

    def __init__(self, particles = [], logp = [], theta=None, keep_rejected=True):
        """
        Initialise MCMC distribution.

        Parameters:
            particles (Iterable):       List of particles to add in the distribution
            logp (Iterable):            Scores (log probabilities) of the particles
            theta (Dict):               Additional particles data
            keep_rejected (bool):       Store rejected particles
        """
        self.rejected = None
        if keep_rejected:
            self.rejected = Distribution()
        super().__init__(particles, logp, theta)

    def update(self, particle, iteration, data):
        """
        Adds particle to distribution. If particle already exists, update its data.

        Parameter:
            particle (Hashable):    Particle to add
            iteration (int):        Iteration number at which the particle was generated.
            data (dict):            Particle data
        """
        data.update({'iteration': iteration})
        super().update(particle, data)
            
        if self.rejected is not None and (not data['accepted'] and data['operation'] != 'initial'):
            particle = data['proposed_state'].to_key()
            self.rejected.update(particle, {'logp': data['score_proposed'], 'iterations': [iteration], 'operation': data['operation'],
                                        'timestamp': data['timestamp']})

