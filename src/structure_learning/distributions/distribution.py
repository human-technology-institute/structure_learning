from abc import abstractmethod
from typing import List, Tuple, Iterable, Dict, Union, TypeVar, Type
from copy import deepcopy, copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from structure_learning.scores import Score
from structure_learning.data_structures import DAG

D = TypeVar('Distribution')
class Distribution:
    """
    Base class for distributions.
    """
    def __init__(self, particles: Iterable = [], logp: Iterable = [], theta: Dict = None):
        """
        Initialise distribution. Particles are stored internally as a dictionary to store their information. 

        Parameters:
            particles (Iterable):       List of particles to add in the distribution
            logp (Iterable):            Scores (log probabilities) of the particles
            theta (Dict):               Additional particles data
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
    
    def clear(self):
        self.particles = {}

    def normalise(self, prop='freq', log=False):
        """
        Normalise the current set of particles in the distribution.
        """
        if len(self.particles) == 0:
            return
        self.p = self.prop(prop)
        if log:
            self.p = np.exp(self.p - np.max(self.p))
        Z = np.sum(self.p)
        self.p /= Z
        keys = list(self.particles.keys())
        for particle, _p in zip(keys, self.p):
            self.particles[particle]['p'] = _p.item()
        return self
    
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
    
    def __len__(self):
        return len(self.particles)
    
    def update(self, particle, data):
        """
        Adds particle to distribution. If particle already exists, update its data.

        Parameter:
            particle (Hashable):    Particle to add
            data (dict):            Particle data
        """
        if particle in self:
            for k,v in data.items():    
                if k in ('freq', 'logp'):
                    continue
                self.particles[particle][k].append(v)
                
            self.particles[particle]['freq'] += (1 if 'freq' not in data else data['freq'])
        else:
            self.particles[particle] = {}
            for k,v in data.items():
                self.particles[particle][k] = [v] if not isinstance(v, list) and k!='logp' else v
            self.particles[particle]['freq'] = (1 if 'freq' not in data else data['freq'])

    def hist(self, prop='freq', normalise=False):
        """
        Returns the histogram of particles.

        Parameters:
            normalise (bool):       If True, return normalised counts.
        """
        k, v = list(self.particles.keys()), self.prop(prop)
        if normalise:
            v = np.array(v)/sum(v)
        return k, v

    def plot(self, prop='freq', sort=True, normalise=False, limit=-1, ax=None):
        particles, count = self.hist(prop=prop, normalise=normalise)
        if sort:
            sort_idx = np.argsort(count)
            particles, count = np.array(particles)[sort_idx], np.array(count)[sort_idx]
        limit = limit if limit > 0 else len(particles)
        if ax is None:
            fig = plt.figure()
            ax = fig.subplots(1,1)
            ax.grid(alpha=0.5)
            ax.set_axisbelow(True)
        bars = ax.bar(particles[-limit:], count[-limit:])
        ax.set_xlabel('Particles')
        ax.set_ylabel('Proportion')
        ax.set_xticklabels(particles[-limit:], rotation=90)
        return bars, particles[-limit:], count[-limit:]
    
    def top(self, prop='freq', n=1):
        k, v = self.hist(prop=prop)
        idx = np.argsort(v)
        return np.array(k)[idx][-n:]

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
        
    @classmethod
    def compute_distribution(cls, data: pd.DataFrame, score: Score):
        dags = DAG.generate_all_dags(len(data.columns), list(data.columns))

        logp = []
        scorer = score(data=data)
        for dag in dags:
            scorer.graph = dag
            logp.append(scorer.compute(dag)['score'])

        dist = Distribution(particles=[dag.to_key() for dag in dags], logp=logp).normalise(prop='logp', log=True)
        dist.normalise = lambda: dist
        return dist
    
class MCMCDistribution(Distribution):

    def __init__(self, particles: Iterable = [], logp: Iterable = [], theta: Dict = None, keep_rejected: bool = True):
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
        data.update({'iteration': iteration, 'logp': data['score_current'] if 'logp' not in data else data['logp']})
        super().update(particle, data)
            
        if self.rejected is not None and (not data['accepted'] and data['operation'] != 'initial'):
            particle = data['proposed_state'].to_key()
            self.rejected.update(particle, {'logp': data['score_proposed'], 'iteration': iteration, 
                                            'operation': data['operation'], 'timestamp': data['timestamp']})
            
    def to_opad(self, plus=False):
        opad = OPAD(plus=plus)
        opad.particles = deepcopy(self.particles)
        opad.rejected = copy(self.rejected)
        opad._add_rejected_particles_()
        opad.normalise()
        return opad
    
    @classmethod
    def from_iterates(cls, iterates: dict):
        dist = MCMCDistribution()
        for iteration, data in iterates.items():
            particle = data['graph'].to_key()
            dist.update(particle=particle, iteration=iteration, data=data)
        dist.normalise()
        return dist

class OPAD(MCMCDistribution):
    """
    This class implements the OPAD re-weighing mechanism described in 
    """
    def __init__(self, particles: Iterable = [], logp: Iterable = [], theta: Dict = [], plus = False):
        super().__init__(particles, logp, theta)
        self.plus = plus
        self.normalise()

    def normalise(self):
        self._add_rejected_particles_()
        return super().normalise(prop='logp', log=True)
    
    def _add_rejected_particles_(self):
        if self.plus: # add rejected to particles
            if len(self.rejected) > 0:
                print('Adding rejected particles')
                for particle, data in self.rejected.particles.items():
                    super(MCMCDistribution, self).update(particle, data)
                self.rejected.clear()

    def update(self, particle, iteration, data, normalise=True):
        """
        Add new particles to the distribution and renormalise.
        """
        super().update(particle, iteration, data)
        if normalise:
            self.normalise()

    @classmethod
    def from_mcmc(cls, dist: Distribution, plus=False):
        return dist.to_opad(plus=plus)
    
    def plot(self, prop='p', sort=True, normalise=False, limit=-1):
        return super().plot(prop=prop, sort=sort, normalise=normalise, limit=limit)
    
    @classmethod
    def compute_normalisation(cls, logp: Union[List, np.ndarray], return_constants=True):
        """
        Compute the normalisation factor given the log scores.

        Parameters:
            logp (list | np.ndarray):   The log scores 
            return_constants (bool):    If True, also returns log(Z) and max score.

        Returns:
            (np.array):                          Normalised scores
            (float):                             Normalisation factor
            (np.array):                          Maximum score
        """
        max_logp = np.max(logp)
        logp = np.array(logp)
        diff = np.exp(logp - max_logp)
        Z = diff.sum()
        p = diff/Z
        return p if not return_constants else (p, np.log(Z), max_logp)
