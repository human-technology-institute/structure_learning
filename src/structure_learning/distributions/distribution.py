"""
This module defines classes for representing and manipulating distributions, including MCMC-based distributions and OPAD re-weighting mechanisms.

Classes:
    Distribution: Base class for distributions, providing methods for particle management, normalization, and visualization.
    MCMCDistribution: Extends Distribution to support MCMC-specific operations, including rejected particle handling.
    OPAD: Implements the OPAD re-weighting mechanism for MCMC distributions.

The module also includes utility methods for computing distributions from data and scores, as well as normalization factors.
"""

from typing import List, Iterable, Dict, Union, TypeVar, Type
from copy import deepcopy, copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import heapq
from structure_learning.scores import Score
from structure_learning.data_structures import DAG, Graph

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
        """
        Retrieve the log probabilities of all particles in the distribution.

        Returns:
            np.ndarray: Array of log probabilities.
        """
        return self.prop('logp')
    
    def clear(self):
        """
        Clear all particles from the distribution.
        """
        self.particles = {}

    def normalise(self, prop='freq', log=False):
        """
        Normalise the current set of particles in the distribution.
        """
        if len(self.particles) == 0:
            return
        self.p = self.prop(prop)
        prior = self.prop('prior')
        if len(prior) == 0:
            prior = 1
        if log:
            self.p = self.p + prior
            self.p = np.exp(self.p - np.max(self.p))
        else:
            self.p = self.p*np.exp(prior)
        
        keys = list(self.particles.keys())
        weights = np.array([(1. if 'weight' not in self.particles[particle] else self.particles[particle]['weight']) for particle in keys])
        
        self.p = (self.p*weights).astype(float)
        Z = np.sum(self.p)
        self.p /= Z
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
        return np.array([v[name] for v in self.particles.values() if name in v and v[name] is not None])
    
    def __contains__(self, particle):
        """
        Checks if particle is in the distribution.
        """
        return particle in self.particles
    
    def __len__(self):
        """
        Get the number of particles in the distribution.

        Returns:
            int: The number of particles.
        """
        return len(self.particles)
   
    def update(self, particle, data, **kwargs):
        """
        Adds particle to distribution. If particle already exists, update its data.

        Parameter:
            particle (Hashable):    Particle to add
            data (dict):            Particle data
        """
        if particle in self:
            for k,v in data.items():    
                if k in ('freq', 'logp', 'weight', 'prior', 'score_current'):
                    continue
                if k not in self.particles[particle]:
                    continue
                self.particles[particle][k].append(v)
                
            self.particles[particle]['freq'] += (1 if 'freq' not in data else data['freq'])
        else:
            self.particles[particle] = {}
            for k,v in data.items():
                self.particles[particle][k] = [v] if not isinstance(v, list) and k not in ('logp', 'weight', 'prior') else v
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

    def plot(self, prop='freq', sort=True, normalise=False, limit=-1, ax=None, showxticklabels=False):
        """
        Plot a histogram of the particles in the distribution.

        Parameters:
            prop (str): The property to plot (default is 'freq').
            sort (bool): Whether to sort the particles by the property values.
            normalise (bool): Whether to normalise the property values.
            limit (int): The maximum number of particles to display (default is -1, which shows all).
            ax (matplotlib.axes.Axes): The axes to plot on (default is None).

        Returns:
            tuple: Bars, particles, and counts displayed in the plot.
        """
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
        bars = sns.barplot(x=particles[-limit:], y=count[-limit:], dodge=True, ax=ax)
        ax.set_xlabel('Particles')
        ax.set_ylabel('Proportion')
        ax.set_xticklabels(particles[-limit:] if showxticklabels else [], rotation=90)
        return bars, particles[-limit:], count[-limit:]
    
    @classmethod
    def plot_multiple(cls, dists: List[Type['D']], prop='freq', normalise=False, limit=-1, ax=None, labels=None):
        """
        Plot multiple distributions on the same axes.

        Parameters:
            dists (List[Distribution]): List of distributions to plot.
            prop (str): The property to plot (default is 'freq').
            sort (bool): Whether to sort the particles by the property values.
            normalise (bool): Whether to normalise the property values.
            limit (int): The maximum number of particles to display (default is -1, which shows all).
            ax (matplotlib.axes.Axes): The axes to plot on (default is None).

        Returns:
            list: List of bar containers for each distribution.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.subplots(1,1)
            ax.grid(alpha=0.5)
            ax.set_axisbelow(True)
        
        all_particles = {}
        for idx,dist in enumerate(dists):
            dist_name = idx if labels is None else labels[idx]
            particles, count = dist.hist(prop=prop, normalise=normalise)
            sort_idx = np.argsort(count)
            limit = limit if limit > 0 else len(particles)
            particles, count = np.array(particles)[sort_idx][-limit:], np.array(count)[sort_idx][-limit:]
            for particle,cnt in zip(particles,count):
                if particle not in all_particles:
                    all_particles[particle] = {}
                all_particles[particle][dist_name] = cnt

        a = pd.DataFrame.from_dict(all_particles, orient='index').fillna(pd.NA)
        a['index'] = list(a.index)
        a_melt = a.melt(id_vars='index', value_vars=set(a.columns) - set(['index']), value_name=prop, var_name='Sampler')
        bars = sns.barplot(x='index', y=prop, hue='Sampler', data=a_melt, dodge=True, ax=ax)
        ax.set_xlabel('Particles')
        ax.set_ylabel('Proportion')
        ax.set_xticklabels(a_melt['index'], rotation=90)
        ax.legend(title='Samplers')
        return bars, a
    
    def top(self, prop='freq', n=1, mass=None):
        """
        Retrieve the top N particles based on a specified property.

        Parameters:
            prop (str): The property to sort by (default is 'freq').
            n (int): The number of top particles to retrieve.
            mass (float): If specified and prop='p', return the top particles with combined mass greater than this value.

        Returns:
            np.ndarray: Array of the top N particles.
        """
        k, v = self.hist(prop=prop)
        idx = np.argsort(v)
        if mass is None or prop != 'p':
            return np.array(k)[idx][-n:][::-1]  # Return the top N particles in descending order
        else:
            sorted_v = np.array(v)[idx[::-1]]
            cumulative_mass = np.cumsum(sorted_v)
            selected_idx = mass > cumulative_mass
            if 0 < mass < 1:
                selected_idx[np.count_nonzero(selected_idx)+1] = True  # Ensure at least one particle is selected
            return np.array(k)[idx[::-1][selected_idx]]
        
    def sample(self, size=1):
        """
        Sample particles from the distribution based on their probabilities.

        Parameters:
            size (int): The number of particles to sample.

        Returns:
            list: List of sampled particles.
        """
        particles = list(self.particles.keys())
        probs = self.prop('p')
        return np.random.choice(particles, size=size, p=probs).tolist()

    # arithmetic
    def __copy__(self):
        """
        Create a copy of the current distribution.

        Returns:
            Distribution: A copy of the distribution.
        """
        dclone = Distribution()
        dclone.particles = deepcopy(self.particles)
        return dclone
    
    def copy(self):
        return self.__copy__()

    def __add__(self, other: Type['D']) -> Type['D']:
        """
        Add two distributions together by combining their particles and frequencies.

        Parameters:
            other (Distribution): The distribution to add.

        Returns:
            Distribution: The resulting distribution after addition.
        """
        dsum = self.__copy__()
        for particle, data in other.particles.items():
            dsum.update(particle, data, iteration=data.get('iteration', []), normalise=False)
            dsum.particles[particle]['freq'] = data['freq'] + (0 if particle not in self else self.particles[particle]['freq'])
        dsum.normalise()
        return dsum

    def __sub__(self, other: Type['D']) -> Type['D']:
        """
        Subtract one distribution from another by removing particles present in the other distribution.

        Parameters:
            other (Distribution): The distribution to subtract.

        Returns:
            Distribution: The resulting distribution after subtraction.
        """
        dsub = Distribution()
        for particle, data in self.particles.items():
            if particle not in other:
                dsub.update(particle, data)
        return dsub
        
    @classmethod
    def compute_distribution(cls, data: pd.DataFrame, score: Score, graph_type='dag', blocklist:np.ndarray=None) -> Type['D']:
        """
        Compute a distribution from data and a scoring function.

        Parameters:
            data (pd.DataFrame): The dataset to compute the distribution from.
            score (Score): The scoring function to evaluate particles.
            graph_type (str): The type of graph to use ('dag' or 'cpdag').

        Returns:
            Distribution: The computed distribution.
        """
        dags = DAG.generate_all_dags(len(data.columns), list(data.columns))
        if blocklist is not None:
            dags = [dag for dag in dags if not (dag.incidence*blocklist).any()]
            
        if graph_type=='cpdag':
            cpdags = {}

        particles = {}
        particle_weights = {}
        scorer = score(data=data)
        for dag in dags:
            particle = dag.to_key()
            if graph_type=='cpdag':
                particle = dag.to_cpdag(blocklist=blocklist).to_key()
            if particle not in particles:
                scorer.graph = dag
                particles[particle] = scorer.compute(dag)['score']
                particle_weights[particle] = {'weight': 1}
            else:
                particle_weights[particle]['weight'] += 1

        dist = TrueDistribution(particles=list(particles.keys()), logp=particles.values(), theta=particle_weights.values())
        
        return dist  
        
    # pickle
    def save(self, filename: str, compression='gzip'):  
        """
        Saves the Graph object to a file.

        Parameters:
            filename (str): Path to the output file.
        """
        with open(filename, 'wb') as f:
            import compress_pickle as pickle
            pickle.dump(self, f, compression=compression)

    @classmethod
    def load(cls, filename: str, compression='gzip'):
        """
        Loads a Graph object from a file.

        Parameters:
            filename (str): Path to the input file.

        Returns:
            Graph: Loaded Graph object.
        """
        with open(filename, 'rb') as f:
            import compress_pickle as pickle
            return pickle.load(f, compression=compression)

class TrueDistribution(Distribution):

    def __init__(self, particles = [], logp = [], theta = None):
        super().__init__(particles, logp, theta)
        super().normalise(prop='logp', log=True)
    
    def normalise(self):
        return self
    

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

    def update(self, particle, data, iteration, **kwargs):
        """
        Adds particle to distribution. If particle already exists, update its data.

        Parameter:
            particle (Hashable):    Particle to add
            iteration (int):        Iteration number at which the particle was generated.
            data (dict):            Particle data
        """
        _data = {'iteration': iteration, 'logp': data['score_current'] if 'logp' not in data else data['logp'], 'prior': data.get('current_state_prior', None) if 'prior' not in data else data['prior'], 'timestamp': data['timestamp']}
        if 'weight' in data:
            _data['weight'] = data['weight']
        super().update(particle, _data, iteration=iteration, **kwargs)
            
        if self.rejected is not None and ('accepted' in data and not data['accepted']):
            if data['proposed_state'] is not None:
                particle = data['proposed_state'].to_key()
                self.rejected.update(particle, {'logp': data['score_proposed'], 'iteration': iteration, 'timestamp': data['timestamp'], 'prior': data.get('proposed_state_prior', None)})
            
    def to_opad(self, plus=False):
        """
        Convert the current MCMC distribution to an OPAD distribution.

        Parameters:
            plus (bool): If True, include rejected particles in the OPAD distribution.

        Returns:
            OPAD: The OPAD distribution.
        """
        opad = OPAD(plus=plus)
        opad.particles = deepcopy(self.particles)
        opad.rejected = copy(self.rejected)
        opad._add_rejected_particles_()
        opad.normalise()
        return opad
    
    @classmethod
    def from_iterates(cls, iterates: dict):
        """
        Create an MCMCDistribution from iteration data.

        Parameters:
            iterates (dict): A dictionary where keys are iteration numbers and values are data about the particles.

        Returns:
            MCMCDistribution: The resulting MCMC distribution.
        """
        dist = MCMCDistribution()
        for iteration, data in iterates.items():
            particle = data['graph'].to_key() if isinstance(data['graph'], Graph) else data['graph']
            dist.update(particle=particle, iteration=iteration, data=data)
        dist.normalise()
        return dist
    
    def to_iterates(self):
        """
        Convert an MCMCDistribution to iteration data.

        Returns:
            dict: A dictionary where keys are iteration numbers and values are data about the particles.
        """
        iterates = {}
        for particle, data in self.particles.items():
            for iteration, timestamp in zip(data['iteration'], data['timestamp']):
                iterates[iteration] = {
                    'graph': particle,
                    'current_state': particle,
                    'score_current': data['logp'],
                    'timestamp': timestamp,
                    'current_state_prior': data.get('prior', None),
                    'freq': data['freq'],
                }

        if self.rejected is not None:
            for particle, data in self.rejected.particles.items():
                for iteration, timestamp in zip(data['iteration'], data['timestamp']):
                    iterates[iteration].update({
                        'proposed_state': particle,
                        'score_proposed': data['logp'],
                        'timestamp': timestamp,
                        'proposed_state_prior': data.get('prior', None),
                        'freq': data['freq'],
                    })
        return iterates
    
    def __copy__(self):
        """
        Create a shallow copy of the current distribution.

        Returns:
            Distribution: A shallow copy of the distribution.
        """
        dclone = MCMCDistribution()
        dclone.particles = deepcopy(self.particles)
        dclone.rejected = copy(self.rejected)
        dclone.normalise()
        return dclone

class OPAD(MCMCDistribution):
    """
    This class implements the OPAD re-weighing mechanism described in 
    """
    def __init__(self, particles: Iterable = [], logp: Iterable = [], theta: Dict = [], plus = False):
        super().__init__(particles, logp, theta)
        self.plus = plus
        self.normalise()

    def normalise(self):
        """
        Normalise the OPAD distribution by adding rejected particles and computing probabilities.

        Returns:
            OPAD: The normalised OPAD distribution.
        """
        self._add_rejected_particles_()
        return super().normalise(prop='logp', log=True)
    
    def _add_rejected_particles_(self):
        """
        Add rejected particles to the distribution if the `plus` attribute is True.
        """
        if self.plus: # add rejected to particles
            if len(self.rejected) > 0:
                print('Adding rejected particles')
                for particle, data in self.rejected.particles.items():
                    super(MCMCDistribution, self).update(particle, data)
                self.rejected.clear()

    def update(self, particle, data, iteration, normalise=True):
        """
        Add new particles to the OPAD distribution and optionally renormalise.

        Parameters:
            particle (Hashable): The particle to add.
            iteration (int): The iteration number at which the particle was generated.
            data (dict): Data associated with the particle.
            normalise (bool): If True, renormalise the distribution after adding the particle.
        """
        super().update(particle, data, iteration=iteration, normalise=normalise)
        if normalise:
            self.normalise()

    @classmethod
    def from_mcmc(cls, dist: Distribution, plus=False):
        """
        Create an OPAD distribution from an MCMC distribution.

        Parameters:
            dist (Distribution): The MCMC distribution to convert.
            plus (bool): Whether to include rejected particles in the OPAD distribution.

        Returns:
            OPAD: The resulting OPAD distribution.
        """
        return dist.to_opad(plus=plus)
    
    def plot(self, prop='p', sort=True, normalise=False, limit=-1):
        """
        Plot a histogram of the particles in the OPAD distribution.

        Parameters:
            prop (str): The property to plot (default is 'p').
            sort (bool): Whether to sort the particles by the property values.
            normalise (bool): Whether to normalise the property values.
            limit (int): The maximum number of particles to display (default is -1, which shows all).

        Returns:
            tuple: Bars, particles, and counts displayed in the plot.
        """
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

    def __copy__(self):
        """
        Create a copy of the current distribution.

        Returns:
            Distribution: A copy of the distribution.
        """
        dclone = OPAD(plus=self.plus)
        dclone.particles = deepcopy(self.particles)
        dclone.rejected = copy(self.rejected)
        dclone.normalise()
        return dclone
    
    def to_iterates(self):
        if not self.plus:
            return super(MCMCDistribution, self).to_iterates()
        raise NotImplementedError("OPAD+ distributions cannot be converted to iterates.")

class FixedSizeDistribution(OPAD):
    
    def __init__(self, particles: Iterable = [], logp: Iterable = [], theta: Dict = [], max_size: int = 1000000):
        super().__init__(particles, logp, theta, False)
        self.max_size = max_size
        self._top_particles = []

        # build min-heap
        for particle, data in self.particles.items():
            score = data['logp']
            heapdata = (score, particle)

            if len(self._top_particles) < self.max_size:
                heapq.heappush(self._top_particles, heapdata)
            else:
                heapq.heappushpop(self._top_particles, heapdata)

        # rebuild particles dict
        self._particles = {}
        for score, particle in self._top_particles:
            self._particles[particle] = self.particles[particle]

        self.particles = self._particles

    def update(self, particle, data, iteration, normalise=False):

        _data = {'iteration': iteration, 'logp': data['score_current'] if 'logp' not in data else data['logp'], 'prior': data.get('current_state_prior', None) if 'prior' not in data else data['prior'], 'timestamp': data['timestamp']}
        if 'weight' in data:
            _data['weight'] = data['weight']

        super().update(particle, _data, iteration=iteration, normalise=False)
        # print('Adding particle:', particle, len(self.particles), len(self._top_particles))
        if particle not in self.particles:
                score = data['score_current']
                heapdata = (score, particle)

                if len(self._top_particles) < self.max_size:
                    heapq.heappush(self._top_particles, heapdata)
                else:
                    min_score, min_particle = heapq.heappushpop(self._top_particles, heapdata)
                    del self.particles[min_particle]
                    # print('Removing particle:', min_particle, len(self.particles), len(self._top_particles))
            
        if data['proposed_state'] is not None:
            particle = data['proposed_state'].to_key()
            super().update(particle, {'logp': data['score_proposed'], 'iteration': iteration, 'timestamp': data['timestamp'], 'prior': data.get('proposed_state_prior', None)}, iteration=iteration, normalise=False)
            # print('Adding particle:', particle, len(self.particles), len(self._top_particles))
            if particle not in self.particles:
                score = data['score_proposed']
                heapdata = (score, particle)

                if len(self._top_particles) < self.max_size:
                    heapq.heappush(self._top_particles, heapdata)
                else:
                    min_score, min_particle = heapq.heappushpop(self._top_particles, heapdata)
                    del self.particles[min_particle]
                    # print('Removing particle:', min_particle, len(self.particles), len(self._top_particles))

        if normalise:
            self.normalise()
