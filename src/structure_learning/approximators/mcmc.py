"""
This module implements the MCMC class, which serves as a base class for Markov Chain Monte Carlo (MCMC) simulations.

The MCMC class provides methods for initializing simulations, running iterations, updating results, and converting results into different formats such as distributions or OPAD objects. It is designed to be extended by subclasses that implement specific MCMC behaviors.

Classes:
    MCMC: Abstract base class for MCMC simulations.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Union, Tuple
import time
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from structure_learning.scores import Score, BGeScore, BDeuScore
from structure_learning.proposals import StructureLearningProposal
from structure_learning.data import Data
from structure_learning.distributions import MCMCDistribution, OPAD
from .pc import PC
from .approximator import Approximator

State = TypeVar('State')

class MCMC(Approximator):
    """
    Base class for Markov Chain Monte Carlo (MCMC) simulations.
    Inheriting classes must implement the `step` method to define the behavior of a single MCMC iteration.
    
    """

    RESULT_TYPE_DIST = 'distribution'
    RESULT_TYPE_OPAD = 'opad'
    RESULT_TYPE_OPAD_PLUS = 'opad+'
    RESULT_TYPE_ITER = 'iterates'
    def __init__(self, data: pd.DataFrame, initial_state: State, max_iter: int = 30000, score_object: Union[str, Score] = None,
                 proposal_object: Union[str, StructureLearningProposal] = None, 
                 pc_init: bool = True, pc_significance_level = 0.01, pc_ci_test = 'pearsonr',
                 blacklist: np.ndarray = None, whitelist: np.ndarray = None, seed: int = None, 
                 result_type: str = RESULT_TYPE_DIST, graph_type='dag', burn_in: float = 0.1):
        """
        Initialize the MCMC instance.

        Parameters:
            data (pd.DataFrame):                                        Dataset for the MCMC simulation.
            initial_state (State):                                      Initial state for the simulation.
            max_iter (int):                                             Maximum number of iterations. Default is 30000.
            score_object (Union[str, Score]):                           Scoring object or type. Default is None.
            proposal_object (Union[str, StructureLearningProposal]):    Proposal object or type. Default is None.
            pc_init (bool):                                             Whether to initialize using the PC algorithm. Default is True.
            pc_significance_level (float):                              Significance level for the PC algorithm. Default is 0.01.
            pc_ci_test (str):                                           Conditional independence test for the PC algorithm. Default is 'pearsonr'.
            blacklist (np.ndarray):                                     Mask for edges to ignore in the proposal.
            whitelist (np.ndarray):                                     Mask for edges to include in the proposal.
            plus1 (bool):                                               Whether to use plus1 neighborhood. Default is False.
            seed (int):                                                 Random seed for reproducibility. Default is 32.
            result_type (str):                                          Type of result to generate. Default is 'distribution'.
            graph_type (str):                                           Type of graph ('dag' or 'cpdag'). Default is 'dag'.
        """

        if data is None or not isinstance(data, (Data, pd.DataFrame)):
            raise Exception("Data (as pandas dataframe or Data) must be provided")
        super().__init__(data)
        self.node_labels = list(data.columns)
        self._node_label_to_idx = {node:idx for idx,node in enumerate(self.node_labels)}
        self.num_nodes = len(self.node_labels)
        self.pc_graph = None
        if seed is None:
            seed = np.random.randint(100000)
        self.seed = seed
        self._rng = np.random.default_rng(seed=seed)
        self._trace = []

        if score_object is None:
            print('Using default BGe score')
            score_object = BGeScore(data=data)
        elif isinstance(score_object, str):
            if score_object=='BGeScore' or score_object.lower() == 'bge':
                score_object = BGeScore(data=data)
            elif score_object=='BDeuScore' or score_object.lower() in ['bde', 'bdeu']:
                score_object = BDeuScore(data=data)
            else:
                raise Exception(f"Unsupported score {score_object}")
        elif not isinstance(score_object, Score):
            raise Exception(f"Unsupported score {score_object}")
        self.score_object = score_object

        if pc_init:
            print('Running PC algorithm')
            pc = PC(data=score_object.data, significance_level=pc_significance_level, ci_test=pc_ci_test)
            self._pc_state, self.pc_graph = pc.run()
        
        self.initial_state = initial_state
        self.proposal_object = proposal_object
        self.blacklist = blacklist
        self.whitelist = whitelist
        self.max_iter = max_iter
        self.n_accepted = 0
        self.scores = None
        self.result_type = result_type
        self.graph_type = graph_type
        self.results = {}
        if result_type == MCMC.RESULT_TYPE_DIST:
            self.results = MCMCDistribution()
        elif result_type == MCMC.RESULT_TYPE_OPAD:
            self.results = OPAD()
        elif result_type == MCMC.RESULT_TYPE_OPAD_PLUS:
            self.results = OPAD(plus=True)
        self._start_time = time.time()
        self._cpdag_sizes = {}
        self._cpdags = {}
        self._to_string = f"MCMC_n_{self.num_nodes}_iter_{self.max_iter}"
        self.iteration = 0
        if not (0 <= burn_in < 1):
            raise ValueError("Burn-in must be a float between 0 and 1")
        self.burn_in = burn_in*max_iter

        self.config_dict = {
            'initial_state': initial_state,
            'max_iter': max_iter,
            'score_object': type(score_object).__name__,
            'proposal_object': type(proposal_object).__name__ if isinstance(proposal_object, StructureLearningProposal) else proposal_object,
            'pc_significance_level': pc_significance_level,
            'pc_ci_test': pc_ci_test,
            'seed': seed,
            'result_type': result_type,
            'graph_type': graph_type
        }

    def run(self, intervals=-1) -> Tuple[dict, float]:
        """
        Execute the MCMC simulation.

        Returns:
            Tuple[dict, float]: Results of the simulation and acceptance ratio.
        """
        results = []
        tqdm_bar = tqdm(total=self.max_iter, desc='MCMC iterations', unit='iter')
        while True:
            if self.iteration > self.max_iter:
                break

            if self.iteration>0 and self.iteration%intervals==0 and intervals > 0:
                if self.result_type in (self.RESULT_TYPE_DIST, self.RESULT_TYPE_OPAD, self.RESULT_TYPE_OPAD_PLUS):
                    self.results.normalise()
                results.append((self.results.copy(), self.n_accepted/self.max_iter))

            result = self.step()
            if self.iteration >= self.burn_in:
                self.update_results(self.iteration, result)
            self.iteration += 1
            tqdm_bar.update(1)
        tqdm_bar.close()

        if self.result_type in (self.RESULT_TYPE_DIST, self.RESULT_TYPE_OPAD, self.RESULT_TYPE_OPAD_PLUS):
            self.results.normalise()
        return self.results if intervals < 0 else results, self.n_accepted/self.max_iter

    @abstractmethod
    def step(self) -> dict:
        """
        Perform one iteration of the MCMC simulation.

        Returns:
            dict: Information about the current iteration.
        """
        pass

    def update_results(self, iteration, info):
        """
        Update the results of the MCMC simulation with information from the current iteration.

        Parameters:
            iteration (int): Current iteration number.
            info (dict): Information about the current iteration.
        """
        key = info['graph'].to_key()
        if self.graph_type=='cpdag':
            if key not in self._cpdags:
                cpdag = info['graph'] = info['graph'].to_cpdag(blocklist=self.blacklist)
                key = info['graph'].to_key()
            else:
                key = cpdag = self._cpdags[key]
            if key not in self._cpdag_sizes:
                self._cpdag_sizes[key] = len(info['graph'])
            info['weight'] = self._cpdag_sizes[key]
            if self.result_type == self.RESULT_TYPE_OPAD_PLUS:
                if info['proposed_state'] is not None:
                    info['proposed_state'] = info['proposed_state'].to_cpdag(blocklist=self.blacklist)
                    proposed_key = info['proposed_state'].to_key()
                    if proposed_key not in self._cpdag_sizes:
                        self._cpdag_sizes[proposed_key] = len(info['proposed_state'])
                    info['proposed_state_weight'] = self._cpdag_sizes[proposed_key]
        if self.result_type in (self.RESULT_TYPE_DIST,):
            self.results.update(particle=key, iteration=iteration, data=info.copy())
        elif self.result_type in (self.RESULT_TYPE_OPAD, self.RESULT_TYPE_OPAD_PLUS):
            self.results.update(particle=key, iteration=iteration, data=info.copy(), normalise=False)
        elif self.result_type == 'iterates':
            self.results[iteration] = info
        else:
            raise Exception("Unsupported result type")
        self._trace.append(info['score_current'])

    def get_graphs(self, results):
        """
        Retrieve a list of sampled graphs from the MCMC results.

        Parameters:
            results (dict): Results of the MCMC simulation.

        Returns:
            list: Sampled graphs.
        """
        return self.get_chain_info(results)

    def get_chain_info(self, results, key='graph'):
        """
        Extract chain information from the MCMC results.

        Parameters:
            results (dict): Results of the MCMC simulation.
            key (str): Key to extract information for. Default is 'graph'.

        Returns:
            list: Chain information.
        """
        if self.result_type == 'distribution':
            return list(results.particles.keys()) if key=='graph' else results.prop(key)
        else:
            return [result[key] for _,(i,result) in enumerate(results.items())]

    def __str__(self):
        return self._to_string

    @property
    def trace(self):
        return self._trace
    
    def traceplot(self, ax=None):
        """
        Generate a trace plot of the MCMC simulation.

        Parameters:
            ax (matplotlib.axes.Axes): Matplotlib axis to plot on. Default is None.

        Returns:
            list: Plot object.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.subplots(1,1)
            ax.grid(alpha=0.5)
            ax.set_axisbelow(True)
        plot = ax.plot(self.trace)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Log score')
        return plot

    def to_cpdag_distribution(self):
        if self.graph_type == 'cpdag':
            return self.results
        elif self.graph_type == 'dag':
            if self.result_type == self.RESULT_TYPE_ITER:
                new_results = {}
                for key, value in self.results.items():
                    pass
            elif self.result_type in (self.RESULT_TYPE_OPAD, self.RESULT_TYPE_OPAD_PLUS):
                pass
            elif self.RESULT_TYPE_DIST:
                pass
            else:
                raise Exception("Unsupported result type for CPDAG conversion", self.result_type)
        else:
            raise Exception("Unknow graph type", self.graph_type)
    
    def to_distribution(self):
        """
        Convert the MCMC results to a distribution.

        Returns:
            MCMCDistribution: Distribution object.
        """
        return MCMCDistribution.from_iterates(self.results) if self.result_type == self.RESULT_TYPE_ITER else self.results
    
    def to_opad(self, plus=False):
        """
        Convert the MCMC results to an OPAD object.

        Parameters:
            plus (bool): Whether to use the plus1 neighborhood. Default is False.

        Returns:
            OPAD: OPAD object.
        """
        if self.result_type == self.RESULT_TYPE_ITER:
            return MCMCDistribution.from_iterates(self.results).to_opad(plus=plus)
        elif self.result_type == self.RESULT_TYPE_DIST :
            return self.results.to_opad(plus=plus)
        else:
            return self.results
        
    def config(self):
        """
        Get the configuration of the MCMC instance.

        Returns:
            dict: Configuration dictionary.
        """
        return {'sampler_type': self.__class__.__name__, 'config': self.config_dict}
