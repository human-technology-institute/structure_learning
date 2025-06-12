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
from matplotlib import pyplot as plt
from structure_learning.scores import Score, BGeScore, BDeuScore
from structure_learning.proposals import StructureLearningProposal
from structure_learning.data import Data
from structure_learning.distributions import MCMCDistribution, OPAD
from .pc import PC

State = TypeVar('State')

class MCMC(ABC):
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
                 blacklist: np.ndarray = None, whitelist: np.ndarray = None, plus1: bool = False, seed: int = 32, 
                 result_type: str = RESULT_TYPE_DIST, graph_type='dag'):
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
        self.data  = data
        self.node_labels = list(data.columns)
        self._node_label_to_idx = {node:idx for idx,node in enumerate(self.node_labels)}
        self.num_nodes = len(self.node_labels)
        self.pc_graph = None
        self._rng = np.random.default_rng(seed=seed)
        self._trace = []

        if score_object is None:
            print('Using default BGe score')
            score_object = BGeScore(data=data)
        elif isinstance(score_object, str):
            if score_object.lower() == 'bge':
                score_object = BGeScore(data=data)
            elif score_object.lower() in ['bde', 'bdeu']:
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
        self._to_string = f"MCMC_n_{self.num_nodes}_iter_{self.max_iter}"

    def run(self) -> Tuple[dict, float]:
        """
        Execute the MCMC simulation.

        Returns:
            Tuple[dict, float]: Results of the simulation and acceptance ratio.
        """
        for iter in range(self.max_iter):
            result = self.step()
            self.update_results(iter, result)
        if self.result_type in (self.RESULT_TYPE_DIST, self.RESULT_TYPE_OPAD):
            self.results.normalise()
        return self.results, self.n_accepted/self.max_iter

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
        info['graph'] = info['graph'] if self.graph_type=='dag' else info['graph'].to_cpdag()
        key = info['graph'].to_key()
        if self.graph_type=='cpdag':
            if key not in self._cpdag_sizes:
                self._cpdag_sizes[key] = len(info['graph'])
            info['weight'] = self._cpdag_sizes[key]
        if self.result_type in (self.RESULT_TYPE_DIST, self.RESULT_TYPE_OPAD, self.RESULT_TYPE_OPAD_PLUS):
            self.results.update(particle=key, iteration=iteration, data=info.copy())
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
            return results.prop(key)
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