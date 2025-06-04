"""

"""
from abc import ABC, abstractmethod
from typing import TypeVar, Union, Tuple
import time
import pandas as pd
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from structure_learning.scores import Score, BGeScore, BDeuScore
from structure_learning.proposals import StructureLearningProposal, GraphProposal, PartitionProposal
from structure_learning.data_structures import OrderedPartition, DAG
from structure_learning.data import Data
from structure_learning.distributions import MCMCDistribution, OPAD
from .pc import PC
from structure_learning.utils.graph_utils import initial_graph_pc

State = TypeVar('State')

class MCMC(ABC):
    """
    Base class for MCMC.
    Inheriting classes must implement the following methods:
        step()
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
        Initilialise MCMC instance.

        Parameters:
            data (pd.DataFrame):                            Dataset. Optional if score_object is given.
            initial_state (State):                          Initial state for the MCMC simulation.
                                                            If None, simulation starts with a random graph or a graph
                                                            constructed from PC algorithm.
            max_iter (int):                                 The number of MCMC iterations to run. Default: 30000.
            score_object (Score):                           A score object implementing compute(). If None, BGeScore is used
                                                            (data must be provided). Default: None.
            proposal_object (StructureLearningProposal):    A proposal object. If None, a GraphProposal instance is used.
                                                            Default: None.
            pc_init (bool):                                 If True and initial_graph is not given, PC algorithm will be used
                                                            to generate initial graph.
            blacklist (numpy.ndarray):                      Mask for edges to ignore in the proposal
            whitelist (numpy.ndarray):                      Mask for edges to include
            plus1 (bool):                                   Use plus1 neighborhood
            result_type (str):                              Options: iterates | distribution (default) | opad | opad+
            graph_type (str):                               Options: dag (default) | cpdag
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
            # self._pc_state, self.pc_graph = initial_graph_pc(score_object.data, True)
        
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
        Run MCMC simulation.

        Returns:
            (dict): dictionary (iteration number as keys) of MCMC state chain information
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
        Perform one MCMC iteration
        """
        pass

    def update_results(self, iteration, info):
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
        Returns list of sampled graphs from MCMC simulation results.

        Parameters:
            results (dict): MCMC simulation results.

        Returns:
            (list): sampled graphs
        """
        return self.get_chain_info(results)

    def get_chain_info(self, results, key='graph'):
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
        return MCMCDistribution.from_iterates(self.results) if self.result_type == self.RESULT_TYPE_ITER else self.results
    
    def to_opad(self, plus=False):
        if self.result_type == self.RESULT_TYPE_ITER:
            return MCMCDistribution.from_iterates(self.results).to_opad(plus=plus)
        elif self.result_type == self.RESULT_TYPE_DIST :
            return self.results.to_opad(plus=plus)
        else:
            return self.results