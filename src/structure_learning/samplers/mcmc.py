"""

"""
from abc import ABC, abstractmethod
from typing import TypeVar, Union, Tuple
import time
import pandas as pd
import networkx as nx
import numpy as np
from structure_learning.scores import Score, BGeScore, BDeuScore
from structure_learning.proposals import StructureLearningProposal, GraphProposal, PartitionProposal
from structure_learning.data_structures import OrderedPartition, DAG, MCMCDistribution, Graph
from structure_learning.utils.graph_utils import initial_graph_pc, generate_DAG
from structure_learning.utils.partition_utils import build_partition

State = TypeVar('State')

class MCMC(ABC):
    """
    Base class for MCMC.
    Inheriting classes must implement the following methods:
        step()
    """
    def __init__(self, data: pd.DataFrame, initial_state: State, max_iter: int = 30000, score_object: Union[str, Score] = None,
                 proposal_object: Union[str, StructureLearningProposal] = None, pc_init: bool = True,
                 blacklist: np.ndarray = None, whitelist: np.ndarray = None, plus1: bool = False, seed: int = 32, 
                 result_type: str = 'distribution'):
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
            result_type (str):                              Save results as either dictionary or distribution object.
        """

        if data is None or not isinstance(data, pd.DataFrame):
            raise Exception("Data (as pandas dataframe) must be provided")
        self.data  = data
        self.node_labels = list(data.columns)
        self.num_nodes = len(self.node_labels)
        self.pc_graph = None
        self._rng = np.random.default_rng(seed=seed)

        if score_object is None:
            print('Using default BGe score')
            score_object = BGeScore(data=data, incidence=None)
        elif isinstance(score_object, str):
            if score_object.lower() == 'bge':
                score_object = BGeScore(data=data, incidence=None)
            elif score_object.lower() in ['bde', 'bdeu']:
                score_object = BDeuScore(data=data, incidence=None)
            else:
                raise Exception(f"Unsupported score {score_object}")
        elif not isinstance(score_object, Score):
            raise Exception(f"Unsupported score {score_object}")
        self.score_object = score_object

        if pc_init:
            print('Running PC algorithm')
            self._pc_state, self.pc_graph = initial_graph_pc(score_object.data, True)

        if proposal_object is None or isinstance(proposal_object, str):
            if isinstance(proposal_object, str) and proposal_object not in ['graph', 'partition']:
                raise Exception('Unsupported proposal', proposal_object)
            if initial_state is not None:
                if isinstance(initial_state, np.ndarray):
                    proposal_object = GraphProposal(initial_state=initial_state, blacklist=blacklist, whitelist=whitelist, seed=seed)
                elif isinstance(initial_state, OrderedPartition):
                    proposal_object = PartitionProposal(initial_state=initial_state, blacklist=blacklist, whitelist=whitelist, seed=seed)
                else:
                    print('Invalid initial state')
            else:
                initial_state = (self._pc_state if proposal_object == 'partition' else self.pc_graph) if pc_init else DAG.generate_random(self.node_labels, 0.5, seed)
                if whitelist is not None:
                    initial_state[whitelist > 0] = 1
                if blacklist is not None:
                    initial_state[blacklist > 0] = 0
                if proposal_object == 'partition':
                    initial_state = build_partition(incidence=initial_state, node_labels=self.node_labels)
                    proposal_object = PartitionProposal(initial_state, whitelist=whitelist, blacklist=blacklist, seed=seed)
                else:
                    proposal_object = GraphProposal(initial_state=initial_state, blacklist=blacklist, whitelist=whitelist, seed=seed)
        elif not isinstance(proposal_object, StructureLearningProposal):
            raise Exception('Unsupported proposal', proposal_object)
        self.initial_state = initial_state
        self.proposal_object = proposal_object
        self.blacklist = blacklist
        self.whitelist = whitelist
        self.max_iter = max_iter
        self.n_accepted = 0
        self.scores = None
        self.result_type = result_type
        self.results = MCMCDistribution() if result_type == 'distribution' else {}
        self._start_time = time.time()
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
        return self.results, self.n_accepted/self.max_iter

    @abstractmethod
    def step(self) -> dict:
        """
        Perform one MCMC iteration
        """
        pass

    def update_results(self, iteration, info):
        if self.result_type == 'distribution':
            self.results.update(info['graph'].to_key(), iteration, info)
        else:
            self.results[iteration] = info

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
            return results.prop('key')
        else:
            return [result[key] for _,(i,result) in enumerate(results.items())]

    def __str__(self):
        return self._to_string
