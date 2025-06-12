"""

"""
from abc import ABC, abstractmethod
from typing import TypeVar, Tuple, List
import numpy as np

State = TypeVar("State")
class StructureLearningProposal(ABC):
    """
    Base class for proposal classes for structure learning using MCMC.
    All inheriting classes must implement the following methods:
        propose() -> graph : numpy.ndarray, operation : str
        compute_acceptance_ratio() -> float
    """
    INITIAL = 'initial'
    STAY_STILL = 'stay_still'

    operations = [STAY_STILL]

    def __init__(self, initial_state : State, blacklist = None, whitelist = None, seed: int = 32):
        """
        Initialise StructureLearningProposal instance.

        Parameters:
            graph (networkx.DiGraph): graph
            blacklist (numpy.ndarray): mask for edges to ignore in the proposal
            whitelist (numpy.ndarray): mask for edges to include in the proposal
        """
        self.initial_state = initial_state
        self.current_state = initial_state
        self.blacklist = blacklist
        self.whitelist = whitelist
        self.proposed_state = None
        self.operation = None
        self._rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def propose(self) -> Tuple[State, str]:
        """
        Propose a DAG
        """
        pass

    @abstractmethod
    def compute_acceptance_ratio(self) -> float:
        pass

    @abstractmethod
    def get_nodes_to_rescore(self) -> List[str]:
        pass

    def accept(self):
        self.current_state = self.proposed_state
        self.proposed_state = None
