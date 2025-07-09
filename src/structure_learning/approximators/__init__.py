from .mcmc import *
from .structure_mcmc import StructureMCMC
from .partition_mcmc import PartitionMCMC
from .pc import PC
from .hillclimb import HillClimb
from .greedy import GreedySearch
from .approximator import Approximator

approximators = {
    'StructureMCMC': StructureMCMC,
    'PartitionMCMC': PartitionMCMC,
    'PC': PC,
    'HillClimb': HillClimb,
    'GreedySearch': GreedySearch
}

def get_approximator(name: str):
    """
    Get a approximator class by its name.

    Parameters:
        name (str): Name of the approximator.

    Returns:
        Approximator class corresponding to the given name.
    """
    if name not in approximators:
        raise ValueError(f"Approximator '{name}' is not defined. Available samplers: {list(approximators.keys())}")
    return approximators[name]