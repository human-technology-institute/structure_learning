from .mcmc import *
from .structure_mcmc import StructureMCMC
from .partition_mcmc import PartitionMCMC
from .pc import PC
from .hillclimb import HillClimb
from .greedy import GreedySearch

samplers = {
    'StructureMCMC': StructureMCMC,
    'PartitionMCMC': PartitionMCMC,
    'PC': PC,
    'HillClimb': HillClimb,
    'GreedySearch': GreedySearch
}

def get_sampler(name: str):
    """
    Get a sampler class by its name.

    Parameters:
        name (str): Name of the sampler.

    Returns:
        Sampler class corresponding to the given name.
    """
    if name not in samplers:
        raise ValueError(f"Sampler '{name}' is not defined. Available samplers: {list(samplers.keys())}")
    return samplers[name]