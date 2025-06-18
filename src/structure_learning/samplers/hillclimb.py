from typing import Union
import pgmpy
import graphical_models as gm
import networkx as nx
import pandas as pd
from structure_learning.data import Data
from structure_learning.scores import Score
from structure_learning.data_structures.dag import DAG
from structure_learning.data_structures.cpdag import CPDAG
from .sampler import Sampler

class HillClimb(Sampler):
    
    def __init__(self, data: Union[Data, pd.DataFrame], score: Score):
        super().__init__(data)

    def run(self):
        pass
    
    def config(self):
        """
        Returns the configuration of the HillClimb algorithm.
        """
        return {}

