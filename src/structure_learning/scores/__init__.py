"""
mcmc.score

This module contains implementations of different scoring
schemes for graphs in structure learning using MCMC.
The abstract class is defined as Score and future 
additional scoring schemes must inherit from it.
"""
from .score import Score
from .bd import BDScore
from .bdeu import BDeuScore
from .bge import BGeScore
