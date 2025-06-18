from abc import ABC, abstractmethod
from typing import Union
import pandas as pd 
from structure_learning.data import Data

class Sampler(ABC):

    def __init__(self, data: Union[Data, pd.DataFrame], **kwargs):
        self.data = data if isinstance(data, Data) else Data(data)
        super().__init__()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def config(self):
        pass
