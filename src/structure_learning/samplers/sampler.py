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

    # pickle
    def save(self, filename: str, compression='gzip'):
        """
        Saves the Graph object to a file.

        Parameters:
            filename (str): Path to the output file.
        """
        with open(filename, 'wb') as f:
            import compress_pickle as pickle
            pickle.dump(self, f, compression=compression)

    @classmethod
    def load(cls, filename: str, compression='gzip'):
        """
        Loads a Graph object from a file.

        Parameters:
            filename (str): Path to the input file.

        Returns:
            Graph: Loaded Graph object.
        """
        with open(filename, 'rb') as f:
            import compress_pickle as pickle
            return pickle.load(f, compression=compression)
