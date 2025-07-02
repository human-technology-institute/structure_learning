"""
This module provides classes and utilities for handling datasets and data structures.

Classes:
    LazyDataset: A dictionary-like class that lazily loads datasets from files.
    Data: A class for managing and analyzing datasets with support for variable types and transformations.

Dependencies:
    - numpy
    - pandas
    - sklearn.preprocessing.StandardScaler
    - sklearn.model_selection.KFold
    - pathlib.Path
"""

from typing import Union, List, Dict
import os
from collections import UserDict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pathlib import Path

path = Path(__file__).parent.absolute()

class LazyDataset(UserDict):

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve the value associated with the given key. If the value is a file path, it will be loaded as a pandas DataFrame.

        Parameters:
            key (Any): Key to retrieve the value for.

        Returns:
            Any: The value associated with the key, or a pandas DataFrame if the value is a file path.
        """
        item = super().__getitem__(key)
        if type(item) == str: # data not read yet
            item = pd.read_excel(os.path.join(path, item))
            super().__setitem__(key, item)
        return item

    def __setitem__(self, key: Any, item: Any) -> None:
        return super().__setitem__(key, item)

datasets = LazyDataset()

datasets['sachs'] = 'datafiles/sachs/1. cd3cd28.xls'

class Data:

    CONTINUOUS_TYPE = 'continuous'
    BINARY_TYPE = 'binary'
    ORDINAL_TYPE = 'ordinal'
    CATEGORICAL_TYPE = 'categorical'

    """
    A class for managing and analyzing datasets with support for variable types and transformations.

    Attributes:
        CONTINUOUS_TYPE (str): Type identifier for continuous variables.
        BINARY_TYPE (str): Type identifier for binary variables.
        ORDINAL_TYPE (str): Type identifier for ordinal variables.
        CATEGORICAL_TYPE (str): Type identifier for categorical variables.
        values (pd.DataFrame): The dataset values.
        variables (List): List of variable names.
        variable_types (Dict): Dictionary mapping variable names to their types.
    """

    def __init__(self, values: Union[np.ndarray, pd.DataFrame], variables: List = None, variable_types: Dict = None):
        """
        Initialize the Data object with dataset values, variable names, and types.

        Parameters:
            values (Union[np.ndarray, pd.DataFrame]): Dataset values.
            variables (List, optional): List of variable names. Required if values is a numpy array.
            variable_types (Dict, optional): Dictionary mapping variable names to their types.

        Raises:
            Exception: If variable names are not supplied when values is a numpy array.
        """
        if isinstance(values, pd.DataFrame):
            self.variables = list(values.columns)
        else:
            self.variables = variables
            if self.variables is None:
                raise Exception('Variable names not supplied')
        
        self.variable_types = {variable: self.CONTINUOUS_TYPE for variable in self.variables}
        if variable_types is not None:
            for var,type in variable_types.items():
                self.variable_types[var] = type
   
        self.values = pd.DataFrame(values, columns=self.variables) if isinstance(values, np.ndarray) else values

    @property
    def columns(self):
        """
        Get the list of variable names.

        Returns:
            List: List of variable names.
        """
        return self.variables
    
    def __getitem__(self, *args):
        """
        Retrieve data for the specified key(s).

        Parameters:
            *args: Key(s) to retrieve data for.

        Returns:
            Any: Data corresponding to the specified key(s).
        """
        return self.values.__getitem__(*args)
    
    def __len__(self):
        """
        Get the number of rows in the dataset.

        Returns:
            int: Number of rows in the dataset.
        """
        return len(self.values)
    
    def __copy__(self):
        """
        Create a copy of the Data object.

        Returns:
            Data: A copy of the Data object.
        """
        clone = Data(self.values.copy(), self.variables.copy(), self.variable_types.copy())
        return clone
    
    @property
    def shape(self):
        """
        Get the shape of the dataset.

        Returns:
            Tuple[int, int]: Shape of the dataset (rows, columns).
        """
        return self.values.shape
    
    def normalise(self, variables: List = None):
        """
        Normalize the specified variables in the dataset.

        Parameters:
            variables (List, optional): List of variable names to normalize. Defaults to all continuous variables.

        Returns:
            Data: A new Data object with normalized variables.
        """
        _scaler = StandardScaler()
        variables = variables if variables is not None else self.variables
        # only normalise continuous variables
        variables = [variable for variable in variables if variable in self.variables and self.variable_types[variable]==self.CONTINUOUS_TYPE]
        x = _scaler.fit_transform(self.values[variables])
        transformed_data = self.__copy__()
        transformed_data.values.loc[:,variables] = x
        transformed_data._scaler = _scaler
        return transformed_data

    def k_fold(self, k=5, shuffle=True, seed=None):
        """
        Perform k-fold splitting of the dataset.

        Parameters:
            k (int): Number of folds. Default is 5.
            shuffle (bool): Whether to shuffle the data before splitting. Default is True.
            seed (int, optional): Random seed for reproducibility.

        Yields:
            Tuple[Data, Data]: Training and testing datasets for each fold.
        """
        fold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for i, (train_index, test_index) in enumerate(fold.split(self.values)):
            train_data = pd.DataFrame(self.values.iloc[train_index, :], columns=self.variables)
            test_data = pd.DataFrame(self.values.iloc[test_index, :], columns=self.variables)
            yield Data(values=train_data, variable_types=self.variable_types), Data(values=test_data, variable_types=self.variable_types)

    # visualise

    # analyse