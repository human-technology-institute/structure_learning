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
    MULTINOMIAL_TYPE = 'multinomial'

    def __init__(self, values: Union[np.ndarray, pd.DataFrame], variables: List = None, variable_types: Dict = None):
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
        return self.variables
    
    def __getitem__(self, *args):
        return self.values.__getitem__(*args)
    
    def __len__(self):
        return len(self.values)
    
    def __copy__(self):
        clone = Data(self.values.copy(), self.variables.copy(), self.variable_types.copy())
        return clone
    
    @property
    def shape(self):
        return self.values.shape
    
    def normalise(self, variables: List = None):
        _scaler = StandardScaler()
        variables = variables if variables is not None else self.variables
        x = _scaler.fit_transform(self.data[variables])
        transformed_data = self.__copy__()
        transformed_data.values.loc[:,variables] = x
        transformed_data._scaler = _scaler
        return transformed_data

    def k_fold(self, k=5, shuffle=True, seed=None):
        fold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for i, (train_index, test_index) in enumerate(fold.split(self.values)):
            train_data = pd.DataFrame(self.values.iloc[train_index, :], columns=self.variables)
            test_data = pd.DataFrame(self.values.iloc[test_index, :], columns=self.variables)
            yield Data(values=train_data, variable_types=self.variable_types), Data(values=test_data, variable_types=self.variable_types)

    # visualise

    # analyse