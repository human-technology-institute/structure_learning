import os
from collections import UserDict
from typing import Any
import pandas as pd
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