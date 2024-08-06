import os
import pandas as pd
from pathlib import Path

path = Path(__file__).parent.absolute()

datasets = {}

datasets['sachs'] = pd.read_excel(os.path.join(path, 'datafiles/sachs/1. cd3cd28.xls'))