
import yaml
import numpy as np
import pandas as pd
from multiprocessing import Pool
from structure_learning.samplers import get_sampler

class Experiment:

    def __init__(self, experiment_name: str = None, data: pd.DataFrame = None, samplers: list = None, n_threads: int = 1, seed: int = 42):
        self._load_config(dict(experiment_name=experiment_name, data=data, samplers=samplers, n_threads=n_threads, seed=seed))

    def run(self):
        with Pool(processes=self.n_threads) as pool:
            return pool.map(self.run_sampler, self.samplers)

    def run_sampler(self, sampler):
        """
        Run a specific sampler.

        Parameters:
            sampler: The sampler instance to run.
        """
        return sampler.run()
    
    @classmethod
    def from_yaml(cls, yaml_file: str):
        """
        Load experiment configuration from a YAML file.

        Parameters:
            yaml_file (str): Path to the YAML file containing experiment configuration.
        """
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        exp = Experiment()
        exp._load_config(config)
        return exp
    
    def to_yaml(self, yaml_file: str):
        """
        Save experiment configuration to a YAML file.

        Parameters:
            yaml_file (str): Path to the YAML file where the configuration will be saved.
        """
        config = self._to_dict()
        with open(yaml_file, 'w') as file:
            yaml.safe_dump(config, file)

    def _to_dict(self): 
        pass

    def _load_config(self, config):
        # check required keys
        required_keys = ['experiment_name', 'data', 'samplers']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key} in configuration")
            self.__setattr__(key, config[key])

        # process optional keys
        optional_keys = ['n_threads', 'seed']
        for key in optional_keys:
            if key in config:
                self.__setattr__(key, config[key])
            else:
                self.__setattr__(key, None)
        np.random.seed(config.get('seed', 42))

        # load data
        self.data = pd.read_csv(self.data) if isinstance(self.data, str) else self.data
        self.variables = list(self.data.columns) if 'variables' not in config else config['variables']
        self.variable_types = [] if 'variable_types' not in config else config['variable_types']

        self.samplers = []
        for sampler in config.get('samplers', []):
            if 'sampler_type' not in sampler:
                raise ValueError("Each sampler must have a specified 'sampler_type'")
            n_chains = sampler.get('n_chains', 1)
            for chain in range(n_chains):
                _sampler = get_sampler(sampler['sampler_type'])(data=self.data, **sampler.get('config', {}))
                self.samplers.append(_sampler)