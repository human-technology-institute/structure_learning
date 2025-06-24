
import yaml
import numpy as np
import pandas as pd
from multiprocessing import Pool
from structure_learning.distributions import Distribution
from structure_learning.samplers import get_sampler, Sampler
from structure_learning.evaluation.metrics import get_metric, SHD, MAE, MSE, KLD, JSD, RHat

class Experiment:

    def __init__(self, experiment_name: str = None, data: pd.DataFrame = None, samplers: list = None, metrics: list=None, ground_truth: str=None, snapshot_interval:int = -1, n_threads: int = 1, seed: int = 42):
        self._load_config(dict(experiment_name=experiment_name, data=data, samplers=samplers, metrics=metrics, ground_truth=ground_truth, n_threads=n_threads, seed=seed, snapshot_interval=snapshot_interval))

    def run(self):
        with Pool(processes=self.n_threads) as pool:
            self.results = pool.map(self.run_sampler, self.samplers)
        return self.results
        
    def evaluate(self):
        # self.results dim is n_samplers * n_snapshots
        n_snapshots = max([len(res[0]) for res in self.results]) if self.snapshot_interval > 0 else 1
        evaluation = {}
        for snapshot in range(n_snapshots):
            evaluation[(1+snapshot)*self.snapshot_interval] = {}
            distributions = []
            for result in self.results:
                if self.snapshot_interval < 0:
                    if isinstance(result, Distribution):
                        distributions.append(result)
                    elif isinstance(result[0], Distribution):
                        distributions.append(result[0])
                else: # result is a list
                    if isinstance(result[0], Distribution): #List[Distribution]
                        idx = min(len(result), snapshot)
                        distributions.append(result[idx])
                    elif isinstance(result[0][0], Distribution): #List[Distribution], acceptance_ratio
                        idx = min(len(result[0]), snapshot)
                        distributions.append(result[0][idx])
                    elif isinstance(result[0][0][0], Distribution): #List[Distribution, acceptance_ratio], acceptance_ratio
                        idx = min(len(result[0]), snapshot)
                        distributions.append(result[0][idx][0])

            for mname, metric in self.metrics:
                ### 
                # TO DO: fix this explicit typing check
                ###
                if isinstance(metric, (MAE, MSE, KLD, JSD)) and self.ground_truth_type=='distribution':
                    if not hasattr(self, 'ground_truth'):
                        raise Exception("Ground truth distribution not provided for", mname)
                    evaluation[(1+snapshot)*self.snapshot_interval][mname] = [metric.compute(dist, self.ground_truth) for dist in distributions]
                elif isinstance(metric, SHD) and self.ground_truth_type=='graph':
                    if not hasattr(self, 'ground_truth'):
                        raise Exception("Ground truth graph not provided for", metric)
                    evaluation[(1+snapshot)*self.snapshot_interval][mname] = [metric.compute(dist, self.ground_truth) for dist in distributions]
                elif isinstance(metric, RHat):
                    evaluation[(1+snapshot)*self.snapshot_interval][mname] = metric.compute(distributions)
                else:
                    raise Exception('Unknown configuration for metric', mname)
        return evaluation

    def run_sampler(self, sampler: Sampler):
        """
        Run a specific sampler.

        Parameters:
            sampler: The sampler instance to run.
        """
        return sampler.run(self.snapshot_interval)
    
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
        if yaml_file is not None:
            with open(yaml_file, 'w') as file:
                yaml.safe_dump(config, file)
        else:
            return config

    def _to_dict(self): 
        config = {}
        # required keys
        required_keys = ['experiment_name', 'samplers']
        for key in required_keys:
            config[key] = self.__getattribute__(key)

        # data 
        config['data'] = self.data_str

        # optional keys
        optional_keys = ['n_threads', 'seed', 'snapshot_interval']
        for key in optional_keys:
            if hasattr(self, key):
                config[key] = self.__getattribute__(key)

        # ground truth
        config['ground_truth'] = self.ground_truth_str

        # samplers
        config['samplers'] = [sampler.config() for sampler in self.samplers]

        # metrics
        config['metrics'] = [mname for mname,_ in self.metrics]

        return config

    def _load_config(self, config):
        # check required keys
        required_keys = ['experiment_name', 'data', 'samplers']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key} in configuration")
            self.__setattr__(key, config[key])

        # process optional keys
        optional_keys = ['n_threads', 'seed', 'ground_truth', 'snapshot_interval']
        for key in optional_keys:
            if key in config:
                self.__setattr__(key, config[key])
            else:
                self.__setattr__(key, None)
        np.random.seed(config.get('seed', 42))

        # load data
        self.data_str = self.data if isinstance(self.data, str) else ""
        self.data = pd.read_csv(self.data) if isinstance(self.data, str) else self.data
        self.variables = list(self.data.columns) if 'variables' not in config else config['variables']
        self.variable_types = [] if 'variable_types' not in config else config['variable_types']

        # ground_truth
        self.ground_truth_str = None
        if self.ground_truth is not None:
            self.ground_truth_str = self.ground_truth
            self.ground_truth = np.load(self.ground_truth_str, allow_pickle=True).item()
            self.ground_truth_type = 'distribution' if isinstance(self.ground_truth, Distribution) else 'graph'

        # samplers
        self.samplers = []
        for sampler in config.get('samplers', []):
            if 'sampler_type' not in sampler:
                raise ValueError("Each sampler must have a specified 'sampler_type'")
            n_chains = sampler.get('n_chains', 1)
            for chain in range(n_chains):
                _sampler = get_sampler(sampler['sampler_type'])(data=self.data, **sampler.get('config', {}))
                self.samplers.append(_sampler)

        # metrics
        self.metrics = []
        for metric in config.get('metrics', []):
            self.metrics.append((metric, get_metric(metric)))