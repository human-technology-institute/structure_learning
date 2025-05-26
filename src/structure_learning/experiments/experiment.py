from typing import Union
import os
from urllib.parse import unquote, urlparse
import multiprocessing as mp
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mlflow
from structure_learning.mcmc import MCMC, StructureMCMC, PartitionMCMC
from structure_learning.data import SyntheticDataset
from structure_learning.utils.graph_utils import generate_DAG, plot_graph_from_adj_mat

class MCMCExperiment:

    def __init__(self, mcmc: Union[MCMC, str], data: pd.DataFrame = None, score: str = 'bge', max_iter: int = 30000, n_chains: int = 1, logging: bool = True, seed: int = 32,
                 blacklist = None, pc_init = True,
                 # if data is None, generate synthetic data from given graph
                 graph: Union[pd.DataFrame, np.ndarray] = None, n_samples: int = 100, n_variables: int = 5,
                 # experiment details
                 experiment_name: str = None, parallel: bool = True, n_processes: int = 4, log_interval: int = 1, artifact_interval: int = 100,
                 # values to log
                 log_values = {'score': 'score_current', 'acceptance probability': 'acceptance_prob'}
                 ):
        # set seed
        self.seed = seed
        np.random.seed(self.seed)

        self.max_iter = max_iter
        self.n_chains = n_chains
        self.logging = logging
        self.parallel = parallel
        self.n_processes = n_processes
        self.experiment_name = experiment_name
        self.artifact_interval = artifact_interval
        self.log_interval = log_interval
        self.log_values = {'score': 'score_current', 'acceptance probability': 'acceptance_prob'}
        self.log_values.update(log_values)

        if data is None:
            # generate synthetic data
            dag = graph.values if isinstance(graph, pd.DataFrame) else graph
            if dag is None:
                dag = generate_DAG(n_variables, 0.5, seed)
            labels = list(graph.columns) if isinstance(graph, pd.DataFrame) else [str(i+1) for i in range(dag.shape[1])]
            data, w = SyntheticDataset.simulate_data_from_dag(dag, n_samples, len(labels), labels, [0,1], 0.01)
        self.data = data

        if isinstance(mcmc, MCMC):
            self.mcmc = [mcmc]
        elif isinstance(mcmc, str):
            self.seeds = np.random.randint(10001, size=self.n_chains).tolist()
            if 'structure' in mcmc.lower():
                self.mcmc = [StructureMCMC(data=data, score_object=score, blacklist=blacklist, pc_init=pc_init, seed=seed) for seed in self.seeds]
            elif 'partition' in mcmc.lower():
                self.mcmc = [PartitionMCMC(data=data, score_object=score, blacklist=blacklist, pc_init=pc_init, seed=seed) for seed in self.seeds]
            else:
                raise NotImplementedError(mcmc)
        else:
            raise NotImplementedError(mcmc)

    def run(self):
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        if self.parallel:
            pool = mp.get_context("fork").Pool(processes=min(self.n_processes, 4))
            pool.map(self._run_one_chain, range(self.n_chains))
        else:
            for chain in self.n_chains:
                self._run_one_chain(chain)

    def _run_one_chain(self, chain: int):
        with mlflow.start_run(nested=True):
            # log parameters and input
            mlflow.log_param('seed', self.seeds[chain])
            mlflow.log_param('n_chains', self.n_chains)
            mlflow.log_param('parallel', self.parallel)
            mlflow.log_param('max_iter', self.max_iter)
            mlflow.log_input(mlflow.data.from_pandas(self.data))

            mcmc = self.mcmc[chain]
            uri = unquote(urlparse(mlflow.get_artifact_uri()).path)
            for iter in range(self.max_iter):
                # run one step
                result = mcmc.step()
                mcmc.update_results(iter, result)

                # log this step
                if self.logging:
                    if iter % self.log_interval == 0:
                        for k,v in self.log_values.items():
                            mlflow.log_metric(k, result[v], iter)

                    # save graph as artifact
                    if self.artifact_interval >= 1 and iter % self.artifact_interval == 0:
                        path = os.path.join(uri, f'{iter}.png')
                        g = result['graph']
                        plot_graph_from_adj_mat(g, list(self.data.columns))
                        plt.savefig(path)
                        mlflow.log_artifact(path)
                        plt.close()

            # save chain results
            results = mcmc.results
            acceptance = mcmc.n_accepted/mcmc.max_iter
            path = os.path.join(uri, f'{chain}.npz')
            np.savez(path, results=results, acceptance=acceptance)
            mlflow.log_artifact(path)
            print(f'Saved npz file to {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MCMCExperiment', description='Run MCMC experiment(s) on given dataset')
    parser.add_argument('-d', '--dataset', type=str, default=None, dest='data', help='Name of dataset file (xlsx, csv) to be loaded using pandas. If none is supplied, synthetic data is generated.')
    parser.add_argument('-v', '--variables', type=str, nargs='+', dest='variables', help='Variables to use. Leave blank to use all')
    parser.add_argument('-t', '--type', choices=['partition', 'structure'], default='partition', help='MCMC type', dest='mcmc_type')
    parser.add_argument('-n', '--n_chains', type=int, default=1, dest='n_chains', help='Number of chains to run')
    parser.add_argument('-s', '--score', type=str, default='bge', dest='score', choices=['bge', 'bdeu'], help='Graph score to use')
    parser.add_argument('-i', '--iterations', type=int, default=30000, dest='max_iter', help='Number of MCMC iterations')
    parser.add_argument('-g', '--graph', dest='graph', type=str, default=None, help='Filename of file (csv, xlsx) containing initial graph to use')
    parser.add_argument('--experiment_name', type=str, dest='experiment_name')
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=10000, help='Number of samples to generate (if dataset is not provided)')
    parser.add_argument('--n_variables', dest='n_variables', type=int, default=5, help='Number of variables for synthetic data generation (if dataset if not provided and variables option is empty)')
    parser.add_argument('-gt', '--graph', dest='gt_graph', type=str, default=None, help='Filename of file (csv, xlsx) containing ground truth graph if available')
    parser.add_argument('--seed', dest='seed', type=int, default=32, help='Seed for numpy')
    parser.add_argument('-p', '--parallel', type=bool, default=True, help='Run chains in parallel')
    parser.add_argument('-np', '--nprocesses', type=int, default=4, help='Number of parallel processes to execute chains. Max 4')
    parser.add_argument('-l', '--logging', type=bool, default=True, help='Log scores and graphs using mlflow')

    args = parser.parse_args()

    # load data
    data = args['data']
    if data is not None:
        _, extension = os.path.splitext(data)
        if extension == 'xlsx':
            data = pd.read_excel(data)
        elif extension == 'csv':
            data = pd.read_csv(data)
        else:
            raise NotImplementedError(f'Unrecognized file format ({extension}) for data')

    # load graph if available
    graph = data['gt_graph']
    if graph is not None:
        _, extension = os.path.splitext(graph)
        if extension == 'xlsx':
            graph = pd.read_excel(graph)
        elif extension == 'csv':
            graph = pd.read_csv(graph)
        else:
            raise NotImplementedError(f'Unrecognized file format ({extension}) for graph')

    # setup experiment
    experiment = MCMCExperiment(mcmc=args['mcmc_type'], data=data, score=args['score'], max_iter=args['max_iter'], n_chains=args['n_chains'], logging=args['logging'], seed=args['seed'],
                                graph=graph, n_samples=args['n_samples'], experiment_name=args['experiment_name'], parallel=args['parallel'], n_processes=args['n_processes'])