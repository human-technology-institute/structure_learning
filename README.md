# Graph Structure Learning

Structure learning for graphs involves discovering the underlying relationships and dependencies among variables in a dataset, typically represented as nodes and edges in a graph. This process is crucial for understanding causal relationships, modeling probabilistic dependencies, and making predictions. The `structure_learning` library provides a suite of algorithms for learning graph structures, enabling researchers and practitioners to analyze complex data effectively.

## Implemented Algorithms

### Structure MCMC
Structure MCMC (Markov Chain Monte Carlo) is a Bayesian method for sampling graph structures from the posterior distribution, allowing for robust probabilistic structure learning.

### Partition MCMC
Partition MCMC leverages the combinatorial structure of directed acyclic graphs (DAGs) to improve convergence and unbiased sampling, supporting operations like DAG partitioning and scoring.

### Greedy Search
Greedy search iteratively adds, removes, or reverses edges to optimize a graph scoring function, providing a fast and straightforward approach to structure learning.

### Hillclimb Search
Hillclimb search is a local optimization algorithm that modifies graph structures to maximize a graph scoring function, ensuring convergence to a local optimum.

### PC Algorithm
The PC algorithm is a constraint-based method for causal discovery that uses conditional independence tests to construct graph structures.

## Implemented Scores

### BGe Score
The Bayesian Gaussian equivalent (BGe) score is designed for continuous data. It evaluates the fit of a graph structure to the data by considering the likelihood of the data under a Gaussian distribution and incorporating a prior over graph structures. 

### BDeu Score
The Bayesian Dirichlet equivalent uniform (BDeu) score is tailored for discrete data. It uses a Dirichlet prior to compute the likelihood of the data given a graph structure, ensuring a uniform prior over possible structures. This score is ideal for datasets with categorical variables.

## Implemented Metrics

### KL Divergence
Kullback-Leibler (KL) divergence measures the difference between two probability distributions. It is commonly used to evaluate how well a learned distribution approximates the true distribution.

### JS Divergence
Jensen-Shannon (JS) divergence is a symmetric measure of similarity between two probability distributions, derived from KL divergence. It is particularly useful for comparing distributions in probabilistic models.

### MSE
Mean Squared Error (MSE) quantifies the average squared difference between predicted and actual values.

### MAE
Mean Absolute Error (MAE) calculates the average absolute difference between predicted and actual values, providing a robust metric for assessing prediction accuracy.

### Rhat
Rhat is a convergence diagnostic metric for MCMC chains. It evaluates the consistency of multiple chains and helps determine whether the chains have converged to the target distribution.

# Installation
## Dependencies
This code depends on the following libraries: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `torch`, `networkx`, `pcalg`, and `igraph`.

## Via conda
For convenience, the file `conda_environment.yml` can be used to resolve these dependencies: `conda env create -f conda_environment.yml`.

Activate the conda environment and add the path to `structure_learning/src` to be able to import the MCMC modules.

## Via pip

Alternatively, you can set up a virtual environment using `venv` and then build and install the package using pip. Follow these steps:

1. **Create a virtual environment**:
```sh
python -m venv venv
source venv/bin/activate
```

2. **Navigate to the directory `structure_learning`. Execute the following commands:**
```sh
pip install -q build
python -m build
```

If successful, you should have `structure_learning-1.0.0-py3-none-any.whl` and `structure_learning-1.0.0.tar.gz` inside `dist`.
Finally, run

```sh
pip install dist/structure_learning-1.0.0-py3-none-any.whl
```

# Running Structure MCMC

```python
from structure_learning.samplers.mcmc import StructureMCMC

# initialise structure MCMC
M = StructureMCMC(max_iter=n_iterations, data=data, score_object='bge') # data is a pd.DataFrame/Data object

# run MCMC
mcmc_results, acceptance = M.run()

# get chain of graphs
graphs = M.get_graphs(mcmc_results)
```

# Running Partition MCMC

```python
from structure_learning.samplers.partition_mcmc import PartitionMCMC

# initialise partition MCMC
M = PartitionMCMC(data=data, max_iter=n_iterations, score_object='bge') # data is a pd.DataFrame/Data object

# run MCMC
mcmc_results, acceptance = M.run()

# get chain of graphs
graphs = M.get_graphs(mcmc_results)
```

# Streamlined experiments

Experiments involving different samplers can be configured using the Experiment class.
```
samplers = [
    {
        "sampler_type": "StructureMCMC",
        "n_chains": 2,
        "config": {
            "max_iter": 5000,
            "score_object": "bge",
            "pc_init": False,
            "result_type": "distribution",
            "graph_type": "dag",
        }
    },
    {
        "sampler_type": "PartitionMCMC",
        "n_chains": 2,
        "config": {
            "max_iter": 5000,
            "score_object": "bge",
            "result_type": "distribution",
            "graph_type": "dag",
            "searchspace": "FULL"
        }
    }
]
exp = Experiment(experiment_name='test', data=synthetic_data.data, samplers=samplers, ground_truth='true_distribution.npy', metrics=['mae', 'mse', 'rhat', 'kld', 'jsd'], n_threads=4, seed=42)
res = exp.run()
metrics = exp.evaluate()
```
# Other libraries
In addition to this Python implementation, similar approaches can be found in the R packages [BiDAG](https://cran.r-project.org/package=BiDAG) and **bnlearn**, which provide tools for structure learning in Bayesian networks, including MCMC methods.

# TODO
- Support for parallel computation
- Extend scoring methods for hybrid data types

# Reference
1. **Friedman, N., & Koller, D. (2003).** "Being Bayesian about network structure: A Bayesian approach to structure discovery in Bayesian networks." *Machine Learning*, 50(1), 95-125.

2. **Madigan, D., & York, J. (1995).** "Bayesian graphical models for discrete data." *International Statistical Review/Revue Internationale de Statistique*, 215-232.

3. **Giudici, P., & Castelo, R. (2003).** "Improving Markov Chain Monte Carlo model search for data mining." *Machine Learning*, 50(1), 127-158.