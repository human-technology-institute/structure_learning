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

## System Requirements

This package requires **Graphviz** to be installed on your system before installing the Python dependencies.

**macOS:**
```bash
brew install graphviz
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install graphviz graphviz-dev
```

**CentOS/RHEL/Fedora:**
```bash
sudo yum install graphviz graphviz-devel  # CentOS/RHEL
# or
sudo dnf install graphviz graphviz-devel  # Fedora
```

## For Users

```bash
# Clone the repository
git clone <repository-url>
cd structure_learning

# Install system dependencies
brew install graphviz  # macOS
# or: sudo apt-get install graphviz graphviz-dev  # Linux

# Set environment variables for pygraphviz (macOS with Homebrew)
export CPPFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"

# Install the package
pip install .
```

## For Developers

```bash
# Clone the repository
git clone <repository-url>
cd structure_learning

# Install system dependencies (see above)
brew install graphviz  # macOS example

# Set environment variables for pygraphviz (macOS with Homebrew)
export CPPFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode with dev dependencies
pip install -r requirements-dev.txt
pip install -e .

# Verify installation
python -c "import structure_learning; print('✅ Installation successful!')"

# Run tests (optional)
python -m pytest src/structure_learning/tests/ -v
```

## Using as a Git Submodule

For development within another project:

```bash
# In your main project
git submodule add <repository-url> libs/structure_learning
cd libs/structure_learning

# Install dependencies
brew install graphviz  # or appropriate for your system

# Set environment variables for pygraphviz (macOS with Homebrew)
export CPPFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"

pip install -e .
```

## Troubleshooting

**Error: `fatal error: 'graphviz/cgraph.h' file not found`**

This means pygraphviz can't find the Graphviz headers. Try:

**macOS (Homebrew):**
```bash
export CPPFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"
pip install pygraphviz
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install graphviz graphviz-dev
pip install pygraphviz
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install graphviz graphviz-devel
pip install pygraphviz
```

**Import Error: `ModuleNotFoundError`**

Make sure you installed in development mode:
```bash
pip install -e .
```

# Running Structure MCMC

```python
from structure_learning.approximators import StructureMCMC

# initialise structure MCMC
M = StructureMCMC(max_iter=n_iterations, data=data, score_object='bge') # data is a pd.DataFrame/Data object

# run MCMC
mcmc_results, acceptance = M.run()

```

# Running Partition MCMC

```python
from structure_learning.approximators import PartitionMCMC

# initialise partition MCMC
M = PartitionMCMC(data=data, max_iter=n_iterations, score_object='bge') # data is a pd.DataFrame/Data object

# run MCMC
mcmc_results, acceptance = M.run()

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
exp = Experiment(experiment_name='test', data=data, samplers=samplers, ground_truth='true_distribution.npy', metrics=['mae', 'mse', 'rhat', 'kld', 'jsd'], n_threads=4, seed=42)
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
