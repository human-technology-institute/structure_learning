# Structure MCMC for Causal Discovery

Structure MCMC (Markov Chain Monte Carlo) is a Bayesian method used for learning the structure of probabilistic graphical models, such as Bayesian networks. This technique involves sampling from the posterior distribution of possible structures given the data, allowing for the estimation of the most probable graph structures that represent the dependencies among variables. The method effectively navigates the high-dimensional space of possible graph structures using MCMC, making it suitable for complex models where traditional methods might struggle.

Structure MCMC is one of the first and most straightforward approaches for sampling graphs, compared to more complex methods such as Partition MCMC, Layering, or Order MCMC. Its simplicity and foundational role make it an essential tool for understanding and developing more advanced graph sampling techniques.

Structure MCMC has proven to be a powerful tool for probabilistic structure learning, enabling the discovery of complex dependencies in data-rich environments. By iterating through possible structures and utilizing Bayesian scoring methods, it provides robust and interpretable models for a wide range of applications.

In addition to this Python implementation, similar approaches can be found in the R packages [BiDAG](https://cran.r-project.org/package=BiDAG) and **bnlearn**, which provide tools for structure learning in Bayesian networks, including MCMC methods.

# Installation
## Dependencies
This code depends on the following libraries: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `networkx`, `igraph`, and `gmpy2`. 

## Via conda
For convenience, the file `conda_environment.yml` can be used to resolve these dependencies: `conda env create -f conda_environment.yml`.

Activate the conda environment and add the path to `structure_mcmc/src` to be able to import the mcmc modules.

## Via pip

Alternatively, you can set up a virtual environment using `venv` and then build and install the package using pip. Follow these steps:

1. **Create a virtual environment**:
```sh
python -m venv venv
source venv/bin/activate
```

2. **Navigate to the directory `structure_mcmc`. Execute the following commands:**
```sh
pip install -q build
python -m build
```

If successful, you should have `mcmc-0.1.0-py3-none-any.whl` and `mcmc-0.1.0.tar.gz` inside `dist`.
Finally, run

```sh
pip install dist/mcmc-0.1.0-py3-none-any.whl
```

# Running Structure MCMC

```
    import numpy as np
    from mcmc.mcmc import StructureMCMC
    from mcmc.proposals import GraphProposal
    from mcmc.scores import BGeScore

    ...

    # start with a random initial graph
    initial_graph = np.random.choice([0,1], size=(n_nodes, n_nodes))*np.tri(n_nodes, n_nodes, -1)
    p = np.random.permutation(n_nodes)
    initial_graph = initial_graph[p, :]
    initial_graph = initial_graph[:, p]

    # create score and proposal objects
    score = BGeScore(data, initial_graph)
    proposal = GraphProposal(initial_graph)

    # initialise structure MCMC
    M = StructureMCMC(initial_graph, n_iterations, proposal, score)

    # run MCMC
    mcmc_results, acceptance = M.run()

    # get chain of graphs
    graphs = M.get_mcmc_res_graphs(mcmc_results)
```

# Reference
1. **Friedman, N., & Koller, D. (2003).** "Being Bayesian about network structure: A Bayesian approach to structure discovery in Bayesian networks." *Machine Learning*, 50(1), 95-125.
   
2. **Madigan, D., & York, J. (1995).** "Bayesian graphical models for discrete data." *International Statistical Review/Revue Internationale de Statistique*, 215-232.

3. **Giudici, P., & Castelo, R. (2003).** "Improving Markov Chain Monte Carlo model search for data mining." *Machine Learning*, 50(1), 127-158.

# TODO
- Add Partition MCMC
- Support for parallel computation