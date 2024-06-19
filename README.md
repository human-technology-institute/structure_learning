# MCMC for Structure Learning

# Installation
## Dependencies
This code depends on the following libraries: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `networkx`, `igraph`, and `gmpy2`. For convenience, the file `conda_environment.yml` can be used to resolve these dependencies: `conda env create -f conda_environment.yml`.

Activate the conda environment and add the path to `structure_learning/src` to be able to import the mcmc modules.
## Via pip
Alternatively, you can build via pip. Navigate to the directory `structure_learning`. Execute the following commands:
```
pip install -q build
python -m build
```
If successful, you should have `mcmc-0.1.0-py3-none-any.whl` and `mcmc-0.1.0.tar.gz` inside `dist`.
Finally, run
```
pip install dist/mcmc-0.1.0-py3-none-any.whl
```
# Running MCMC
## Structure MCMC
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

# TODO
- Add Partition MCMC
- Support for parallel computation