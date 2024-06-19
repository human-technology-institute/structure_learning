# MCMC for Structure Learning

# Installation
## Dependencies
This code depends on the following libraries: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `networkx`, `igraph`, and `gmpy2`. For convenience, the file `conda_environment.yml` can be used to resolve these dependencies: `conda env create -f conda_environment.yml`.

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