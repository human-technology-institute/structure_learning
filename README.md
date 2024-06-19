# MCMC for Structure Learning

# Installation
## Dependencies
This code depends on the following libraries: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `networkx`, `igraph`, and `gmpy2`. For convenience, the file `conda_environment.yml` can be used to resolve these dependencies: `conda env create -f conda_environment.yml`.

## Git submodule

## Pip


# Running MCMC
## Structure MCMC
```
    initial_graph = np.random.choice([0,1], size=(n_nodes, n_nodes))*np.tri(n_nodes, n_nodes, -1)
    score = BGeScore(synthetic_data.data, initial_graph)
    proposal = GraphProposal(initial_graph)
    M = StructureMCMC(initial_graph, n_iterations, proposal, score)
    mcmc_results, acceptance = M.run()
    graphs = M.get_mcmc_res_graphs(mcmc_results)
```

# TODO
- Add Partition MCMC
- Support for parallel computation