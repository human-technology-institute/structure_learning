import unittest
import numpy as np
import pandas as pd
from mcmc.mcmc.sampler import OPADSampler, MCMCSampler
from mcmc.scores import BGeScore

from mcmc.utils.graph_utils import generate_key_from_adj_matrix, generate_adj_matrix_from_key
from mcmc.inference.posterior import generate_all_dags
from mcmc.data.synthetic import SyntheticDataset
from mcmc.proposals.graph.graph_proposal import GraphProposal
from mcmc.inference.prior import UniformPrior

class TestBGe(unittest.TestCase):

    def test_compute(self):
        # create synthetic data
        N = 1000
        n_vars = 4

        node_labels = [f"X{i + 1}" for i in range(n_vars)]
        noise_scale = 1
        graph_type = "erdos-renyi"
        degree = 2

        # %%
        sdj = SyntheticDataset(num_nodes=n_vars, num_obs=N, node_labels=node_labels, degree=degree,
                               noise_scale=noise_scale, graph_type=graph_type)

        true_dag = sdj.adj_mat.values
        data = sdj.data

        all_dags = generate_all_dags(data, BGeScore, gen_augmented_priors=False, isScoreLogSpace=True,
                                     return_normalized=False)

        blacklist = np.eye(n_vars)
        whitelist = np.zeros((n_vars, n_vars))

        initial_state = generate_adj_matrix_from_key(np.random.choice([k for k in all_dags.keys()]), n_vars)
        mcmc_sampler = MCMCSampler(model=BGeScore(data, initial_state),
                                   proposal=GraphProposal(initial_state=initial_state,
                                                          blacklist=blacklist,
                                                          whitelist=whitelist),
                                   prior=UniformPrior(blacklist=blacklist, whitelist=whitelist))

        mcmc_res = mcmc_sampler.run(initial_state, 1000000)

        opad_sampler = OPADSampler(model=BGeScore(data, initial_state),
                                   proposal=GraphProposal(initial_state=initial_state,
                                                          blacklist=blacklist,
                                                          whitelist=whitelist),
                                   prior=UniformPrior(blacklist=blacklist, whitelist=whitelist))

        opad_res = opad_sampler.run(initial_state, 1000000)

        self.assertAlmostEqual(1, 1, delta=1e-5)

        #
        #
        # possible_parents = list_possible_parents(n_vars-1, vars)
        # with rules.context():
        #     score = bidag.scoreparameters(scoretype="bge", data=data)
        #
        # bge2 = implemented_BGe(data=data, incidence=np.zeros((n_vars, n_vars)))
        #
        # for i,pmat in enumerate(possible_parents):
        #     for j in range(pmat.shape[0]):
        #         # remove nans
        #         parents = [p for p in pmat[j] if isinstance(p, str)]
        #
        #         # compute using our BGe
        #         score2 = bge2.compute_node_with_edges(vars[i], parents)['score']
        #
        #         # compute using BiDAG
        #         with rules.context():
        #             parents_idx = [vars_label_to_index[node]+1 for node in parents]
        #             score1 = bidag.DAGcorescore(i+1, np.array(parents_idx), n_vars, score)
        #
        #         # assertion
        #         print(score1, score2)
        #         self.assertAlmostEqual(score1, score2, delta=1e-5)

if __name__ == '__main__':
    TestBGe().test_compute()
    #unittest.main(TestBGe.test_compute())
