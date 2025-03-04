import unittest
import rpy2
import rpy2.robjects as robjects
import rpy2.rinterface as rinterface
from rpy2.robjects.packages import importr, data
from rpy2.robjects.methods import getmethod
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd
from mcmc.scores import BGeScore as implemented_BGe
from mcmc.utils.score_utils import list_possible_parents
from mcmc.utils.graph_utils import node_label_to_index

rules = default_converter + numpy2ri.converter + pandas2ri.converter
bidag = importr('BiDAG')

class TestBGe(unittest.TestCase):

    def test_compute(self):
        # create synthetic data
        N = 1000
        n_vars = 5
        vars = [str(i) for i in range(n_vars)]
        vars_label_to_index = node_label_to_index(vars)
        data = pd.DataFrame(np.random.choice([0,1], (N, n_vars)), columns=vars)

        possible_parents = list_possible_parents(n_vars-1, vars)
        with rules.context():
            score = bidag.scoreparameters(scoretype="bge", data=data)

        bge2 = implemented_BGe(data=data, incidence=np.zeros((n_vars, n_vars)))

        for i,pmat in enumerate(possible_parents):
            for j in range(pmat.shape[0]):
                # remove nans
                parents = [p for p in pmat[j] if isinstance(p, str)]

                # compute using our BGe
                score2 = bge2.compute_node_with_edges(vars[i], parents)['score']

                # compute using BiDAG
                with rules.context():
                    parents_idx = [vars_label_to_index[node]+1 for node in parents]
                    score1 = bidag.DAGcorescore(i+1, np.array(parents_idx), n_vars, score)

                # assertion
                print(score1, score2)
                self.assertAlmostEqual(score1, score2, delta=1e-5)

if __name__ == '__main__':
    unittest.main()
