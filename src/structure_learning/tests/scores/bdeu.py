import unittest
from pgmpy.estimators.StructureScore import BDeu as target_bdeu
import numpy as np
import pandas as pd
from structure_learning.scores import BDeuScore as implemented_bdeu
from structure_learning.samplers import PartitionMCMC

class TestBDeu(unittest.TestCase):

    def test_compute(self):

        # create synthetic data
        N = 1000
        n_vars = 5
        vars = [str(i) for i in range(n_vars)]
        data = pd.DataFrame(np.random.choice([0,1], (N, n_vars)), columns=vars)
        possible_parents = PartitionMCMC._list_possible_parents(n_vars-1, vars)
        bdeu1 = target_bdeu(data=data)
        bdeu2 = implemented_bdeu(data=data)

        for i,pmat in enumerate(possible_parents):
            for j in range(pmat.shape[0]):
                # remove nans
                parents = [p for p in pmat[j] if isinstance(p, str)]

                # compute using our BDeu
                score2 = bdeu2.compute_node_with_edges(vars[i], parents)['score']
                # compute using pgmpy
                score1 = bdeu1.local_score(vars[i], parents)

                # assertion
                self.assertAlmostEqual(score1, score2, delta=1e-5)

if __name__ == '__main__':
    unittest.main()
