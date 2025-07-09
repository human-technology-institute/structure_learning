"""
This module contains unit tests for the BGe (Bayesian Gaussian equivalent) score implementation.

The tests validate the correctness of the BGe score by comparing the implementation
against the reference implementation in the R package BiDAG. The tests ensure that
both implementations produce equivalent scores for various parent configurations.
"""

import unittest
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd
from structure_learning.scores import BGeScore as implemented_BGe
from structure_learning.approximators import PartitionMCMC

rules = default_converter + numpy2ri.converter + pandas2ri.converter
bidag = importr('BiDAG')

class TestBGe(unittest.TestCase):
    """
    Unit tests for the BGe (Bayesian Gaussian equivalent) score implementation.
    
    This test class compares our implementation of the BGe score against the 
    reference implementation in the R package BiDAG. It verifies that both 
    implementations produce equivalent scores for various parent configurations.
    """

    def test_compute(self):
        """
        Test the BGe score computation against BiDAG's implementation.
        
        Creates synthetic data and compares the BGe scores computed by our 
        implementation against the scores computed by BiDAG's implementation 
        for various parent configurations. The scores should match within a 
        small numerical tolerance.
        """
        # create synthetic data
        N = 1000
        n_vars = 5
        vars = [str(i) for i in range(n_vars)]
        vars_label_to_index = {v:idx for idx,v in enumerate(vars)}
        data = pd.DataFrame(np.random.choice([0,1], (N, n_vars)), columns=vars)

        possible_parents = PartitionMCMC._list_possible_parents(n_vars-1, vars)
        with rules.context():
            score = bidag.scoreparameters(scoretype="bge", data=data)

        bge2 = implemented_BGe(data=data)

        for i,pmat in enumerate(possible_parents):
            for j in range(pmat.shape[0]):
                # remove nans
                parents = [p for p in pmat[j] if isinstance(p, str)]

                # compute using our BGe
                score2 = bge2.compute_node_with_edges(vars[i], parents, vars_label_to_index)['score']

                # compute using BiDAG
                with rules.context():
                    parents_idx = [vars_label_to_index[node]+1 for node in parents]
                    score1 = bidag.DAGcorescore(i+1, np.array(parents_idx), n_vars, score)

                # assertion
                print(score1, score2)
                self.assertAlmostEqual(score1, score2, delta=1e-7)

if __name__ == '__main__':
    unittest.main()
