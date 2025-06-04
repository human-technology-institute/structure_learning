import numpy as np
from .graph import Graph 

class CPDAG(Graph):
    
    def __init__(self, incidence = None, nodes = None):
        super().__init__(incidence, nodes)
        self.undirected_edges = list()
        self._undirected_edges_idx = list()
        rows, cols = np.nonzero(np.tril(incidence + incidence.T, 0)==2)
        for r, c in zip(rows, cols):
            self._undirected_edges_idx.append((r,c))
            self.undirected_edges.append((nodes[r], nodes[c]))
        self._n_dags = None
        self.dags = None

    @classmethod
    def from_dag(cls, dag):
        return dag.to_cpdag()
    
    def __contains__(self, dag):
        cpdag = dag.to_cpdag()
        return cpdag == self
    
    def __eq__(self, other):
        return (self.incidence==other.incidence).all()
    
    def __iter__(self):
        return self

    def __next__(self):
        pass

    def enumerate_dags(self, generate=True):
        if self.dags is None:
            self._n_dags = 0
            self.dags = []
            dim = len(self.undirected_edges)
            fstr = "{0:0" + str(dim) + "b}"
            for i in range(2**dim):
                binstr = fstr.format(i)
                incidence = self.incidence.copy()
                for j,flip in enumerate(binstr):
                    r, c = self._undirected_edges_idx[j]
                    if flip=="1":
                        incidence[r,c] = False
                    else:
                        incidence[c,r] = False
                if not Graph.has_cycle(incidence):
                    self._n_dags += 1
                    if generate:
                        g = Graph(incidence=incidence, nodes=self.nodes)
                        self.dags.append(g)
                        yield g

    def __len__(self):
        [g for g in self.enumerate_dags(generate=True)]
        return self._n_dags
