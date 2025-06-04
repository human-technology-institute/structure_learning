from .graph import Graph 

class CPDAG(Graph):
    
    def __init__(self, incidence = None, nodes = None):
        super().__init__(incidence, nodes)
        self.undirected_edges = set()

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

    def enumerate_dags(self):
        pass

