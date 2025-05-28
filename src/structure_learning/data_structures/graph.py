from typing import Union, List, Tuple, Set, Type, TypeVar
from math import comb, factorial
import re
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

G = TypeVar('Graph')
class Graph:
    """
    Class wrapper for graphs.
    """
    def __init__(self, incidence: Union[np.ndarray, pd.DataFrame] = None, nodes: Union[List, Tuple] = None):
        """
        Initialise a graph instance.

        Parameters:
            incidence (np.ndarray | pd.DataFrame):      adjacency matrix
            nodes (list | tuple):                       node labels. If incidence is a dataframe, this parameter is ignored.
        """
        self.incidence, self.nodes = None, None
        if incidence is not None:
            self.incidence, self.nodes = (incidence.astype(bool), list(nodes)) if isinstance(incidence, np.ndarray) else (incidence.values.astype(bool), list(incidence.columns))
        self._edges = None
        self._node_to_index_dict = None

    @property
    def dim(self):
        """
        Returns the number of nodes in the Graph object
        """
        return len(self.nodes)
    
    @property
    def shape(self):
        return (len(self.nodes),len(self.nodes))

    @property
    def edges(self):
        """
        Returns list of edges
        """
        rows, columns = np.nonzero(self.incidence > 0)
        self._edges = set()
        for r,c in zip(rows, columns):
            self._edges.add((self.nodes[r], self.nodes[c]))
        return self._edges
    
    def _update_node_index(self):
        """
        Update node-index dictionary
        """

        self._node_to_index_dict = {} if self.nodes is None else {node:idx for idx,node in enumerate(self.nodes)}

    def _node_to_index(self, nodes):
        """
        Map node names to index in list
        """
        if self._node_to_index_dict is None:
            self._update_node_index()
        unwrap = not isinstance(nodes, list) and not isinstance(nodes, tuple)
        if unwrap:
            nodes = [nodes]
        idx = [self._node_to_index_dict.get(node) for node in nodes]
        return idx if len(idx) > 1 or not unwrap else idx[0]
    
    def find_parents(self, node, return_index=False):
        node_idx = self._node_to_index(node)
        if node_idx is None:
            return []
        parent_indices = np.nonzero(self.incidence[:, node_idx])[0].tolist()
        parents = np.array(self.nodes)[parent_indices]
        return (parents, parent_indices) if return_index else parents
    
    # modifiers
    def add_node(self, node: str):
        """
        Add node to graph.
        """
        self.add_nodes([node])

    def add_nodes(self, nodes: Union[List[str], Tuple[str]]):
        """
        Add nodes to graph.
        """
        nodes = set(nodes) - set(self.nodes)
        if len(nodes) == 0:
            return
        new_incidence = np.zeros((self.dim + len(nodes), self.dim + len(nodes)), dtype=bool)
        new_incidence[:self.dim,:self.dim] = self.incidence
        self.nodes.extend(nodes)
        self.incidence = new_incidence
        self._update_node_index()

    def remove_node(self, node: str):
        """
        Remove node from graph.
        """
        self.remove_nodes([node])

    def remove_nodes(self, nodes: Union[List[str], Tuple[str]]):
        """
        Remove nodes from graph.
        """
        idx = [i for i in self._node_to_index(nodes) if i is not None]
        new_incidence = np.delete(self.incidence, idx, axis=0)
        new_incidence = np.delete(new_incidence, idx, axis=1)
        self.incidence = new_incidence
        [self.nodes.pop(i) for i in idx]
        self._update_node_index()

    def add_edge(self, edge: Union[List[str], Tuple[str]]):
        """
        Add an edge to graph.
        """
        self.add_edges([edge])

    def add_edges(self, edges: Union[List[Tuple], Tuple[Tuple]]):
        """
        Add edges to graph.
        """
        nodes = set([node for edge in edges for node in edge])
        self.add_nodes(nodes)
        for edge in edges:
            node1, node2 = self._node_to_index(edge)
            self.incidence[node1, node2] = True
        self._update_node_index()

    def remove_edge(self, edge: Union[List, Tuple]):
        """
        Remove an edge from graph.
        """
        self.remove_edges([edge])

    def remove_edges(self, edges: Union[List[Tuple], Tuple[Tuple]]):
        """
        Remove edges from graph.
        """
        for edge in edges:
            node1, node2 = self._node_to_index(edge)
            self.incidence[node1, node2] = False

    # converters
    @classmethod
    def from_numpy(cls, incidence: np.ndarray, nodes: Union[List, Tuple, np.ndarray]) -> Type[G]:
        """
        Create Graph object from numpy array.
        """
        if len(incidence) != len(nodes):
            raise Exception("The number of node labels must match the dimensions of the graph")

        return Graph(incidence=incidence, nodes=nodes)
    
    @classmethod
    def from_pandas(cls, graph: pd.DataFrame) -> Type[G]:
        """
        Create Graph object from Pandas DataFrame.
        """
        nodes = list(graph.columns)
        if set(nodes) != set(graph.index):
            raise Exception("Column and index names should be similar")
        return Graph.from_numpy(graph.loc[nodes, nodes].values, nodes)
    
    @classmethod
    def from_nx(cls, graph: nx.DiGraph) -> Type[G]:
        """
        Create Graph object from networkx graph.
        """
        incidence = graph.to_numpy_array()
        nodes = list(graph.nodes)
        return Graph(incidence=incidence, nodes=nodes)
    
    def to_numpy(self, return_node_labels=False):
        """
        Convert Graph object to numpy array.
        """
        return self.incidence if not return_node_labels else (self.incidence, self.nodes)

    def to_pandas(self):
        """
        Convert Graph object to Pandas Dataframe.
        """
        return pd.DataFrame(self.incidence, index=self.nodes, columns=self.nodes)
    
    def to_nx(self):
        """
        Convert Graph object to networkx DiGraph.
        """
        G =  nx.from_numpy_array(self.incidence, create_using=nx.DiGraph)
        mapping = {old_label: new_label for old_label, new_label in zip(G.nodes(), self.nodes)}
        G = nx.relabel_nodes(G, mapping)

        return G
    
    def __str__(self):
        """
        Convert Graph object to string.
        """
        return self.to_key()

    def to_key(self, type:str = 'default'):
        """
        Create key for Graph object.

        Parameters:
            type (str): type of key to use
        """
        key = ''
        if self.incidence is not None:
            if type == 'default':
                num_nodes = str(self.incidence.shape[0])
                s = "".join(map(str, self.incidence.flatten().astype(int)))
                key = re.sub("(.{" + num_nodes + "})", "\\1 ", s)
                key = key if key[-1] != " " else key[:-1]
            else:
                raise Exception("Unsupported key type")
        return key

    def from_key(key: str, type:str = 'default', nodes: Union[List, Tuple, np.ndarray] = None) -> Type[G]:
        """
        Create graph from key.

        Parameters:
            key (str): string representation of the graph
            type:
            nodes (list | tuple | np.ndarray): list of nodes
        Returns:
            (numpy.ndarray)
        """
        num_nodes = len(nodes)
        if type == 'default':
            # Split the string by spaces
            s_list = key.split()

            # Convert each string in the list to a list of integers
            int_list = [[int(char) for char in string] for string in s_list]

            # Convert the list of lists to a numpy array
            array = np.array(int_list)
            return Graph(incidence=array.reshape(num_nodes, num_nodes), nodes=nodes)
        else:
            raise Exception("Unsupported key type")
        return None

    # arithmetic
    def __mul__(self, g: Type['G']):
        if self.nodes != g.nodes:
            raise Exception("Graph do not have matching nodes")
        product = self.incidence * g.incidence
        return Graph(product, self.nodes)

    def __rmul__(self, g: Type['G']):
        if self.nodes != g.nodes:
            raise Exception("Graph do not have matching nodes")
        product = g.incidence * self.incidence
        return Graph(product, self.nodes)
    
    def __eq__(self, g: Type['G']):
        if self.nodes != g.nodes:
            raise Exception("Graph do not have matching nodes")
        return (self.incidence == g.incidence).all()

    # I/O
    def to_csv(self, filename):
        self.to_pandas().to_csv(filename)

    @classmethod
    def from_csv(cls, filename):
        return cls.from_pandas(pd.read_csv(filename))

    # visualisation
    def plot(self, title="Graph", figsize = (5,3), node_size=2000, node_color="skyblue", k=5):
        """
        Plot a networkx graph.

        Parameters:
            title (str): figure title
            figsize (tuple (int)): figure size
            node_size (int): node size
            node_color (matplotlib.colors): node color
            k (int): distance between nodes
        """
        G = self.to_nx()
        pos = nx.spring_layout(G, k=k)

        plt.figure(figsize=figsize)
        nx.draw(G, with_labels=True, arrowsize=20, arrows=True, node_size=node_size,
                node_color=node_color, pos=pos)
        plt.gca().margins(0.20)
        plt.title(title)
        plt.axis("off")

    # utils
    def compare(self, other: Type[G], operation: str):
        """
        Checks edge operation on two adjacency matrices.

        Parameters:
            other (Graph): graph to compare with
            operation (str): edge operation [add_edge, delete_edge, reverse_edge]

        Returns:
            (str)|(list (str)): label(s) of node(s) connected to relevant edge(s)
                                under specified operation, or error message

        """
        g1, g2 = self.incidence, other.incidence
        if set(self.nodes) != set(other.nodes):
            raise Exception("The graphs have different set of nodes")
        
        node_idx = {idx:node for idx,node in enumerate(self.nodes)}
        if g1.shape != g2.shape:
            return "[ERROR] Graphs are not the same size"

        if operation == 'add_edge':
            diff = np.where((g1 == 0) & (g2 != 0))
            if len(diff[0]) > 0:
                return node_idx[diff[1][0]]

            return "[ERROR] No edge added"

        if operation == 'delete_edge':
            diff = np.where((g1 != 0) & (g2 == 0))
            if len(diff[0]) > 0:
                return node_idx[diff[1][0]]

            return "[ERROR] No edge deleted"

        if operation == 'reverse_edge':
            added_edges = np.where((g1 == 0) & (g2 != 0))
            deleted_edges = np.where((g1 != 0) & (g2 == 0))
            reversed_edges = list(set(zip(added_edges[1], added_edges[0])) & set(zip(deleted_edges[0], deleted_edges[1])))[0]

            if reversed_edges:
                return [node_idx[reversed_edges[0]], node_idx[reversed_edges[1]]]

            return "[ERROR] No edge reversed"

        return "[ERROR] Edge Operation Not Found"
    
    @classmethod
    def has_cycle(cls, graph: Union[np.ndarray, Type[G]]) -> bool:
        """
        Check if the graph represented by the adjacency matrix has a cycle.

        Parameters:
            graph (numpy.ndarray): adjacency matrix of the graph

        Returns:
            (bool): True if the graph has a cycle, False otherwise
        """
        adj_matrix = graph if isinstance(graph, np.ndarray) else graph.incidence
        def is_cyclic_util(v, visited, rec_stack, adj_matrix):
            visited[v] = True
            rec_stack[v] = True

            # Consider all adjacent vertices
            for i in range(len(adj_matrix)):
                if adj_matrix[v][i] != 0:
                    if not visited[i]:
                        if is_cyclic_util(i, visited, rec_stack, adj_matrix):
                            return True
                    elif rec_stack[i]:
                        return True

            rec_stack[v] = False
            return False

        num_vertices = len(adj_matrix)

        visited = [False] * num_vertices
        rec_stack = [False] * num_vertices

        for node in range(num_vertices):
            if not visited[node]:
                if is_cyclic_util(node, visited, rec_stack, adj_matrix):
                    return True

        return False
    