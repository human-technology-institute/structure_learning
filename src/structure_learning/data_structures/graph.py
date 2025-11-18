"""
This module provides the Graph class and associated methods for representing and manipulating graph structures.

The Graph class supports operations such as adding/removing nodes and edges, finding parent nodes, converting between different graph representations (numpy, pandas, networkx), and visualizing graphs.

It also includes utility methods for checking graph properties like cycles and performing edge operations.
"""

from typing import Union, List, Tuple, Type, TypeVar
import re
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pygraphviz as pgv

G = TypeVar('Graph')
class Graph:
    """
    Class wrapper for graphs.
    """
    def __init__(self, incidence: Union[np.ndarray, pd.DataFrame] = None, nodes: Union[List, Tuple] = None, weights: Union[np.ndarray, pd.DataFrame] = None):
        """
        Initialise a graph instance.

        Parameters:
            incidence (np.ndarray | pd.DataFrame): Adjacency matrix representing the graph structure.
            nodes (list | tuple): Node labels. If incidence is a DataFrame, this parameter is ignored.
            weights (np.ndarray | pd.DataFrame): Edge weights. Defaults to the adjacency matrix if not provided.
        """
        self.incidence, self.nodes = None, None
        if incidence is not None:
            self.incidence, self.nodes = (incidence, nodes) if isinstance(incidence, np.ndarray) else (incidence.values, list(incidence.columns))

        self.weights = weights
        if self.weights is None and self.incidence is not None:
            self.weights = self.incidence.copy()
            self.incidence = self.incidence!=0
        self._edges = None
        self._node_to_index_dict = None

    @property
    def dim(self):
        """
        Returns the number of nodes in the Graph object.

        Returns:
            int: Number of nodes.
        """
        return len(self.nodes)
    
    @property
    def shape(self):
        """
        Returns the shape of the adjacency matrix.

        Returns:
            tuple: Shape of the adjacency matrix (rows, columns).
        """
        return (len(self.nodes),len(self.nodes))

    @property
    def edges(self):
        """
        Returns the list of edges in the graph.

        Returns:
            set: Set of edges represented as tuples (node1, node2).
        """
        rows, columns = np.nonzero(self.incidence > 0)
        self._edges = set()
        for r,c in zip(rows, columns):
            self._edges.add((self.nodes[r], self.nodes[c]))
        return self._edges
    
    def v_structures(self):
        """
        Identifies v-structures in the graph.

        Returns:
            set: Set of v-structures represented as tuples (node1, node2, node3).
        """
        v = set()
        incidence = self.incidence
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                for k in range(self.dim):
                    if i==k or j==k:
                        continue
                    if (incidence[i,k] and incidence[k,i]) or (incidence[j,k] and incidence[k,j]):
                        continue
                    if incidence[i,k] and incidence[j,k] and not (incidence[i,j] or incidence[j,i]):
                        v.add((i,k,j))
        return v
    
    def has_edge(self, node1, node2, undirected=False) -> bool:
        """
        Checks if an edge exists in the graph.

        Parameters:
            node1 (str): First node label.
            node2 (str): Second node label.
            undirected (bool): If True, checks for an undirected edge (i.e., edge exists in either direction).

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        if self._node_to_index_dict is None:
            self._update_node_index()
        node1, node2 = self._node_to_index([node1, node2])
        directed_edge = self.incidence[node1, node2] if node1 is not None and node2 is not None else False
        if undirected:  # Check for undirected edge
            undirected_edge = self.incidence[node1, node2] or self.incidence[node2, node1] if node1 is not None and node2 is not None else False
            return undirected_edge
        return directed_edge
    
    def _update_node_index(self):
        """
        Updates the internal mapping of node names to indices.
        """

        self._node_to_index_dict = {} if self.nodes is None else {node:idx for idx,node in enumerate(self.nodes)}

    def _node_to_index(self, nodes):
        """
        Maps node names to their indices in the adjacency matrix.

        Parameters:
            nodes (str | list | tuple): Node names to map.

        Returns:
            int | list: Index or list of indices corresponding to the nodes.
        """
        if self._node_to_index_dict is None:
            self._update_node_index()
        unwrap = not isinstance(nodes, list) and not isinstance(nodes, tuple)
        if unwrap:
            nodes = [nodes]
        idx = [self._node_to_index_dict.get(node) for node in nodes]
        return idx if len(idx) > 1 or not unwrap else idx[0]
    
    def find_parents(self, node, return_index=False):
        """
        Finds the parent nodes of a given node.

        Parameters:
            node (str): The node for which to find parents.
            return_index (bool): If True, return indices of parent nodes along with their labels.

        Returns:
            list | tuple: List of parent node labels, or a tuple containing parent node labels and their indices if return_index is True.
        """
        node_idx = self._node_to_index(node)
        if node_idx is None:
            return []
        parent_indices = np.nonzero(self.incidence[:, node_idx])[0].tolist()
        parents = np.array(self.nodes)[parent_indices]
        return (parents, parent_indices) if return_index else parents
    
    @classmethod
    def find_parent_nodes(cls, incidence):
        """
        Finds parent nodes (nodes with no incoming edges).

        Parameters:
            incidence (np.ndarray): Adjacency matrix.

        Returns:
            set: Set of parent node indices.
        """
        n = len(incidence)
        parent_nodes = set(range(n))
        exist_edge = set([True, 1, 1.0])
        for i in range(n):
            for j in range(n):
                if incidence[j][i] in exist_edge:
                    parent_nodes.discard(i)
        return parent_nodes
    
    # modifiers
    def add_node(self, node: str):
        """
        Adds a single node to the graph.

        Parameters:
            node (str): Node label.
        """
        self.add_nodes([node])

    def add_nodes(self, nodes: Union[List[str], Tuple[str]]):
        """
        Adds multiple nodes to the graph.

        Parameters:
            nodes (list | tuple): List or tuple of node labels.
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
        Removes a single node from the graph.

        Parameters:
            node (str): Node label.
        """
        self.remove_nodes([node])

    def remove_nodes(self, nodes: Union[List[str], Tuple[str]]):
        """
        Removes multiple nodes from the graph.

        Parameters:
            nodes (list | tuple): List or tuple of node labels.
        """
        idx = [i for i in self._node_to_index(nodes) if i is not None]
        new_incidence = np.delete(self.incidence, idx, axis=0)
        new_incidence = np.delete(new_incidence, idx, axis=1)
        self.incidence = new_incidence
        [self.nodes.pop(i) for i in idx]
        self._update_node_index()

    def add_edge(self, edge: Union[List[str], Tuple[str]]):
        """
        Adds a single edge to the graph.

        Parameters:
            edge (list | tuple): Edge represented as a tuple (node1, node2).
        """
        self.add_edges([edge])

    def add_edges(self, edges: Union[List[Tuple], Tuple[Tuple]]):
        """
        Adds multiple edges to the graph.

        Parameters:
            edges (list | tuple): List or tuple of edges, each represented as a tuple (node1, node2).
        """
        nodes = set([node for edge in edges for node in edge])
        self.add_nodes(nodes)
        for edge in edges:
            node1, node2 = self._node_to_index(edge)
            self.incidence[node1, node2] = True
        self._update_node_index()

    def remove_edge(self, edge: Union[List, Tuple]):
        """
        Removes a single edge from the graph.

        Parameters:
            edge (list | tuple): Edge represented as a tuple (node1, node2).
        """
        self.remove_edges([edge])

    def remove_edges(self, edges: Union[List[Tuple], Tuple[Tuple]]):
        """
        Removes multiple edges from the graph.

        Parameters:
            edges (list | tuple): List or tuple of edges, each represented as a tuple (node1, node2).
        """
        for edge in edges:
            node1, node2 = self._node_to_index(edge)
            self.incidence[node1, node2] = False

    # converters
    @classmethod
    def from_numpy(cls, incidence: np.ndarray, nodes: Union[List, Tuple, np.ndarray] ) -> Type[G]:
        """
        Creates a Graph object from a numpy array.

        Parameters:
            incidence (np.ndarray): Adjacency matrix.
            nodes (list | tuple | np.ndarray): Node labels.

        Returns:
            Graph: Graph object.
        """
        if len(incidence) != len(nodes):
            raise Exception("The number of node labels must match the dimensions of the graph")

        return Graph(incidence=incidence, nodes=nodes)
    
    @classmethod
    def from_pandas(cls, graph: pd.DataFrame) -> Type[G]:
        """
        Creates a Graph object from a Pandas DataFrame.

        Parameters:
            graph (pd.DataFrame): DataFrame representing the adjacency matrix.

        Returns:
            Graph: Graph object.
        """
        nodes = list(graph.columns)
        if set(nodes) != set(graph.index):
            raise Exception("Column and index names should be similar")
        return Graph.from_numpy(graph.loc[nodes, nodes].values, nodes)
    
    @classmethod
    def from_nx(cls, graph: nx.DiGraph) -> Type[G]:
        """
        Creates a Graph object from a networkx graph.

        Parameters:
            graph (nx.DiGraph): networkx DiGraph object.

        Returns:
            Graph: Graph object.
        """
        incidence = graph.to_numpy_array()
        nodes = list(graph.nodes)
        return Graph(incidence=incidence, nodes=nodes)
    
    def to_numpy(self, return_node_labels=False):
        """
        Converts the Graph object to a numpy array.

        Parameters:
            return_node_labels (bool): If True, return node labels along with the adjacency matrix.

        Returns:
            np.ndarray | tuple: Adjacency matrix, or a tuple containing the adjacency matrix and node labels.
        """
        return self.incidence if not return_node_labels else (self.incidence, self.nodes)

    def to_pandas(self):
        """
        Converts the Graph object to a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame representing the adjacency matrix.
        """
        return pd.DataFrame(self.incidence, index=self.nodes, columns=self.nodes)
    
    def to_nx(self):
        """
        Converts the Graph object to a networkx DiGraph.

        Returns:
            nx.DiGraph: networkx DiGraph object.
        """
        G =  nx.from_numpy_array(self.incidence, create_using=nx.DiGraph)
        mapping = {old_label: new_label for old_label, new_label in zip(G.nodes(), self.nodes)}
        G = nx.relabel_nodes(G, mapping)

        return G
    
    # pickle
    def save(self, filename: str, compression='gzip'):
        """
        Saves the Graph object to a file.

        Parameters:
            filename (str): Path to the output file.
        """
        with open(filename, 'wb') as f:
            import compress_pickle as pickle
            pickle.dump(self, f, compression=compression)

    @classmethod
    def load(cls, filename: str, compression='gzip') -> Type[G]:
        """
        Loads a Graph object from a file.

        Parameters:
            filename (str): Path to the input file.

        Returns:
            Graph: Loaded Graph object.
        """
        with open(filename, 'rb') as f:
            import compress_pickle as pickle
            return pickle.load(f, compression=compression)
    
    def __str__(self):
        """
        Converts the Graph object to a string representation.

        Returns:
            str: String representation of the graph.
        """
        return self.to_key()

    def to_key(self, type: str = 'default'):
        """
        Creates a key for the Graph object.

        Parameters:
            type (str): Type of key to use. Defaults to 'default'.

        Returns:
            str: Key representing the graph.
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

    @classmethod
    def from_key(cls, key: str, type: str = 'default', nodes: Union[List, Tuple, np.ndarray] = None) -> Type[G]:
        """
        Creates a Graph object from a key.

        Parameters:
            key (str): String representation of the graph.
            type (str): Type of key. Defaults to 'default'.
            nodes (list | tuple | np.ndarray): Node labels.

        Returns:
            Graph: Graph object.
        """
        num_nodes = len(nodes)
        if type == 'default':
            # Split the string by spaces
            s_list = key.split()

            # Convert each string in the list to a list of integers
            int_list = [[int(char) for char in string] for string in s_list]

            # Convert the list of lists to a numpy array
            array = np.array(int_list)
            return cls(incidence=array.reshape(num_nodes, num_nodes), nodes=nodes)
        else:
            raise Exception("Unsupported key type")
        return None

    # arithmetic
    def __mul__(self, g: Type['G']):
        """
        Multiplies two Graph objects element-wise.

        Parameters:
            g (Graph): Another Graph object.

        Returns:
            Graph: Resulting Graph object.
        """
        if self.nodes != g.nodes:
            raise Exception("Graph do not have matching nodes")
        product = self.incidence * g.incidence
        return Graph(product, self.nodes)

    def __rmul__(self, g: Type['G']):
        """
        Multiplies two Graph objects element-wise (reverse operation).

        Parameters:
            g (Graph): Another Graph object.

        Returns:
            Graph: Resulting Graph object.
        """
        if self.nodes != g.nodes:
            raise Exception("Graph do not have matching nodes")
        product = g.incidence * self.incidence
        return Graph(product, self.nodes)
    
    def __eq__(self, g: Type['G']):
        """
        Checks equality of two Graph objects.

        Parameters:
            g (Graph): Another Graph object.

        Returns:
            bool: True if the graphs are equal, False otherwise.
        """
        if self.nodes != g.nodes:
            raise Exception("Graph do not have matching nodes")
        return (self.incidence == g.incidence).all()

    # I/O
    def to_csv(self, filename):
        """
        Saves the Graph object to a CSV file.

        Parameters:
            filename (str): Path to the output CSV file.
        """
        self.to_pandas().to_csv(filename)

    def __copy__(self):
        g = self.__class__(incidence=self.incidence.copy(), nodes=self.nodes.copy())
        g.weights = self.weights.copy() if self.weights is not None else None
        return g 
    
    def copy(self):
        return self.__copy__()

    @classmethod
    def from_csv(cls, filename):
        """
        Creates a Graph object from a CSV file.

        Parameters:
            filename (str): Path to the input CSV file.

        Returns:
            Graph: Graph object.
        """
        return cls.from_pandas(pd.read_csv(filename))

    # visualisation
    def plot(self, filename=None, text=None, edge_colors: dict = None, edge_weights: dict = None, node_clusters: dict = None, max_penwidth: int =5, show_weights: bool = False):
        """
        Plot a networkx graph.

        Parameters:
            filename (str): Path to save the plot image. If None, the plot is not saved.
            text (str): Additional text to display on the plot.
            edge_colors (dict): Dictionary mapping edges to colors.
            edge_weights (dict): Dictionary mapping edges to weights.
        """
        G = self.to_nx()
        G_gvz = nx.nx_agraph.to_agraph(G)
        if text is not None:
            G_gvz.add_node('info',label=text, shape='note', style='filled', fillcolor='lightgrey')
        if edge_weights is not None:
            w = [abs(v) for v in edge_weights.values()]
            w_min, w_max = min(w), max(w)
        for r,c in zip(*np.nonzero(self.incidence)):
            if  G_gvz.has_edge(self.nodes[r], self.nodes[c]):
                edge = G_gvz.get_edge(self.nodes[r], self.nodes[c])
                if self.incidence[c,r] and G_gvz.has_edge(self.nodes[c], self.nodes[r]):
                    G_gvz.remove_edge(self.nodes[c], self.nodes[r])
                    edge = G_gvz.get_edge(self.nodes[r], self.nodes[c])
                    edge.attr['dir'] = 'none'
                else:
                    if edge_colors is not None and ((self.nodes[r], self.nodes[c]) in edge_colors and (self.nodes[c], self.nodes[r]) not in edge_colors):
                        edge.attr['color'] = edge_colors[(self.nodes[r], self.nodes[c])]
                        edge.attr['arrowhead'] = 'tee' if edge_colors[(self.nodes[r], self.nodes[c])]=="#FE5600" else 'vee'
                    if edge_weights is not None:
                        if (self.nodes[r], self.nodes[c]) in edge_weights and show_weights:
                            edge.attr['label'] = str(round(edge_weights[(self.nodes[r], self.nodes[c])], 2))
                        edge.attr['penwidth'] = (max_penwidth - 1)*(np.abs(edge_weights[(self.nodes[r], self.nodes[c])] if (self.nodes[r], self.nodes[c]) in edge_weights else edge_weights[(self.nodes[c], self.nodes[r])]) - w_min + 1e-7)/(w_max - w_min + 1e-7) + 1
        if node_clusters is not None:
            for cluster_id, nodes in node_clusters.items():
                G_gvz.add_subgraph(nodes, name=f'Cluster {cluster_id}', style='filled', color='lightgrey')
        G_gvz.layout('dot')
        if filename is not None:
            G_gvz.draw(filename, format='png')
        return G_gvz

    # utils
    def compare(self, other: Type[G], operation: str):
        """
        Compares two Graph objects based on edge operations.

        Parameters:
            other (Graph): Another Graph object.
            operation (str): Edge operation ('add_edge', 'delete_edge', 'reverse_edge').

        Returns:
            str | list: Labels of nodes connected to relevant edges under the specified operation, or an error message.
        """
        g1, g2 = self.incidence.astype(int), other.incidence.astype(int)
        if set(self.nodes) != set(other.nodes):
            raise Exception("The graphs have different set of nodes")
        
        node_idx = {idx:node for idx,node in enumerate(self.nodes)}
        if g1.shape != g2.shape:
            return "[ERROR] Graphs are not the same size"
        diff = g1 - g2

        if operation == 'add_edge':
            added_edges = np.where(diff < 0)
            if len(added_edges[0]) > 0:
                return [(node_idx[r],node_idx[c]) for r,c in zip(added_edges[0], added_edges[1])]

            return "[ERROR] No edge added"

        if operation == 'delete_edge':
            deleted_edges = np.where(diff > 0)
            if len(deleted_edges[0]) > 0:
                return [(node_idx[r],node_idx[c]) for r,c in zip(deleted_edges[0], deleted_edges[1])]

            return "[ERROR] No edge deleted"

        if operation == 'reverse_edge':
            added_edges = np.where(diff < 0)
            deleted_edges = np.where(diff > 0)
            reversed_edges = list(set(zip(added_edges[1], added_edges[0])) & set(zip(deleted_edges[0], deleted_edges[1])))

            if reversed_edges:
                return [(node_idx[r],node_idx[c]) for r,c in reversed_edges]

            return "[ERROR] No edge reversed"

        return "[ERROR] Edge Operation Not Found"
    
    @classmethod
    def has_cycle(cls, graph: Union[np.ndarray, Type[G]]) -> bool:
        """
        Checks if the graph has a cycle.

        Parameters:
            graph (np.ndarray | Graph): Adjacency matrix or Graph object.

        Returns:
            bool: True if the graph has a cycle, False otherwise.
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

