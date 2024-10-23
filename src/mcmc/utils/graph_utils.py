"""

"""
import itertools
import re
import zipfile
from collections import Counter
from math import comb, factorial
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import igraph
import pcalg
from conditional_independence import partial_correlation_suffstat, partial_correlation_test
import matplotlib.pyplot as plt

def count_dags(n : int):
    """
    Count all possible DAGs. Given $n$ nodes, the possible number of DAGs
    that can be built is given by
        a(n) = \sum_{k=1}^{n} (-1)^{(k+1)} \binom{n}{k} 2^{k(n-k)} a(n-k)

    Parameter:
        n (int): number of nodes

    Returns:
        (int): number of possible directed graphs
    """
    if n == 0:
        return 1

    total = 0
    for k in range(1, n + 1):
        total += (-1)**(k+1) * comb(n, k) * (2**(k*(n-k))) * count_dags(n-k)
    return total

def update_matrix(matrix1, matrix2):
    """
    Update matrix1 with the values from matrix2 only where matrix1 is zero.

    Parameters:
        matrix1 (numpy.ndarray): First matrix to update.
        matrix2 (numpy.ndarray): Second matrix with values to use for updating.

    Returns:
        (numpy.ndarray): Updated matrix
    """
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    # Check if both matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same dimensions.")

    # Update matrix1 only where its elements are zero
    matrix1[matrix1 == 0] = matrix2[matrix1 == 0]

    return matrix1

def convert_pkl_graph_to_csv(pickle_path : str):
    """
    Save pickled object to csv.

    Parameter:
        pickle_path (str): Path to pickle file
    """
    with open(pickle_path, 'rb') as pickle_file:
        graph = pickle.load(pickle_file)

    adjacency_matrix = graph_to_adjmatrix(graph).astype(int)
    df = pd.DataFrame(adjacency_matrix, columns=list(graph.nodes()))
    csv_path = pickle_path.replace('.pkl', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Graph saved to {csv_path}")

def save_graphs(graphs: list, directory: str, filename : str, num_nodes : int):
    """
    Save graphs as zipped GraphML files

    Parameter:
        graphs (list): a list of graphs nx.DiGraph
        filename (str): a string representing the filename to be saved
        num_nodes (int): number of nodes in graphs
    """

    # Save each graph to a separate GraphML file
    for idx, graph in enumerate(graphs):
        nx.write_graphml(graph, f"{directory}/{num_nodes}nodes/graph_{idx}.graphml")

    # Create a ZIP archive containing all the GraphML files
    with zipfile.ZipFile(f"{directory}/{num_nodes}nodes/{filename}.zip", "w") as zipf:
        for idx in range(len(graphs)):
            zipf.write(f"{directory}/{num_nodes}nodes/graph_{idx}.graphml")

def generate_all_dags_from_ordering(nodes : list):
    """
    Generate all DAGs from a topological ordering of nodes.

    Parameter:
        nodes (list): nodes in topological order

    Returns:
        A generator of DAGs
    """
    # Generate all permutations of edges based on the given topological order
    all_edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    unique_graphs = set()

    for edges in itertools.chain.from_iterable(itertools.combinations(all_edges, r) for r in range(len(all_edges)+1)):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        # Check for cycles, since we only want DAGs
        if not nx.is_directed_acyclic_graph(G):
            continue

        # Generate a sorted adjacency matrix as a tuple and check if it's already in our set
        adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        matrix_tuple = tuple(map(tuple, adj_matrix))
        if matrix_tuple not in unique_graphs:
            unique_graphs.add(matrix_tuple)
            yield G

def collect_node_scores(graph_score):
    """
    Collect node scores as dictionary with nodes as keys.

    Parameter:
        graph_score (dict): dictionary of score information generated from Score objects

    Returns:
        (dict): dictionary with nodes as keys and scores as values
    """
    scores = {node: info['score'] for node, info in graph_score['parameters'].items()}
    return scores


def compare_graphs(g1: np.ndarray, g2: np.ndarray, op, index_to_node_label):
    """
    Checks edge operation on two adjacency matrices.

    Parameters:
        g1 (numpy.ndarray): adjacency matrix of first graph
        g2 (numpy.ndarray): adjacency matrix of second graph
        op (str): edge operation [add_edge, delete_edge, reverse_edge]
        index_to_node_label (dict)

    Returns:
        (str)|(list (str)): label(s) of node(s) connected to relevant edge(s)
                            under specified operation, or error message

    """
    if g1.shape != g2.shape:
        return "[ERROR] Graphs are not the same size"

    if op == 'add_edge':
        diff = np.where((g1 == 0) & (g2 != 0))
        if len(diff[0]) > 0:
            return index_to_node_label[diff[1][0]]

        return "[ERROR] No edge added"

    if op == 'delete_edge':
        diff = np.where((g1 != 0) & (g2 == 0))
        if len(diff[0]) > 0:
            return index_to_node_label[diff[1][0]]

        return "[ERROR] No edge deleted"

    if op == 'reverse_edge':
        added_edges = np.where((g1 == 0) & (g2 != 0))
        deleted_edges = np.where((g1 != 0) & (g2 == 0))
        reversed_edges = list(set(zip(added_edges[1], added_edges[0])) & set(zip(deleted_edges[0], deleted_edges[1])))[0]

        if reversed_edges:
            return [index_to_node_label[reversed_edges[0]], index_to_node_label[reversed_edges[1]]]

        return "[ERROR] No edge reversed"

    return "[ERROR] Edge Operation Not Found"

def has_cycle(adj_matrix):
    """
    Check if the graph represented by the adjacency matrix has a cycle.

    Parameters:
        adj_matrix (numpy.ndarray): adjacency matrix of the graph

    Returns:
        (bool): True if the graph has a cycle, False otherwise
    """
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

def generate_dag_from_ordering(ordering: list, node_label_to_index: dict, edge_prob: float = 0.5):
    """
    Generate a random DAG from topological ordering.

    Parameters:
        ordering (list): node topological ordering
        node_label_to_index (dict): node-index mapping
        edge_prob (float): probability of an edge being present in the generated graph

    Returns:
        (numpy.ndarray): adjacency_matrix
    """
    node_labels = list(node_label_to_index.keys())
    num_nodes = len(node_labels)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                # Convert node names to indices
                index_i = node_label_to_index[ordering[i]]
                index_j = node_label_to_index[ordering[j]]

                # Add directed edge in the adjacency matrix
                adjacency_matrix[index_i, index_j] = 1

    return adjacency_matrix

def get_adjacency_matrix(G: nx.DiGraph):
    """
    Get DataFrame adjacency from networkx graph.

    Parameter:
        G (networkx.DiGraph): graph

    Returns:
        (pandas.DataFrame): adjacency matrix
    """
    adj_matrix = nx.adjacency_matrix(G)
    dense_adj_matrix = adj_matrix.toarray()

    nodes = list(G.nodes())
    adj_df = pd.DataFrame(dense_adj_matrix, index=nodes, columns=nodes)

    return adj_df

def intersection(d1, d2):
    """
    Intersection of two dictionaries.

    Parameters:
        d1 (dict): first dictionary
        d2 (dict): second dictionary

    Returns:

    """
    d2_cp = d2.copy()
    d1_keys = set(d1.keys())
    d2_keys = set(d2_cp.keys())
    shared_keys = d1_keys.intersection(d2_keys)

    # sum the values of d1
    total = sum(d1.values())

    for key in shared_keys:
        d2_cp[key] = d1[key] / total

    return d2_cp

def update_graph_frequencies(graph_list: list, result_index: dict):
    """
    Given a list of graphs, returns a dictionary with the number of times a graph occurs

    Parameters:
        graph_list (list): List of graphs.
        dag_dict (dict): Dictionary to be updated with graph frequencies.

    Returns:
        (dict): Updated dictionary with normalized frequencies.
    """
    # Count occurrences of each graph string
    graph_str_counter = Counter(generate_key_from_adj_matrix(graph) for graph in graph_list)

    # convert graph_str_counter to a dictionary
    graph_str_dict = dict(graph_str_counter)
    result = intersection(graph_str_dict, result_index)

    return result

def load_pickle_file(filename : str):
    """
    Load pickled object.

    Parameter:
        filename (str): name of pickle file
    """
    with open( filename, "rb" ) as f:
        res = pickle.load(f)
    return res

def edges_to_adj_np(edges, node_labels):
    """
    Create adjacency matrix from a list of edges.

    Parameters:
        edges (list (tuple)): list of graph edges
        node_labels (list):  node labels

    Returns:
        (numpy.ndarray): adjacency matrix
    """
    # Create a copy of the graph to avoid modifying the original
    G_temp = nx.DiGraph()

    G_temp.add_nodes_from(node_labels)

    # Add edges to the temporary graph
    G_temp.add_edges_from(edges)

    # Check if the node sets match
    if set(node_labels) != set(G_temp.nodes()):
        raise ValueError("The node sets of the given graph and edge list do not match.")

    # Create an adjacency matrix with the order of nodes from G
    adj_matrix = nx.to_numpy_array(G_temp, nodelist=node_labels)

    return adj_matrix.astype(int)

def plot_graph(G : nx.DiGraph, title="Graph", figsize = (5,3), node_size=2000, node_color="skyblue", k=5):
    """
    Plot a networkx graph.

    Parameters:
        G (networkx.DiGraph): graph
        title (str): figure title
        figsize (tuple (int)): figure size
        node_size (int): node size
        node_color (matplotlib.colors): node color
        k (int): distance between nodes
    """
    pos = nx.spring_layout(G, k=k)

    plt.figure(figsize=figsize)
    nx.draw(G, with_labels=True, arrowsize=20, arrows=True, node_size=node_size,
            node_color=node_color, pos=pos)
    plt.gca().margins(0.20)
    plt.title(title)
    plt.axis("off")

def plot_graph_from_adj_mat(adj_matrix : np.ndarray , node_labels : list, title="Graph",
                            figsize = (5,3), node_size=2000, node_color="skyblue", k=5):
    """
    Plot graph from adjacency matrix.

    Parameters:
        adj_matrix (numpy.ndarray): graph adjacency matrix
        node_labels (list (str)): node labels
        title (str): figure title
        figsize (tuple (int)): figure size
        node_size (int): node size
        node_color (matplotlib.colors): node color
        k (int): distance between nodes
    """
    key = generate_key_from_adj_matrix( adj_matrix )
    G = generate_graph_from_key( key, node_labels)

    plot_graph(G = G, title=title, figsize = figsize, node_size=node_size, node_color=node_color, k=k)

def compute_ancestor_matrix(adj_matrix : np.ndarray):
    """
    Compute ancestor matrix from adjacency matrix.

    Parameter:
        adj_matrix (numpy.ndarray): graph adjancency matrix

    Returns:
        (numpy.ndarray): ancestor matrix
    """
    num_nodes = adj_matrix.shape[0]

    # Initialize the ancestor matrix as the adjacency matrix
    ancestor_matrix = np.copy(adj_matrix)

    # Compute powers of the adjacency matrix and update the ancestor matrix
    power_matrix = np.copy(adj_matrix)
    for _ in range(num_nodes - 1):
        power_matrix = np.dot(power_matrix, adj_matrix)
        # If a path exists (i.e., value > 0), set it to 1
        power_matrix[power_matrix > 0] = 1
        ancestor_matrix = np.logical_or(ancestor_matrix, power_matrix).astype(int)

        res = ancestor_matrix.tolist()
        res = np.array(res)
        res = res.T

    return res

def edges_to_adjacency_matrix(edges, G):
    """
    Convert edges to adjacency matrix.

    Parameters:
        edges (list (tuple)): list of edges to add
        G (networkx.DiGraph): graph

    Returns:
        (numpy.ndarray): adjacency matrix
    """
    # Create a copy of the graph to avoid modifying the original
    G_temp = nx.DiGraph()

    G_temp.add_nodes_from(G.nodes())

    # Add edges to the temporary graph
    G_temp.add_edges_from(edges)

    # Check if the node sets match
    if set(G.nodes()) != set(G_temp.nodes()):
        raise ValueError("The node sets of the given graph and edge list do not match.")

    # Create an adjacency matrix with the order of nodes from G
    adj_matrix = nx.to_numpy_array(G_temp, nodelist=G.nodes())

    return adj_matrix.astype(int)

def graph_to_adjmatrix(G):
    """
    Convert a networkx graph to an adjacency matrix.

    Parameter:
        G (nx.DiGraph): a directed graph

    Returns:
        (numpyp.ndarray): adjacency matrix
    """
    return nx.to_numpy_array(G)

def generate_key_from_graph(G : nx.DiGraph):
    """
    Generate a unique key from a graph.

    Parameter:
        G (nx.DiGraph): Directed graph.

    Returns:
        (str): Unique key for the graph.
    """
    num_nodes = str(len(G.nodes()))

    s = "".join(map(str, nx.to_numpy_array(G).flatten().astype(int)))

    # Insert a space every three characters
    key = re.sub("(.{" + num_nodes + "})", "\\1 ", s)
    key = key if key[-1] != " " else key[:-1]
    return key

def generate_edges_from_adj_matrix(adj_matrix):
    """
    Generate list of edges (as tuples) of the graph from adjacency matrix.

    Parameter:
        adj_matrix (numpy.ndarray): graph adjacency matrix

    Returns:
        (list (tuple (int))): list of edges
    """
    edges = []
    for i in range( adj_matrix.shape[0] ):
        for j in range( adj_matrix.shape[1] ):
            if adj_matrix[i, j] == 1:
                edges.append((i, j))
    return edges


def generate_key_from_adj_matrix(adj_matrix):
    """
    Generate a unique key from an adjacency matrix.

    Parameter:
        adj_matrix (np.ndarray): Adjacency matrix.

    Returns:
        str: Unique key for the adjacency matrix.
    """

    num_nodes = str(adj_matrix.shape[0])

    # Convert the graph to a string
    s = "".join(map(str, adj_matrix.flatten().astype(int)))

    # Insert a space every three characters
    key = re.sub("(.{" + num_nodes + "})", "\\1 ", s)
    key = key if key[-1] != " " else key[:-1]
    return key

def plot_igraph(g : igraph, title="Graph", figsize = (5,3), node_size=2000, node_color="skyblue", k=5):
    """
    Plot igraph graph.

    Parameters:
        g (igraph): graph
        title (str): figure title
        figsize (tuple (int)): figure size
        node_size (int): node size
        node_color (str): node color
        k (int): distance between nodes
    """
    g_nx = nx.DiGraph(g.get_edgelist(), create_using=nx.DiGraph, )
    mapping = {old_label: "X" + str(old_label) for old_label in g_nx.nodes()}
    g_nx = nx.relabel_nodes(g_nx, mapping)
    plot_graph(g_nx,title=title, figsize = figsize, node_size=node_size, node_color=node_color, k=k)

def generate_graph_from_key(key, node_labels):
    """
    Create networkx graph from key. Inverse of generate_key_from_graph.

    Parameters:
        key (str): string representation of the graph
        node_labels (list (str)): node labels

    Returns:
        (networkx.DiGraph)
    """
    # Create a graph from the adjacency matrix
    adj_matrix = generate_adj_matrix_from_key( key, len(node_labels) )
    G =  nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Create a mapping from old labels to new labels
    mapping = {old_label: new_label for old_label, new_label in zip(G.nodes(), node_labels)}

    # Relabel the nodes
    G = nx.relabel_nodes(G, mapping)
    return G

def generate_adj_matrix_from_key(key, num_nodes):
    """
    Create adjacency matrix from key. Inverse of generate_key_from_adj_matrix.

    Parameters:
        key (str): string representation of the graph
        num_nodes (int): number of nodes
    Returns:
        (numpy.ndarray)
    """
    # Split the string by spaces
    s_list = key.split()

    # Convert each string in the list to a list of integers
    int_list = [[int(char) for char in string] for string in s_list]

    # Convert the list of lists to a numpy array
    array = np.array(int_list)
    return array.reshape(num_nodes, num_nodes)

def inspect_graph_from_index(distr : dict, indx : int, node_labels : list = None):
    """
    Plot graph from graph distribution.

    Parameters:
        distr (dict): graph distribution
        indx (int): index of graph to plot
        node_labels (list): node labels
    """
    key = list(distr.keys())[indx]

    # if node_labels is not None, then use the node labels
    if node_labels is not None:
        G = generate_graph_from_key(key, node_labels)
    else:
        node_labels = [f"X{i}" for i in range(int(np.sqrt(len(key))))]
        G = generate_graph_from_key(key, node_labels)
    plot_graph(G)

def inspect_graph_from_key(key: str, node_labels : list = None):
    """
    Plot graph from string representation.

    Parameters:
        key (str): string representation of graph
        node_labels (list): node labels
    """
    # if node_labels is not None, then use the node labels
    if node_labels is not None:
        G = generate_graph_from_key(key, node_labels )
    else:
        node_labels = [f"X{i}" for i in range(int(np.sqrt(len(key))))]
        G = generate_graph_from_key(key, node_labels )
    plot_graph(G)

# make a function that creates an identity matrix of size N
def create_identity_matrix(N):
    """
    Create an NxN identity matrix.

    Parameters:
        N (int): dimension of the matrix

    Returns:
        (numpy.ndarray): identity matrix
    """
    return np.identity(N).astype(int)

# create a function that creates a NxN matrix with all 1s
def create_ones_matrix(N):
    """
    Create an NxN ones matrix.

    Parameters:
        N (int): dimension of the matrix

    Returns:
        (numpy.ndarray): ones matrix
    """
    return np.ones((N,N)).astype(int)

def create_zeros_matrix(N):
    """
    Create an NxN zero matrix.

    Parameters:
        N (int): dimension of the matrix

    Returns:
        (numpy.ndarray): zero matrix
    """
    return np.zeros((N,N)).astype(int)

def convert_adj_mat_to_graph(incidence: pd.DataFrame):
    """
    Convert pandas DataFrame adjacency matrix to networkx graph.

    Parameters:
        incidence (pandas.DataFrame): adjacency matrix

    Returns:
        (networkx.DiGraph): networkx graph
    """
    labels = list(incidence.columns)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(labels)

    # Add edges based on adjacency matrix
    for i in range(incidence.shape[0]):
        for j in range(incidence.shape[1]):
            if incidence.iloc[i, j] == 1:
                G.add_edge(labels[i], labels[j])

    return G

def convert_adj_mat_np_to_graph(incidence: np.ndarray, labels : list):
    """
    Convert numpy adjacency matrix to networkx graph.

    Parameters:
        incidence (numpy.ndarray): adjacency matrix
        labels (list): node labels

    Returns:
        (networkx.DiGraph): networkx graph
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(labels)

    # Add edges based on adjacency matrix
    for i in range(incidence.shape[0]):
        for j in range(incidence.shape[1]):
            if incidence[i, j] == 1:
                G.add_edge(labels[i], labels[j])
    return G

def add_edges_to_graph(edges, G_init):
    """
    Add edges to existing networkx graph.

    Parameters:
        edges (list (tuple)): list of edges to add
        G_init (networkx.DiGraph): graph

    Returns:
        (networkx.DiGraph): new graph with added edges
    """
    G = G_init.copy()
    G.add_edges_from(edges)
    return G


def remove_edges_from_graph(edges, G_init):
    """
    Remove edges from graph.

    Parameters:
        edges (list (tuple)): list of edges to remove
        G_init (networkx.DiGraph): graph

    Returns:
        (networkx.DiGraph): new graph
    """
    G = G_init.copy()
    G.remove_edges_from(edges)
    return G


def count_parents(adj_matrix, node):
    """
    Count the number of parents for a given node in a directed graph.

    Parameters
        adj_matrix (numpy.ndarray): adjacency matrix of the graph
        node (int): index of the node to count parents for

    Returns:
        (int): number of parents (incoming edges) of the specified node
    """
    if node < 0 or node >= adj_matrix.shape[0]:
        raise ValueError("Node index out of bounds")

    # Count the non-zero entries in the node's column
    return np.count_nonzero(adj_matrix[:, node])

def find_parents(adj_matrix, node):
    """
    Find the indices of parent nodes for a given node in a directed graph.

    Parameters:
        adj_matrix (numpy.ndarray): adjacency matrix of the graph
        node (int): index of the node to find parents for

    Returns:
        (list (int)): list of indices of parent nodes of the specified node
    """
    if node < 0 or node >= adj_matrix.shape[0]:
        raise ValueError("Node index out of bounds")

    # Find the indices of non-zero entries in the node's column
    parent_indices = np.nonzero(adj_matrix[:, node])[0]
    return parent_indices.tolist()

def initialize_adj_matrix(edges, N):
    """
    Initialise graph adjacency matrix from list of edges.

    Parameters:
        edges (list (tuple (int))): list of graph edges
        N (int): number of nodes

    Returns:
        (numpy.ndarray): adjacency matrix
        (list (int)): in-degree of each node
    """
    adjMatrix = [[0 for _ in range(N)] for _ in range(N)]
    indegree = [0] * N

    for (src, dest) in edges:
        adjMatrix[src][dest] = 1
        indegree[dest] += 1

    return adjMatrix, indegree

def findAllTopologicalOrders(adjMatrix, indegree, path, discovered, N, results, indx_to_node):
    """

    Parameters:

    """
    for v in range(N):
        if indegree[v] == 0 and not discovered[v]:
            for u in range(N):
                if adjMatrix[v][u] == 1:
                    indegree[u] -= 1

            path.append(indx_to_node[v])
            discovered[v] = True
            findAllTopologicalOrders(adjMatrix, indegree, path, discovered, N, results, indx_to_node)

            for u in range(N):
                if adjMatrix[v][u] == 1:
                    indegree[u] += 1

            path.pop()
            discovered[v] = False

    if len(path) == N:
        results.append(path.copy())

def all_valid_orderings(adjMatrix, indx_to_node):
    """

    Parameters:

    Returns:

    """
    N = len(adjMatrix)
    indegree = calculate_in_degree(adjMatrix)

    discovered = [False] * N
    path = []
    results = []

    findAllTopologicalOrders(adjMatrix, indegree, path, discovered, N, results, indx_to_node)
    return results

def calculate_in_degree(adjMatrix):
    """
    Compute in-degree for each node in a graph.

    Parameters:
        adjMatrix (numpy.ndarray): graph adjacency matrix

    Returns:
        (list (int)): node in-degree
    """
    N = len(adjMatrix)  # Assuming the matrix is square
    in_degree = [0] * N

    for j in range(N):
        for i in range(N):
            if adjMatrix[i][j] > 0:  # If there's an edge from i to j
                in_degree[j] += 1

    return in_degree

def generate_DAG(N : int, prob : float , random_seed: int = None):
    """
    Generate a random DAG, represented by a lower triangular adjacency matrix.

    Parameters:
        N (int): number of nodes
        prob (float): edge probability
        random_seed (int): seed for numpy RNG

    Returns:
        (numpy.ndarray): adjacency matrix

    """
    if random_seed is not None:
        np.random.seed(random_seed)
    adjmat = np.zeros((N, N))
    adjmat[np.tril_indices_from(adjmat, k=-1)] = np.random.binomial(1, prob, size=int(N * (N - 1) / 2))
    return adjmat

def rDAG(n : int, p : float , labels : str, random_seed: int = 42):
    """
    Generate a random DAG, represented by a lower triangular adjacency matrix (DataFrame).

    Parameters:
        N (int): number of nodes
        prob (float): edge probability
        labels (list): node labels
        random_seed (int): seed for numpy RNG

    Returns:
        (pandas.DataFrame): adjacency matrix
    """
    np.random.seed(random_seed)
    adjmat = np.zeros((n, n))
    adjmat[np.tril_indices_from(adjmat, k=-1)] = np.random.binomial(1, p, size=int(n * (n - 1) / 2))
    return pd.DataFrame(adjmat, columns=labels,index=labels)

def count_all_valid_orderings(graph : np.ndarray):
    """
    Count all the valid node orderings given a graph.

    Parameters:
        graph (numpy.ndarray): graph adjacency matrix

    Returns:
        (int): number of valid node orderings
    """
    # Convert adjacency matrix to adjacency list for easier manipulation
    n = len(graph)
    adj_list = {i: [] for i in range(n)}
    in_degree = [0] * n

    # if the graph has no edges, then return N!
    if all([all([not graph[i][j] for j in range(n)]) for i in range(n)]):
        return factorial(n)

    for i in range(n):
        for j in range(n):
            if graph[i][j] == 1:
                adj_list[i].append(j)
                in_degree[j] += 1

    # Helper function to use backtracking for counting orderings
    def count_orderings(visited, in_degree):
        # Check if all nodes are visited, meaning a valid ordering is found
        if all(visited):
            return 1
        count = 0
        for node in range(n):
            # If in_degree is 0 and node is not visited, choose it next
            if in_degree[node] == 0 and not visited[node]:
                # Choose the node, update visited and in_degree
                visited[node] = True
                for next_node in adj_list[node]:
                    in_degree[next_node] -= 1

                # Continue with the next choice
                count += count_orderings(visited, in_degree)

                # Backtrack: undo the choice
                visited[node] = False
                for next_node in adj_list[node]:
                    in_degree[next_node] += 1
        return int(count)

    return count_orderings([False] * n, in_degree)

def is_valid_ordering(adjacency_matrix: np.ndarray, ordering: str, node_label_to_indx : dict):
    """
    Checks valid node ordering.

    Parameters:
        adjacency_matrix (numpy.ndarray): graph adjacency matrix
        ordering (str): node ordering
        node_label_to_indx: mapping of node labels to indices

    Returns:
        (bool) True if valid ordering, False otherwise
    """
    for i, node in enumerate(ordering):
        for node_after in ordering[i+1:]:

            if node_after not in node_label_to_indx:
                print(f"KeyError: {node_after}")  # Debugging print
                return False

        for successor in np.where(adjacency_matrix[node_label_to_indx[node], :] == 1)[0]:
            if successor in [node_label_to_indx[n] for n in ordering[:i]]:
                return False

    return True

def find_all_paths_from_roots(graph):
    """
    Find all paths in the given DAG starting from root nodes, represented by an adjacency matrix.

    Parameters:
        graph: A 2D list representing the adjacency matrix of the DAG.

    Returns:
        (list (list)) Each list represents a path starting from any root node.
    """
    def dfs(current_path, node):
        # If the current node has no outgoing edges, we've reached the end of a path
        if not any(graph[node]):
            all_paths.append(current_path + [node])
            return

        for next_node, has_edge in enumerate(graph[node]):
            if has_edge:
                dfs(current_path + [node], next_node)

    def is_root(node):
        return all(graph[i][node] == 0 for i in range(n))

    all_paths = []
    n = len(graph)  # Number of nodes in the graph

    # Identify root nodes
    root_nodes = [node for node in range(n) if is_root(node)]

    # Start DFS from each root node
    for start_node in root_nodes:
        dfs([], start_node)

    return all_paths

def random_index_from_ones(matrix : np.ndarray):
    """
    Return a random index where the matrix element is 1 (edge).

    Parameters:
        matrix (numpy.ndarray): matrix

    Returns:
        (numpy.ndarray): index
    """
    ones_indices = np.argwhere(matrix == 1)
    if len(ones_indices) == 0:
        return None  # Return None if there are no 1s in the matrix.
    random_choice = np.random.choice(len(ones_indices))
    return ones_indices[random_choice]

def index_to_node_label(node_labels):
    """
    Map node index to label.

    Parameters:
        node_labels (list (str)): node labels

    Returns:
        (dict): dictionary with index as keys and labels as values
    """
    return {indx: node_label for indx, node_label in enumerate(node_labels)}

def node_label_to_index(node_labels):
    """
    Map node label to index.

    Parameters:
        node_labels (list (str)): node labels

    Returns:
        (dict): dictionary with labels as keys and index as values
    """
    return {node_label: indx for indx, node_label in enumerate(node_labels)}

def indep_test_func(data, i, j, k):
    """
    Conditional independence test on data variables (columns)

    """
    suffstat = partial_correlation_suffstat(data)
    return partial_correlation_test(suffstat, i, j, k)['p_value']

def cpdag_to_dag(cpdag):
    """
    Returns a DAG from a CPDAG
    """
    nodes = np.arange(cpdag.shape[0])
    num_parents = cpdag.sum(axis=0)
    idx = np.argsort(num_parents)
    nodes = nodes[idx]

    dag = cpdag.copy()
    for i,node in enumerate(nodes):
        for j,node2 in enumerate(nodes[i+1:]):
            if cpdag[node,node2] == 1:
                dag[node2,node] = 0

    return dag

def initial_graph_pc(data: pd.DataFrame, return_cpdag=False):
    """
    Runs PC Algorithm and returns a DAG
    """
    (g, sep_set) = pcalg.estimate_skeleton(indep_test_func=indep_test_func, data_matrix=data.values, alpha=0.01)
    g = pcalg.estimate_cpdag(skel_graph=g, sep_set=sep_set)
    cpdag = nx.to_numpy_array(g)
    dag = cpdag_to_dag(cpdag)
    return (dag, cpdag) if return_cpdag else dag