"""

"""
from typing import Union
import networkx as nx
import numpy as np
from mcmc.utils.graph_utils import create_identity_matrix, create_ones_matrix, \
                                    random_index_from_ones, compute_ancestor_matrix, update_matrix
from mcmc.proposals import StructureLearningProposal

class GraphProposal(StructureLearningProposal):
    """
    Graph Proposal by adding, deleting, reversing edges.
    """

    operations = ["add_edge", "delete_edge", "reverse_edge", "stay_still"]
    """
    Possible edge operations.
    """
    def __init__(self, G_curr : Union[np.ndarray, nx.DiGraph], blacklist = None, whitelist = None):
        """
        Initialise GraphProposal instance.

        Parameters:
            G_curr (networkx.DiGraph | numpy.ndarray): graph
            blacklist (numpy.ndarray): mask for edges to ignore in the proposal
            whitelist (numpy.ndarray): mask for edges to include in the proposal
        """
        super().__init__(G_curr, blacklist, whitelist) # initialize the parent class

        self._max_parents = self.num_nodes - 1

        self._G_curr = nx.adjacency_matrix(G_curr).toarray() if isinstance(G_curr, nx.DiGraph) else G_curr
        self._G_prop = None

        self._G_curr_neigh = -1
        self._G_prop_neigh = -1

        # Matrices used in further computations
        self._fullmatrix = create_ones_matrix(self.num_nodes)
        self._Ematrix = create_identity_matrix(self.num_nodes)

    def _propose_neighbor_by_addition(self, index_mat : np.ndarray, incidence : np.ndarray):
        """
        Propose new graph by adding an edge.

        Parameters:
            index_mat (numpy.ndarray): index matrix where 1 denotes possible edge to add
            incidence (numpy.ndarray): adjacency matrix

        Returns:
            (numpy.ndarray): adjacency matrix of new graph
        """
        try:
            r, c = random_index_from_ones(index_mat)
        except Exception as e:
            print("Incidence matrix")
            print(incidence)
            print("Index matrix")
            print(index_mat)
            raise Exception("The incidence matrix is not valid!") from e

        # update the incidence matrix
        new_incidence = incidence.copy()
        new_incidence[r, c] = 1
        return new_incidence

    def _propose_neighbor_by_reverse(self, index_mat : np.ndarray, incidence : np.ndarray):
        """
        Propose new graph by reversing an edge.

        Parameters:
            index_mat (numpy.ndarray): index matrix where 1 denotes possible edge to reverse
            incidence (numpy.ndarray): adjacency matrix

        Returns:
            (numpy.ndarray): adjacency matrix of new graph
        """
        try:
            r, c = random_index_from_ones(index_mat)
        except Exception as e:
            print("Incidence matrix")
            print(incidence)
            print("Index matrix")
            print(index_mat)
            raise Exception("The incidence matrix is not valid!") from e

        # update the incidence matrix
        new_incidence = incidence.copy()
        new_incidence[r, c] = 0
        new_incidence[c, r] = 1

        return new_incidence

    def _propose_neighbor_by_deletion(self, index_mat : np.ndarray, incidence : np.ndarray ):
        """
        Propose new graph by deleting an edge.

        Parameters:
            index_mat (numpy.ndarray): index matrix where 1 denotes possible edge to delete
            incidence (numpy.ndarray): adjacency matrix

        Returns:
            (numpy.ndarray): adjacency matrix of new graph
        """
        try:
            r, c = random_index_from_ones(index_mat)
        except Exception as e:
            print("Incidence matrix")
            print(incidence)
            print("Index matrix")
            print(index_mat)
            raise Exception("The incidence matrix is not valid!") from e

        # update the incidence matrix
        new_incidence = incidence.copy()
        new_incidence[r, c] = 0

        return new_incidence

    def _compute_nbhood(self, incidence : np.ndarray):
        """
        Computes neighbor graphs obtainable by edge deletion, addition, and reversal.

        Parameter:
            incidence (numpy.ndarray): adjacency matrix

        Returns:
            (int): total number of neighboring graphs
            (numpy.ndarray): index matrix for edge deletion
            (numpy.ndarray): index matrix for edge addition
            (numpy.ndarray): index matrix for edge reversal
            (int): number of neighboring graphs obtainable by edge deletion
            (int): number of neighboring graphs obtainable by edge addition
            (int): number of neighboring graphs obtainable by edge reversal
        """
        ancestor = compute_ancestor_matrix(incidence)

        # 1.) Number of neighbour graphs obtained by edge deletions
        deletion = incidence.copy() - self.whitelist
        num_deletion = np.sum(deletion)

        # 2.) Number of neighbour graphs obtained by edge additions
        add = self._fullmatrix - self._Ematrix - incidence  - ancestor  - self.blacklist

        add[add < 0] = 0
        try:
            indx = np.where( np.sum(incidence, axis = 0) > self._max_parents - 1 )[0][0]
            add[:,indx] = 0
            num_addition = np.sum(add)# eliminate cycles
        except:
            num_addition = np.sum(add)

        # 3.) Number of neighbour graphs obtained by edge reversals
        reversal = (incidence - self.whitelist) - ((incidence - self.whitelist).T @ ancestor).T - self.blacklist.T
        reversal[reversal < 0] = 0 # replace all negative values by zero

        try:
            reversal[indx, :] = 0
            num_reversal = np.sum(reversal)
        except:
            num_reversal = np.sum(reversal)

        # Total number of neighbour graphs
        currentnbhood =  num_deletion + num_addition + 1 + num_reversal

        return currentnbhood, deletion, add, reversal, num_deletion, num_addition, num_reversal

    def prob_Gcurr_Gprop_f(self):
        """
        Computes transition probability of going from proposed graph to current graph Q(G_curr|G_prop).

        Returns:
            (float): transition probability Q(G_curr|G_prop)
        """
        num_neigh, _, _, _ , _, _, _ = self._compute_nbhood(self.proposed_graph)
        self._G_prop_neigh = num_neigh

        # Q(G_curr -> G_prop) = 1 / (number of neighbors of G_prop)
        self.prob_Gcurr_Gprop = 1/self._G_prop_neigh

        return self.prob_Gcurr_Gprop

    def prob_Gprop_Gcurr_f(self):
        """
        Computes transition probability of going from current graph to proposed graph Q(G_prop|G_curr).

        Returns:
            (float): transition probability Q(G_prop|G_curr)
        """
        # Q(G_prop -> G_curr) = 1 / (number of neighbors of G_curr)
        self.prob_Gprop_Gcurr = 1 / self._G_curr_neigh

        return self.prob_Gprop_Gcurr

    def propose_DAG(self):
        """
        Propose a DAG.

        Returns:
            (numpy.ndarray): adjacency matrix of proposed graph
            (str): operation that generated the proposed graph
        """
        # add the whitelist to the current incidence matrix
        self._G_curr = update_matrix(self.current_graph, self.current_graph + self.whitelist)

        # compute all possible neighbours
        num_neighbours, del_indx_mat, add_indx_mat, rev_indx_mat, num_deletion, num_addition, num_reversal = self._compute_nbhood(self.current_graph)

        # set the number of neighbours
        self._G_curr_neigh = num_neighbours

        # Randomly sample an operation
        operation = np.random.choice(self.operations, size=1, p=[num_addition/num_neighbours, num_deletion/num_neighbours, num_reversal/num_neighbours, 1/num_neighbours])

        # initialise new_incidence as zeros NxN
        self._G_prop = np.zeros((self.num_nodes, self.num_nodes))

        if operation == "add_edge":
            self._G_prop = self._propose_neighbor_by_addition(add_indx_mat, self.current_graph)
        elif operation == "delete_edge":
            self._G_prop = self._propose_neighbor_by_deletion(del_indx_mat, self.current_graph)
        elif operation == "reverse_edge":
            self._G_prop = self._propose_neighbor_by_reverse(rev_indx_mat, self.current_graph)
        elif operation == "stay_still":
            self._G_prop = self.current_graph
        else:
            raise Exception(f"The operation '{operation}' is not valid!")

        self._operation = operation

        # compute the proposal distribution
        self.prob_Gcurr_Gprop_f()
        self.prob_Gprop_Gcurr_f()

        return self.proposed_graph, operation
