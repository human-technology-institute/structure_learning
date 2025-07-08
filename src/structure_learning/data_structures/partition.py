"""
partition.py

This module defines the Partition and OrderedPartition classes for representing and manipulating partitions of nodes, including visualization and conversion utilities. It provides methods for creating, modifying, and visualizing partitions, as well as constructing partitions from graphs, numpy arrays, and string representations.
"""

from typing import List
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .graph import Graph

class Partition:
    """
    Represents a single partition of nodes.
    """
    def __init__(self, ID : int, nodes : set ):
        """
        Initialize a Partition with an ID and a set of nodes.
        Args:
            ID (int): Partition identifier.
            nodes (set): Set of node labels.
        """
        self.ID = ID
        self._nodes = nodes

    @property
    def size(self):
        """
        Returns the number of nodes in the partition.
        """
        return len(self.nodes)

    @property
    def nodes(self):
        """
        Returns the set of nodes in the partition.
        """
        return self._nodes

    @nodes.setter
    def nodes(self, n):
        """
        Set the nodes of the partition.
        Args:
            n (set): New set of nodes.
        """
        self._nodes = n

    def copy(self):
        """
        Return a deep copy of the partition.
        """
        return Partition(self.ID, self.nodes.copy())

    def remove_single_node(self, node : str):
        """
        Remove a single node from the partition.
        Args:
            node (str): Node label to remove.
        """
        self.nodes.remove(node)

    def remove_nodes(self, nodes : set):
        """
        Remove multiple nodes from the partition.
        Args:
            nodes (set): Set of node labels to remove.
        """
        self.nodes = self.nodes.difference(nodes)

    def add_single_node(self, node : str):
        """
        Add a single node to the partition.
        Args:
            node (str): Node label to add.
        """
        self.nodes.add(node)

    def add_nodes(self, nodes : set):
        """
        Add multiple nodes to the partition.
        Args:
            nodes (set): Set of node labels to add.
        """
        self.nodes = self.nodes.union(nodes)

    def replace_partition(self, new_partition):
        """
        Replace the current partition with another partition.
        Args:
            new_partition (Partition): Partition to copy from.
        """
        self.nodes = new_partition.nodes
        self.ID = new_partition.get_ID()

    def get_ID(self):
        """
        Return the partition's ID.
        """
        return self.ID

    def set_ID(self, ID):
        """
        Set the partition's ID.
        Args:
            ID (int): New partition ID.
        """
        self.ID = ID

    def set_ID(self, ID):
        """
        Set the partition's ID.
        Args:
            ID (int): New partition ID.
        """
        self.ID = ID

    def __str__(self):
        """
        Return a string representation of the partition.
        """
        return f"{self.nodes}"

    def info(self):
        """
        Print information about the partition.
        """
        print(f"Partition ID {self.get_ID()}: {self.nodes} | Size: {self.size} " )

    def plot(self, fig_size=(2.7, 6)):
        """
        Visualize the partition as a rectangle with nodes as circles.
        Args:
            fig_size (tuple): Figure size for matplotlib.
        """
        nodes = list(self.nodes)
        partition_id = self.get_ID()
        partition_size = self.size

        fig, ax = plt.subplots(figsize=fig_size)

        # Calculate rectangle dimensions within adjusted limits
        rect_height = len(nodes) + 1.5  # Adding some space for the partition ID at the top
        rect_width =  1  # Slimmer rectangle within adjusted limits
        rect_x = 0.5
        rect_y = 0.5  # Starting a little above the bottom

        # Draw rectangle
        rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
        ax.add_patch(rect)

        # Add partition ID and size at the top of the rectangle
        ax.text(rect_x + rect_width / 2, rect_y + rect_height - 0.5, f"ID: {partition_id} | Size: {partition_size}", ha='center')

        # Plot nodes as circles inside the rectangle, vertically aligned, using seaborn's color palette
        circle_radius = 0.2  # Radius for the circles representing nodes
        for i, node in enumerate(nodes, start=1):
            circle_y = rect_y + i  # Y position for each circle
            circle = patches.Circle((rect_x + rect_width / 2, circle_y), circle_radius, edgecolor='b', facecolor=sns.color_palette("pastel")[0])
            ax.add_patch(circle)
            ax.text(rect_x + rect_width / 2, circle_y, node, ha='center', va='center')

        xlim_left = -0.05
        xlim_right = 2
        ylim_bottom = 0
        ylim_top = len(nodes) + 2

        ax.set_xlim(xlim_left, xlim_right)  # Adjusted limits to shrink plot area
        ax.set_ylim(ylim_bottom, ylim_top)  # Adjusted limits to shrink plot area
        ax.axis('off')  # Hide axis

        plt.show()

class OrderedPartition:
    """
    Represents an ordered collection of Partition objects.
    """
    def __init__(self, partitions: List[Partition]):
        """
        Initialize with a list of Partition objects.
        Args:
            partitions (list): List of Partition objects.
        """
        self._partitions = partitions

    @property
    def size(self) -> int:
        """
        Return the number of partitions.
        """
        return len(self.partitions)

    @property
    def all_nodes(self):
        """
        Return a set of all nodes in all partitions.
        """
        return set.union(*(partition.nodes for partition in self.partitions))

    @property
    def partitions(self) -> List[Partition]:
        """
        Return the list of Partition objects.
        """
        return self._partitions

    @partitions.setter
    def partitions(self, parts: list):
        """
        Set the list of Partition objects.
        Args:
            parts (list): List of Partition objects.
        """
        self._partitions = parts

    def get_all_nodes_from_right(self, indx):
        """
        Return all nodes from partition at index `indx` to the end.
        Args:
            indx (int): Start index.
        """
        return set.union(*(partition.nodes for partition in self.partitions[indx:]))

    def get_all_nodes_adj_right(self, indx):
        """
        Return all nodes in the partition at index `indx`.
        Args:
            indx (int): Index.
        """
        return set.union(*(partition.nodes for partition in self.partitions[indx:indx+1]))

    def get_all_nodes_adj_left(self, indx):
        """
        Return all nodes in the partition at index `indx-1`.
        Args:
            indx (int): Index.
        """
        return set.union(*(partition.nodes for partition in self.partitions[indx-1:indx]))

    def get_all_nodes_from_left(self, indx):
        """
        Return all nodes from the start up to (but not including) index `indx`.
        Args:
            indx (int): End index.
        """
        return set.union(*(partition.nodes for partition in self.partitions[:indx]))

    def copy(self):
        """
        Return a deep copy of the OrderedPartition.
        """
        return OrderedPartition([partition.copy() for partition in self.partitions])

    def replace_partition(self, new_partition : Partition, index : int):
        """
        Replace the partition at `index` with `new_partition`.
        Args:
            new_partition (Partition): Replacement partition.
            index (int): Index to replace.
        """
        self.partitions[index].replace_partition(new_partition)

    def add_node_to_partition(self, part_indx : int, node : str):
        """
        Add a node to the partition at `part_indx`.
        Args:
            part_indx (int): Partition index.
            node (str): Node label.
        """

        if part_indx < 0 or part_indx >= len(self.partitions):
            #print(f"Error: Attempt to access invalid partition index {part_indx} | {len(self.partitions)}.")
            part_indx = len(self.partitions)-1

        self.partitions[part_indx].add_nodes(node)

    def remove_node_from_partition(self, part_id : int, node : str):
        """
        Remove a node from the partition at `part_id`.
        Args:
            part_id (int): Partition index.
            node (str): Node label.
        """

        self.partitions[part_id].remove_nodes(node)
        #if the partition is empty, delete it
        self.remove_partition(part_id) if self.partitions[part_id].size == 0 else self.partitions

    def remove_empty_partitions(self):
        """
        Remove all empty partitions from the collection.
        """
        self.partitions = [partition for partition in self.partitions if partition.size > 0]

    def join_partition(self, part_id : int):
        """
        Join the partition at `part_id` with its left neighbor.
        Args:
            part_id (int): Partition index.
        """
        # we always join to the left of the sampled partition id
        try:
            current_partition = self.get_partitions()[part_id]
            target_partition = self.get_partitions()[part_id-1]
        except IndexError:
            print(f"[ERROR] Partition {part_id} or {part_id-1} not found")
            return

        # If both partitions are found, replace the node set of the replacement partition
        target_partition.nodes = target_partition.nodes.union(current_partition.nodes)

        # remove current partition
        self.remove_partition(part_id)
        self.update_IDs()

    def insert_partition(self, part_id : int, nodes : set):
        """
        Insert a new partition at `part_id` with the given nodes.
        Args:
            part_id (int): Index to insert at.
            nodes (set): Nodes for the new partition.
        """
        new_partition = Partition(part_id, nodes)
        self.partitions.insert(part_id, new_partition)
        for new_id, partition in enumerate(self.get_partitions(), start=part_id):
            partition.set_ID(new_id)
        self.update_IDs()

    def remove_partition(self, part_id):
        """
        Remove the partition at `part_id`.
        Args:
            part_id (int): Partition index.
        """
        self.partitions.remove(self.partitions[part_id])
        self.update_IDs()

    def find_node(self, node_label):
        """
        Return the index of the partition containing `node_label`, or None if not found.
        Args:
            node_label (str): Node label to search for.
        """
        for index, partition in enumerate(self.partitions):
            if node_label in partition.nodes:
                return index
        return None

    def get_all_nodes(self):
        """
        Return a set of all nodes in all partitions.
        """
        return self.all_nodes

    def get_partition_by_indx(self, index):
        """
        Return the partition at the given index.
        Args:
            index (int): Partition index.
        """
        return self.partitions[index]

    def get_partitions(self):
        """
        Return the list of Partition objects.
        """
        return self.partitions

    def update_IDs (self):
        """
        Update the IDs of all partitions to be consecutive starting from 0.
        """
        for new_id, partition in enumerate(self.get_partitions(), start=0):
            partition.set_ID(new_id)

    def __str__(self):
        """
        Return a string representation of the OrderedPartition.
        """
        return ' '.join(str(partition) for partition in self.partitions)

    def print_partitions(self):
        """
        Return a string with all partitions.
        """
        res = ' '.join(str(self.get_partitions()[i]) for i in range(len(self.get_partitions())))
        return res

    def info(self):
        """
        Print information about all partitions.
        """
        for i in range(len(self.get_partitions())):
            print(f"Partition ID {self.get_partitions()[i].get_ID()}: {self.get_partitions()[i].nodes} | Size: {self.get_partitions()[i].size} | Partition Indx : {i}" )

    @classmethod
    def from_graph(cls, g: Graph):
        """
        Create an OrderedPartition from a Graph object.
        Args:
            g (Graph): Graph instance.
        """
        return cls.from_numpy(g.incidence, g.nodes)

    @classmethod
    def from_numpy(cls, incidence: np.ndarray , node_labels : list):
        """
        Create an OrderedPartition from an incidence matrix and node labels.
        Args:
            incidence (np.ndarray): Incidence matrix.
            node_labels (list): List of node labels.
        """
         
        partitions = []

        graph_matrix = incidence.copy()
        remaining_nodes = set(range(len(graph_matrix)))
        partition_id = 1

        while remaining_nodes:
            parent_nodes = Graph.find_parent_nodes(graph_matrix)
            if not parent_nodes:
                break

            partition_labels = [node_labels[node] for node in parent_nodes if node in remaining_nodes]
            partition_labels = set(partition_labels)

            new_partition = Partition(partition_id, partition_labels)
            partitions.append(new_partition)

            """Remove outgoing edges from a set of nodes."""
            for node in parent_nodes:
                for i in range(len(graph_matrix)):
                    graph_matrix[node][i] = 0
                    
            remaining_nodes -= parent_nodes
            partition_id += 1

        return OrderedPartition(partitions)
    
    @classmethod
    def from_string(cls, string:str):
        """
        Create an OrderedPartition from a string representation.
        Args:
            string (str): String representation.
        """
        tokens = string.split("} {")
        tokens = [ token.replace("{", "").replace("}", "") for token in tokens ]
        id = 0
        partitions = []
        for token in tokens:
            partitions.append( Partition(id, set(token.split(","))))
            id = id + 1
        return OrderedPartition(partitions)
    
    def to_party_permy_posy(self):
        """
        Return the party, permy, and posy representations of the partition.
        """
        num_partitions =  len(self.get_partitions())

        party = []
        permy = []
        posy = []

        for i in range(num_partitions):
            num_nodes = self.partitions[i].size
            party.append( num_nodes )
            posy.append( [i]*num_nodes )
            permy.append(list(self.partitions[i].nodes))

        flatten = lambda l: [item for sublist in l for item in sublist]
        posy = flatten(posy)
        permy = flatten(permy)
        return party, permy, posy
    
    def plot(self, fig_size=(2.7, 6), title = "Ordered Partition"):
        """
        Visualize all partitions side by side.
        Args:
            fig_size (tuple): Figure size for matplotlib.
            title (str): Title for the plot.
        """
        partitions = self.get_partitions()

        # Calculate the total width of the figure based on the number of partitions and the individual fig_size width
        total_width = fig_size[0] * len(partitions) * 0.7  # Adding some space between partitions
        total_height = fig_size[1]

        # Setup figure with the calculated total width and height
        fig, axs = plt.subplots(1, len(partitions), figsize=(total_width, total_height))

        if len(partitions) == 1:  # Ensure axs is iterable for a single partition
            axs = [axs]

        for ax, partition in zip(axs, partitions):
            nodes = list(partition.nodes)
            partition_id = partition.get_ID()
            partition_size = partition.size

            # Calculate rectangle dimensions within adjusted limits
            rect_height = len(nodes) + 1.5  # Adding some space for the partition ID at the top
            rect_width =  1  # Slimmer rectangle within adjusted limits
            rect_x = 0.5
            rect_y = 0.5  # Starting a little above the bottom

            # Draw rectangle
            rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
            ax.add_patch(rect)

            # Add partition ID and size at the top of the rectangle
            ax.text(rect_x + rect_width / 2, rect_y + rect_height - 0.5, f"ID: {partition_id} | Size: {partition_size}", ha='center')

            # Plot nodes as circles inside the rectangle, vertically aligned, using seaborn's color palette
            circle_radius = 0.2  # Radius for the circles representing nodes
            for i, node in enumerate(nodes, start=1):
                circle_y = rect_y + i  # Y position for each circle
                circle = patches.Circle((rect_x + rect_width / 2, circle_y), circle_radius, edgecolor='b', facecolor=sns.color_palette("pastel")[0])
                ax.add_patch(circle)
                ax.text(rect_x + rect_width / 2, circle_y, node, ha='center', va='center')

            xlim_left = -0.05
            xlim_right = 2
            ylim_bottom = 0
            ylim_top = len(nodes) + 2

            ax.set_xlim(xlim_left, xlim_right)  # Adjusted limits to shrink plot area
            ax.set_ylim(ylim_bottom, ylim_top)  # Adjusted limits to shrink plot area
            ax.axis('off')  # Hide axis

        # add a title
        fig.suptitle(title, fontsize=16)

        plt.tight_layout()  # Optimize layout to reduce unnecessary plot area
        plt.show()
        
    # pickle
    def save(self, filename: str):
        """
        Saves the Graph object to a file.

        Parameters:
            filename (str): Path to the output file.
        """
        with open(filename, 'wb') as f:
            import pickle
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """
        Loads a Graph object from a file.

        Parameters:
            filename (str): Path to the input file.

        Returns:
            Graph: Loaded Graph object.
        """
        with open(filename, 'rb') as f:
            import pickle
            return pickle.load(f)