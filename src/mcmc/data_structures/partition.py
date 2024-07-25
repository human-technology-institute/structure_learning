import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Partition:

    def __init__(self, ID : int, nodes : set ):
        self.ID = ID
        self._nodes = nodes

    @property
    def size(self):
        return len(self.nodes)

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, n):
        self._nodes = n

    def copy(self):
        return Partition(self.ID, self.nodes.copy())

    def remove_single_node(self, node : str):
        self.nodes.remove(node)

    def remove_nodes(self, nodes : set):
        self.nodes = self.nodes.difference(nodes)

    def add_single_node(self, node : str):
        self.nodes.add(node)

    def add_nodes(self, nodes : set):
        self.nodes = self.nodes.union(nodes)

    def replace_partition(self, new_partition):
        self.nodes = new_partition.nodes
        self.ID = new_partition.get_ID()

    def get_ID(self):
        return self.ID

    def set_ID(self, ID):
        self.ID = ID

    def set_ID(self, ID):
        self.ID = ID

    def __str__(self):
        return f"{self.nodes}"

    def info(self):
        print(f"Partition ID {self.get_ID()}: {self.nodes} | Size: {self.size} " )

    def visualize_partition(self, fig_size=(2.7, 6)):
        """
        Visualizes the given partition with an optimized layout, reducing the plot area outside the rectangle.
        The nodes are plotted as circles vertically aligned inside a slimmer rectangle.
        The partition ID and size are displayed at the top of the rectangle.
        Allows control over the figure size through the fig_size parameter.

        Parameters:
            partition: An instance of the Partition class.
            fig_size: A tuple (width, height) specifying the size of the figure.
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
    def __init__(self, partitions: list[Partition]):
        self._partitions = partitions

    @property
    def size(self):
        return len(self.partitions)

    @property
    def all_nodes(self):
        return set.union(*(partition.nodes for partition in self.partitions))

    @property
    def partitions(self) -> list[Partition]:
        return self._partitions

    @partitions.setter
    def partitions(self, parts: list):
        self._partitions = parts

    def get_all_nodes_from_right(self, indx):
        return set.union(*(partition.nodes for partition in self.partitions[indx:]))

    def get_all_nodes_adj_right(self, indx):
        return set.union(*(partition.nodes for partition in self.partitions[indx:indx+1]))

    def get_all_nodes_adj_left(self, indx):
        return set.union(*(partition.nodes for partition in self.partitions[indx-1:indx]))

    def get_all_nodes_from_left(self, indx):
        return set.union(*(partition.nodes for partition in self.partitions[:indx]))

    def copy(self):
        return OrderedPartition([partition.copy() for partition in self.partitions])

    def replace_partition(self, new_partition : Partition, index : int):
        self.partitions[index].replace_partition(new_partition)

    def add_node_to_partition(self, part_indx : int, node : str):

        if part_indx < 0 or part_indx >= len(self.partitions):
            #print(f"Error: Attempt to access invalid partition index {part_indx} | {len(self.partitions)}.")
            part_indx = len(self.partitions)-1

        self.partitions[part_indx].add_nodes(node)

    def remove_node_from_partition(self, part_id : int, node : str):

        self.partitions[part_id].remove_nodes(node)
        #if the partition is empty, delete it
        self.remove_partition(part_id) if self.partitions[part_id].size == 0 else self.partitions

    def remove_empty_partitions(self):
        self.partitions = [partition for partition in self.partitions if partition.size > 0]

    def join_partition(self, part_id : int):
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
        new_partition = Partition(part_id, nodes)
        self.partitions.insert(part_id, new_partition)
        for new_id, partition in enumerate(self.get_partitions(), start=part_id):
            partition.set_ID(new_id)
        self.update_IDs()

    def remove_partition(self, part_id):
        self.partitions.remove(self.partitions[part_id])
        self.update_IDs()

    def find_node(self, node_label):
        for index, partition in enumerate(self.partitions):
            if node_label in partition.nodes:
                return index
        return None

    def get_all_nodes(self):
        return self.all_nodes

    def get_partition_by_indx(self, index):
        return self.partitions[index]

    def get_partitions(self):
        return self.partitions

    def update_IDs (self):
        for new_id, partition in enumerate(self.get_partitions(), start=0):
            partition.set_ID(new_id)

    def __str__(self):
        return ' '.join(str(partition) for partition in self.partitions)

    def print_partitions(self):
        res = ' '.join(str(self.get_partitions()[i]) for i in range(len(self.get_partitions())))
        return res

    def info(self):
        for i in range(len(self.get_partitions())):
            print(f"Partition ID {self.get_partitions()[i].get_ID()}: {self.get_partitions()[i].nodes} | Size: {self.get_partitions()[i].size} | Partition Indx : {i}" )

    def visualize_partitions(self, fig_size=(2.7, 6), title = "Ordered Partition"):
        """
        Visualizes a set of partition objects side by side with an optimized layout.
        Each partition's nodes are plotted as circles vertically aligned inside a slimmer rectangle.
        The partition ID and size are displayed at the top of each rectangle.
        Allows control over the figure size through the fig_size parameter for each individual partition visualization.

        Args:
        - partitions: A list of Partition class instances to be visualized.
        - fig_size: A tuple (width, height) specifying the size of the figure for each partition.
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