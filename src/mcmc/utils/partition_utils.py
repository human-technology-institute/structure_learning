import math
import numpy as np
from mcmc.data_structures.partition import Partition, OrderedPartition

def find_parent_nodes(graph_matrix):
    """Find parent nodes (nodes with no incoming edges)."""
    n = len(graph_matrix)
    parent_nodes = set(range(n))
    for i in range(n):
        for j in range(n):
            if graph_matrix[j][i] == 1:
                parent_nodes.discard(i)
    return parent_nodes

def remove_outgoing_edges(graph_matrix, nodes):
    """Remove outgoing edges from a set of nodes."""
    for node in nodes:
        for i in range(len(graph_matrix)):
            graph_matrix[node][i] = 0

def build_partition(incidence : np.ndarray , node_labels : list):
    """Partition the graph based on the connectivity."""
    partitions = []

    graph_matrix = incidence.copy()
    remaining_nodes = set(range(len(graph_matrix)))
    partition_id = 1



    while remaining_nodes:
        parent_nodes = find_parent_nodes(graph_matrix)
        if not parent_nodes:
            break

        partition_labels = [node_labels[node] for node in parent_nodes if node in remaining_nodes]
        partition_labels = set(partition_labels)

        new_partition = Partition(partition_id, partition_labels)
        partitions.append(new_partition)

        remove_outgoing_edges(graph_matrix, parent_nodes)
        remaining_nodes -= parent_nodes
        partition_id += 1

    return OrderedPartition(partitions)

def build_ordered_partition_from_srt( my_srt ):
    tokens = my_srt.split("} {")
    tokens = [ token.replace("{", "").replace("}", "") for token in tokens ]

    id = 0
    partitions = []
    for token in tokens:
        partitions.append( Partition(id, set(token.split(","))))
        id = id + 1
    return OrderedPartition(partitions)


# def convert_partition_to_party_posy( part_g : OrderedPartition):

#     num_partitions = part_g.get_num_part()

#     party = []
#     posy = []
#     for i in range(num_partitions):

#         num_nodes = part_g.get_partitions()[i].size
#         party.append( num_nodes )
#         posy.append( [i]*num_nodes )

#     flatten = lambda l: [item for sublist in l for item in sublist]
#     posy = flatten(posy)

#     return party, posy


def convert_partition_to_party_permy_posy( part_g : OrderedPartition):

    num_partitions =  len(part_g.get_partitions())

    party = []
    permy = []
    posy = []

    for i in range(num_partitions):
        num_nodes = part_g.get_partitions()[i].size
        party.append( num_nodes )
        posy.append( [i]*num_nodes )
        permy.append(list(part_g.get_partitions()[i].get_nodes()))

    flatten = lambda l: [item for sublist in l for item in sublist]
    posy = flatten(posy)
    permy = flatten(permy)

    return party, permy, posy


# old party function from BiDAG
# posy is a list that states to which partition index does a node belong to
# party is a list that states the size of each partition
# n is the total number of nodes (or the sum of the elements in the party list)
def calculate_join_possibilities(n, party, posy):
    m = len(party)
    join_possibs = [0] * n
    for k in range(n):
        join_possibs[k] = m - 1
        node_element = posy[k]
        if party[node_element] == 1:  # Nodes in a partition element of size 1
            if node_element < m - 1:
                if party[node_element + 1] == 1:  # And if the next partition element is also size 1
                    join_possibs[k] = m - 2  # We only allow them to jump to the left to count the swap only once
    return join_possibs


# old partyhole function from BiDAG
# posy is a list that states to which partition index does a node belong to
# party is a list that states the size of each partition
# n is the total number of nodes (or the sum of the elements in the party list)
def calculate_partition_transitions(n, party, posy):
    m = len(party)
    hole_possibs = [0] * n
    for k in range(n):
        node_element = posy[k]
        if party[node_element] == 1:  # Nodes in a partition element of size 1 cannot move to the neighbouring holes
            hole_possibs[k] = m - 1
            if node_element < m - 1:
                if party[node_element + 1] == 1:  # And if the next partition element is also size 1
                    hole_possibs[k] = m - 2  # We only allow them to jump to the left to count the swap only once
        elif party[node_element] == 2:  # Nodes in a partition element of size 2 cannot move to the hole on the left
            hole_possibs[k] = m  # Since this would count the same splitting twice
        else:
            hole_possibs[k] = m + 1
    return hole_possibs

# This function calculates permutations excluding nodes in the same partition element
# given the partition

# parttopermdiffelemposs<-function(n,party){
# 	m<-length(party)
# 	possibs<-rep(0,m-1)
# 	if(m>1){
# 		remainder<-n
# 		for (i in 1:(m-1)){
# 			remainder<-remainder-party[i]
# 			possibs[i]<-party[i]*remainder
# 		}
# 	}
# 	return(possibs)
# }

def possible_permutations(n, party):
    m = len(party)
    possibs = [0] * (m)
    if m > 1:
        remainder = n
        for i in range(m):
            remainder = remainder - party[i]
            possibs[i] = party[i]*remainder
    return possibs

# parttopermneighbourposs<-function(n,party){
# 	m<-length(party)
# 	possibs<-rep(0,m-1)
# 	if(m>1){
# 		for (i in 1:(m-1)){
# 			possibs[i]<-party[i]*party[i+1]
# 		}
# 	}
# 	return(possibs)
# }

def possible_permutations_neighbors(n, party):
    m = len(party)
    possibs = [0] * (m-1)
    if m > 1:
        for i in range(m-1):
            possibs[i] = party[i]*party[i+1]
    return possibs

# partysteps<-function(n,party){
# 	partypossibs<-rep(0,n)
# 	kbot<-1
# 	m<-length(party)
# 	for (j in 1:m){
# 		elemsize<-party[j]
# 		ktop<-kbot+elemsize-1
# 		partypossibs[kbot:ktop]<-choose(elemsize,1:elemsize)
# 		kbot<-ktop+1
# 	}
# 	return(partypossibs[1:(n-1)])
# }
def possible_splits_joins(n, party):
    partypossibs = [0] * n
    kbot = 0
    m = len(party)
    for j in range(m):
        elemsize = party[j]
        ktop = kbot + elemsize - 1
        for k in range(kbot, ktop):
            partypossibs[k] = math.comb(elemsize, k-kbot+1)
        kbot = ktop + 1
    return partypossibs