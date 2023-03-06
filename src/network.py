# network-economy is a simulation program for the Network Economy ABM
# Copyright (C) 2020 Théo Dessertaine and Mitja Devetak
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
The ``network`` module
======================

This module deals with various network generation useful for the model.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_base_case_tech_matrix(n_firms, c_tot, value=1):
    """
    Generates a base case technology matrix according to the parameters c_tot -> number 
    of nonzero entries of each row and each row sums to 1 and diagonal is 0.
    :param n_firms: number of firms,
    :param c_tot: number of nonzero entries of each column and each row
    :param value: value of the nonzero entries.
    :return: Technology matrix.
    """
    technology_matrix = np.zeros((n_firms, n_firms))
    for i in range(n_firms):
        ind = np.random.choice(np.arange(0, n_firms-1), c_tot, replace=False)
        for j in ind:
            if j < i:
                technology_matrix[j, i] = value
            else:
                technology_matrix[j+1, i] = value

    return technology_matrix

def generate_connectivity_matrix(technology_matrix, connectivity_parameter):
    """''''
    Generates a connectivity matrix from a technology matrix T and a connectivity parameter c.
    Randomly selects c non-zero elements of each column as the non-zero elements 
    of the corresponding column of W.
    :param technology_matrix: Technology matrix.
    :param connectivity_parameter: Connectivity parameter.
    :return: Connectivity matrix.
    """
    W = np.zeros(technology_matrix.shape)
    for j in range(technology_matrix.shape[1]):
        ind = np.random.choice(np.nonzero(technology_matrix[:, j])[0], connectivity_parameter, replace=False)
        W[ind, j] = technology_matrix[ind, j]
    return W

def plot_connectivity_network(W, save_path=None):
    """
    Plots the network of connectivity matrix W.
    :param W: Connectivity matrix.
    :return: None.
    """
    G = nx.DiGraph()
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] != 0:
                G.add_edge(i,j)
    nx.draw(G, with_labels=True)
    plt.draw()
    if save_path is not None:
        plt.savefig(save_path)
        # clear the figure
        plt.clf()
    else:
        plt.show()
    return None


# given a connectivity matrix W compute the out weight of each node
# (w_out)_i = sum_j W_ij
def compute_out_weight(W):
    w_out = np.sum(W, axis=1)
    return w_out

# given a connectivity matrix W compute the in weight of each node
# (w_in)_i = sum_j W_ji
def compute_in_weight(W):
    w_in = np.sum(W, axis=0)
    return w_in

# given a connectivity matrix W compute the total weight of each node compute the graph laplacian defined as:
# L = diag(total_weight) - W - W^T
def compute_laplacian(W, total_weight):
    L = np.diag(total_weight) - W - W.T
    return L

# given a graph laplacian L and the imbalance of each node
# compute the least squares solution of the linear system Lx = imbalance
# and normalize so that the smallest entry in x is 0
def compute_trophic_level(laplacian, imbalance):
    x = np.linalg.lstsq(laplacian, imbalance, rcond=None)[0]
    x = x - np.min(x)
    return x

# given a connectivity matrix W and the trophic level of each node
# compute the trophic incoherence defined as:
# sum_i sum_j W_ij (h_j - h_i - 1)^2 / sum_i sum_j W_ij
def compute_trophic_incoherence_from_level(W, trophic_level):
    n = W.shape[0]
    trophic_incoherence = 0
    for i in range(n):
        for j in range(n):
            if W[i, j] != 0:
                trophic_incoherence += W[i, j] * (trophic_level[j] - trophic_level[i] - 1) ** 2
    trophic_incoherence = trophic_incoherence / np.sum(W)
    return trophic_incoherence

# given a connectivity matrix W compute trophic incoherence
def compute_trophic_incoherence(W):
    in_weight = compute_in_weight(W)
    out_weight = compute_out_weight(W)
    total_weight = in_weight + out_weight
    laplacian = compute_laplacian(W, total_weight)
    imbalance = in_weight - out_weight
    trophic_level = compute_trophic_level(laplacian, imbalance)
    trophic_incoherence = compute_trophic_incoherence_from_level(W, trophic_level)
    return trophic_incoherence

def generate_communities_supply(n_firms, n_communities, c):
    """
    Generates a connectivity matrix with communities.
    :param n_firms: number of firms.
    :param n_communities: number of communities.
    :param c: connectivity parameter.
    :return: Connectivity matrix.
    """

    if n_firms % n_communities != 0:
        raise ValueError('n_firms must be a multiple of n_communities')
    
    firms_per_community = n_firms // n_communities

    W = np.zeros((n_firms, n_firms))
    T = np.zeros((n_firms, n_firms))

    for i in range(n_communities):
        for j in range(n_communities):
            T[i * firms_per_community:(i + 1) * firms_per_community, j * firms_per_community:(j + 1) * firms_per_community] = generate_base_case_tech_matrix(firms_per_community, c, 1/c)
    
    # generate connectivity matrix where each community is connected only to itself
    for i in range(n_communities):
        W[i*firms_per_community:(i+1)*firms_per_community, i*firms_per_community:(i+1)*firms_per_community] = T[i*firms_per_community:(i+1)*firms_per_community, i*firms_per_community:(i+1)*firms_per_community]
    
    return W, T

def _compute_total_elements_in_partition(U):
    """
    Computes the total number of elements in a partition.
    :param U: Partition.
    :return: Total number of elements.
    """
    total_elements = 0
    for i in range(len(U)):
        total_elements += len(U[i])
    return total_elements

def mutual_information_of_two_partitions(U,V):
    """
    Computes the mutual information between two partitions.
    :param U: Partition 1.
    :param V: Partition 2.
    :return: Mutual information.
    """
    u_lenght = len(U)
    v_lenght = len(V)

    total_elements = _compute_total_elements_in_partition(U)
    mutual_information = 0

    for i in range(u_lenght):
        for j in range(v_lenght):
            P = len(np.intersect1d(U[i], V[j])) / total_elements
            if P != 0:
                mutual_information += P*np.log(P / ((len(U[i]) / total_elements) * (len(V[j]) / total_elements)))
    
    return mutual_information
    
def is_subset(network_1, network_2):
    """
    Checks if network_1 is a subset of network_2.
    :param network_1: Network 1.
    :param network_2: Network 2.
    :return: True if network_1 is a subset of network_2, False otherwise.
    """
    return np.all(network_1 <= network_2)

def is_superset(network_1, network_2):
    """
    Checks if network_1 is a superset of network_2.
    :param network_1: Network 1.
    :param network_2: Network 2.
    :return: True if network_1 is a superset of network_2, False otherwise.
    """
    return np.all(network_1 >= network_2)

# def generate_random_technology_matrix(n_firms, c_tot):
#     """
#     Generates a random technology matrix according to the parameters c_tot -> number of nonzero entries of each row and
#     each row sums to 1 and diagonal is 0.
#     :param n_firms: number of firms,
#     :param c_tot: number of nonzero entries of each row and each row sums to 1 and diagonal is 0.
#     :return: Technology matrix.
#     """   
#     T = np.zeros((n_firms, n_firms))
#     for i in range(n_firms):
#         ind = np.random.choice(np.arange(0, n_firms), c_tot, replace=False)
#         ind = ind[ind != i]
#         T[i, ind] = np.random.dirichlet(np.ones(c_tot), size=1)
#     return T



# def undir_rrg(d, n):
#     """
#     Generates an undirected d-regular network on n nodes.
#     :param d: node connectivity,
#     :param n: number of nodes.
#     :return: Adjacency matrix of the network.
#     """
#     return np.array(nx.convert_matrix.to_numpy_array(nx.random_regular_graph(d, n)))


# def dir_rrg(d, n):
#     """
#     Generates a directed d-regular network on n nodes with in and out connectivity d. Note: bad implementation but
#     still really efficient because of very few cases for which the generated network does not satisfy the while
#     loop condition.
#     :param d: node connectivity,
#     :param n: number of nodes.
#     :return: Adjacency matrix of the network.
#     """
#     A = np.zeros((n, n))
#     while not ((np.sum(A, axis=0) == d).all() and (np.sum(A, axis=1) == d).all()):
#         A = np.zeros((n, n))
#         ind = np.random.choice(np.arange(1, n), d, replace=False)
#         A[0, ind] = 1
#         for k in range(1, n):
#             sums = np.sum(A, axis=0)
#             m = np.where(np.min(sums) == sums)[0]
#             m = m[m != k]
#             if len(m) < d:
#                 r = len(m)
#                 it = r
#                 it_aux = 1
#                 while it < d:
#                     maux = np.where(np.min(sums) + it_aux == sums)[0]
#                     maux = maux[maux != k]
#                     if len(maux) < d - it:
#                         m = np.append(m, maux)
#                     else:
#                         aux = np.random.choice(maux, d - it, replace=False)
#                         m = np.append(m, aux)
#                     it += min(len(maux), d - it)
#                     it_aux += 1
#                 ind = m
#             else:
#                 ind = np.random.choice(m, d, replace=False)
#             A[k, ind] = 1
#     return A


# def mdir_rrg(d, n):
#     """
#     Generates a directed regular network with in and out average connectivity d.
#     :param d: average node connectivity,
#     :param n: number of nodes.
#     :return: Adjacency matrix of the network.
#     """
#     A1 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(d, n))
#     A2 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(d, n))
#     return np.triu(A1) + np.tril(A2)


# def er(n, p, directed=False):
#     """
#     Generates an undirected (or directed) Erdös-Renyi network with link probability p.
#     :param p: probability for link presence,
#     :param n: number of nodes,
#     :param directed: whether or not the network is directed, default False.
#     :return: Adjacency matrix of the network.
#     """
#     return np.array(nx.convert_matrix.to_numpy_array(nx.binomial_graph(n, p, directed=directed)))


# def create_net(net_str, directed, n, d):
#     """
#     Generates the prescribed network.
#     :param net_str: type of network - 'regular' for regular, 'm-regular' for multi-regular, 'er' for Erdös-Renyi,
#     :param directed: whether or not the network is directed,
#     :param n: number of nodes,
#     :param d: average connectivity.
#     :return: Adjacency matrix of the network.
#     """
#     if directed:
#         if net_str == 'regular':
#             return dir_rrg(d, n)
#         elif net_str == 'm_regular':
#             return mdir_rrg(d, n)
#         elif net_str == 'er':
#             return er(n, d / n, directed=directed)
#     else:
#         if net_str == 'regular':
#             return undir_rrg(d, n)
#         elif net_str == 'er':
#             return er(n, d / n)
#         else:
#             raise Exception("Not coded yet")


# # Graphical representation of networks from stack overflow
# # https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx

# def community_layout(g, partition):
#     """
#     Computes the layout for a modular graph.
#     :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
#     :param partition: dict mapping int node -> int community graph partitions.
#     :return: dict mapping int node -> (float x, float y) node positions.
#     """
#     pos_communities = _position_communities(g, partition, scale=3.)
#     pos_nodes = _position_nodes(g, partition, scale=1.)

#     # combine positions
#     pos = dict()
#     for node in g.nodes():
#         pos[node] = pos_communities[node] + pos_nodes[node]
#     return pos


# def _position_communities(g, partition, **kwargs):
#     """
#     Creates a weighted graph, in which each node corresponds to a community,
#     and each edge weight to the number of edges between communities
#     :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
#     :param partition: dict mapping int node -> int community graph partitions,
#     :return: dict mapping int node -> (float x, float y) community positions.
#     """
#     between_community_edges = _find_between_community_edges(g, partition)
#     communities = set(partition.values())
#     hypergraph = nx.DiGraph()
#     hypergraph.add_nodes_from(communities)
#     for (ci, cj), edges in between_community_edges.items():
#         hypergraph.add_edge(ci, cj, weight=len(edges))

#     # find layout for communities
#     pos_communities = nx.spring_layout(hypergraph, **kwargs)

#     # set node positions to position of community
#     pos = dict()
#     for node, community in partition.items():
#         pos[node] = pos_communities[community]
#     return pos


# def _find_between_community_edges(g, partition):
#     """
#     Determines edges between communities.
#     :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
#     :param partition: dict mapping int node -> int community graph partitions,
#     :return: dict mapping tuple (community, community) -> (float x, float y) edges.
#     """
#     edges = dict()

#     for (ni, nj) in g.edges():
#         ci = partition[ni]
#         cj = partition[nj]

#         if ci != cj:
#             try:
#                 edges[(ci, cj)] += [(ni, nj)]
#             except KeyError:
#                 edges[(ci, cj)] = [(ni, nj)]

#     return edges


# def _position_nodes(g, partition, **kwargs):
#     """"
#     Positions nodes within communities.
#     :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
#     :param partition: dict mapping int node -> int community graph partitions,
#     :return: dict mapping int node -> (float x, float y) node positions.
#     """
#     communities = dict()
#     for node, community in partition.items():
#         try:
#             communities[community] += [node]
#         except KeyError:
#             communities[community] = [node]

#     pos = dict()
#     for ci, nodes in communities.items():
#         subgraph = g.subgraph(nodes)
#         pos_subgraph = nx.spring_layout(subgraph, **kwargs)
#         pos.update(pos_subgraph)

#     return pos

