import sys

sys.path.append("../src")
from firms import Firms
from dynamics import Dynamics
from network import *
import numpy as np
from networkx.algorithms import community

# fix random seed of numpy
np.random.seed(33)

# set up firms
n_firms = 8
n_periods = 1000
a = np.ones(n_firms) * 0.5
b = np.ones(n_firms) * 0.9
z = np.ones(n_firms)
tau = np.zeros(n_firms)

c = 2
c_prime = 2
n_communities = 2

W, T = generate_communities_supply(n_firms, n_communities, c)

# pot connectivity matrix
plot_connectivity_network(W)

# set up firms
firms = Firms(a, z, b, tau, W, T)

# set up dynamics
dynamics = Dynamics(firms, n_periods, verbose=True)

# compute the initial community
U = community.greedy_modularity_communities(nx.from_numpy_array(dynamics.firms.supply_network))

# run simulation
dynamics.compute_dynamics()

# plot network
plot_connectivity_network(dynamics.firms.supply_network)

V = community.greedy_modularity_communities(nx.from_numpy_array(dynamics.firms.supply_network))

# get which was the final round
final_round = dynamics.r

# plot the evolution of household utility
household_utility = dynamics.household_utility[: final_round * n_firms]
plt.plot(household_utility)
plt.show()

# rewiring series
rewiring_series = dynamics.rewiring_occourences_series[: final_round * n_firms]

# count the elements that are not -1
n_rewirings = np.count_nonzero(rewiring_series != -1)

# print the mutual information of the partition
# TODO : this is not working
print("Mutual information of the partition: ", mutual_information_of_two_partitions(U, V))
