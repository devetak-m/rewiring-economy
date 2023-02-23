import sys
sys.path.append('../src')
from firms import Firms
from dynamics import Dynamics
from network import *
import numpy as np

# fix random seed of numpy
np.random.seed(33)

# set up firms
n_firms = 10
n_periods = 100
a = np.ones(n_firms) * 0.5
b = np.ones(n_firms) * 0.9
z = np.ones(n_firms)
tau = np.zeros(n_firms)

c = 1
c_prime = 1

# generaet technology matrix
T = generate_base_case_tech_matrix(n_firms, c + c_prime, 1/c)

# generate connectivity matrix
W = generate_connectivity_matrix(T, c)

# pot connectivity matrix
plot_connectivity_network(W)

# set up firms
firms = Firms(a, z, b, tau, W, T)

# set up dynamics
dynamics = Dynamics(firms, n_periods)

# run simulation
dynamics.compute_dynamics()

# plot network
plot_connectivity_network(dynamics.firms.W)

# get which was the final round
final_round = dynamics.r

# plot the evolution of household utility
household_utility = dynamics.household_utility[:final_round * n_firms]
plt.plot(household_utility)
plt.show()

# rewiring series
rewiring_series = dynamics.rewiring_occourences_series[:final_round * n_firms]

# count the elements that are not -1
n_rewirings = np.count_nonzero(rewiring_series != -1)

print(f"Number of rewirings: {n_rewirings}")
