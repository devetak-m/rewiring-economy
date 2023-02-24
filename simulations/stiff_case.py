import sys
sys.path.append('../src')
from firms import Firms
from dynamics import Dynamics
from network import *
import numpy as np

# fix random seed of numpy
np.random.seed(33)

# set up firms
n_firms = 20
n_periods = 100
a = np.ones(n_firms) * 0.5
b = np.ones(n_firms) * 0.9
z = np.ones(n_firms)
tau = - np.ones(n_firms)

c = 2
c_prime = 4

# generaet technology matrix
T = generate_base_case_tech_matrix(n_firms, c + c_prime, 1/c)

# generate connectivity matrix
W = generate_connectivity_matrix(T, c)

# pot connectivity matrix
plot_connectivity_network(W)

# set up firms
firms = Firms(a, z, b, tau, W, T)

# set up dynamics
dynamics = Dynamics(firms, n_periods, stiffness = 1)

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

if dynamics.r < dynamics.rmax:
    print("Simulation stopped due to equilibrium")
else:
    print("Simulation stopped due to the end of the simulation period")

# profit series
profit_series = dynamics.P_series[:final_round * n_firms]

# get the profits of firm 0, 9 and 12
firm_0 = profit_series[:,0]
firm_9 = profit_series[:,9]
firm_12 = profit_series[:,12]

# plot the profits of firm 0, 9 and 12
plt.plot(firm_0, label = "firm 0")
plt.plot(firm_9, label = "firm 9")
plt.plot(firm_12, label = "firm 12")
plt.legend()
plt.show()

# plot the evolution of trophic incoherence
trophic_incoherence_series = dynamics.trophic_incoherence_series[:final_round * n_firms]
plt.plot(trophic_incoherence_series)
plt.show()

