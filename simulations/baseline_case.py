import sys
sys.path.append('../src')
from firms import Firms
from dynamics import Dynamics
from network import *
import numpy as np

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
