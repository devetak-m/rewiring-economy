import sys
sys.path.append('../src')
from firms import Firms
from dynamics import Dynamics
from network import *
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# fix random seed of numpy
np.random.seed(33)

# set up firms
n_firms = 10
n_periods = 300
a = np.ones(n_firms) * 0.5
b = np.ones(n_firms) * 1.6
z = np.ones(n_firms)
tau = - np.ones(n_firms)

c = 2
c_prime = 2

# generaet technology matrix
technology_matrix = generate_base_case_tech_matrix(n_firms, c + c_prime, 1/c)

# generate connectivity matrix
supply_matrix = generate_connectivity_matrix(technology_matrix, c)

# set up firms
firms = Firms(a, z, b, tau, supply_matrix, technology_matrix)

# set up dynamics
dynamics = Dynamics(firms, n_periods)

# run simulation
dynamics.compute_dynamics()

# get which was the final round
final_round = dynamics.current_round

print(f"Final round: {final_round}")
if final_round == n_periods:
    print("Warning: simulation did not converge")
else:
    print("Simulation converged")

# compute number of rewirings
rewiring_series = dynamics.rewiring_occourences_series[:final_round * n_firms]
n_rewirings = np.count_nonzero(rewiring_series != -1)
print(f"Number of rewirings: {n_rewirings}")

# compute number of rewirings per firms
number_of_rewirings_per_firm = np.zeros(n_firms)
for i in range(n_firms):
    number_of_rewirings_per_firm[i] = np.sum(dynamics.rewiring_occourences_series[:final_round * n_firms] == i)

# get the network time series
supply_network_series = dynamics.supply_network_series[:final_round].T

# get only the time series corresponding to the non-zero entries in the technology matrix
technology_matrix_series = technology_matrix.copy()
# remove diagonal
technology_matrix_series = technology_matrix_series[~np.eye(technology_matrix_series.shape[0],dtype=bool)].reshape(technology_matrix_series.shape[0],-1)    
technology_matrix_series = technology_matrix_series.flatten()
indices = np.where(technology_matrix_series != 0)
supply_network_series = supply_network_series[indices]
supply_network_series[supply_network_series != 0] = 1

# compute autocorrelation
nlags = 1
autocorrelation = np.zeros((len(supply_network_series), nlags+1))
for i in range(supply_network_series.shape[0]):
   autocorrelation[i,:] = sm.tsa.stattools.acf(supply_network_series[i, :], nlags=nlags)

# remove nan rows
autocorrelation = autocorrelation[~np.isnan(autocorrelation).any(axis=1)]

# compute mean autocorrelation
autocorrelation = np.mean(autocorrelation, axis=0)
x = np.arange(0, nlags+1, 1)
plt.plot(x, autocorrelation)
plt.xlabel("Regression lag")
plt.ylabel("Autocorrelation")
# add a horizontal line at y=0
plt.axhline(y=0, color='k')
plt.show()  