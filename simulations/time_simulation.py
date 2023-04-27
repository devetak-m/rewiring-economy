import cProfile, pstats, io
from pstats import SortKey
import pandas as pd
import numpy as np
import sys

sys.path.append("../src")
from firms import Firms
from dynamics import Dynamics
from network import *


pr = cProfile.Profile()
pr.enable()

# fix random seed of numpy
np.random.seed(33)

# set up firms
n_firms = 50
n_periods = 300
a = np.ones(n_firms) * 0.5
b = np.ones(n_firms) * 0.9
z = np.ones(n_firms)
tau = np.ones(n_firms).astype(int)

c = 4
c_prime = 8

# generaet technology matrix
T = generate_base_case_tech_matrix(n_firms, c + c_prime, 1 / c)

# generate connectivity matrix
W = generate_connectivity_matrix(T, c)

# set up firms
firms = Firms(a, z, b, tau, W, T)

# set up dynamics
dynamics = Dynamics(firms, n_periods)

# run simulation
dynamics.compute_dynamics()

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
