from joblib import Parallel, delayed
import numpy as np
import os
import sys
sys.path.append('../src')
import matplotlib.pyplot as plt
from firms import Firms
from dynamics import Dynamics
from network import *


def run_simulation(parameters):
    # save location
    save_location = "/Users/mitjadevetak/Desktop/projects/rewiring-economy/results/"
    # folder name
    folder_name = f"n_firms_{parameters[0]}_n_supplier_{parameters[1]}_n_potential_supplier_{parameters[2]}_tau_{parameters[3]}_stiffness_{parameters[4]}"
    # Path
    path = os.path.join(save_location, folder_name)
    # create the folder    
    os.mkdir(path)
    # fix random seed of numpy
    np.random.seed(33)

    # set up firms
    n_firms = parameters[0]
    n_supplier = parameters[1]
    n_potential_supplier = parameters[2]
    tau = parameters[3]
    stiffness = parameters[4]

    a = np.ones(n_firms) * 0.5
    b = np.ones(n_firms) * 0.9
    z = np.ones(n_firms)
    tau = np.ones(n_firms) * tau
    # typecase tau to int
    tau = tau.astype(int)
    

    T = generate_base_case_tech_matrix(n_firms, n_supplier + n_potential_supplier, 1/n_supplier)
    W = generate_connectivity_matrix(T, n_supplier)

    # save a plot of the initial network W
    plot_connectivity_network(W, save_location + folder_name + "/initial_network.png")

    # set up firms
    firms = Firms(a, z, b, tau, W, T)

    # set up dynamics
    dynamics = Dynamics(firms, 200, stiffness = stiffness)

    # run simulation
    dynamics.compute_dynamics()

    # save a plot of the final network W
    plot_connectivity_network(dynamics.firms.W, save_location + folder_name + "/final_network.png")

    # save a plot of the household utility
    household_utility = dynamics.household_utility[:dynamics.r * n_firms]
    plt.plot(household_utility)
    plt.xlabel("Round")
    plt.ylabel("Household utility")
    plt.savefig(save_location + folder_name + "/household_utility.png")
    plt.close()

    # save a plot of the trophic incoherence
    trophic_level = dynamics.trophic_incoherence_series[:dynamics.r * n_firms]
    plt.plot(trophic_level)
    plt.xlabel("Round")
    plt.ylabel("Trophic level")
    plt.savefig(save_location + folder_name + "/trophic_level.png")

    # save the rewiring series
    rewiring_series = dynamics.rewiring_occourences_series[:dynamics.r * n_firms]
    np.save(save_location + folder_name + "/rewiring_series.npy", rewiring_series)

    # compute the total number of rewirings
    n_rewirings = np.count_nonzero(rewiring_series != -1)

    # in a txt file save the total number of rewirings and the final and initial thropic level
    with open(save_location + folder_name + "/info.txt", "w") as f:
        f.write(f"Total number of rewirings: {n_rewirings}\n")
        f.write(f"Initial trophic level: {trophic_level[0]}\n")
        f.write(f"Final trophic level: {trophic_level[-1]}\n")


if __name__ == '__main__':
    # Define the parameters
    
    # network sizes
    n_firms = [10, 50, 100, 200]
    # number of suppliers
    n_suppliers = [2, 4, 6]
    # number of potential suppliers
    n_potential_suppliers = [2, 4, 6]
    # foresight of firms
    taus = [-1, 0, 1, 2]
    # stiffness of the network
    stiffness = [0.8, 0.90, 0.95, 0.99, 1]

    # # make a small test
    # n_firms = [10]
    # # number of suppliers
    # n_suppliers = [2]
    # # number of potential suppliers
    # n_potential_suppliers = [2]
    # # foresight of firms
    # taus = [-1]
    # # stiffness of the network
    # stiffness = [1]

    # create the list of parameters
    all_parameters = []
    for n_firm in n_firms:
        for n_supplier in n_suppliers:
            for n_potential_supplier in n_potential_suppliers:
                for tau in taus:
                    for stiff in stiffness:
                        all_parameters.append([n_firm, n_supplier, n_potential_supplier, tau, stiff])

    # run the simulations
    Parallel(n_jobs=12)(delayed(run_simulation)(pameters) for pameters in all_parameters)
    
    # for parameters in all_parameters:
    #     run_simulation(parameters)