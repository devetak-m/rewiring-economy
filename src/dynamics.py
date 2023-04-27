# network-economy is a simulation program for the Network Economy ABM
# Copyright (C) 2020 Mitja Devetak
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
The ``dynamics`` module
======================

This module contains the class that implements the dynamics of the network economy.
"""
import numpy as np


class Dynamics:
    """Class that implements the dynamics of the network economy."""

    def __init__(self, firms, rmax, stiffness=1, verbose=False):
        # an instance of the Firms class on which the dynamics is computed
        self.firms = firms

        # number of firms
        self.n_firms = len(firms.a)

        # maximum number of rounds

        # stiffness parameter
        self.stiffness = stiffness
        self.verbose = verbose

        self.rmax = rmax

        # current round
        self.current_round = 0

        # current time
        self.current_time = 0

        # initialize time series

        self.profits_series = np.zeros((self.rmax * self.n_firms, self.n_firms))
        self.p_series = np.zeros((self.rmax * self.n_firms, self.n_firms))
        self.x_series = np.zeros((self.rmax * self.n_firms, self.n_firms))
        self.l_series = np.zeros((self.rmax * self.n_firms, self.n_firms))
        self.h_series = np.zeros((self.rmax * self.n_firms, 1))
        self.household_utility = np.zeros((self.rmax * self.n_firms, 1))
        self.trophic_incoherence_series = np.zeros((self.rmax * self.n_firms, 1))

        self.supply_network_series = np.zeros((self.rmax, self.n_firms * (self.n_firms - 1)))

        # initialize observables
        self.rewiring_occourences_series = -np.ones((self.rmax * self.n_firms, 1))

    def compute_possible_rewirings(self, i):
        """Compute possible rewirings for firm i in the network.
        Returns a list of lists containing elements of the form [j,k, l]
        where j is the firm that is removed, k is the firm that is wired to and
        l is the share in production of firm k product to be assigned to firm i
        if the rewiring is performed. [i,i,0] is the 'do nothing' case"""

        possible_rewirings = []

        # append the 'do nothing' case
        possible_rewirings.append([i, i, 0])
        for j in range(self.n_firms):
            if self.firms.supply_network[j, i] != 0:
                # find all the firms that could be wired to i
                # but are not already wired to i from the technology matrix
                for k in range(self.n_firms):
                    if self.firms.technology_network[k, i] != 0 and self.firms.supply_network[k, i] == 0:
                        possible_rewirings.append([j, k, self.firms.technology_network[k, i]])

        return possible_rewirings

    def compute_round(self):
        """Compute one round of the dynamics.
        return a flag that is True if the network has changed and False otherwise"""

        # create flag
        network_changed = False
        # update the time
        self.current_time = 0

        # create a list of firms to be updated
        firms_to_update = list(range(self.n_firms))

        # shuffle the list
        np.random.shuffle(firms_to_update)

        # loop over firms
        for i in firms_to_update:
            # compute possible rewirings
            possible_rewirings = self.compute_possible_rewirings(i)
            current_profit = self.firms.profits[i]
            current_rewiring = [i, i, 0]
            current_connectivity_matrix = self.firms.supply_network.copy()

            # loop over possible rewirings
            for rewiring in possible_rewirings[1:]:
                old_supplier = rewiring[0]
                new_supplier = rewiring[1]
                capacity_new_supplier = rewiring[2]
                capacity_old_supplier = self.firms.supply_network[old_supplier, i]
                # change the network matrix
                current_connectivity_matrix[old_supplier, i] = 0
                current_connectivity_matrix[new_supplier, i] = capacity_new_supplier
                # compute expected profits
                expected_profits = self.firms.compute_expected_profits(i, current_connectivity_matrix)
                # if the expected profit is higher than the current one, update
                if self.stiffness * expected_profits > current_profit:
                    current_profit = expected_profits
                    current_rewiring = rewiring
                    network_changed = True
                    self.rewiring_occourences_series[self.current_round * self.n_firms + self.current_time, :] = i
                # revert the network matrix
                current_connectivity_matrix[old_supplier, i] = capacity_old_supplier
                current_connectivity_matrix[new_supplier, i] = 0

            # update the network matrix
            self.firms.supply_network[current_rewiring[0], i] = 0
            self.firms.supply_network[current_rewiring[1], i] = current_rewiring[2]

            # update the firms economy
            self.firms.update_equilibrium()

            # update the time series
            current_position = self.current_round * self.n_firms + self.current_time
            self.profits_series[current_position, :] = self.firms.profits
            self.p_series[current_position, :] = self.firms.prices
            self.x_series[current_position, :] = self.firms.sales
            self.l_series[current_position, :] = self.firms.labour_hired
            self.h_series[current_position, :] = self.firms.wage
            self.household_utility[current_position, :] = self.firms.compute_household_utility()
            self.trophic_incoherence_series[current_position, :] = self.firms.compute_trophic_incoherence()

            # update the time
            self.current_time += 1

        time_series_network = self.firms.supply_network.copy()
        # remove diagonal
        time_series_network = time_series_network[~np.eye(time_series_network.shape[0], dtype=bool)].reshape(
            time_series_network.shape[0], -1
        )
        self.supply_network_series[self.current_round, :] = time_series_network.flatten()

        # update the round
        self.current_round += 1

        return network_changed

    def compute_dynamics(self):
        """Compute the dynamics until the network is stable"""

        # create flag
        network_changed = True

        # loop over rounds
        while network_changed and self.current_round < self.rmax:
            if self.current_round % 99 == 0 and self.verbose:
                print("Computing round: ", self.current_round + 1)
            network_changed = self.compute_round()
