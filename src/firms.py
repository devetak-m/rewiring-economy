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
The ``firms`` module
======================

This module contains the class that implements the firms of the network economy.
"""
import numpy as np
import network as net


class Firms:
    """Class that implements the firms of the network economy."""

    def __init__(self, a, z, b, tau, supply_network, technology_network):
        # a : labor intensity vector
        # z : productivity vector
        # b : return to scale vector
        # tau : foresight of firms vector
        # W : current supply network
        # T : technology matrix
        self.a = a
        self.z = z
        self.b = b
        self.tau = tau
        self.supply_network = supply_network
        self.technology_network = technology_network

        # store equilibrium values
        self.profits = None
        self.prices = None
        self.sales = None
        self.labour_hired = None
        self.wage = None

        # compute equilibrium values
        self.profits, self.prices, self.sales, self.labour_hired, self.wage = self.compute_equilibrium()

    # Setters for class instances

    def update_a(self, a):
        self.a = a

    def update_z(self, z):
        self.z = z

    def update_b(self, b):
        self.b = b

    def update_tau(self, tau):
        self.tau = tau

    def update_supply_network(self, new_supply_network, recalculate_equilibrium=True):
        # check that the new supply network is a subset of the technology network
        if not net.is_subset(new_supply_network, self.technology_network):
            raise ValueError("New supply network is not a subset of the current technology network.")
        self.supply_network = new_supply_network
        # update equilibrium values
        if recalculate_equilibrium:
            self.update_equilibrium()

    def update_technology_network(self, new_technology_network):
        # check that the new technology network is a superset of the supply network
        if not net.is_superset(new_technology_network, self.supply_network):
            raise ValueError("New technology network is not a superset of the current supply network.")
        self.technology_network = new_technology_network

    def update_a(self, a):
        self.a = a

    def update_z(self, z):
        self.z = z

    def update_b(self, b):
        self.b = b

    def update_tau(self, tau):
        self.tau = tau

    def update_equilibrium(self):
        self.profits, self.prices, self.sales, self.labour_hired, self.wage = self.compute_equilibrium()

    # Compute profits at equilibrium
    def compute_equilibrium(self):
        # compute W_tilde matrix
        W_tilde = self.compute_W_tilde()
        # solve for V
        V = self.compute_V(W_tilde)
        # compute wage h
        h = self.compute_h(V)
        # compute constant for log prices
        C = self.compute_C(V, h)
        # compute W_prime
        W_prime = self.compute_W_prime()
        # compute log prices
        lp = self.compute_lp(W_prime, C)
        # compute prices
        p = self.compute_p(lp)
        # compute production quantities
        x = self.compute_x(V, p)
        # compute ammount of hired labor
        l = self.compute_l(x, p, h)
        # compute g
        g = self.compute_g(p, x)
        # compute profits
        P = self.compute_P(h, x, p, l, g)
        return P, p, x, l, h

    # Compute W_tilde matrix
    # (W_tilde)ij = (1 − a_j)b_j w_ij
    def compute_W_tilde(self):
        return (1 - self.a) * self.b * self.supply_network

    # Solve for V
    # V = (I − W_tilde)^(-1) * (1/n)
    def compute_V(self, W_tilde):
        # V = np.linalg.inv(np.eye(W_tilde.shape[0]) - W_tilde) @ (np.ones(W_tilde.shape[0]))
        V = np.linalg.solve(np.eye(W_tilde.shape[0]) - W_tilde, np.ones(W_tilde.shape[0]))
        V = V / V.shape[0]
        return V

    # Compute wage h
    # h = sum_j (1 - a_j) * b_j * V_j * 1/n
    def compute_h(self, V):
        # h = 0
        # for j in range(V.shape[0]):
        #     h += (1 - self.a[j]) * self.b[j] * V[j]
        h = np.sum((1 - self.a) * self.b * V)
        return h / V.shape[0]

    # Compute constant for log prices
    # C_i = log(z_i^(-1) * b_i^(-b_i) * v_i^(1 - bi) * h^(a_ib_i))
    def compute_C(self, V, h):
        return np.log(self.z ** (-1) * self.b ** (-self.b) * V ** (1 - self.b) * h ** (self.a * self.b))

    # Compute W_prime matrix
    # W_prime(i,j) = (1 - a(i)) * b(i) * W(j,i)
    # W_prime_ij = (1 - a_i) * W_ji * b_i
    def compute_W_prime(self):
        return ((1 - self.a) * self.supply_network * self.b).T

    # Compute log prices
    # lp = (I - W_prime)^(-1) * C
    def compute_lp(self, W_prime, C):
        lp = np.linalg.solve(np.eye(W_prime.shape[0]) - W_prime, C)
        return lp

    def compute_p(self, lp):
        return np.exp(lp)

    # Compute production quantities
    # x_i = V_i/p_i
    def compute_x(self, V, p):
        return V / p

    # Compute ammount of hired labor
    # l_i = a_i * b_i * p_i * x_i / h
    def compute_l(self, x, p, h):
        l = self.a * self.b * p * x / h
        return l

    # Compute g
    # g_ji = (1 - a_i) * W_ji * b_i * p_i * x_i / p_j
    def compute_g(self, p, x):
        g = (1 - self.a) * self.supply_network * self.b * p * x / p[:, None]
        return g

    # Compute profits
    # P_i = p_i * x_i - l_i * h - sum_j p_j * g_ji
    def compute_P(self, h, x, p, l, g):
        P = p * x - l * h - g.T @ p
        return P

    # compute expected equilibrium profits for firm i with foresight tau_i and netowrk W'
    def compute_expected_profits(self, i, W_new):
        tau_i = self.tau[i]

        # if tau = -1, then firm i has perfect foresight
        if tau_i == -1:
            # compute profits with W = W'
            # store current W
            W = self.supply_network
            # store current state
            current_profits, current_prices, current_sales, current_labour_hired, current_wage = (
                self.profits,
                self.prices,
                self.sales,
                self.labour_hired,
                self.wage,
            )
            # update W
            self.update_supply_network(W_new)
            # compute profits
            new_profits = self.profits
            # W = W again
            self.update_supply_network(W, recalculate_equilibrium=False)
            # profits = profits again
            self.profits, self.prices, self.sales, self.labor, self.wage = (
                current_profits,
                current_prices,
                current_sales,
                current_labour_hired,
                current_wage,
            )
            return new_profits[i]

        else:
            # store current W
            W = self.supply_network

            # firm has imperfect foresight
            # compute indices of firms that are known by firm under inspection
            indices_known = self.compute_indices_known(i, tau_i)
            # compute indices of firms that are not known by firm under inspection
            indices_unknown = self.compute_indices_unknown(indices_known)
            # update W
            self.update_supply_network(W_new, recalculate_equilibrium=False)
            # compute expected profits with W = W_new
            K1 = self.compute_K1(indices_known, indices_unknown)
            eW_tilde = self.compute_eW_tilde(indices_known)
            eV = self.compute_eV(eW_tilde, K1)
            K2 = self.compute_K2(indices_known)
            eh = self.compute_eh(eV, K2, indices_known)
            lK3 = self.compute_lK3(indices_known, indices_unknown)
            eC = self.compute_eC(eV, eh, lK3, indices_known)
            eW_prime = self.compute_eW_prime(indices_known)
            # compute expected log prices
            elp = self.compute_elp(eW_prime, eC)
            # compute expected prices
            ep = self.compute_p(elp)
            ep_all = self.compute_ep_all(ep, indices_known, indices_unknown)
            # compute expected production
            ex = self.compute_x(eV, ep)
            ex_all = self.compute_ex_all(ex, indices_known, indices_unknown)
            # compute expected labor
            el = self.compute_l(ex_all, ep_all, eh)
            eg = self.compute_g(ep_all, ex_all)
            # compute expected profits
            eP = self.compute_P(eh, ex_all, ep_all, el, eg)
            # W = W again
            self.update_supply_network(W, recalculate_equilibrium=False)
            return eP[i]

    # compute indices of firms that are known by firm under inspection
    def compute_indices_known(self, i, tau):
        indices_known = [i]
        if tau == 0:
            return indices_known
        for j in range(self.supply_network.shape[0]):
            if (self.supply_network[i, j] > 0 and j != i) or (self.supply_network[j, i] > 0 and j != i):
                indices_known.append(j)
        for k in range(1, tau):
            new_indices = []
            for j in indices_known:
                for l in range(self.supply_network.shape[0]):
                    if ((self.supply_network[j, l] > 0 and l != j) or (self.supply_network[l, j] > 0 and l != j)) and (
                        l not in indices_known
                    ):
                        new_indices.append(l)
            indices_known += new_indices
        return indices_known

    # compute indices of firms that are not known by firm under inspection
    def compute_indices_unknown(self, indices_known):
        indices_unknown = []
        for j in range(self.supply_network.shape[0]):
            if j not in indices_known:
                indices_unknown.append(j)
        return indices_unknown

    # compute K1
    # K1_k = sum_{j in J} (1 - a_j) * b_j * W_kj * V_j
    def compute_K1(self, indices_known, indices_unknown):
        K1 = np.sum(
            (1 - self.a[indices_unknown])
            * self.b[indices_unknown]
            * self.supply_network[indices_known, :][:, indices_unknown]
            * self.prices[indices_unknown]
            * self.sales[indices_unknown],
            axis=1,
        )
        return K1

    # compute W_tilde matrix for known firms
    # (eW_tilde)_ij = (1 - a_j) * b_j * w_ij
    def compute_eW_tilde(self, indices_known):
        eW_tilde = (
            (1 - self.a[indices_known])
            * self.b[indices_known]
            * self.supply_network[indices_known, :][:, indices_known]
        )
        return eW_tilde

    # compute expected V for known firms
    # V = (I - eW_tilde)^(-1) * (1/n + K1)
    def compute_eV(self, eW_tilde, K1):
        I = np.eye(len(eW_tilde))
        n = self.profits.shape[0]
        return np.linalg.solve(I - eW_tilde, np.ones(len(eW_tilde)) / n + K1)

    # compute K2 constant for known firms
    # K2 = sum_k a_k * b_k * v_k / h
    def compute_K2(self, indices_known):
        K2 = np.sum(
            self.a[indices_known] * self.b[indices_known] * self.prices[indices_known] * self.sales[indices_known]
        )
        return K2 / self.wage

    # compute expected wage h
    # eh = (1 / K2) sum_k a_k * b_k eV_k
    def compute_eh(self, eV, K2, indices_known):
        eh = np.sum(self.a[indices_known] * self.b[indices_known] * eV)
        return eh / K2

    # compute log(K3) constant for known firms
    # lK3_k = sum_j (1 - a_j) * b_j * w_ji * b_i * log(p_j)
    def compute_lK3(self, indices_known, indices_unknown):
        lK3 = np.sum(
            (1 - self.a[indices_known])
            * self.b[indices_known]
            * self.supply_network[indices_unknown, :][:, indices_known]
            * np.log(self.prices[indices_unknown]).reshape(-1, 1),
            axis=0,
        )
        return lK3

    # compute constant for log prices of known firms
    # Ce_k = log(z_k^(-1) * b_k^(-bk) * v_k^(1 - bk) * h^(a_kb_k)) + lK3_k
    def compute_eC(self, eV, eh, lK3, indices_known):
        eC = (
            np.log(
                self.z[indices_known] ** (-1)
                * self.b[indices_known] ** (-self.b[indices_known])
                * eV ** (1 - self.b[indices_known])
                * eh ** (self.a[indices_known] * self.b[indices_known])
            )
            + lK3
        )
        return eC

    # W_prime(i,j) = (1 - a(i)) * b(i) * W(j,i)
    # W_prime_ij = (1 - a_i) * W_ji * b_i
    # for i and j known
    def compute_eW_prime(self, indices_known):
        return (
            (1 - self.a[indices_known])
            * self.supply_network[indices_known, :][:, indices_known]
            * self.b[indices_known]
        )

    # compute expected log prices for known firms
    # lp = (1 - W_tilde)^(-1) * Ce
    def compute_elp(self, eW_prime, eC):
        return np.linalg.solve(np.eye(len(eW_prime)) - eW_prime, eC)

    # compute expected log prices for all firms
    # ep_j = ep_k if j is known and ep_j = p_j if j is unknown
    def compute_ep_all(self, ep, indices_known, indices_unknown):
        ep_all = np.zeros(self.supply_network.shape[0])
        for i, k in enumerate(indices_known):
            ep_all[k] = ep[i]
        for j in indices_unknown:
            ep_all[j] = self.prices[j]
        return ep_all

    # compute expected log quantities for all firms
    # eq_j = eq_k if j is known and eq_j = q_j if j is unknown
    def compute_ex_all(self, ex, indices_known, indices_unknown):
        ex_all = np.zeros(self.supply_network.shape[0])
        for i, k in enumerate(indices_known):
            ex_all[k] = ex[i]
        for j in indices_unknown:
            ex_all[j] = self.sales[j]
        return ex_all

    # compute household utility
    # - sum_j log(p_j)
    def compute_household_utility(self):
        return -np.sum(np.log(self.prices))

    def compute_trophic_incoherence(self):
        return net.compute_trophic_incoherence(self.supply_network)
