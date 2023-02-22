import numpy as np

class Dynamics:

    def __init__(self, firms, tmax):
        # an instance of the Firms class on which the dynamics is computed
        self.firms = firms

        # number of firms
        self.n_firms = len(firms.a)

        # maximum number of rounds
        self.tmax = tmax

        # current round
        self.r = 0

        # initialize time series
        self.P_series = np.zeros((self.tmax * self.n_firms, self.n_firms))
        self.p_series = np.zeros((self.tmax * self.n_firms, self.n_firms))
        self.x_series = np.zeros((self.tmax * self.n_firms, self.n_firms))
        self.l_series = np.zeros((self.tmax * self.n_firms, self.n_firms))
        self.h_series = np.zeros((self.tmax * self.n_firms, 1))

    def compute_possible_rewirings(self, i):
        """ Compute possible rewirings for firm i in the network.
        Returns a list of lists containing elements of the form [j,k, l]
        where j is the firm that is removed, k is the firm that is wired to and
        l is the share in production of firm k product to be assigned to firm i
        if the rewiring is performed. [i,i,0] is the 'do nothing' case """

        possible_rewirings = []

        # append the 'do nothing' case
        possible_rewirings.append([i, i, 0])
        for j in range(self.n_firms):
            if self.firms.W[j,i] != 0:
                # find all the firms that could be wired to i
                # but are not already wired to i from the technology matrix
                for k in range(self.n_firms):
                    if self.firms.T[k,i] != 0 and self.firms.W[k,i] == 0:
                        possible_rewirings.append([j,k,self.firms.T[k,i]])

        return possible_rewirings

    def compute_round(self):
        """ Compute one round of the dynamics.
         return a flag that is True if the network has changed and False otherwise """


        # create flag
        network_changed = False

        # create a list of firms to be updated
        firms_to_update = list(range(self.n_firms))

        # shuffle the list
        np.random.shuffle(firms_to_update)

        # loop over firms
        for i in firms_to_update:

            # compute possible rewirings
            possible_rewirings = self.compute_possible_rewirings(i)
            current_profit = self.firms.P[i]
            current_rewiring = [i,i,0]
            current_W = self.firms.W.copy()

            # loop over possible rewirings
            for rewiring in possible_rewirings[1:]:
                old_supplier = rewiring[0]
                new_supplier = rewiring[1]
                capacity_new_supplier = rewiring[2]
                capacity_old_supplier = self.firms.W[old_supplier,i]
                # change the network matrix
                current_W[old_supplier,i] = 0
                current_W[new_supplier,i] = capacity_new_supplier
                # compute expected profits
                expected_profits = self.firms.compute_expected_profits(i, current_W)
                # if the expected profit is higher than the current one, update
                if expected_profits > current_profit:
                    current_profit = expected_profits
                    current_rewiring = rewiring
                    network_changed = True
                # revert the network matrix
                current_W[old_supplier,i] = capacity_old_supplier
                current_W[new_supplier,i] = 0
            
            # update the network matrix
            self.firms.W[current_rewiring[0],i] = 0
            self.firms.W[current_rewiring[1],i] = current_rewiring[2]

            # update the firms economy
            self.firms.update_equilibrium()

        return network_changed
    





    