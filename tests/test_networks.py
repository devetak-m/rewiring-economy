import unittest
import numpy as np
import sys
sys.path.append('../src')
from network import *


class TestNetwork(unittest.TestCase):

    def test_1_base_case(self):

        n_firms = 10
        c = 1
        c_prime = 1
        
        # generate base_case_tech_matrix
        T = generate_base_case_tech_matrix(n_firms, c + c_prime, 1/c)

        # check that all diagonals are zero
        for i in range(n_firms):
            self.assertAlmostEqual(T[i,i], 0)
        
        # check that there are exactly c + c_prime nonzero elements in each column
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(T[:,j]), c + c_prime)

        # check that the nonzero elements of each column are 1/c
        for j in range(n_firms):
            self.assertAlmostEqual(np.sum(T[:, j]), (c + c_prime)/c, 3)

        # check that there are exactly c + c_prime nonzero elements in each column
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(T[:, j]), c + c_prime)
        
        # generate a connectivity matrix from T
        W = generate_connectivity_matrix(T, c)

        # check that there are exactly c nonzero elements in each column
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(W[:, j]), c)
        
        # check that the columns of W sum to 1
        for j in range(n_firms):
            self.assertAlmostEqual(np.sum(W[:, j]), 1)
        
        # check that the diagonal of W is zero
        for j in range(n_firms):
            self.assertAlmostEqual(W[j, j], 0)
    
    def test_2_base_case(self):

        n_firms = 100
        c = 2
        c_prime = 4

        # generate base_case_tech_matrix
        T = generate_base_case_tech_matrix(n_firms, c + c_prime, 1 / c)

        # check that all diagonals are zero
        for i in range(n_firms):
            self.assertAlmostEqual(T[i,i], 0)

        # check that there are exactly c + c_prime nonzero elements in each column
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(T[:,j]), c + c_prime)

        # check that there are exactly c + c_prime nonzero elements in each column
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(T[:, j]), c + c_prime)

        # generate a connectivity matrix from T
        W = generate_connectivity_matrix(T, c)

        # check that there are exactly c nonzero elements in each column
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(W[:, j]), c)

        # check that the columns of W sum to 1
        for j in range(n_firms):
            self.assertAlmostEqual(np.sum(W[:, j]), 1)

        # check that the diagonal of W is zero
        for j in range(n_firms):
            self.assertAlmostEqual(W[j, j], 0)


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
