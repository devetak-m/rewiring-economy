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

    def test_3_compute_trophic_level(self):

        W = np.array([[1, 0, 1, 0],
                    [0, 0.5, 0, 0.5],
                    [1,0,0,0],
                    [1, 0.5, 0, 1]])
    
        expected_in_weight = np.array([3, 1, 1, 1.5])
        expected_out_weight = np.array([2, 1, 1, 2.5])
        computed_out_weight = compute_out_weight(W)
        computed_in_weight = compute_in_weight(W)
        for i in range(4):
            self.assertAlmostEqual(expected_out_weight[i], computed_out_weight[i])
            self.assertAlmostEqual(expected_in_weight[i], computed_in_weight[i])
    
        expected_laplacian = np.array([[3, 0, -2, -1],
                                       [0, 1, 0, -1],
                                       [-2, 0, 2, 0],
                                       [-1, -1, 0, 2]])
        computed_laplacian = compute_laplacian(W, expected_out_weight + expected_in_weight)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(expected_laplacian[i, j], computed_laplacian[i, j])
    
        expected_trophic_level = np.array([1, 0, 1, 0])
        computed_trophic_level = compute_trophic_level(expected_laplacian, expected_in_weight - expected_out_weight)
        for i in range(4):
            self.assertAlmostEqual(expected_trophic_level[i], computed_trophic_level[i])
    
        expected_trophic_incoherence = 5.5/6.5
        computed_trophic_incoherence = compute_trophic_incoherence_from_level(W, computed_trophic_level)
        self.assertAlmostEqual(expected_trophic_incoherence, computed_trophic_incoherence)

        compute_trophic_incoherence_2  = compute_trophic_incoherence(W)
        self.assertAlmostEqual(expected_trophic_incoherence, compute_trophic_incoherence_2)
    
    def test_4_communities(self):

        n_firms = 30
        n_communities = 3
        c = 4
        c_prime = 8
        
        W,T = generate_communities_supply(n_firms, n_communities, c)

        # check that there are exactly c + c_prime nonzero elements in each column of T
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(T[:, j]), c + c_prime)
        
        # check that there are exactly c nonzero elements in each column of W
        for j in range(n_firms):
            self.assertEqual(np.count_nonzero(W[:, j]), c)
        
        # check that the columns of W sum to 1
        for j in range(n_firms):
            self.assertAlmostEqual(np.sum(W[:, j]), 1)
        
        # check that the diagonal of W is zero
        for j in range(n_firms):
            self.assertAlmostEqual(W[j, j], 0)
        
        # generate a undirected network from W
        G = nx.from_numpy_array(W)

        # check that the number of communities is n_communities
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        self.assertEqual(len(communities), n_communities)

if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
