import unittest
import numpy as np
import sys
sys.path.append('../src')
from firms import Firms
from dynamics import Dynamics


class TestDynamics(unittest.TestCase):

    def test_1_compute_rewirings(self):
        
        T = np.array(  [[0, 0.2, 0.3, 0.4, 0.5],
                        [0.2, 0, 0.1, 0.2, 0.3],
                        [0.3, 0.1, 0, 0.1, 0.2],
                        [0.4, 0.2, 0.1, 0, 0.1],
                        [0.5, 0.3, 0.2, 0.1, 0]])
        
        W = np.array([  [0, 0.2, 0.3, 0,     0],
                        [0.2, 0, 0,   0,     0.3],
                        [0.3, 0, 0,   0.1,   0],
                        [0.0, 0, 0.1, 0,     0.1],
                        [0, 0.3, 0,   0.1,   0]])
        
        a = np.array([0.4,0.3,0.6,0.2,0.1])
        b = np.array([0.2,0.5,0.9,0.1,0.3])
        z = np.array([1.,2.,3.,4.,5.])
        tau = [0,0,0,0,0]

        firms = Firms(a, z, b, tau, W, T)

        dynamics = Dynamics(firms, 100)

        # expected rewirings
        rewirings_0_expected = np.array([[0,0,0], [1,3,0.4],[1,4,0.5], [2,3,0.4], [2,4,0.5]])
        rewirings_1_expected = np.array([[1,1,0], [0,2,0.1],[0,3,0.2], [4,2,0.1], [4,3,0.2]])
        rewirings_2_expected = np.array([[2,2,0], [0,1,0.1],[0,4,0.2], [3,1,0.1], [3,4,0.2]])
        rewirings_3_expected = np.array([[3,3,0], [2,0,0.4],[2,1,0.2], [4,0,0.4], [4,1,0.2]])
        rewirings_4_expected = np.array([[4,4,0], [1,0,0.5],[1,2,0.2], [3,0,0.5], [3,2,0.2]])

        rewirings_0_computed = dynamics.compute_possible_rewirings(0)
        rewirings_1_computed = dynamics.compute_possible_rewirings(1)
        rewirings_2_computed = dynamics.compute_possible_rewirings(2)
        rewirings_3_computed = dynamics.compute_possible_rewirings(3)
        rewirings_4_computed = dynamics.compute_possible_rewirings(4)

        # check that the expected rewirings are equal to the computed rewirings
        for i in range(rewirings_0_expected.shape[0]):
            for j in range(rewirings_0_expected.shape[1]):
                self.assertAlmostEqual(rewirings_0_expected[i][j], rewirings_0_computed[i][j])
                self.assertAlmostEqual(rewirings_1_expected[i][j], rewirings_1_computed[i][j])
                self.assertAlmostEqual(rewirings_2_expected[i][j], rewirings_2_computed[i][j])
                self.assertAlmostEqual(rewirings_3_expected[i][j], rewirings_3_computed[i][j])
                self.assertAlmostEqual(rewirings_4_expected[i][j], rewirings_4_computed[i][j])

        # check that the matrix W is unchanged
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                self.assertAlmostEqual(W[i,j], firms.supply_network[i,j])
            
        # check that the matrix T is unchanged
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                self.assertAlmostEqual(T[i,j], firms.technology_network[i,j])

    def test_2_compute_round(self):

        # technology matrix 
        T = np.array([[0, 0.3, 0.5],
                     [0.2, 0, 1],
                     [0.3, 0.1, 0]])
        # current supply network
        W = np.array([[0, 0.3, 0],
                        [0.2, 0, 1],
                        [0, 0, 0]])
        
        a = np.array([0.4,0.3,0.6])
        b = np.array([0.2,0.5,0.9])
        z = np.array([1.,2.,3.])
        tau = [0,0,0]
 
        firms = Firms(a, z, b, tau, W, T)
        
        dynamics = Dynamics(firms, 100)

        # expected W after one round
        W_expected = np.array([[0, 0, 0.5],
                               [0.2, 0, 0],
                               [0, 0.1, 0]])
        flag_expected = True        
        expected_P = np.array([0.35436185, 0.27940251, 0.09669302])


        # compute one round
        flag_computed = dynamics.compute_round()
        W_compute = dynamics.firms.supply_network
        P_compute = dynamics.firms.profits

        # check that the expected W is equal to the computed W
        self.assertEqual(flag_computed, flag_expected)
        for i in range(W_expected.shape[0]):
            for j in range(W_expected.shape[1]):
                self.assertAlmostEqual(W_expected[i,j], W_compute[i,j])
        for i in range(expected_P.shape[0]):
            self.assertAlmostEqual(expected_P[i], P_compute[i])

        # check that the matrix T is unchanged
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                self.assertAlmostEqual(T[i,j], firms.technology_network[i,j])
    
if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
