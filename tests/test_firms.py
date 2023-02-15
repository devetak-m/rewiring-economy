import unittest
import numpy as np
from firms import Firms


class TestFirms(unittest.TestCase):

    def test_firm(self):

        a = np.array([0.1, 0.2, 0.4, 0.3])
        z = np.array([0.1, 0.3, 0.5, 0.1])
        b = np.array([0.1, 0.2, 0.3, 0.4])
        tau = [-1, 1, 2]
        W = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        
        T = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        
        firm = Firms(a, z, b, tau, W, T)

        self.assertEqual((firm.compute_W_tilde()== np.array([[0, 0.8, 0, 0],
                                                            [0.8, 0, 0.8, 0],
                                                            [0, 0.8, 0, 0.8],
                                                            [0, 0, 0.8, 0]])).all(), True)
        
        W_tilde = np.array([[0, 0.8, 0, 0],
                            [0.8, 0, 0.8, 0],
                            [0, 0.8, 0, 0.8],
                            [0, 0, 0.8, 0]])
        
        self.assertEqual((firm.compute_V(W_tilde) == np.array([[0.25, 0.25, 0.25, 0.25],
                                                            [0.25, 0.25, 0.25, 0.25],
                                                            [0.25, 0.25, 0.25, 0.25],
                                                            [0.25, 0.25, 0.25, 0.25]])).all(), True)

        V = np.array([0.1, 0.2, 0.3, 0.4])

        self.assertEqual(firm.compute_h(V), 0.25)

        h = 0.25
   




if __name__ == '__main__':
    unittest.main()
