import unittest
import numpy as np
import sys
sys.path.append('../src')
from firms import Firms

class TestFirms(unittest.TestCase):

    def test_firm(self):

        W = np.array([[0,1], [0.5, 0]])
        
        a = np.array([1/4, 1/5])
        b = np.array([1/6, 1/7])
        z = np.array([1/2, 1/3])
        tau = [1,1]
        T = np.ones((2,2))

        # type caste everything to floats
        a = a.astype(float)
        b = b.astype(float)
        z = z.astype(float)
        W = W.astype(float)

        firms = Firms(a, z, b, tau, W, T)
        
        W_tilde_computed = firms.compute_W_tilde()
        V = np.array([117/207, (12 + 117/207)/24])
        V_computed = firms.compute_V(W_tilde_computed)

        for i in range(2):
            self.assertAlmostEqual(V[i], V_computed[i], places=1)

        h = (1 - a[0])*b[0]*V_computed[0] + (1 - a[1])*b[1]*V_computed[1]
        h = h/2
        h_computed = firms.compute_h(V_computed)
        self.assertAlmostEqual(h, h_computed, places=3)
        
        W = np.array([[0, 0.2, 0, 0, 0.8],
                            [1, 0, 0, 0, 0],
                            [0, 0.5, 0, 0.3, 0.2],
                            [0.6, 0.2, 0.2, 0, 0],
                            [0.3, 0, 0, 0.7, 0]])
        
        a = np.array([0.3,0.3,0.7,0.1,0.4])
        b = np.array([0.7, 0.8, 0.9,0.7,0.5])
        z = np.array([2,3,5,1,1])
        tau = [1,1,1,1,1]
        T = np.ones((5,5))

        # type caste everything to floats
        a = a.astype(float)
        b = b.astype(float)
        z = z.astype(float)
        W = W.astype(float)

        firms = Firms(a, z, b, tau, W, T)

        W_tilde = np.array([[0, 0.112, 0, 0, 0.24],
                            [0.49, 0, 0, 0, 0],
                            [0, 0.28, 0, 0.189, 0.06],
                            [0.294, 0.112, 0.054, 0, 0],
                            [0.147, 0, 0, 0.441, 0]])
        W_tilde_computed = firms.compute_W_tilde()
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(W_tilde[i,j], W_tilde_computed[i,j], places=3)

        V = np.array([0.339322646313418, 0.366268096693575, 0.395575430019778, 0.362143958066893, 0.409585914515572])
        V_computed = firms.compute_V(W_tilde)
        for i in range(5):
            self.assertAlmostEqual(V[i], V_computed[i], places=3)
        
        h = 0.165842012976826
        h_computed = firms.compute_h(V)
        self.assertAlmostEqual(h, h_computed, places=3)

        C = np.array([-1.145027010664114, -1.552188110924564, -2.739288221175688, -0.180811957940223, -0.459074641411455])
        C_computed = firms.compute_C(V, h)
        for i in range(5):
            self.assertAlmostEqual(C[i], C_computed[i], places=3)

        W_prime = np.array([[0, 0.49, 0, 0.294, 0.147],
                           [0.112, 0, 0.280, 0.112, 0],
                            [0, 0, 0, 0.054, 0],
                            [0, 0, 0.189, 0, 0.441],
                            [0.24, 0, 0.06, 0, 0]])
        W_prime_computed = firms.compute_W_prime()
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(W_prime[i,j], W_prime_computed[i,j], places=3)
        
        lp = np.array([-3.125690573095744, -2.837019797749766, -2.810549207631804, -1.319647897335476, -1.377873331412342])
        lp_computed = firms.compute_lp(W_prime, C)
        for i in range(5):
            self.assertAlmostEqual(lp[i], lp_computed[i], places=3)
        
        p = np.array([0.043906602433295, 0.058600045980473, 0.060171936429203, 0.267229377578626, 0.252114146578972])
        p_computed = firms.compute_p(lp)
        for i in range(5):
            self.assertAlmostEqual(p[i], p_computed[i], places=3)
        
        x = np.array([7.728282934871435, 6.250303912997331, 6.574085088406625, 1.355180187703505, 1.624605045267755])
        x_computed = firms.compute_x(V, p)
        for i in range(5):
            self.assertAlmostEqual(x[i], x_computed[i], places=3)
        
        l = np.array([0.429672520531784, 0.530048698930959, 1.502710419628612, 0.152856785862969, 0.493947109256091])
        l_computed = firms.compute_l(x, p, h)
        for i in range(5):
            self.assertAlmostEqual(l[i], l_computed[i], places=3)
        
        g = np.array([[0, 0.934302008268644, 0, 0, 2.238857348005475],
                            [2.837337307704143, 0, 0, 0, 0],
                            [0, 1.704367071431450, 0, 1.137493857376078, 0.408415555976809],
                            [0.373315459999500, 0.153508671843576, 0.079935347732429, 0, 0],
                            [0.197848592333742, 0, 0, 0.633464990658403, 0]])
        g_computed = firms.compute_g(p, x)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(g[i,j], g_computed[i,j], places=3)
        
        P = np.array([-0.047844493130192, 0.093764632753555, 0.125001835886250, 0.108643187420068, 0.204792957257786])
        P_computed = firms.compute_P(h, x, p, l , g)
        for i in range(5):
            self.assertAlmostEqual(P[i], P_computed[i], places=3)
        
        # self.P, self.p, self.x, self.l, self.h
        firms_P = firms.profits
        firms_p = firms.prices
        firms_x = firms.sales
        firms_l = firms.labour_hired
        firms_h = firms.wage

        for i in range(5):
            self.assertAlmostEqual(P[i], firms_P[i], places=3)
            self.assertAlmostEqual(p[i], firms_p[i], places=3)
            self.assertAlmostEqual(x[i], firms_x[i], places=3)
            self.assertAlmostEqual(l[i], firms_l[i], places=3)
        
        self.assertAlmostEqual(h, firms_h, places=3)

        firms.compute_equilibrium()
    
    def test_expectations(self):

        W = np.array([[0, 0.2, 0, 0, 0.8],
                            [1, 0, 0, 0, 0],
                            [0, 0.5, 0, 0.3, 0.2],
                            [0.6, 0.2, 0.2, 0, 0],
                            [0.3, 0, 0, 0.7, 0]])
        
        a = np.array([0.3,0.3,0.7,0.1,0.4])
        b = np.array([0.7, 0.8, 0.9,0.7,0.5])
        z = np.array([2,3,5,1,1])
        tau = [1,1,1,1,1]
        T = np.ones((5,5))

        # type caste everything to floats
        a = a.astype(float)
        b = b.astype(float)
        z = z.astype(float)
        W = W.astype(float)


        firms = Firms(a, z, b, tau, W, T)

        W = np.array([[0, 0.8, 0, 0, 0.3],
                            [1, 0, 0, 0, 0],
                            [0, 0.5, 0, 0.3, 0.2],
                            [0.6, 0.2, 0.2, 0, 0],
                            [0.3, 0, 0, 0.7, 0]])

        firms.supply_network = W

        indices_known = [0,1,3,4]
        indices_known_computed = firms.compute_indices_known(0, firms.tau[0])
        for i in range(len(indices_known)):
            self.assertEqual(indices_known[i], indices_known_computed[i])

        indices_unknown = [2]
        indices_unknown_computed = firms.compute_indices_unknown(indices_known)
        for i in range(len(indices_unknown)):
            self.assertEqual(indices_unknown[i], indices_unknown_computed[i])
        
        K1 = np.array([0,0,0.021361073221068, 0])
        K1_computed = firms.compute_K1(indices_known, indices_unknown)
        for i in range(len(K1)):
            self.assertAlmostEqual(K1[i], K1_computed[i], places=3)
        
        eW_tilde = np.array([[0, 0.448, 0, 0.09],
                                    [0.49, 0, 0, 0],
                                    [0.294, 0.112, 0, 0],
                                    [0.147, 0, 0.441, 0]])
        eW_tilde_computed = firms.compute_eW_tilde(indices_known)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(eW_tilde[i,j], eW_tilde_computed[i,j], places=3)

        eV = np.array([0.421122423023432, 0.406349987281481, 0.390682264165483, 0.434195874681422])
        eV_computed = firms.compute_eV(eW_tilde, K1)
        for i in range(4):
            self.assertAlmostEqual(eV[i], eV_computed[i], places=3)
        
        K2 = 1.606525114581803
        K2_computed = firms.compute_K2(indices_known)
        self.assertAlmostEqual(K2, K2_computed, places=3)

        eh = 0.186829721170264
        eh_computed = firms.compute_eh(eV, K2, indices_known)
        self.assertAlmostEqual(eh, eh_computed, places=3)

        lK3 = np.array([0, -0.786953778136905, -0.531193800242411, -0.168632952457908])
        lK3_computed = firms.compute_lK3(indices_known, indices_unknown)
        for i in range(4):
            self.assertAlmostEqual(lK3[i], lK3_computed[i], places=3)

        eC = np.array([-1.055211337141992, -2.289773154514842, -0.680908577502449, -0.574700655238758])
        eC_computed = firms.compute_eC(eV, eh, lK3, indices_known)
        for i in range(4):
            self.assertAlmostEqual(eC[i], eC_computed[i], places=3)
        
        eW_prime = np.array([[0, 0.448, 0, 0.09],
                                    [0.49, 0, 0, 0],
                                    [0.294, 0.112, 0, 0],
                                    [0.147, 0, 0.441, 0]])
        eW_prime_computed = firms.compute_eW_prime(indices_known)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(eW_prime[i,j], eW_prime_computed[i,j], places=3)

        elp = np.array([-2.880207338610663, -3.701074750434067, -1.942209907102599, -1.854605703046772])
        elp_computed = firms.compute_elp(eW_prime, eC)
        for i in range(4):
            self.assertAlmostEqual(elp[i], elp_computed[i], places=3)

        ep = np.array([0.056123125136911, 0.024696969123350, 0.143386728042801, 0.156514643757050])
        ep_computed = firms.compute_p(elp)
        for i in range(4):
            self.assertAlmostEqual(ep[i], ep_computed[i], places=3)

        ex = np.array([7.503545499223542, 16.453435449991755, 2.724675215748445, 2.774154956103680])
        ex_computed = firms.compute_x(eV, ep)
        for i in range(4):
            self.assertAlmostEqual(ex[i], ex_computed[i], places=3)

        ep_all = np.array([0.056123125136911, 0.024696969123350, 0.060171936429203, 0.143386728042801, 0.156514643757050])
        ep_all_computed = firms.compute_ep_all(ep,  indices_known, indices_unknown)
        for i in range(5):
            self.assertAlmostEqual(ep_all[i], ep_all_computed[i], places=3)

        ex_all = np.array([7.503545499223542, 16.453435449991755, 6.574085088406625, 2.724675215748445, 2.774154956103680])
        ex_all_computed = firms.compute_ex_all(ex, indices_known, indices_unknown)
        for i in range(5):
            self.assertAlmostEqual(ex_all[i], ex_all_computed[i], places=3)

        el = np.array([0.473349252361868, 0.521994018599850, 1.333901904640455, 0.146377986972752, 0.464803856647333])
        el_computed = firms.compute_l(ex_all, ep_all, eh)
        for i in range(5):
            self.assertAlmostEqual(el[i], el_computed[i], places=3)
    
        eg = np.array([[0, 3.243668164557984, 0, 0, 0.696283904825306],
                                [8.355275752698917, 0, 0, 0, 0],
                                [0, 1.890881417331194, 0, 1.227132652015495, 0.432955195177025],
                                [0.863468983907154, 0.317401751171425, 0.148975246960735, 0, 0],
                                [0.395522071918948, 0, 0, 1.100797180131059, 0]])
        eg_computed = firms.compute_g(ep_all, ex_all)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(eg[i,j], eg_computed[i,j], places=3)

        eP = np.array([-0.059378261646304, -0.032507998982519, 0.125001835886250, 0.117204679249645, 0.282227318542925])
        eP_computed = firms.compute_P(eh, ex_all, ep_all, el , eg)
        for i in range(5):
            self.assertAlmostEqual(eP[i], eP_computed[i], places=3)
        
        ePi = firms.compute_expected_profits(0, W)

        self.assertAlmostEqual(ePi, eP[0], places=3)

if __name__ == '__main__':
    unittest.main()


