import numpy as np
import network as net


class Firms():

    def __init__(self, a, z, b, tau, W, T):

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
        self.W = W
        self.T = T

        # store equilibrium values
        self.P = None
        self.p = None
        self.x = None
        self.l = None
        self.h = None

        # compute equilibrium values
        self.P, self.p, self.x, self.l, self.h = self.compute_equilibrium()       
    
    # Setters for class instances

    def update_a(self, a):
        self.a = a
    
    def update_z(self, z):
        self.z = z
    
    def update_b(self, b):
        self.b = b

    def update_tau(self, tau):
        self.tau = tau
    
    def update_W(self, W):
        self.W = W
    
    def update_T(self, T):
        self.T = T

    def update_P(self, P):
        self.P = P
    
    def update_p(self, p):
        self.p = p
    
    def update_x(self, x):
        self.x = x
    
    def update_l(self, l):
        self.l = l
    
    def update_h(self, h):
        self.h = h
    
    def update_equilibrium(self):
        self.P, self.p, self.x, self.l, self.h = self.compute_equilibrium()
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
        P = self.compute_P(h, x, p, l , g)
        return P, p, x, l, h
    
    # Compute W_tilde matrix
    # (W_tilde)ij = (1 − a_j)b_j w_ij
    def compute_W_tilde(self):
        W_tilde = np.zeros(self.W.shape)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                W_tilde[i, j] = (1 - self.a[j]) * self.b[j] * self.W[i, j]
        return W_tilde
    
    # Solve for V
    # V = (I − W_tilde)^(-1) * (1/n)
    def compute_V(self, W_tilde):
        V = np.linalg.inv(np.eye(W_tilde.shape[0]) - W_tilde) @ (np.ones(W_tilde.shape[0]))

        V = V / V.shape[0]
        return V
    
    # Compute wage h
    # h = sum_j (1 - a_j) * b_j * V_j * 1/n
    def compute_h(self, V):
        h = 0
        for j in range(V.shape[0]):
            h += (1 - self.a[j]) * self.b[j] * V[j]
        return h/V.shape[0]
    
    # Compute constant for log prices
    # C_i = log(z_i^(-1) * b_i^(-b_i) * v_i^(1 - bi) * h^(a_ib_i))
    def compute_C(self, V, h):
        C = np.zeros(V.shape[0])
        for i in range(V.shape[0]):
            C[i] = np.log(self.z[i]**(-1) * self.b[i] ** (-self.b[i]) * V[i]**(1 - self.b[i]) * h**(self.a[i] * self.b[i]))
        return C
    
    # Compute W_prime matrix
    # W_prime(i,j) = (1 - a(i)) * b(i) * W(j,i)        
    # W_prime_ij = (1 - a_i) * W_ji * b_i
    def compute_W_prime(self):
        W_prime = np.zeros(self.W.shape)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                W_prime[i, j] = (1 - self.a[i]) * self.W[j, i] * self.b[i]
        return W_prime

    # Compute log prices
    # lp = (I - W_prime)^(-1) * C
    def compute_lp(self, W_prime, C):
        lp = np.linalg.inv(np.eye(W_prime.shape[0]) - W_prime) @ C
        return lp

    def compute_p(self, lp):
        return np.exp(lp)
    
    # Compute production quantities
    # x_i = V_i/p_i
    def compute_x(self, V, p):
        x = np.zeros(V.shape[0])
        for i in range(V.shape[0]):
            x[i] = V[i] / p[i]
        return x
    # Compute ammount of hired labor
    # l_i = a_i * b_i * p_i * x_i / h
    def compute_l(self, x, p, h):
        l = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            l[i] = self.a[i] * self.b[i] * p[i] * x[i] / h
        return l
    
    # Compute g
    # g_ji = (1 - a_i) * W_ji * b_i * p_i * x_i / p_j
    def compute_g(self, p, x):
        g = np.zeros(self.W.shape)
        for i in range(p.shape[0]):
            for j in range(p.shape[0]):
                g[j, i] = (1 - self.a[i]) * self.W[j, i] * self.b[i] * p[i] * x[i] / p[j]
        return g

    # Compute profits
    # P_i = p_i * x_i - l_i * h - sum_j p_j * g_ji
    def compute_P(self, h, x, p, l , g):
        P = np.zeros(p.shape[0])
        for i in range(p.shape[0]):
            P[i] = p[i] * x[i] - l[i] * h
            for j in range(p.shape[0]):
                P[i] -= p[j] * g[j, i]
        return P
    
    # compute expected equilibrium profits for firm i with foresight tau_i and netowrk W'
    def compute_expected_profits(self, i, W_new):
        
        tau_i = self.tau[i]

        # if tau = -1, then firm i has perfect foresight
        if tau_i == -1:
            # compute profits with W = W'
            # store current W
            W = self.W
            # update W
            self.update_W(W_new)
            # compute profits
            P, p, x, l, h  = self.compute_equilibrium()
            # W = W again
            self.update_W(W)
            return P[i]
        
        else:
            # store current W
            W = self.W
            
            # firm has imperfect foresight
            # compute indices of firms that are known by firm under inspection
            indices_known = self.compute_indices_known(i, tau_i)
            # compute indices of firms that are not known by firm under inspection
            indices_unknown = self.compute_indices_unknown(indices_known)
            # update W
            self.update_W(W_new)
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
            eP = self.compute_P(eh, ex_all, ep_all, el , eg)
            # W = W again
            self.update_W(W)
            return eP[i]
    
    # compute indices of firms that are known by firm under inspection
    def compute_indices_known(self, i, tau):
        indices_known = [i]
        if tau == 0:
            return indices_known
        for j in range(self.W.shape[0]):
            if (self.W[i, j] > 0 and j != i) or (self.W[j, i] > 0 and j != i):
                indices_known.append(j)
        for k in range(1,tau):
            new_indices = []
            for j in indices_known:
                for l in range(self.W.shape[0]):
                    if ((self.W[j, l] > 0 and l != j) or (self.W[l, j] > 0  and l != j)) and (l not in indices_known):
                        new_indices.append(l)
            indices_known += new_indices
        return indices_known
    
    # compute indices of firms that are not known by firm under inspection
    def compute_indices_unknown(self, indices_known):
        indices_unknown = []
        for j in range(self.W.shape[0]):
            if j not in indices_known:
                indices_unknown.append(j)
        return indices_unknown
    
    # compute K1
    # K1_k = sum_{j in J} (1 - a_j) * b_j * W_kj * V_j
    def compute_K1(self, indices_known, indices_unknown):
        K1 = np.zeros(len(indices_known))
        for i, k in enumerate(indices_known):
            for j in indices_unknown:
                K1[i] += (1 - self.a[j]) * self.b[j] * self.W[k, j] * self.p[j] * self.x[j]
        return K1
    


    # compute W_tilde matrix for known firms
    # (eW_tilde)_ij = (1 - a_j) * b_j * w_ij
    def compute_eW_tilde(self, indices_known):
        eW_tilde = np.zeros((len(indices_known), len(indices_known)))
        for i, k in enumerate(indices_known):
            for j, u in enumerate(indices_known):
                eW_tilde[i, j] = (1 - self.a[u]) * self.b[u] * self.W[k, u]
        return eW_tilde

    # compute expected V for known firms
    # V = (I - eW_tilde)^(-1) * (1/n + K1)
    def compute_eV(self, eW_tilde, K1):
        I = np.eye(len(eW_tilde))
        n = self.P.shape[0]
        eV = np.linalg.inv(I - eW_tilde) @ (np.ones(len(eW_tilde)) / n + K1)
        return eV
    
    # compute K2 constant for known firms
    # K2 = sum_k a_k * b_k * v_k / h
    def compute_K2(self, indices_known):
        K2 = np.sum(self.a[indices_known] * self.b[indices_known] * self.p[indices_known] * self.x[indices_known])
        return K2/self.h    

    # compute expected wage h
    # eh = (1 / K2) sum_k a_k * b_k eV_k
    def compute_eh(self, eV, K2, indices_known):
        eh = 0
        for i, k in enumerate(indices_known):
            eh += self.a[k] * self.b[k] * eV[i]
        return eh / K2

    # compute log(K3) constant for known firms
    # lK3_k = sum_j (1 - a_j) * b_j * w_ji * b_i * log(p_j)
    def compute_lK3(self, indices_known, indices_unknown):
        lK3 = np.zeros(len(indices_known))
        for i, k in enumerate(indices_known):
            for j in indices_unknown:
                lK3[i] += (1 - self.a[k]) * self.b[k] * self.W[j, k] * np.log(self.p[j])
        return lK3

    # compute constant for log prices of known firms
    # Ce_k = log(z_k^(-1) * b_k^(-bk) * v_k^(1 - bk) * h^(a_kb_k)) + lK3_k  
    def compute_eC(self, eV, eh, lK3, indices_known):
        eC = np.zeros(len(indices_known))
        for i, k in enumerate(indices_known):
            eC[i] = np.log(self.z[k]**(-1) * self.b[k]**(-self.b[k]) * eV[i]**(1 - self.b[k]) * eh**(self.a[k] * self.b[k])) + lK3[i]
        return eC

    # W_prime(i,j) = (1 - a(i)) * b(i) * W(j,i)        
    # W_prime_ij = (1 - a_i) * W_ji * b_i
    # for i and j known
    def compute_eW_prime(self, indices_known):
        eW_prime = np.zeros((len(indices_known), len(indices_known)))
        for i, k in enumerate(indices_known):
            for j, u in enumerate(indices_known):
                eW_prime[i, j] = (1 - self.a[u]) * self.W[k,u] * self.b[u]
        return eW_prime

    # compute expected log prices for known firms
    # lp = (1 - W_tilde)^(-1) * Ce
    def compute_elp(self, eW_prime, eC):
        I = np.eye(len(eW_prime))
        elp = np.linalg.inv(I - eW_prime) @ eC
        return elp
    
    # compute expected log prices for all firms
    # ep_j = ep_k if j is known and ep_j = p_j if j is unknown
    def compute_ep_all(self, ep, indices_known, indices_unknown):
        ep_all = np.zeros(self.W.shape[0])
        for i, k in enumerate(indices_known):
            ep_all[k] = ep[i]
        for j in indices_unknown:
            ep_all[j] = self.p[j]
        return ep_all
    
    # compute expected log quantities for all firms
    # eq_j = eq_k if j is known and eq_j = q_j if j is unknown
    def compute_ex_all(self, ex, indices_known, indices_unknown):
        ex_all = np.zeros(self.W.shape[0])
        for i, k in enumerate(indices_known):
            ex_all[k] = ex[i]
        for j in indices_unknown:
            ex_all[j] = self.x[j]
        return ex_all
    
    def compute_household_utility(self):
        # compute household utility
        # - sum_j log(p_j)
        return -np.sum(np.log(self.p))
    
    def compute_trophic_incoherence(self):
        return net.compute_trophic_incoherence(self.W)
    
        
