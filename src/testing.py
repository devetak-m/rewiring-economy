import numpy as np

# solving eqaution 4,5 and 6 from the paper to compute equilibrium of economy

# input parameters
# n : number of firms
# a : labor intensity vector
# z : productivity vector
# b : return to scale vectors
# W : current supply network

# compute W_tilde matrix
# (W_tilde)ij = (1 − a_j)b_j w_ij 
def compute_W_tilde(a, b, W):
    W_tilde = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_tilde[i, j] = (1 - a[j]) * b[j] * W[i, j]
    return W_tilde

# solve for V
# V = (I − W_tilde)^(-1) * (1/n)
def compute_V(W_tilde):
    V = np.linalg.inv(np.eye(W_tilde.shape[0]) - W_tilde)
    V = V / V.shape[0]
    return V

# compute wage h
# h = sum_j (1 - a_j) * b_j * V_j * 1/n
def compute_h(a, b, V):
    h = 0
    for j in range(V.shape[0]):
        h += (1 - a[j]) * b[j] * V[j]
    return h/V.shape[0]

# compute constant for log prices
# C_i = log(z_i^(-1) * b_i^(-bi) * v_i^(1 - bi) * h^(a_ib_i))
def compute_C(a, b, z, V, h):
    C = np.zeros(V.shape[0])
    for i in range(V.shape[0]):
        C[i] = np.log(z[i]**(-1) * b[i]**(-b[i]) * V[i]**(1 - b[i]) * h**(a[i] * b[i]))
    return C

# compute log prices
# lp = (1 - W_tilde)^(-1) * C
def compute_lp(W_tilde, C):
    lp = np.linalg.inv(np.eye(W_tilde.shape[0]) - W_tilde)
    lp = np.dot(lp, C)
    return lp

# compute prices
# p = exp(lp)
def compute_p(lp):
    p = np.exp(lp)
    return p

# compute production quantities
# x_i = v_i/p_i
def compute_x(V, p):
    x = V / p
    return x

# compute ammount of hired labor
# l_i = a_i * b_i * x_i * p_i / h
def compute_l(a, b, x, p, h):
    l = a * b * x * p / h
    return l

# compute ammonut of bought quantities
# g_ij = (1 - a_i) * w_ji * b_i * x_i * p_i / p_j
def compute_g(a, b, x, p, W):
    g = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            g[i, j] = (1 - a[i]) * W[j, i] * b[i] * x[i] * p[i] / p[j]
    return g

# compute profits
# P_i = p_i * x_i - l_i * h - sum_j g_ij * p_j
def compute_P(a, b, x, p, h, W):
    P = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        P[i] = p[i] * x[i] - a[i] * b[i] * x[i] * p[i] / h
        for j in range(W.shape[1]):
            P[i] -= (1 - a[i]) * W[j, i] * b[i] * x[i] * p[i] / p[j]
    return P

# solving eqautions 8, 9 and 10 from the paper to compute expected equilibrium of economy

# input parameters
# i : index of firm under inspection
# tau : foresight of firm under inspection
# n : number of firms
# a : labor intensity vector
# z : productivity vector
# b : return to scale vectors
# W : current supply network under inspection
# p : current prices
# x : current production quantities
# h : current wage

# compute indices of firms that are known by firm under inspection
# (i.e. firms that are less than tau steps away in the network from firm under inspection)
def compute_indices_known(i, tau, W):
    indices_known = [i]
    if tau == 0:
        return indices_known
    for j in range(W.shape[0]):
        if W[i, j] > 0 and j != i:
            indices_known.append(j)
    for k in range(1,tau):
        new_indices = []
        for j in indices_known:
            for l in range(W.shape[0]):
                if W[j, l] > 0 and l != j and l not in indices_known:
                    new_indices.append(l)
        indices_known += new_indices
    return indices_known

# compute indices of firms that are not known by firm under inspection
# (i.e. firms that are more than tau steps away in the network from firm under inspection)
def compute_indices_unknown(i, indices_known, W):
    indices_unknown = []
    for j in range(W.shape[0]):
        if j not in indices_known and j != i:
            indices_unknown.append(j)
    return indices_unknown

# compute K1 constant for unknown firms
# for known k and unknown j
# K1_k = sum_j (1 - a_j) * b_j * w_kj * v_j
def compute_K1(a, b, W, V, indices_known, indices_unknown):
    K1 = np.zeros(len(indices_known))
    for i, k in enumerate(indices_known):
        for j in indices_unknown:
            K1[i] += (1 - a[j]) * b[j] * W[k, j] * V[j]
    return K1

# compute W_tilde matrix for known firms
# (eW_tilde)_ij = (1 - a_j) * b_j * w_ij
def compute_eW_tilde_known(a, b, W, indices_known):
    eW_tilde = np.zeros((len(indices_known), len(indices_known)))
    for i, k in enumerate(indices_known):
        for j, u in enumerate(indices_known):
            eW_tilde[i, j] = (1 - a[u]) * b[u] * W[k, u]
    return eW_tilde

# compute expected V for known firms
# V = (I - eW_tilde)^(-1) * (I/n + K1)
def compute_eV_known(eW_tilde, K1):
    eV = np.linalg.inv(np.eye(eW_tilde.shape[0]) - eW_tilde)
    eV = eV / eV.shape[0] + eV * K1
    return eV

# compute K2 constant for known firms
# K2 = sum_k a_k * b_k * v_k / h
def compute_K2(a, b, V, h, indices_known):
    K2 = 0
    for k in indices_known:
        K2 += a[k] * b[k] * V[k] / h
    return K2

# compute expected wage h
# eh = (1 / K2) sum_k a_k * b_k eV_k
def compute_eh(a, b, eV, K2, indices_known):
    eh = 0
    for k in indices_known:
        eh += a[k] * b[k] * eV[k]
    eh = eh / K2
    return eh

# compute log(K3) constant for known firms
# lK3_k = sum_j (1 - a_j) * b_j * w_ji * b_i * log(p_j)
def compute_lK3(a, b, W, p, indices_known, indices_unknown):
    lK3 = np.zeros(len(indices_known))
    for i, k in enumerate(indices_known):
        for j in indices_unknown:
            lK3[i] += (1 - a[j]) * b[j] * W[j, k] * b[k] * np.log(p[j])
    return lK3

# compute constant for log prices of known firms
# Ce_k = log(z_k^(-1) * b_k^(-bk) * v_k^(1 - bk) * h^(a_kb_k)) + lK3_k
def compute_eC(a, b, z, V, h, lK3, indices_known):
    Ce = np.zeros(len(indices_known))
    for i, k in enumerate(indices_known):
        Ce[i] = np.log(z[k]**(-1) * b[k]**(-b[k]) * V[k]**(1 - b[k]) * h**(a[k] * b[k])) + lK3[i]
    return Ce

# compute expected log prices for known firms
# lp = (1 - W_tilde)^(-1) * Ce
def compute_elp(eW_tilde, eC):
    elp = np.linalg.inv(np.eye(eW_tilde.shape[0]) - eW_tilde)
    elp = elp * eC
    return elp

# compute expected prices for known firms
# ep = exp(elp)
def compute_ep(elp):
    ep = np.exp(elp)
    return ep

# compute expected prices for all firms
# ep_j = ep_k if j is known and ep_j = p_j if j is unknown
def compute_ep_all(ep, p, indices_known, indices_unknown):
    ep_all = np.zeros(len(p))
    for k in indices_known:
        ep_all[k] = ep[k]
    for j in indices_unknown:
        ep_all[j] = p[j]
    return ep_all

# compute expected production quantities for known firms
# ex = ev / ep
def compute_ex(eV, ep):
    ex = eV / ep
    return ex

# compute expected production quantities for all firms
# ex_j = ex_k if j is known and ex_j = x_j if j is unknown
def compute_ex_all(ex, x, indices_known, indices_unknown):
    ex_all = np.zeros(len(x))
    for k in indices_known:
        ex_all[k] = ex[k]
    for j in indices_unknown:
        ex_all[j] = x[j]
    return ex_all


