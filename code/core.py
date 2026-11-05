"""Core formulas for the constrained noisy network paper."""
import numpy as np
from scipy.stats import norm
import networkx as nx


def A(z):
    return norm.pdf(z) - z * norm.sf(z)


def mu(z):
    return norm.sf(z) / A(z)


def psi(z):
    return norm.pdf(z) / A(z)


def var_S(z):
    return 1.0 + psi(z) - mu(z) ** 2


def f_z(s, z):
    return np.where(s > z, (s - z) * norm.pdf(s), 0.0) / A(z)


def laplacian(G, nodes=None):
    if nodes is None:
        nodes = sorted(G.nodes())
    return nx.laplacian_matrix(G, nodelist=nodes, weight="weight").toarray().astype(float)


def graph_data(L):
    N = L.shape[0]
    J = np.ones((N, N)) / N
    C = np.linalg.inv(L + J) - J
    C = 0.5 * (C + C.T)
    d = np.diag(C)
    R = d[:, None] + d[None, :] - 2 * C
    return C, R


def gaussian_sampler(L, rng, n):
    """Samples of gamma_C on 1-perp: Y = U2 diag(lam^-1/2) xi."""
    lam, U = np.linalg.eigh(L)
    U2 = U[:, 1:]
    lam2 = lam[1:]
    xi = rng.standard_normal((n, L.shape[0] - 1))
    return (xi / np.sqrt(lam2)) @ U2.T


def slice_length(Y, eps, lo, hi):
    """q_eps(y) = [min_i(r_i - eps y_i) - max_i(l_i - eps y_i)]_+ for rows of Y."""
    up = np.min(hi[None, :] - eps * Y, axis=1)
    dn = np.max(lo[None, :] - eps * Y, axis=1)
    return np.clip(up - dn, 0.0, None)
