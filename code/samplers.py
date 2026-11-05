"""Exact importance samplers for the finite-box stationary law."""
import numpy as np
from scipy.stats import norm
from core import A, laplacian, graph_data, gaussian_sampler, slice_length


def sample_fz(z, n, rng, tol=1e-13):
    """Inverse-CDF samples from f_z(s) = (s-z)_+ phi(s) / A(z) by bisection on the survival function."""
    U = rng.random(n)
    Az = A(z)
    lo = np.full(n, float(z))
    hi = np.full(n, float(max(z, 0.0) + 40.0))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        G = (norm.pdf(mid) - z * norm.sf(mid)) / Az
        big = G > U
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
        if np.max(hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


def contact_is(L, u, v, lo, hi, eps, z_prop, n, rng):
    """Self-normalized IS for Y = P X / eps under the finite-box law.

    Proposal: Y = (C a / sqrt R) S + V0 with S ~ f_{z_prop} and V0 the Gaussian residual.
    Returns Y (n x N) and normalized weights.
    """
    N = L.shape[0]
    C, Rm = graph_data(L)
    a = np.zeros(N); a[u] = 1.0; a[v] = -1.0
    R = a @ C @ a
    h = C @ a / np.sqrt(R)
    Yp = gaussian_sampler(L, rng, n)
    Sp = Yp @ a / np.sqrt(R)
    V0 = Yp - np.outer(Sp, h)
    S = sample_fz(z_prop, n, rng)
    Y = np.outer(S, h) + V0
    q = slice_length(Y, eps, lo, hi)
    logprop = np.log(np.clip(S - z_prop, 1e-300, None)) - np.log(A(z_prop))  # log f/phi
    w = q * np.exp(-logprop)
    w = np.where(S > z_prop, w, 0.0)
    return Y, w / w.sum(), R, h


def weighted_stats(Y, w, L, u, v, R):
    a = np.zeros(L.shape[0]); a[u] = 1.0; a[v] = -1.0
    S = Y @ a / np.sqrt(R)
    Q = np.einsum('ni,ij,nj->n', Y, L, Y)
    ES = np.sum(w * S)
    ES2 = np.sum(w * S * S)
    EQ = np.sum(w * Q)
    return dict(ES=ES, ES2=ES2, VarS=ES2 - ES ** 2, EQ=EQ)


def ess_fraction(w):
    return 1.0 / (len(w) * np.sum(w * w))
