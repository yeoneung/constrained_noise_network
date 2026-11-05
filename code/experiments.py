"""All numerical experiments. Results saved to results/."""
import numpy as np, networkx as nx, json, os, time, sys
from scipy.integrate import quad
from scipy.stats import norm
from core import *
from samplers import *

os.makedirs("results", exist_ok=True)
SEED = 20261029
NB, NS = 8, 2 ** 17


def graphs8():
    G = {
        "Path": (nx.path_graph(8), 0, 7),
        "Cycle": (nx.cycle_graph(8), 0, 4),
        "Star": (nx.star_graph(7), 1, 2),
        "Complete": (nx.complete_graph(8), 0, 7),
    }
    return G


def contact_walls(N, u, v, eps, R, z, slack=1.0):
    lo = -slack * np.ones(N); hi = slack * np.ones(N)
    lo[u] = eps * np.sqrt(R) * z; hi[v] = 0.0
    return lo, hi


def batch_stats(L, u, v, lo, hi, eps, z, rng):
    out = []
    for b in range(NB):
        Y, w, R, h = contact_is(L, u, v, lo, hi, eps, z, NS, rng)
        st = weighted_stats(Y, w, L, u, v, R)
        out.append([st["ES2"], st["VarS"], st["EQ"], ess_fraction(w)])
    out = np.array(out)
    return out.mean(0), out.std(0, ddof=1) / np.sqrt(NB)


def exp_crossover():
    """E1: four graphs, z grid, eps = 0.05; plus table at eps in {0.15,0.10,0.05}."""
    ss = np.random.SeedSequence(SEED)
    rng = np.random.default_rng(ss.spawn(1)[0])
    zgrid = np.round(np.arange(-3.0, 3.0001, 0.25), 4)
    res = {}
    for name, (G, u, v) in graphs8().items():
        L = laplacian(G); C, Rm = graph_data(L); R = Rm[u, v]
        M, E = [], []
        for z in zgrid:
            lo, hi = contact_walls(8, u, v, 0.05, R, z)
            m, e = batch_stats(L, u, v, lo, hi, 0.05, z, rng)
            M.append(m); E.append(e)
        res[name] = dict(R=R, mean=np.array(M), se=np.array(E))
        print(name, "done", flush=True)
    np.savez("results/crossover.npz", zgrid=zgrid,
             **{f"{k}_mean": v["mean"] for k, v in res.items()},
             **{f"{k}_se": v["se"] for k, v in res.items()},
             **{f"{k}_R": v["R"] for k, v in res.items()})
    # table
    tab = []
    for name, (G, u, v) in graphs8().items():
        L = laplacian(G); C, Rm = graph_data(L); R = Rm[u, v]
        for eps in [0.15, 0.10, 0.05]:
            for z in [-2.0, 0.0, 2.0]:
                lo, hi = contact_walls(8, u, v, eps, R, z)
                m, e = batch_stats(L, u, v, lo, hi, eps, z, rng)
                tab.append(dict(graph=name, R=R, eps=eps, z=z, ES2=m[0], ES2_se=e[0], VarS=m[1], VarS_se=e[1],
                                EQ=m[2], EQ_se=e[2], ess=m[3]))
    json.dump(tab, open("results/table_contact.json", "w"), indent=1)
    print("table done", flush=True)


def exp_two_node():
    """E2: exact two-node quadrature, a0 = b0 = 1."""
    a0 = b0 = 1.0
    def w(t):
        return np.clip(np.minimum.reduce([t, a0 + 0 * t, b0 + 0 * t, a0 + b0 - t]), 0, None)
    def moments(eps, delta):
        d0 = max(delta, 0.0)
        f = lambda d, k: d ** k * w(d - delta) * np.exp(-(d * d - d0 * d0) / (2 * eps * eps))
        lo, hi = delta, delta + a0 + b0
        sc = eps * eps / delta if delta > eps else eps
        pts = [lo, hi, 0.0, delta + min(a0, b0), delta + max(a0, b0)] + [d0 + sc * k for k in (0.5, 1, 2, 4, 8, 16, 32, 64)]
        pts = sorted(set([p for p in pts if lo <= p <= hi]))
        Z = sum(quad(f, pts[i], pts[i + 1], args=(0,), epsabs=0, epsrel=1e-12, limit=400)[0] for i in range(len(pts) - 1))
        M2 = sum(quad(f, pts[i], pts[i + 1], args=(2,), epsabs=0, epsrel=1e-12, limit=400)[0] for i in range(len(pts) - 1))
        return M2 / Z
    zz = np.linspace(-3, 3, 241)
    out = {"z": zz}
    for eps in [0.2, 0.1, 0.05, 0.025]:
        out[f"eps{eps}"] = np.array([moments(eps, eps * z) / eps ** 2 for z in zz])
    dd = np.linspace(-0.3, 0.3, 241)
    for eps in [0.1, 0.05, 0.025]:
        out[f"phys{eps}"] = np.array([moments(eps, d) for d in dd])
    out["delta"] = dd
    # three-regime check at eps small
    chk = {}
    for eps in [0.05, 0.02, 0.01]:
        chk[str(eps)] = dict(neg=moments(eps, -0.5) / eps ** 2, zero=moments(eps, 0.0) / eps ** 2,
                             pos=(moments(eps, 0.5) - 0.25) / eps ** 2)
    json.dump(chk, open("results/two_node_check.json", "w"), indent=1)
    np.savez("results/two_node.npz", **out)
    print("two node done", chk, flush=True)


def exp_spatial():
    """E3: small-world network, all-pair raw and centered excess at z = 0 and z = 1."""
    G = nx.connected_watts_strogatz_graph(40, 4, 0.15, seed=7)
    L = laplacian(G); N = L.shape[0]; C, Rm = graph_data(L)
    u, v = np.unravel_index(np.argmax(Rm), Rm.shape)
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[2])
    eps = 0.05
    out = dict(R=Rm, u=u, v=v, edges=np.array(list(G.edges())))
    for z in [0.0, 1.0]:
        lo, hi = contact_walls(N, u, v, eps, Rm[u, v], z)
        raws, cens = [], []
        for b in range(16):
            Y, w, R, h = contact_is(L, u, v, lo, hi, eps, z, NS, rng)
            m = w @ Y
            M2 = (Y * w[:, None]).T @ Y
            raw = np.diag(M2)[:, None] + np.diag(M2)[None, :] - 2 * M2
            cov = M2 - np.outer(m, m)
            cen = np.diag(cov)[:, None] + np.diag(cov)[None, :] - 2 * cov
            raws.append(raw); cens.append(cen)
        raws = np.array(raws); cens = np.array(cens)
        out[f"raw_mean_{z}"] = raws.mean(0); out[f"raw_se_{z}"] = raws.std(0, ddof=1) / 4
        out[f"cen_mean_{z}"] = cens.mean(0); out[f"cen_se_{z}"] = cens.std(0, ddof=1) / 4
        out["h"] = h
        print("spatial z", z, flush=True)
    pos = nx.kamada_kawai_layout(G)
    out["pos"] = np.array([pos[i] for i in range(N)])
    np.savez("results/spatial.npz", **out)


def qp_min(L, lo, hi):
    x = np.clip(np.zeros(L.shape[0]), lo, hi)
    step = 1.0 / np.linalg.eigvalsh(L).max()
    for _ in range(500000):
        xn = np.clip(x - step * (L @ x), lo, hi)
        if np.max(np.abs(xn - x)) < 1e-15:
            break
        x = xn
    return x


def incompat_setup(L, lo, hi):
    x = qp_min(L, lo, hi); g = L @ x; tol = 1e-9
    act = np.where((np.abs(x - lo) < tol) | (np.abs(x - hi) < tol))[0]
    A_ = np.array([i for i in act if abs(g[i]) > 1e-9])
    assert len(A_) == len(act)
    F = np.array([i for i in range(L.shape[0]) if i not in A_])
    m = np.zeros(L.shape[0]); m[A_] = 1.0 / g[A_]
    m[F] = -np.linalg.solve(L[np.ix_(F, F)], L[np.ix_(F, A_)] @ m[A_])
    return x, g, A_, F, m


def incompat_is(L, lo, hi, x, g, A_, F, eps, n, rng):
    N = L.shape[0]
    zeta = np.zeros((n, N))
    zeta[:, A_] = np.sign(g[A_]) * rng.exponential(1.0 / np.abs(g[A_]), size=(n, len(A_)))
    LFF = L[np.ix_(F, F)]
    zeta[:, F] = rng.standard_normal((n, len(F))) @ np.linalg.cholesky(np.linalg.inv(LFF)).T
    D = np.zeros(N); D[A_] = eps ** 2; D[F] = eps
    X = x[None, :] + zeta * D[None, :]
    inside = np.all((X >= lo) & (X <= hi), axis=1)
    za, zf = zeta[:, A_], zeta[:, F]
    logw = -0.5 * eps ** 2 * np.einsum('ni,ij,nj->n', za, L[np.ix_(A_, A_)], za) \
           - eps * np.einsum('ni,ij,nj->n', za, L[np.ix_(A_, F)], zf)
    w = np.where(inside, np.exp(logw - logw[inside].max()), 0.0)
    return X, w / w.sum()


def incompat_configs():
    N = 8
    base_lo, base_hi = -2 * np.ones(N), 2 * np.ones(N)
    cfg = {}
    lo, hi = base_lo.copy(), base_hi.copy(); lo[0] = 1.0; hi[7] = 0.0; cfg[2] = (lo, hi)
    lo, hi = base_lo.copy(), base_hi.copy(); lo[0] = 1.0; hi[4] = 0.0; lo[7] = 1.0; cfg[3] = (lo, hi)
    lo, hi = base_lo.copy(), base_hi.copy(); lo[0] = 1.0; hi[2] = 0.0; lo[5] = 1.0; hi[7] = 0.0; cfg[4] = (lo, hi)
    return cfg


def exp_incompatible():
    """E4: path N = 8, |A| in {2,3,4}; excess energy and pairwise expansion."""
    L = laplacian(nx.path_graph(8))
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(4)[3])
    epss = [0.04, 0.02, 0.01, 0.005, 0.0025]
    out = {}
    tab = []
    for k, (lo, hi) in incompat_configs().items():
        x, g, A_, F, m = incompat_setup(L, lo, hi)
        Hs = 0.5 * x @ L @ x
        b = np.zeros(8); b[1] = 1; b[6] = -1  # pair (2,7)
        coef = 2 * (b @ x) * (b @ m) + b[F] @ np.linalg.solve(L[np.ix_(F, F)], b[F])
        M, E = [], []
        for eps in epss:
            vals = []
            for bb in range(NB):
                X, w = incompat_is(L, lo, hi, x, g, A_, F, eps, NS, rng)
                Q = np.einsum('ni,ij,nj->n', X, L, X)
                exc = (Q - 2 * Hs) / eps ** 2
                zF = (X[:, F] - x[F]) / eps; zA = (X[:, A_] - x[A_]) / eps ** 2
                EzF = w @ zF - zF.mean(0)
                EzA = w @ zA - zA.mean(0) + 1.0 / g[A_]
                tq = (zF @ b[F] + eps * (zA @ b[A_])) ** 2
                bA, gA = b[A_], g[A_]
                prop_tq = b[F] @ np.linalg.solve(L[np.ix_(F, F)], b[F]) + eps ** 2 * (np.sum(bA ** 2 / gA ** 2) + np.sum(bA / gA) ** 2)
                Etq = w @ tq - tq.mean() + prop_tq
                pair = 2 * (b @ x) * (b[F] @ EzF / eps + bA @ EzA) + Etq
                vals.append([w @ exc, pair, ess_fraction(w)])
            vals = np.array(vals)
            M.append(vals.mean(0)); E.append(vals.std(0, ddof=1) / np.sqrt(NB))
            tab.append(dict(A=int(k), eps=eps, exc=M[-1][0], exc_se=E[-1][0], pair=M[-1][1], pair_se=E[-1][1],
                            pair_pred=coef, ess=M[-1][2], gmin=float(np.min(np.abs(g[A_])))))
        out[k] = dict(x=x, g=g, A=A_, coef=coef, mean=np.array(M), se=np.array(E))
        print("incompat |A|=", k, "x*", np.round(x, 4), "g", np.round(g, 4), "coef", coef, flush=True)
    np.savez("results/incompat.npz", epss=np.array(epss),
             **{f"A{k}_mean": v["mean"] for k, v in out.items()},
             **{f"A{k}_se": v["se"] for k, v in out.items()},
             **{f"A{k}_x": v["x"] for k, v in out.items()},
             **{f"A{k}_g": v["g"] for k, v in out.items()},
             **{f"A{k}_coef": v["coef"] for k, v in out.items()})
    json.dump(tab, open("results/table_incompat.json", "w"), indent=1)


def full_samples(L, lo, hi, eps, Y, w, rng, k):
    idx = rng.choice(len(w), size=k, p=w)
    Ys = Y[idx]
    up = np.min(hi[None, :] - eps * Ys, axis=1); dn = np.max(lo[None, :] - eps * Ys, axis=1)
    m = dn + rng.random(k) * (up - dn)
    return m[:, None] + eps * Ys


def exp_ladder():
    """Energy histograms in the three regimes for the path, N = 8."""
    L = laplacian(nx.path_graph(8)); C, Rm = graph_data(L)
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(5)[4])
    eps = 0.02; n = 2 ** 20
    out = {}
    # overlap: l_u = -0.5, r_v = 0.5, others +-1
    lo, hi = -np.ones(8), np.ones(8); lo[0] = -0.5; hi[7] = 0.5
    Y = gaussian_sampler(L, rng, n); q = slice_length(Y, eps, lo, hi); w = q / q.sum()
    out["overlap_Q"] = np.einsum('ni,ij,nj->n', Y, L, Y); out["overlap_w"] = w
    # contact z = 0
    lo, hi = contact_walls(8, 0, 7, eps, Rm[0, 7], 0.0)
    Y, w, R, h = contact_is(L, 0, 7, lo, hi, eps, 0.0, n, rng)
    out["contact_Q"] = np.einsum('ni,ij,nj->n', Y, L, Y); out["contact_w"] = w
    # separated |A| = 2
    lo, hi = incompat_configs()[2]
    x, g, A_, F, m = incompat_setup(L, lo, hi)
    X, w = incompat_is(L, lo, hi, x, g, A_, F, 0.005, n, rng)
    out["sep_exc"] = (np.einsum('ni,ij,nj->n', X, L, X) - x @ L @ x) / 0.005 ** 2; out["sep_w"] = w
    np.savez("results/ladder.npz", **out)
    print("ladder done", flush=True)


def exp_schematic():
    """Stationary samples per regime on a path of N = 6 for Figure 1."""
    N = 6; L = laplacian(nx.path_graph(N)); C, Rm = graph_data(L)
    rng = np.random.default_rng(7)
    eps = 0.06
    u, v = 1, 4
    R = Rm[u, v]
    base_lo = np.array([-0.55, -0.05, -0.35, -0.70, -0.20, -0.60])
    base_hi = np.array([0.85, 0.75, 0.85, 0.40, 0.55, 0.65])
    cfg = {"overlap": (-0.05, 0.55), "contact": (0.25, 0.25), "separated": (0.55, -0.05)}
    out = {}
    for k, (lu, rv) in cfg.items():
        lo_, hi_ = base_lo.copy(), base_hi.copy(); lo_[u] = lu; hi_[v] = rv
        z = (lu - rv) / (eps * np.sqrt(R))
        Y, w, R_, h = contact_is(L, u, v, lo_, hi_, eps, z, 2 ** 18, rng)
        out[f"{k}_X"] = full_samples(L, lo_, hi_, eps, Y, w, rng, 40)
        out[f"{k}_lo"] = lo_; out[f"{k}_hi"] = hi_
        out[f"{k}_xstar"] = qp_min(L, lo_, hi_)
        out[f"{k}_ess"] = ess_fraction(w)
    np.savez("results/schematic.npz", **out)
    print("schematic done", {k: out[f"{k}_ess"] for k in cfg}, flush=True)


def exp_dynamics():
    """E5: reflected Euler (mirror) scheme, path N = 8, contact z = 0."""
    N = 8; L = laplacian(nx.path_graph(N)); C, Rm = graph_data(L); R = Rm[0, 7]
    kappa, eps = 1.0, 0.05; D = kappa * eps ** 2
    lo, hi = contact_walls(N, 0, 7, eps, R, 0.0)
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(6)[5])
    M = 400; h = 0.005; T = 1500.0; burn = 150.0
    nsteps = int(T / h); nburn = int(burn / h)
    X = np.zeros((M, N)); X[:, 0] = 0.01; X[:, 7] = -0.01
    sq = np.sqrt(2 * D * h)
    rec_every = 20
    traj = []
    Ssum = S2sum = Qsum = 0.0; cnt = 0
    repS = np.zeros(M); repS2 = np.zeros(M); repQ = np.zeros(M); repn = 0
    hist_edges = np.linspace(0, 5, 101); hist = np.zeros(100)
    Qhist_edges = np.linspace(0, 30, 121); Qhist = np.zeros(120)
    run_t, run_S2, run_Q = [], [], []
    lam, Uv = np.linalg.eigh(L)
    E = (Uv * np.exp(-kappa * lam * h)) @ Uv.T
    var = np.where(lam > 1e-12, D / np.where(lam > 1e-12, kappa * lam, 1.0) * (1 - np.exp(-2 * kappa * lam * h)), 2 * D * h)
    Sig_half = (Uv * np.sqrt(var)) @ Uv.T
    for n in range(nsteps):
        X = X @ E + rng.standard_normal((M, N)) @ Sig_half
        # mirror reflection, repeated until inside
        for _ in range(3):
            X = np.where(X < lo, 2 * lo - X, X)
            X = np.where(X > hi, 2 * hi - X, X)
        if n % rec_every == 0:
            if n * h <= 400:
                traj.append(X[0].copy())
            if n >= nburn:
                S = (X[:, 0] - X[:, 7]) / (eps * np.sqrt(R))
                Q = np.einsum('ni,ij,nj->n', X, L, X) / eps ** 2
                Ssum += S.sum(); S2sum += (S * S).sum(); Qsum += Q.sum(); cnt += M
                repS += S; repS2 += S * S; repQ += Q; repn += 1
                hist += np.histogram(S, bins=hist_edges)[0]
                Qhist += np.histogram(Q, bins=Qhist_edges)[0]
                if (n // rec_every) % 50 == 0:
                    run_t.append(n * h); run_S2.append(S2sum / cnt); run_Q.append(Qsum / cnt)
    se = lambda r: float(np.std(r / repn, ddof=1) / np.sqrt(M))
    res = dict(ES=Ssum / cnt, ES2=S2sum / cnt, EQ=Qsum / cnt, count=cnt, ES_se=se(repS), ES2_se=se(repS2), EQ_se=se(repQ), M=M, h=h, T=T, burn=burn)
    json.dump(res, open("results/dynamics.json", "w"), indent=1)
    np.savez("results/dynamics.npz", traj=np.array(traj), dt=h * rec_every, hist=hist, edges=hist_edges,
             Qhist=Qhist, Qedges=Qhist_edges, run_t=np.array(run_t), run_S2=np.array(run_S2), run_Q=np.array(run_Q),
             lo=lo, hi=hi, eps=eps, R=R)
    print("dynamics", res, flush=True)


if __name__ == "__main__":
    which = sys.argv[1:]
    fns = dict(crossover=exp_crossover, two=exp_two_node, spatial=exp_spatial, incompat=exp_incompatible,
               ladder=exp_ladder, schematic=exp_schematic, dynamics=exp_dynamics)
    for w_ in [x for x in which if x in fns]:
        t = time.time(); fns[w_](); print(w_, "%.1fs" % (time.time() - t), flush=True)


def exp_ladder_means():
    L = laplacian(nx.path_graph(8)); C, Rm = graph_data(L)
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(7)[6])
    epss = [0.04, 0.02, 0.01, 0.005, 0.0025]
    out = {"epss": np.array(epss)}
    for reg in ["overlap", "contact"]:
        M, E = [], []
        for eps in epss:
            vals = []
            for b in range(NB):
                if reg == "overlap":
                    lo, hi = -np.ones(8), np.ones(8); lo[0] = -0.5; hi[7] = 0.5
                    Y = gaussian_sampler(L, rng, NS); q = slice_length(Y, eps, lo, hi); w = q / q.sum()
                else:
                    lo, hi = contact_walls(8, 0, 7, eps, Rm[0, 7], 0.0)
                    Y, w, R, h = contact_is(L, 0, 7, lo, hi, eps, 0.0, NS, rng)
                vals.append(w @ np.einsum('ni,ij,nj->n', Y, L, Y))
            M.append(np.mean(vals)); E.append(np.std(vals, ddof=1) / np.sqrt(NB))
        out[f"{reg}_mean"] = np.array(M); out[f"{reg}_se"] = np.array(E)
    np.savez("results/ladder_means.npz", **out)
    print({k: np.round(v, 4) for k, v in out.items()}, flush=True)


if __name__ == "__main__" and "ladder_means" in sys.argv:
    exp_ladder_means()
