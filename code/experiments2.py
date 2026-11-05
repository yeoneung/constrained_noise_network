"""Revision experiments: independent samplers, multiple contacts, time steps, footprints."""
import numpy as np, networkx as nx, json, sys, time
from scipy.special import ndtr, ndtri
from core import *
from samplers import *
from experiments import graphs8, contact_walls, qp_min, incompat_configs, incompat_setup, incompat_is, SEED

def revision_rng(child):
    """Use the streams from the published full run regardless of CLI order."""
    return np.random.default_rng(np.random.SeedSequence(SEED + 1, spawn_key=(child,)))


# ---------------------------------------------------------------- rejection sampler
def rejection_contact(L, lo, hi, eps, target, max_prop, rng, batch=2 ** 20):
    """Exact acceptance-rejection for Y = PX/eps: propose gamma_C, accept w.p. q/qmax."""
    qmax = np.min(hi - lo)
    acc, nprop = [], 0
    while sum(len(a) for a in acc) < target and nprop < max_prop:
        Y = gaussian_sampler(L, rng, batch)
        q = slice_length(Y, eps, lo, hi)
        keep = rng.random(batch) < q / qmax
        acc.append(Y[keep]); nprop += batch
    Y = np.concatenate(acc)
    return Y, nprop, len(Y) / nprop


def exp_rejection():
    rng = revision_rng(0)
    out, foot = [], {}
    for name, (G, u, v) in graphs8().items():
        L = laplacian(G); C, Rm = graph_data(L); R = Rm[u, v]
        a = np.zeros(8); a[u] = 1; a[v] = -1
        for z in [-2.0, 0.0, 2.0]:
            lo, hi = contact_walls(8, u, v, 0.05, R, z)
            target = 200000 if z == 0.0 else 40000
            Y, nprop, rate = rejection_contact(L, lo, hi, 0.05, target, 6e8, rng)
            S = Y @ a / np.sqrt(R); Q = np.einsum('ni,ij,nj->n', Y, L, Y)
            grp = np.array_split(np.arange(len(Y)), 8)
            def mse(f):
                vals = np.array([f(g) for g in grp]); return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(8))
            ES2 = mse(lambda g: np.mean(S[g] ** 2)); VS = mse(lambda g: np.var(S[g])); EQ = mse(lambda g: np.mean(Q[g]))
            out.append(dict(graph=name, z=z, n_acc=int(len(Y)), n_prop=int(nprop), rate=rate,
                            ES2=ES2[0], ES2_se=ES2[1], VarS=VS[0], VarS_se=VS[1], EQ=EQ[0], EQ_se=EQ[1]))
            print(name, z, len(Y), "rate %.5f" % rate, "ES2 %.4f±%.4f" % ES2, "Var %.4f±%.4f" % VS, "EQ %.4f±%.4f" % EQ, flush=True)
            if z == 0.0 and name in ("Cycle", "Star"):
                h = C @ a / np.sqrt(R)
                edges = sorted((min(i, j), max(i, j)) for i, j in G.edges())
                rows = []
                for (i, j) in edges:
                    d = Y[:, i] - Y[:, j]
                    raw = np.array([np.mean(d[g] ** 2) for g in grp]) - Rm[i, j]
                    cen = np.array([np.var(d[g]) for g in grp]) - Rm[i, j]
                    rows.append(dict(edge=[int(i), int(j)], share=float((h[i] - h[j]) ** 2), R=float(Rm[i, j]),
                                     raw=float(raw.mean()), raw_se=float(raw.std(ddof=1) / np.sqrt(8)),
                                     cen=float(cen.mean()), cen_se=float(cen.std(ddof=1) / np.sqrt(8))))
                foot[name] = rows
    json.dump(out, open("results/rejection.json", "w"), indent=1)
    json.dump(foot, open("results/footprint.json", "w"), indent=1)


# ---------------------------------------------------------------- Gibbs sampler, several contacts
def gibbs_box(L, a, b, K, sweeps, burn, rng, x0):
    """Coordinatewise Gibbs sampler for density exp(-xi'L xi/2) on the box prod [a_k, b_k] (xi units)."""
    N = L.shape[0]
    d = np.diag(L).copy(); W = -(L - np.diag(d))
    sd = 1.0 / np.sqrt(d)
    X = np.tile(x0, (K, 1))
    acc = dict(n=0, Q=np.zeros(K), vir=np.zeros((K, N)), m1=np.zeros((K, N)), m2=np.zeros((N, N)))
    for s in range(sweeps):
        for k in range(N):
            m = X @ W[k] / d[k]
            al = (a[k] - m) / sd[k]; be = (b[k] - m) / sd[k]
            U = rng.random(K)
            upper = al > 0
            # lower branch: u in (Phi(al), Phi(be))
            pl, pu = ndtr(al), ndtr(be)
            xl = ndtri(np.clip(pl + U * (pu - pl), 1e-300, 1 - 1e-16))
            # upper branch: use survival function for accuracy
            sl, su = ndtr(-be), ndtr(-al)
            xu = -ndtri(np.clip(sl + U * (su - sl), 1e-300, 1 - 1e-16))
            X[:, k] = m + sd[k] * np.where(upper, xu, xl)
        if s >= burn:
            LX = X @ L
            acc["n"] += 1
            acc["Q"] += np.sum(X * LX, axis=1)
            acc["vir"] += X * LX
            acc["m1"] += X
            acc["m2"] += X.T @ X
    return acc


def limit_W_moments(L, U, V, zeta, n, rng, nb=8):
    """Moments of nu_W by slice integration: Y ~ gamma_C weighted by the t-slice length, t uniform."""
    N = L.shape[0]
    res = []
    for b in range(nb):
        Y = gaussian_sampler(L, rng, n)
        upper = np.min(np.stack([zeta[v] - Y[:, v] for v in V]), axis=0)
        lower = np.max(np.stack([zeta[u] - Y[:, u] for u in U]), axis=0)
        w = np.clip(upper - lower, 0, None); w = w / w.sum()
        tm = 0.5 * (upper + lower)                     # E[t | y]
        tv = (np.clip(upper - lower, 0, None)) ** 2 / 12  # Var[t | y]
        Xi_mean = Y + tm[:, None]
        Q = np.einsum('ni,ij,nj->n', Y, L, Y)
        LY = Y @ L.T
        vir = (Y + tm[:, None]) * LY                     # E[xi_k (L xi)_k | y], L xi = L y
        m2 = (Xi_mean * w[:, None]).T @ Xi_mean + np.sum(w * tv) * np.ones((N, N))
        res.append(dict(Q=w @ Q, vir=w @ vir, m1=w @ Xi_mean, m2=m2))
    agg = {k: np.mean([r[k] for r in res], axis=0) for k in res[0]}
    se = {k: np.std([r[k] for r in res], axis=0, ddof=1) / np.sqrt(nb) for k in res[0]}
    return agg, se


def exp_multicontact():
    rng = revision_rng(2)
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5), ordering="sorted")
    L = laplacian(G); N = L.shape[0]
    U, V = [0, 4], [22]
    eps = 0.05
    out = {}
    for case, zeta in [("exact", {0: 0.0, 4: 0.0, 22: 0.0}), ("offset", {0: 0.5, 4: -0.5, 22: 0.3})]:
        lo = -np.ones(N); hi = np.ones(N)
        for u in U: lo[u] = eps * zeta[u]
        for v in V: hi[v] = eps * zeta[v]
        a = lo / eps; b = hi / eps            # xi units, c = 0
        x0 = np.clip(np.zeros(N), a, b) + 0.0
        x0 = np.full(N, 0.5 * (max(a[U]) + min(b[V])))
        t = time.time()
        K, sweeps, burn = 4000, 4000, 1000
        acc = gibbs_box(L, a, b, K, sweeps, burn, rng, x0)
        n = acc["n"]
        Qc = acc["Q"] / n                     # per chain
        virc = acc["vir"] / n
        m1c = acc["m1"] / n
        gibbs = dict(Q=float(Qc.mean()), Q_se=float(Qc.std(ddof=1) / np.sqrt(K)),
                     vir=virc.mean(0), vir_se=virc.std(0, ddof=1) / np.sqrt(K),
                     m1=m1c.mean(0), m1_se=m1c.std(0, ddof=1) / np.sqrt(K),
                     m2=acc["m2"] / (n * K))
        lim, lim_se = limit_W_moments(L, U, V, zeta, 2 ** 18, rng)
        # pairwise mean squares for edges
        edges = sorted((min(i, j), max(i, j)) for i, j in G.edges())
        def pair(m2, i, j): return m2[i, i] + m2[j, j] - 2 * m2[i, j]
        ge = np.array([pair(gibbs["m2"], i, j) for i, j in edges]); le = np.array([pair(lim["m2"], i, j) for i, j in edges])
        out[case] = dict(gibbs_Q=gibbs["Q"], gibbs_Q_se=gibbs["Q_se"], lim_Q=float(lim["Q"]), lim_Q_se=float(lim_se["Q"]),
                         gibbs_vir=gibbs["vir"].tolist(), gibbs_vir_se=gibbs["vir_se"].tolist(),
                         lim_vir=lim["vir"].tolist(), gibbs_m1=gibbs["m1"].tolist(), lim_m1=lim["m1"].tolist(),
                         gibbs_m1_se=gibbs["m1_se"].tolist(),
                         edge_gibbs=ge.tolist(), edge_lim=le.tolist(), edges=[list(e) for e in edges],
                         seconds=time.time() - t)
        print(case, "Gibbs EQ %.4f±%.4f  limit %.4f±%.4f" % (gibbs["Q"], gibbs["Q_se"], lim["Q"], lim_se["Q"]),
              "vir range %.4f..%.4f" % (gibbs["vir"].min(), gibbs["vir"].max()),
              "max|m1 diff| %.4f" % np.max(np.abs(gibbs["m1"] - lim["m1"])),
              "max|edge diff| %.4f" % np.max(np.abs(ge - le)), "%.0fs" % (time.time() - t), flush=True)
    json.dump(out, open("results/multicontact.json", "w"), indent=1)


# ---------------------------------------------------------------- dynamics time steps
def run_dynamics(h, M, T, burn, rng):
    N = 8; L = laplacian(nx.path_graph(N)); C, Rm = graph_data(L); R = Rm[0, 7]
    kappa, eps = 1.0, 0.05; D = kappa * eps ** 2
    lo, hi = contact_walls(N, 0, 7, eps, R, 0.0)
    lam, Uv = np.linalg.eigh(L)
    E = (Uv * np.exp(-kappa * lam * h)) @ Uv.T
    var = np.where(lam > 1e-12, D / np.where(lam > 1e-12, kappa * lam, 1.0) * (1 - np.exp(-2 * kappa * lam * h)), 2 * D * h)
    Sh = (Uv * np.sqrt(var)) @ Uv.T
    X = np.zeros((M, N)); X[:, 0] = 0.01; X[:, 7] = -0.01
    rec = int(round(0.1 / h)); nsteps = int(round(T / h)); nburn = int(round(burn / h))
    sS = np.zeros(M); sS2 = np.zeros(M); sQ = np.zeros(M); cnt = 0
    for n in range(nsteps):
        X = X @ E + rng.standard_normal((M, N)) @ Sh
        for _ in range(3):
            X = np.where(X < lo, 2 * lo - X, X)
            X = np.where(X > hi, 2 * hi - X, X)
        if n >= nburn and n % rec == 0:
            S = (X[:, 0] - X[:, 7]) / (eps * np.sqrt(R)); Q = np.einsum('ni,ij,nj->n', X, L, X) / eps ** 2
            sS += S; sS2 += S * S; sQ += Q; cnt += 1
    f = lambda s: (float(np.mean(s / cnt)), float(np.std(s / cnt, ddof=1) / np.sqrt(M)))
    return dict(h=h, ES=f(sS), ES2=f(sS2), EQ=f(sQ))


def exp_steps():
    rng = revision_rng(5)
    res = []
    for h in [0.01, 0.005, 0.0025]:
        t = time.time(); r = run_dynamics(h, 1000, 1500.0, 150.0, rng); res.append(r)
        print(r, "%.0fs" % (time.time() - t), flush=True)
    # finite-eps stationary values for the same box
    L = laplacian(nx.path_graph(8)); C, Rm = graph_data(L); R = Rm[0, 7]
    lo, hi = contact_walls(8, 0, 7, 0.05, R, 0.0)
    vals = []
    for b in range(16):
        Y, w, R_, h_ = contact_is(L, 0, 7, lo, hi, 0.05, 0.0, 2 ** 18, rng)
        a = np.zeros(8); a[0] = 1; a[7] = -1
        S = Y @ a / np.sqrt(R); Q = np.einsum('ni,ij,nj->n', Y, L, Y)
        vals.append([w @ S, w @ S ** 2, w @ Q])
    vals = np.array(vals)
    stat = dict(ES=(float(vals[:, 0].mean()), float(vals[:, 0].std(ddof=1) / 4)),
                ES2=(float(vals[:, 1].mean()), float(vals[:, 1].std(ddof=1) / 4)),
                EQ=(float(vals[:, 2].mean()), float(vals[:, 2].std(ddof=1) / 4)))
    print("stationary", stat, flush=True)
    json.dump(dict(steps=res, stationary=stat), open("results/steps.json", "w"), indent=1)


# ---------------------------------------------------------------- diagnostics
def exp_diagnostics():
    out = {}
    L = laplacian(nx.path_graph(8))
    for k, (lo, hi) in incompat_configs().items():
        x = qp_min(L, lo, hi)
        g = L @ x; tol = 1e-9
        Am = np.where(np.abs(x - lo) < tol)[0]; Ap = np.where(np.abs(x - hi) < tol)[0]
        A_ = np.concatenate([Am, Ap]); F = np.array([i for i in range(8) if i not in A_])
        xp = x.copy(); xp[Am] = lo[Am]; xp[Ap] = hi[Ap]
        xp[F] = -np.linalg.solve(L[np.ix_(F, F)], L[np.ix_(F, A_)] @ xp[A_])
        gp = L @ xp
        out[str(k)] = dict(pg_change=float(np.max(np.abs(xp - x))), stat_res=float(np.max(np.abs(gp[F]))),
                           sign_ok=bool(np.all(gp[Am] > 0) and np.all(gp[Ap] < 0)),
                           feas=float(max(np.max(lo - xp), np.max(xp - hi))))
    tc = json.load(open("results/table_contact.json")); ti = json.load(open("results/table_incompat.json"))
    out["ess_contact_min"] = min(r["ess"] for r in tc)
    out["ess_incompat_min"] = min(r["ess"] for r in ti)
    out["ess_incompat_by"] = {f"{r['A']}_{r['eps']}": r["ess"] for r in ti}
    json.dump(out, open("results/diagnostics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    fns = dict(rejection=exp_rejection, multi=exp_multicontact, steps=exp_steps, diag=exp_diagnostics)
    for w_ in sys.argv[1:]:
        t = time.time(); fns[w_](); print(w_, "%.1fs" % (time.time() - t), flush=True)
