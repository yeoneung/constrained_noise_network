import numpy as np, networkx as nx, json
from experiments2 import limit_W_moments
from core import laplacian
rng = np.random.default_rng(np.random.SeedSequence(20261030).spawn(1)[0])
G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5), ordering="sorted")
L = laplacian(G)
d = json.load(open("results/multicontact.json"))
for case, zeta in [("exact", {0: 0.0, 4: 0.0, 22: 0.0}), ("offset", {0: 0.5, 4: -0.5, 22: 0.3})]:
    lim, se = limit_W_moments(L, [0, 4], [22], zeta, 2 ** 20, rng, nb=24)
    edges = d[case]["edges"]
    pair = lambda m2, i, j: m2[i, i] + m2[j, j] - 2 * m2[i, j]
    le = np.array([pair(lim["m2"], i, j) for i, j in edges])
    d[case].update(lim_Q=float(lim["Q"]), lim_Q_se=float(se["Q"]), lim_vir=lim["vir"].tolist(), lim_m1=lim["m1"].tolist(),
                   lim_m1_se=se["m1"].tolist(), edge_lim=le.tolist())
    ge = np.array(d[case]["edge_gibbs"]); gm = np.array(d[case]["gibbs_m1"])
    print(case, "lim EQ %.4f±%.4f gibbs %.4f±%.4f" % (lim["Q"], se["Q"], d[case]["gibbs_Q"], d[case]["gibbs_Q_se"]),
          "max|m1| %.4f" % np.max(np.abs(gm - lim["m1"])), "max|edge| %.4f" % np.max(np.abs(ge - le)),
          "lim vir range %.4f..%.4f" % (lim["vir"].min(), lim["vir"].max()),
          "gibbs vir range %.4f..%.4f" % (min(d[case]["gibbs_vir"]), max(d[case]["gibbs_vir"])),
          "max vir se %.4f" % max(d[case]["gibbs_vir_se"]))
json.dump(d, open("results/multicontact.json", "w"), indent=1)
