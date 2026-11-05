"""Check independent identities and selected stored outputs without rerunning simulations."""
from pathlib import Path
import json
import numpy as np
import networkx as nx
from scipy.integrate import quad
from core import f_z, psi, var_S, mu

# The single-contact limiting density is normalized and has the stated moments.
for z in (-2.0, 0.0, 2.0):
    norm = quad(lambda s: f_z(s, z), z, np.inf, epsabs=1e-11)[0]
    m1 = quad(lambda s: s * f_z(s, z), z, np.inf, epsabs=1e-11)[0]
    m2 = quad(lambda s: s * s * f_z(s, z), z, np.inf, epsabs=1e-11)[0]
    np.testing.assert_allclose(norm, 1, atol=1e-10)
    np.testing.assert_allclose(m2, 1 + psi(z), atol=1e-10)
    np.testing.assert_allclose(m2 - m1 * m1, var_S(z), atol=1e-10)

# KKT conditions and grounded covariance for the new incompatible cycle.
L = nx.laplacian_matrix(nx.cycle_graph(8)).toarray().astype(float)
x = np.array([1, .75, .5, .25, 0, .25, .5, .75])
g = L @ x
np.testing.assert_allclose(g, [.5, 0, 0, 0, -.5, 0, 0, 0], atol=1e-14)
np.testing.assert_allclose(x @ L @ x, .5, atol=1e-14)
F = [1, 2, 3, 5, 6, 7]
C = np.linalg.inv(L[np.ix_(F, F)])
np.testing.assert_allclose(C[0, 1], .5, atol=1e-14)
np.testing.assert_allclose(C[0, 0] + C[1, 1] - 2*C[0, 1], .75, atol=1e-14)

# Check the actual saved batch estimates and their reported uncertainties.
rows = json.loads(Path('results/cycle_incompatible.json').read_text())
assert [r['eps'] for r in rows] == [.1, .05, .025]
for r in rows:
    batches = np.array(r['batch_estimates'])
    assert batches.shape == (16, 3)
    np.testing.assert_allclose(batches.mean(0), r['mean'], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(batches.std(0, ddof=1) / 4, r['se'], rtol=1e-13, atol=1e-13)
np.testing.assert_allclose(rows[-1]['mean'], [10, .5, .75], atol=.03, rtol=0)

# The exact two-node calculations have the claimed values in the three regimes.
two = json.loads(Path('results/two_node_check.json').read_text())
np.testing.assert_allclose([two['0.01']['neg'], two['0.01']['zero'], two['0.01']['pos']],
                           [1, 2, 3.9976], atol=.0001, rtol=0)
# The separately regenerated spatial dataset reproduces the manuscript's
# reported rounded summary statistics; it is not the original figure input.
with np.load('results/spatial.npz') as spatial:
    R = spatial['R']
    h = spatial['h']
    pairs = np.triu_indices(len(h), 1)
    profile = ((h[:, None] - h[None, :]) ** 2)[pairs]
    np.testing.assert_allclose(R.max(), 1.7922, atol=0.00005, rtol=0)
    np.testing.assert_allclose(np.mean(profile < .01), .4641, atol=0.00005, rtol=0)
    errors, standard_errors = [], []
    for z, expected in [(0., (0.9993, -0.5710)), (1., (2.9066, -0.7220))]:
        raw = (spatial[f'raw_mean_{z}'] - R)[pairs]
        centered = (spatial[f'cen_mean_{z}'] - R)[pairs]
        slopes = [profile @ raw / (profile @ profile),
                  profile @ centered / (profile @ profile)]
        np.testing.assert_allclose(slopes, expected, atol=0.00005, rtol=0)
        errors.extend([np.max(np.abs(raw - psi(z) * profile)),
                       np.max(np.abs(centered - (psi(z) - mu(z)**2) * profile))])
        standard_errors.extend([np.max(spatial[f'raw_se_{z}'][pairs]),
                                np.max(spatial[f'cen_se_{z}'][pairs])])
    np.testing.assert_allclose(max(errors), .0044, atol=.00005, rtol=0)
    np.testing.assert_allclose(max(standard_errors), .0031, atol=.00005, rtol=0)

print('Response-function integrals, cycle KKT/grounded covariance, stored batch results,')
print('two-node check, and regenerated spatial summaries passed.')
