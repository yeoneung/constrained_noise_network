"""Coordinate Gibbs check of grounded covariance on an incompatible cycle.

Run from the code directory: python cycle_check.py
The JSON output contains 16 independent batch estimates, standard errors,
and first/second-half estimates for a simple mixing check.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr, ndtri


def run(eps, seed, batches=16, chains_per_batch=64, sweeps=4000, burn=1000):
    if not 0 < burn < sweeps:
        raise ValueError('burn must lie between 0 and sweeps')
    rng = np.random.default_rng(seed)
    lo = -2 * np.ones(8)
    hi = 2 * np.ones(8)
    lo[0] = 1.0
    hi[4] = 0.0
    a, b = lo / eps, hi / eps
    xstar = np.array([1, .75, .5, .25, 0, .25, .5, .75])
    x = np.broadcast_to(xstar / eps, (batches, chains_per_batch, 8)).copy()
    # Center around x* before calculating covariance to avoid cancellation.
    # Columns are energy, y2, y3, y2*y3, (y2-y3)^2.
    sums = np.zeros((2, batches, 5))
    counts = np.zeros(2, dtype=int)
    sd = 1 / np.sqrt(2)
    half = burn + (sweeps - burn) // 2
    for sweep in range(sweeps):
        for k in range(8):
            mean = (x[:, :, (k - 1) % 8] + x[:, :, (k + 1) % 8]) / 2
            alpha, beta = (a[k] - mean) / sd, (b[k] - mean) / sd
            u = rng.random((batches, chains_per_batch))
            pl, pu = ndtr(alpha), ndtr(beta)
            sl, su = ndtr(-beta), ndtr(-alpha)
            lower = ndtri(np.clip(pl + u * (pu - pl), 1e-300, 1 - 1e-16))
            upper = -ndtri(np.clip(sl + u * (su - sl), 1e-300, 1 - 1e-16))
            x[:, :, k] = mean + sd * np.where(alpha > 0, upper, lower)
        if sweep >= burn:
            part = int(sweep >= half)
            y2, y3 = x[:, :, 1] - xstar[1] / eps, x[:, :, 2] - xstar[2] / eps
            energy = np.sum((x - np.roll(x, 1, axis=2)) ** 2, axis=2) - .5 / eps**2
            sums[part, :, 0] += energy.mean(axis=1)
            sums[part, :, 1] += y2.mean(axis=1)
            sums[part, :, 2] += y3.mean(axis=1)
            sums[part, :, 3] += (y2 * y3).mean(axis=1)
            sums[part, :, 4] += ((y2 - y3) ** 2).mean(axis=1)
            counts[part] += 1

    def measures(values):
        energy, y2, y3, prod, pair = values.T
        return np.column_stack((energy, prod - y2 * y3, pair - (y2 - y3)**2))

    estimates = measures(sums.sum(axis=0) / counts.sum())
    first = measures(sums[0] / counts[0])
    second = measures(sums[1] / counts[1])
    return {'eps': eps, 'seed': seed, 'batches': batches,
            'chains_per_batch': chains_per_batch, 'sweeps': sweeps, 'burn': burn,
            'columns': ['excess_energy', 'cov_23', 'var_difference_23'],
            'mean': estimates.mean(axis=0).tolist(),
            'se': (estimates.std(axis=0, ddof=1) / np.sqrt(batches)).tolist(),
            'batch_estimates': estimates.tolist(),
            'half_difference': (second - first).mean(axis=0).tolist(),
            'half_difference_se': ((second - first).std(axis=0, ddof=1) / np.sqrt(batches)).tolist()}


if __name__ == '__main__':
    results = []
    for eps, seed in [(0.1, 2026110101), (0.05, 2026110102), (0.025, 2026110103)]:
        result = run(eps, seed)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'batch_estimates'}, indent=2), flush=True)
    output = Path('results/cycle_incompatible.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n')
