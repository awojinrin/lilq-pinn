"""
Random-search configurations for the Component A baselines (Package 1 v2.0 Section 4.3)
======================================================================================

24 configurations per family, drawn with generator seed 12345 and saved
(``search/F1_configs.json``, ``search/F2_configs.json``) before any run.
Each family has its own ``numpy.random.default_rng(12345)``, so adding or
reordering one family never changes the other's draws.

F1 space: width {64, 128, 256}, depth {3, 4, 5}, shared trunk or three
networks, m in {32, 64, 128}, sigma_FF in {0.5, 1, 2, 5}, boundary
conditions hard or soft (soft: lambda_bc in {1, 10, 100}), loss balancing
on/off, eta in {1e-3, 3e-4, 1e-4}, T_Adam in {5,000, 20,000, 50,000},
N_int in {2,000, 8,000, 32,000}, resampling on/off. Hard/soft is drawn
first with equal probability, then lambda_bc if soft.

F2 space: width {32, 64, 100}, depth {3, 4} subject to n_theta <= 20,000
(violators are redrawn), m in {32, 64}, sigma_FF in {0.5, 1, 2},
N_int in {2,000, 8,000}. The 24 F2 configurations are distinct: a draw
that repeats an earlier configuration is redrawn from the same generator,
like a violator (the advisor's reply to wave 1, item 2.1; wave 1's list had
17 distinct). 42 of the space's 72 configurations satisfy the cap.

Usage::

    python -m baselines.search --out-dir <package>/A_calibration/search
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

SEARCH_SEED = 12345
N_CONFIGS = 24
MAX_PARAMS_F2 = 20000

F1_SPACE = {
    'width': [64, 128, 256], 'depth': [3, 4, 5], 'trunk': ['shared', 'separate'],
    'm': [32, 64, 128], 'sigma_ff': [0.5, 1.0, 2.0, 5.0], 'bc': ['hard', 'soft'],
    'lambda_bc': [1.0, 10.0, 100.0], 'balancing': [True, False], 'eta': [1e-3, 3e-4, 1e-4],
    't_adam': [5000, 20000, 50000], 'n_int': [2000, 8000, 32000], 'resample': [True, False],
}
F2_SPACE = {
    'width': [32, 64, 100], 'depth': [3, 4], 'm': [32, 64], 'sigma_ff': [0.5, 1.0, 2.0],
    'n_int': [2000, 8000],
}


def fourier_mlp_params(width, depth, m, n_out=3):
    """Parameters of a Fourier-feature MLP: 2m features -> depth tanh layers -> n_out."""
    return (2 * m * width + width) + (depth - 1) * (width * width + width) + (width * n_out + n_out)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def draw_f1(n=N_CONFIGS, seed=SEARCH_SEED):
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n):
        c = {k: _pick(rng, F1_SPACE[k]) for k in ('width', 'depth', 'trunk', 'm', 'sigma_ff', 'bc')}
        c['lambda_bc'] = _pick(rng, F1_SPACE['lambda_bc']) if c['bc'] == 'soft' else None
        c.update({k: _pick(rng, F1_SPACE[k]) for k in ('balancing', 'eta', 't_adam', 'n_int', 'resample')})
        outputs_per_net = 3 if c['trunk'] == 'shared' else 1
        nets = 1 if c['trunk'] == 'shared' else 3
        c['n_theta'] = nets * fourier_mlp_params(c['width'], c['depth'], c['m'], outputs_per_net)
        configs.append({'id': f'F1_{i:02d}', 'family': 'F1', **c})
    return configs


F2_KEYS = ('width', 'depth', 'm', 'sigma_ff', 'n_int')


def config_key(c, keys=F2_KEYS):
    """What makes two configurations the same (``id`` and the derived
    ``n_theta`` aside)."""
    return tuple(c[k] for k in keys)


def draw_f2(n=N_CONFIGS, seed=SEARCH_SEED, max_params=MAX_PARAMS_F2):
    """``n`` distinct configurations within the cap: a draw over the cap or
    equal to an earlier configuration is replaced by the generator's next
    draw. Returns ``(configs, redrawn_violators, redrawn_duplicates)``."""
    feasible = {k for k in itertools.product(*(F2_SPACE[key] for key in F2_KEYS))
                if fourier_mlp_params(*k[:3]) <= max_params}
    if len(feasible) < n:
        raise ValueError(f"only {len(feasible)} F2 configurations satisfy n_theta <= {max_params}; {n} needed")
    rng = np.random.default_rng(seed)
    configs, seen, violators, duplicates = [], set(), 0, 0
    while len(configs) < n:
        c = {k: _pick(rng, F2_SPACE[k]) for k in F2_KEYS}
        c['n_theta'] = fourier_mlp_params(c['width'], c['depth'], c['m'])
        if c['n_theta'] > max_params:
            violators += 1
            continue
        if config_key(c) in seen:
            duplicates += 1
            continue
        seen.add(config_key(c))
        configs.append({'id': f'F2_{len(configs):02d}', 'family': 'F2', **c})
    return configs, violators, duplicates


def save_search(out_dir):
    """Write both families' configurations; refuses to overwrite a saved
    search (the spec: drawn and saved before any run)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f2, redraws, duplicates = draw_f2()
    payloads = {
        'F1_configs.json': {'seed': SEARCH_SEED, 'space': F1_SPACE, 'configs': draw_f1()},
        'F2_configs.json': {'seed': SEARCH_SEED, 'space': F2_SPACE, 'max_params': MAX_PARAMS_F2,
                            'redrawn_violators': redraws, 'redrawn_duplicates': duplicates,
                            'configs': f2},
    }
    for name, payload in payloads.items():
        path = out_dir / name
        if path.exists():
            if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
                raise FileExistsError(f"{path} exists with different contents; not overwriting")
            continue
        path.write_text(json.dumps(payload, indent=2))
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Save the Component A random-search configurations")
    ap.add_argument('--out-dir', required=True)
    print(f"Wrote {save_search(ap.parse_args().out_dir)}")
