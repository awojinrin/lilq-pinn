"""
Package 5, item 4: the irregular domain (Poisson-Boltzmann around three inclusions)
===================================================================================

Section 5 of the instructions of 13 November 2026. The problem, point sets
and systems are Package 4's (``problems.poisson_boltzmann``,
``experiments/p4_2_pb_inclusions.py``), unchanged.

**The points.** Package 4 shipped 8,000 Halton points in Omega per seed;
4a at p = 42 needs 8P = 14,112 and 4b at P = 1,600 needs 12,800. The
``points`` stage draws ``HALTON_N`` = 16,000 per seed on the laptop
(``pb.halton``, as Package 4) and checks that the first 8,000 are
Package 4's shipped points, bit for bit; the draws are shipped in
``data/package5``. Every Package 4 set is a prefix of them.

**4a** (``lilq``, ``under``, ``time``): Package 4's 2a and 2c at
p = 34, 38, 42, seeds 0, 1, 2, by Package 4's own functions:
- 2a logged with termination disabled, K_max = 60, every iterate stored, the
  rule read from the log;
- 2c: N = P/2 rows, Newton with the minimum-norm step (gelsd);
- the timing of 2a by the protocol of the paper (Section 2: dedicated node,
  warm-up, median of three), stopped at the iterate the rule returns; the
  timed region builds the matrices from the shipped points, as Package 4.

**4b** (``rf``): the same problem, points (Halton seed 0, 8P interior
points, 12 round(sqrt(P)) boundary points), weight 1,000, solver and rule,
with tanh(a x + b y + c) features (``pb.RandomFeatures``): P = 100, 200, 400,
800, 1,600; sigma = 2, 4, 8; draws 0, 1, 2. Logged as 2a (K_max = 60, every
iterate stored); untimed.

**delta_P two ways** (``delta``): delta_L2 and delta_H2 with ``gelsy`` and
with ``gelsd``, both at rcond = eps, for p = 10..30 of Package 4, the
spaces of 4a and of 4b; both and their relative difference
|gelsy - gelsd| / gelsd.

Usage::

    python experiments/p5_4_pb.py points --out data/package5 --shipped data/package4      (laptop, once)
    python experiments/p5_4_pb.py {lilq,under,time,rf,delta} --out <root> --points data/package5 [--p ..] [--seeds ..] [--P ..]
    python experiments/p5_4_pb.py summarize --out <root> [--expected-rf expected_rf_pb.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np
import scipy

from experiments import p4_2_pb_inclusions as p42
from experiments.p4_common import K_MAX, logged_run, read_csv, rel, to_float, write_csv
from lilq.run_metadata import first_stall_iteration, write_run_json
from problems import poisson_boltzmann as pb

ITEM = 'P5_4_pb'
HALTON_N = 16000
P4_HALTON_N = p42.HALTON_N
SEEDS = (0, 1, 2)
SIZES_A = (34, 38, 42)
SIZES_P4 = p42.SIZES
RF_P = (100, 200, 400, 800, 1600)
RF_SIGMA = (2.0, 4.0, 8.0)
RF_DRAWS = (0, 1, 2)
RF_SEED = 0                       # the Halton draw of 4b
RF_K_MAX = K_MAX
DRIVERS = ('gelsy', 'gelsd')
# Section 5's indicative values (ref_pb_large.py, seed 0): within a factor of two
INDICATIVE_4A = {34: {'stall_level': 4.1e-8, 'eH2': 1.7e-8, 'rank_gelsy': 1098},
                 38: {'stall_level': 6.6e-10, 'eH2': 1.8e-10, 'rank_gelsy': 1340},
                 42: {'stall_level': 1.5e-11, 'eH2': 5.0e-12, 'rank_gelsy': 1594}}
INDICATIVE_RULE_STEP = 4
INDICATIVE_FACTOR = 2.0


# ---------------------------------------------------------------- the points

def draw_points(out_dir, shipped, n=HALTON_N, seeds=SEEDS):
    """Draw n Halton points in Omega per seed; check the prefix against Package 4's shipped draws."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for seed in seeds:
        x, y = pb.halton(n, seed)
        with np.load(Path(shipped) / f'halton_seed{seed}.npz') as z:
            m = len(z['x'])
            same = bool(np.array_equal(x[:m], z['x']) and np.array_equal(y[:m], z['y']))
        if not same:
            raise RuntimeError(f'seed {seed}: the first {m} points differ from the shipped draw')
        np.savez(out_dir / f'halton_seed{seed}.npz', x=x, y=y, scipy_version=scipy.__version__, n=n, seed=seed)
        report.append({'seed': seed, 'n': n, 'prefix_of_package4': m, 'prefix_identical': same})
    (out_dir / 'README.json').write_text(json.dumps({'drawn_with_scipy': scipy.__version__, 'seeds': report,
                                                     'note': 'Package 5, item 4: pb.halton(16000, seed); the first '
                                                             '8,000 are data/package4 bit for bit'}, indent=2))
    return report


def compare_points(points_dir, out_root, n=HALTON_N, seeds=SEEDS):
    """This machine's scipy against the shipped 16,000 points (a record only: every set is built
    from the shipped points). Writes P5_4_pb/points/halton_compare.json."""
    shipped = p42.Halton(points_dir)
    rows = []
    for seed in seeds:
        x, y = shipped.take(seed, n)
        gx, gy = pb.halton(n, seed)
        rows.append({'seed': seed, 'identical': bool(np.array_equal(x, gx) and np.array_equal(y, gy)),
                     'max_abs_difference': float(max(np.abs(x - gx).max(), np.abs(y - gy).max()))})
    out = {'scipy_here': scipy.__version__, 'scipy_shipped': shipped.scipy_version(), 'n': n, 'seeds': rows}
    path = Path(out_root) / ITEM / 'points' / 'halton_compare.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- 4b: random features

def rf_dir(out_root, P, sigma, draw):
    return Path(out_root) / ITEM / 'random_features' / f'P{P}_sigma{sigma:g}_draw{draw}'


def rf_run(out_root, halton, P, sigma, draw, norms):
    space = pb.RandomFeatures(P, sigma, draw)
    S = pb.lilq_system(space, RF_SEED, interior=halton.take(RF_SEED, int(round(pb.RATIO * P))))
    d = rf_dir(out_root, P, sigma, draw)
    t0 = time.perf_counter()
    extra = lambda b: dict(zip(('eL2', 'eH2'), norms.errors(b, space)))  # noqa: E731
    _, k_s, _, _, tracker, logger, scaled = logged_run(S, np.zeros(P), d, lambda b: norms.errors(b, space)[0],
                                                       k_max=RF_K_MAX, extra_fn=extra)
    _, beta_ret = p42.returned_iterate(d)
    np.save(d / 'beta_returned.npy', beta_ret)
    meta = p42._meta(S, RF_K_MAX, pb.LAM, pb.RATIO, RF_SEED, halton.scipy_version())
    meta.update(first_stall_iteration=first_stall_iteration(logger.rows), b2_check=tracker.b2_check)
    meta['basis_description'] = {'family': 'tanh(a_j x + b_j y + c_j), a, b, c uniform in [-sigma, sigma]',
                                 'P': P, 'sigma': sigma, 'draw': draw,
                                 'generator': f'numpy.random.default_rng({100 + draw}), in the order a, b, c'}
    write_run_json(d / 'run.json', {**meta, 'package5': {
        'item': '4b', 'P': P, 'sigma': sigma, 'draw': draw, 'halton_seed': RF_SEED,
        'n_interior': S.n_interior, 'n_boundary': S.n_boundary, 'lambda_boundary': pb.LAM,
        'rule_fires_at_step': k_s, 'identity_scaled_max': scaled, 'wall_time_s': time.perf_counter() - t0}})
    return d


def run_rf(out_root, halton, sizes=None, sigmas=None):
    """The 4b runs (resumable). ``sizes`` and ``sigmas`` choose a part of the set, so that one job
    can run parts side by side; the default is the whole set."""
    norms = pb.Norms()
    for P in sizes or RF_P:
        for sigma in sigmas or RF_SIGMA:
            for draw in RF_DRAWS:
                if (rf_dir(out_root, P, sigma, draw) / 'run.json').exists():
                    continue                                   # resume
                d = rf_run(out_root, halton, P, sigma, draw, norms)
                print('4b', d, flush=True)


# ---------------------------------------------------------------- delta_P two ways

def spaces(which=('p4', '4a', '4b'), sizes_rf=None):
    out = []
    if 'p4' in which:
        out += [(f'cheb_p{p}', p) for p in SIZES_P4]
    if '4a' in which:
        out += [(f'cheb_p{p}', p) for p in SIZES_A]
    if '4b' in which:
        out += [(f'rf_P{P}_sigma{s:g}_draw{d}', pb.RandomFeatures(P, s, d))
                for P in sizes_rf or RF_P for s in RF_SIGMA for d in RF_DRAWS]
    return out


def delta_path(out_root, label):
    return Path(out_root) / ITEM / 'delta_P' / f'{label}.json'


def run_delta(out_root, which=('p4', '4a', '4b'), sizes_rf=None):
    norms = pb.Norms()
    for label, space in spaces(which, sizes_rf):
        path = delta_path(out_root, label)
        if path.exists():
            continue
        row = {'space': label, 'P': pb.space_size(space)}
        for driver in DRIVERS:
            t0 = time.perf_counter()
            d0, d2 = norms.best(space, driver)
            row.update({f'delta_L2_{driver}': d0, f'delta_H2_{driver}': d2,
                        f'time_s_{driver}': time.perf_counter() - t0})
        for n in ('L2', 'H2'):
            row[f'rel_diff_{n}'] = rel(row[f'delta_{n}_gelsy'], row[f'delta_{n}_gelsd'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, indent=2))
        print('delta', row, flush=True)


def best_of(out_root, label):
    """(delta_L2, delta_H2) by gelsy, as delivered (Package 4)."""
    r = json.loads(delta_path(out_root, label).read_text())
    return r['delta_L2_gelsy'], r['delta_H2_gelsy']


# ---------------------------------------------------------------- the summary

def summarize(out_root, expected_rf=None):
    root = Path(out_root) / ITEM
    out = {}
    deltas = [json.loads(f.read_text()) for f in sorted((root / 'delta_P').glob('*.json'))]
    if deltas:
        write_csv(root / 'delta_P_two_ways.csv', deltas)
    out['delta'] = deltas
    lilq = []
    for p in SIZES_A:
        for seed in SEEDS:
            d = root / 'lilq' / f'p{p}_seed{seed}'
            if (d / 'run.json').exists() and delta_path(out_root, f'cheb_p{p}').exists():
                lilq.append(p42.run_row(d, best_of(out_root, f'cheb_p{p}')))
    if lilq:
        write_csv(root / 'terminal_4a.csv', lilq)
    out['4a'] = lilq
    ind = []
    for r in lilq:
        if r['seed'] == 0 and r['p'] in INDICATIVE_4A:
            row = {'p': r['p'], 'rule_step': r['rule_step'], 'rule_step_indicative': INDICATIVE_RULE_STEP}
            for k, v in INDICATIVE_4A[r['p']].items():
                ours = to_float(r[k])
                row.update({k: ours, f'{k}_indicative': v, f'{k}_ratio': ours / v,
                            f'{k}_within_factor_2': 1 / INDICATIVE_FACTOR <= ours / v <= INDICATIVE_FACTOR})
            ind.append(row)
    if ind:
        write_csv(root / 'compare_indicative_4a.csv', ind)
    out['4a_indicative'] = ind
    rf = []
    for P in RF_P:
        for sigma in RF_SIGMA:
            for draw in RF_DRAWS:
                d = rf_dir(out_root, P, sigma, draw)
                label = f'rf_P{P}_sigma{sigma:g}_draw{draw}'
                if (d / 'run.json').exists() and delta_path(out_root, label).exists():
                    run = json.loads((d / 'run.json').read_text())
                    rf.append({'P': P, 'sigma': sigma, 'draw': draw, 'N': run['N_total'],
                               **p42.rule_row(d, best_of(out_root, label))})
    if rf:
        write_csv(root / 'terminal_4b.csv', rf)
    out['4b'] = rf
    if expected_rf and rf:
        out['4b_expected'] = compare_rf(rf, expected_rf)
        write_csv(root / 'compare_expected_4b.csv', out['4b_expected'])
    return out


def compare_rf(rows, expected_path):
    """Our 4b rows beside ``expected_rf_pb.json`` (his ``seed`` is our draw; his ``solves`` is rule step + 1)."""
    exp = {(e['P'], float(e['sigma']), e['seed']): e for e in json.loads(Path(expected_path).read_text())}
    out = []
    for r in rows:
        e = exp.get((r['P'], float(r['sigma']), r['draw']))
        if e is None:
            continue
        solves = None if r['rule_step'] is None else r['rule_step'] + 1
        out.append({'P': r['P'], 'sigma': r['sigma'], 'draw': r['draw'],
                    'solves': solves, 'solves_expected': e['solves'],
                    'rank_gelsy': r['rank_gelsy'], 'rank_expected': e['rank'],
                    'stall_level': r['stall_level'], 'stall_expected': e['stall'],
                    'eH2': r['eH2'], 'eH2_expected': e['eH2'], 'eH2_rel_diff': rel(r['eH2'], e['eH2']),
                    'eH2_over_delta': r['eH2_over_delta'], 'eH2_over_delta_expected': e['eH2'] / e['dH2'],
                    'delta_H2': r['delta_H2'], 'delta_H2_expected': e['dH2'],
                    'delta_H2_rel_diff': rel(r['delta_H2'], e['dH2'])})
    return out


# ---------------------------------------------------------------- the command line

def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 5, item 4: the irregular domain.')
    ap.add_argument('stage', choices=('points', 'compare-points', 'lilq', 'under', 'time', 'rf', 'delta', 'summarize'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--points', help='the folder of the 16,000-point halton_seed*.npz (data/package5)')
    ap.add_argument('--shipped', help="points: Package 4's draws (data/package4)")
    ap.add_argument('--p', type=int, nargs='+')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--P', type=int, nargs='+', help='4b sizes (rf, delta)')
    ap.add_argument('--sigma', type=float, nargs='+', help='rf: these sigma only (default: all)')
    ap.add_argument('--which', nargs='+', default=['p4', '4a', '4b'], help="delta: spaces of 'p4', '4a', '4b'")
    ap.add_argument('--expected-rf', help='expected_rf_pb.json')
    a = ap.parse_args(argv)
    if a.stage == 'points':
        print(json.dumps(draw_points(a.out, a.shipped), indent=1))
        return 0
    if a.stage == 'summarize':
        out = summarize(a.out, a.expected_rf)
        print({k: len(v) for k, v in out.items()})
        return 0
    if a.stage == 'delta':
        run_delta(a.out, tuple(a.which), a.P)
        return 0
    if not a.points:
        ap.error('--points is required for this stage')
    if a.stage == 'compare-points':
        print(json.dumps(compare_points(a.points, a.out)))
        return 0
    halton = p42.Halton(a.points)
    if a.stage == 'lilq':
        p42.run_lilq(a.out, halton, a.p or SIZES_A, a.seeds, item=ITEM)
    elif a.stage == 'under':
        p42.run_under(a.out, halton, a.p or SIZES_A, a.seeds, item=ITEM)
    elif a.stage == 'time':
        p42.run_timing(a.out, halton, a.p or SIZES_A, a.seeds, item=ITEM)
    else:
        run_rf(a.out, halton, a.P, a.sigma)
    return 0


if __name__ == '__main__':
    sys.exit(main())
