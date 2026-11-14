"""
Package 4, item 2: Poisson-Boltzmann around three inclusions
============================================================

The advisor's instructions of 12 November 2026, Section 4. The problem,
point sets and systems are ``problems.poisson_boltzmann``. The reference
implementation is ``ref_pb_inclusions.py``; where the text and that script
differ, the script decides.

**The Halton points are drawn once, on the laptop, and shipped**
(``data/package4/halton_seed{s}.npz``, the first ``HALTON_N`` points in Omega for
each seed). The first n points of a draw are the first n of any longer draw,
so every set below is a prefix of these. The laptop's scipy (1.17.0)
reproduces the advisor's points (scipy 1.18.1). Grace's scipy (1.13.1) is
compared with the shipped points by the ``compare-points`` stage and is not
used to build any set.

**2a** (``lilq``):
- p = 10, 14, ..., 30 and seeds 0, 1, 2.
- Logged with termination disabled, K_max = 60.
- Errors at every iterate: relative L2 and H2 on the 70 x 420 polar rule
  (``errors_by_iterate.csv``; ``eps_ref`` in ``iterations.csv`` is the L2
  error).
- The rule (tau_chi 0.1, tau_r 0.01, n_s 2) and the class are read from the
  log.

**2b** (``square``):
- N = P rows, no weights, from zero coefficients.
- Newton's method with LU.
- Scattered or approximate Fekete points; p = 10..26, seeds 0..2.
- Statuses:
  - "converged": the relative change of the coefficients is below 1e-9;
  - "diverged": a non-finite iterate or residual, or a coefficient norm
    above 1e14. The last accepted iterate is reported (zero if none).
  - "singular": the solve fails;
  - "budget": 30 steps.

**2c** (``under``): N = P/2 rows; the minimum-norm Newton step by gelsd; the
stopping test of 2b.

**2d** (``rows``): 2a at p = 14 and 22, seed 0, with 1.5 to 16 interior points
per unknown (the boundary set unchanged). K_max = 30.

**2e** (``weight``): 2a at p = 14, 18 and 22, seed 0, with lambda_d = 1 to 10^4.
K_max = 30.

**2f** (``constants``): for every 2a run and the two extreme 2d sets at p = 22,
c1, c2 and rho_r at every iterate, by the recipe of Package 3, Section 6.
- The norm of Y: the 70 x 420 rule inside, the 4,000 points of
  ``boundary(4000)`` on the boundary, with the block-matched weights (each
  block's total squared weight in the collocation set).
- Columns with |R_ii| < 1e-10 |R_11| are dropped and counted.

**Timing** (``time``): the protocol of Section 2.3, stopped at the iterate the
rule returns. The timed region builds the collocation matrices from the
shipped points and runs the iteration.

**M7** (``m7``): the errors and delta_P of 2a at p = 22, seed 0, on the
140 x 840 rule. **The solution figure** (``solution``): p = 22, seed 0,
on a 321 x 321 grid of the box, with the mask of Omega.

Usage::

    python experiments/p4_2_pb_inclusions.py points --out <root>            (laptop: draw and ship)
    python experiments/p4_2_pb_inclusions.py points --out <root> --points data/package4   (Grace: copy the shipped draws)
    python experiments/p4_2_pb_inclusions.py compare-points --out <root> --points <dir>
    python experiments/p4_2_pb_inclusions.py {lilq,square,under,rows,weight,constants,time,m7,solution} --out <root> --points <dir>
    python experiments/p4_2_pb_inclusions.py summarize --out <root> --expected expected_pb.json
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
import scipy.linalg as sla

from experiments.p4_common import (K_MAX, N_S, iterate_to_rule, kappa_returned, logged_run, m3_summary, read_csv, rel,
                                   stall_class, timed, to_float, write_csv)
from lilq.instrumentation import EPS_MACH, stall_rule_index
from lilq.provenance import capture_blas_thread_env
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
from problems import poisson_boltzmann as pb

ITEM = 'P4_2_pb_inclusions'
SIZES = (10, 14, 18, 22, 26, 30)
SEEDS = (0, 1, 2)
SQUARE_SIZES = (10, 14, 18, 22, 26)
RATIOS = (1.5, 2, 3, 4, 6, 8, 12, 16)
RATIO_SIZES = (14, 22)
WEIGHTS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
WEIGHT_SIZES = (14, 18, 22)
SWEEP_K_MAX = 30
HALTON_N = 8000
NEWTON_K_MAX, NEWTON_TOL, BETA_CAP = 30, 1e-9, 1e14
Y_BOUNDARY_N = 4000
M7_RULE = (140, 840)
GRID_N = 321


# ---------------------------------------------------------------- the shipped Halton points

def draw_points(out_root, shipped=None):
    """The points folder of the deliverables: the Halton draws (drawn here, or copied from the shipped
    ``shipped/halton_seed{s}.npz``) and the evaluation quadrature."""
    out = Path(out_root) / ITEM / 'points'
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        if shipped:
            (out / f'halton_seed{seed}.npz').write_bytes((Path(shipped) / f'halton_seed{seed}.npz').read_bytes())
            continue
        x, y = pb.halton(HALTON_N, seed)
        np.savez(out / f'halton_seed{seed}.npz', x=x, y=y, scipy_version=scipy.__version__, n=HALTON_N, seed=seed)
    Xq, Yq, Wq = pb.fine_quadrature()
    np.savez(out / 'evaluation_quadrature_70x420.npz', x=Xq, y=Yq, w=Wq)
    return out


class Halton:
    """The shipped points: ``take(seed, n)`` is the first n points in Omega."""

    def __init__(self, points_dir):
        self.dir = Path(points_dir)
        self._cache = {}

    def take(self, seed, n):
        if seed not in self._cache:
            with np.load(self.dir / f'halton_seed{seed}.npz') as z:
                self._cache[seed] = (z['x'].copy(), z['y'].copy(), str(z['scipy_version']))
        x, y, _ = self._cache[seed]
        if n > len(x):
            raise ValueError(f'{n} points requested; {len(x)} shipped for seed {seed}')
        return x[:n], y[:n]

    def scipy_version(self, seed=0):
        self.take(seed, 1)
        return self._cache[seed][2]


def compare_points(points_dir, out_root):
    """Grace's scipy against the shipped points (record only; never used to build a set)."""
    shipped = Halton(points_dir)
    rows = []
    for seed in SEEDS:
        x, y = shipped.take(seed, HALTON_N)
        gx, gy = pb.halton(HALTON_N, seed)
        rows.append({'seed': seed, 'identical': bool(np.array_equal(x, gx) and np.array_equal(y, gy)),
                     'max_abs_difference': float(max(np.abs(x - gx).max(), np.abs(y - gy).max()))})
    out = {'scipy_here': scipy.__version__, 'scipy_shipped': shipped.scipy_version(), 'seeds': rows}
    path = Path(out_root) / ITEM / 'points' / 'halton_compare.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def lilq_set(halton, p, seed, ratio=pb.RATIO, lam=pb.LAM):
    n = int(round(ratio * p * p))
    return pb.lilq_system(p, seed, ratio=ratio, lam=lam, interior=halton.take(seed, n))


# ---------------------------------------------------------------- the logged LiL-Q runs

def _meta(S, k_max, lam, ratio, seed, scipy_version):
    thread_env = capture_blas_thread_env()
    return build_run_metadata(
        N_total=S.N, N_composition={'interior': S.n_interior, 'boundary': S.n_boundary}, P_total=S.P,
        P_composition={'u': S.P},
        row_weights=(f'interior: equal, squared weights summing to 1; boundary: lambda_d = {lam:g} times arc length '
                     'per point / total length (Section 4.1)'),
        collocation_construction={'interior': f'first {S.n_interior} scrambled Halton points in Omega '
                                              f'(scipy.stats.qmc.Halton, seed {seed}; drawn with scipy '
                                              f'{scipy_version} and shipped)',
                                  'boundary': f'{S.n_boundary} points on the four curves (boundary(12p))',
                                  'interior_points_per_unknown': ratio},
        basis_description={'family': 'T_i(x/1.32) T_j(y/1.32), i, j < p', 'modes_per_direction': S.p,
                           'ordering': 'x-major'},
        initial_coefficients='zero', solver_driver='gelsy', rcond=EPS_MACH,
        stopping_rule={'type': 'none: K_max iterations; the rule (tau_chi 0.1, tau_r 0.01, n_s 2) is read from the log'},
        K_max=k_max, stopping_reason='iteration_cap', first_stall_iteration=None, b2_check=None,
        kappa_qr_raw_ratio=None, device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1))


def logged(S, run_dir, norms, seed, scipy_version, k_max=K_MAX, lam=pb.LAM, ratio=pb.RATIO, item='2a'):
    t0 = time.perf_counter()
    p = S.p
    extra = lambda b: dict(zip(('eL2', 'eH2'), norms.errors(b, p)))  # noqa: E731
    _, k_s, _, _, tracker, logger, scaled = logged_run(S, np.zeros(S.P), run_dir,
                                                       lambda b: norms.errors(b, p)[0], k_max=k_max, extra_fn=extra)
    _, beta_ret = returned_iterate(run_dir)
    np.save(Path(run_dir) / 'beta_returned.npy', beta_ret)       # the rule's returned iterate (the last if it never fired)
    meta = _meta(S, k_max, lam, ratio, seed, scipy_version)
    meta.update(first_stall_iteration=first_stall_iteration(logger.rows), b2_check=tracker.b2_check)
    write_run_json(Path(run_dir) / 'run.json', {**meta, 'package4': {
        'item': item, 'p': p, 'seed': seed, 'lambda_boundary': lam, 'interior_points_per_unknown': ratio,
        'kappa2': pb.KAP2, 'rule_fires_at_step': k_s, 'identity_scaled_max': scaled,
        'quadrature': f'{norms.nr} x {norms.nth} polar Gauss', 'wall_time_s': time.perf_counter() - t0}})
    return run_dir


def save_set(S, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Xi, Yi, Wi = S.points['interior']
    Xb, Yb, Wb = S.points['boundary']
    np.savez(path, interior_x=Xi, interior_y=Yi, interior_w2=Wi, boundary_x=Xb, boundary_y=Yb, boundary_w2=Wb)


def run_lilq(out_root, halton, sizes=SIZES, seeds=SEEDS, item=ITEM):
    """``item``: the results folder (Package 5, item 4a: the same runs at larger p)."""
    norms = pb.Norms()
    root = Path(out_root) / item
    for p in sizes:
        for seed in seeds:
            S = lilq_set(halton, p, seed)
            save_set(S, root / 'points' / f'lilq_p{p}_seed{seed}.npz')
            d = logged(S, root / 'lilq' / f'p{p}_seed{seed}', norms, seed, halton.scipy_version())
            print('2a', d, flush=True)


def run_rows(out_root, halton):
    norms = pb.Norms()
    for p in RATIO_SIZES:
        for ratio in RATIOS:
            S = lilq_set(halton, p, 0, ratio=ratio)
            d = Path(out_root) / ITEM / 'rows_per_unknown' / f'p{p}_ratio{ratio:g}'
            logged(S, d, norms, 0, halton.scipy_version(), k_max=SWEEP_K_MAX, ratio=ratio, item='2d')
            print('2d', d, flush=True)


def run_weight(out_root, halton):
    norms = pb.Norms()
    for p in WEIGHT_SIZES:
        for lam in WEIGHTS:
            S = lilq_set(halton, p, 0, lam=lam)
            d = Path(out_root) / ITEM / 'boundary_weight' / f'p{p}_lam{lam:g}'
            logged(S, d, norms, 0, halton.scipy_version(), k_max=SWEEP_K_MAX, lam=lam, item='2e')
            print('2e', d, flush=True)


# ---------------------------------------------------------------- square and underdetermined collocation

def newton(S, solver):
    """Newton from zero coefficients (``newton_square`` of the reference script)."""
    import warnings
    b = np.zeros(S.P)
    status, it = 'budget', 0
    t0 = time.perf_counter()
    for k in range(NEWTON_K_MAX):
        A, f = S.linearized(b)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bn = (sla.solve(A, f) if solver == 'lu'
                      else sla.lstsq(A, f, lapack_driver=solver, cond=EPS_MACH)[0])
        except Exception:
            status = 'singular'
            break
        with np.errstate(over='ignore', invalid='ignore'):
            finite = (np.all(np.isfinite(bn)) and np.linalg.norm(bn) <= BETA_CAP
                      and np.all(np.isfinite(S.residual(bn))))
        if not finite:
            status = 'diverged'
            break
        step = np.linalg.norm(bn - b) / np.linalg.norm(bn)
        b, it = bn, k + 1
        if step < NEWTON_TOL:
            status = 'converged'
            break
    return b, it, status, time.perf_counter() - t0


def run_square(out_root, halton, sizes=SQUARE_SIZES, seeds=SEEDS):
    norms = pb.Norms()
    rows = []
    for p in sizes:
        for kind in ('halton', 'fekete'):
            for seed in seeds:
                S = pb.square_system(p, seed, kind, halton_points=lambda n, s=seed: halton.take(s, n))
                b, it, status, t = newton(S, 'lu')
                e0, e2 = norms.errors(b, p) if np.all(np.isfinite(b)) else (float('nan'), float('nan'))
                rows.append({'p': p, 'P': p * p, 'seed': seed, 'points': 'scattered' if kind == 'halton' else 'fekete',
                             'N': S.N, 'status': status, 'steps': it, 'eL2': e0, 'eH2': e2, 'time_s': t})
                print('2b', rows[-1], flush=True)
    out = Path(out_root) / ITEM / 'square'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'square.csv', rows)


def run_under(out_root, halton, sizes=SQUARE_SIZES, seeds=SEEDS, item=ITEM):
    norms = pb.Norms()
    rows = []
    for p in sizes:
        for seed in seeds:
            S = pb.under_system(p, seed, halton_points=lambda n, s=seed: halton.take(s, n))
            b, it, status, t = newton(S, 'gelsd')
            e0, e2 = norms.errors(b, p) if np.all(np.isfinite(b)) else (float('nan'), float('nan'))
            rows.append({'p': p, 'P': p * p, 'seed': seed, 'points': 'scattered', 'N': S.N, 'status': status,
                         'steps': it, 'eL2': e0, 'eH2': e2, 'time_s': t})
            print('2c', rows[-1], flush=True)
    out = Path(out_root) / item / 'underdetermined'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'underdetermined.csv', rows)


# ---------------------------------------------------------------- 2f: constants at the iterates

def y_system(S):
    """The Y-norm rows of Section 6: the 70 x 420 rule inside and boundary(4000), each block carrying its total
    squared weight in the collocation set."""
    Xq, Yq, Wq = pb.fine_quadrature()
    Xb, Yb, Wb = pb.boundary(Y_BOUNDARY_N)
    tot_i = float(np.sum(S.points['interior'][2]))
    tot_b = float(np.sum(S.points['boundary'][2]))
    return pb.System(S.p, Xq, Yq, tot_i * Wq, Xb, Yb, tot_b * Wb)


def constants_for(run_dir, S, out_dir):
    from experiments.p3_2_bl_certified import section6_row
    Sy = y_system(S)
    betas = np.load(Path(run_dir) / 'betas.npy')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_iter = []
    for k in range(len(betas) - 1):
        by_iter.append({'k': k, **section6_row(S, Sy, betas[k], betas[k + 1], ())})
        write_csv(out_dir / 'constants_by_iterate.csv', by_iter)
    write_csv(out_dir / 'rho_r.csv', [{key: x[key] for key in ('k', 'rho_r', 'rho_r_numerator', 'rho_r_denominator',
                                                                'kappa_A_Y', 'round_off')} for x in by_iter])
    return out_dir


def run_constants(out_root, halton, sizes=SIZES, seeds=SEEDS, extremes=True):
    root = Path(out_root) / ITEM
    for p in sizes:
        for seed in seeds:
            name = f'p{p}_seed{seed}'
            print('2f', constants_for(root / 'lilq' / name, lilq_set(halton, p, seed), root / 'constants' / name),
                  flush=True)
    if extremes:
        for ratio in (RATIOS[0], RATIOS[-1]):
            name = f'p22_ratio{ratio:g}'
            print('2f', constants_for(root / 'rows_per_unknown' / name, lilq_set(halton, 22, 0, ratio=ratio),
                                      root / 'constants' / name), flush=True)


# ---------------------------------------------------------------- timing, M7, the solution figure

def run_timing(out_root, halton, sizes=SIZES, seeds=SEEDS, item=ITEM):
    thread_env = capture_blas_thread_env()
    for p in sizes:
        for seed in seeds:
            interior = halton.take(seed, int(round(pb.RATIO * p * p)))

            def once():
                t0 = time.perf_counter()
                S = pb.lilq_system(p, seed, interior=interior)
                _, k_s, _, _ = iterate_to_rule(S, np.zeros(S.P), N_S)
                return time.perf_counter() - t0, None if k_s is None else k_s + 1
            out = timed(once)
            d = Path(out_root) / item / 'lilq' / f'p{p}_seed{seed}'
            d.mkdir(parents=True, exist_ok=True)
            (d / 'timing.json').write_text(json.dumps({**out, 'p': p, 'seed': seed, 'thread_env': thread_env},
                                                      indent=2))
            print(f"timing p={p} seed={seed}: median {out['median_s']:.4g} s, solves {out['solves']}", flush=True)


def returned_iterate(run_dir):
    """The rule's step and the returned coefficients of a logged run."""
    solve = [r for r in read_csv(Path(run_dir) / 'iterations.csv') if r['norm_Rlin_h'] not in ('', None)]
    k_s = stall_rule_index([to_float(r['chi']) for r in solve], [to_float(r['norm_Rlin_h']) for r in solve])
    betas = np.load(Path(run_dir) / 'betas.npy')
    k = (k_s + 1) if k_s is not None else len(betas) - 1
    return k_s, betas[k]


def run_m7(out_root):
    root = Path(out_root) / ITEM
    k_s, beta = returned_iterate(root / 'lilq' / 'p22_seed0')
    out = {}
    for name, rule in (('70x420', (70, 420)), ('140x840', M7_RULE)):
        n = pb.Norms(*rule)
        e0, e2 = n.errors(beta, 22)
        d0, d2 = n.best(22)
        out[name] = {'eL2': e0, 'eH2': e2, 'delta_L2': d0, 'delta_H2': d2}
    a, b = out['70x420'], out['140x840']
    out['relative_change'] = {k: rel(b[k], a[k]) for k in a}
    out['pass'] = bool(max(out['relative_change'].values()) < 0.02)
    (root / 'check_M7.json').write_text(json.dumps(out, indent=2))
    return out


def run_solution(out_root):
    root = Path(out_root) / ITEM
    k_s, beta = returned_iterate(root / 'lilq' / 'p22_seed0')
    g = np.linspace(-pb.BOX, pb.BOX, GRID_N)
    X, Y = np.meshgrid(g, g, indexing='ij')
    u = (pb.basis(X.ravel(), Y.ravel(), 22, ['v'])['v'] @ beta).reshape(X.shape)
    np.savez(root / 'solution_p22_seed0.npz', x=g, y=g, u=u, u_exact=pb.u_exact(X, Y), mask=pb.inside(X, Y),
             returned_iterate=k_s + 1)


# ---------------------------------------------------------------- the summary and the checks

def run_row(run_dir, best):
    """The terminal values of one logged run: at the rule's step, and the errors of the iterate it returns."""
    run = json.loads((Path(run_dir) / 'run.json').read_text())
    p4 = run['package4']
    out = {'p': p4['p'], 'P': run['P_total'], 'N': run['N_total'], 'seed': p4['seed'],
           'lambda_boundary': p4['lambda_boundary'], 'interior_points_per_unknown': p4['interior_points_per_unknown']}
    return {**out, **rule_row(run_dir, best)}


def rule_row(run_dir, best):
    """``run_row`` without the run's identity (Package 5, item 4b reads it for its random-feature runs)."""
    rows = read_csv(Path(run_dir) / 'iterations.csv')
    solve = [r for r in rows if r['norm_Rlin_h'] not in ('', None)]
    errs = read_csv(Path(run_dir) / 'errors_by_iterate.csv')
    k_s = stall_rule_index([to_float(r['chi']) for r in solve], [to_float(r['norm_Rlin_h']) for r in solve])
    out = {'rule_step': k_s, 'returned_iterate': None if k_s is None else k_s + 1,
           'delta_L2': best[0], 'delta_H2': best[1]}
    i = k_s if k_s is not None else len(solve) - 1          # 6.2: report the history as is
    r, e = solve[i], errs[i + 1]
    last = solve[-1]
    out.update({'class': stall_class(r) if k_s is not None else '', 'stall_level': to_float(r['norm_Rlin_h']),
                'chi_at_rule': to_float(r['chi']), 'norm_N_returned': to_float(rows[i + 1]['norm_R_h']),
                'eL2': to_float(e['eL2']), 'eH2': to_float(e['eH2']),
                'eL2_over_delta': to_float(e['eL2']) / best[0], 'eH2_over_delta': to_float(e['eH2']) / best[1],
                'rank_gelsy': r['num_rank_gelsy'], 'rank_svd': r['num_rank_svd'],
                'kappa_retained_at_rule': to_float(r['kappa_retained']), 'kappa_raw_at_rule': to_float(r['kappa_raw']),
                **kappa_returned(solve, i),
                'rank_gelsy_last': last['num_rank_gelsy'], 'rank_svd_last': last['num_rank_svd'],
                'kappa_retained_last': to_float(last['kappa_retained']), 'kappa_raw_last': to_float(last['kappa_raw'])})
    t = Path(run_dir) / 'timing.json'
    if t.exists():
        tj = json.loads(t.read_text())
        out.update({'median_time_s': tj['median_s'], 'timing_spread': tj['spread'], 'timed_solves': tj['solves'],
                    'timing_matches_logged': tj['solves'] == ([k_s + 1] if k_s is not None else [])})
    return out


def summarize(out_root, expected=None):
    root = Path(out_root) / ITEM
    norms = pb.Norms()
    best = {}

    def best_of(p):
        if p not in best:
            best[p] = norms.best(p)
        return best[p]

    checks = {}
    term = [run_row(d, best_of(int(d.name.split('_')[0][1:]))) for d in (root / 'lilq').glob('p*_seed*')
            if (d / 'iterations.csv').exists()]
    term.sort(key=lambda r: (r['p'], r['seed']))
    write_csv(root / 'lilq' / 'terminal.csv', term)
    sweeps = {}
    for folder, key, out_name in (('rows_per_unknown', 'interior_points_per_unknown', 'rows_per_unknown.csv'),
                                  ('boundary_weight', 'lambda_boundary', 'boundary_weight.csv')):
        rows = []
        for d in (root / folder).glob('p*') if (root / folder).exists() else []:
            r = run_row(d, best_of(int(d.name.split('_')[0][1:])))
            rows.append({'p': r['p'], 'P': r['P'], key: r[key], 'N': r['N'], 'rule_step': r['rule_step'],
                         'stall_level': r['stall_level'], 'eL2': r['eL2'], 'eH2': r['eH2'],
                         'eL2_over_delta': r['eL2_over_delta'], 'eH2_over_delta': r['eH2_over_delta'],
                         'class': r['class']})
        rows.sort(key=lambda r: (r['p'], float(r[key])))
        if rows:
            write_csv(root / out_name, rows)
        sweeps[folder] = rows
    # 2f
    if (root / 'constants').exists():
        rows_c = []
        for d in sorted((root / 'constants').iterdir()):
            f = d / 'constants_by_iterate.csv'
            if f.exists():
                c = read_csv(f)
                ok = [x for x in c if x['round_off'] == 'False']
                rows_c.append({'run': d.name, 'iterates': len(c), 'c1_min': min(to_float(x['c1']) for x in c),
                               'c2_max': max(to_float(x['c2']) for x in c),
                               'c2_over_c1_max': max(to_float(x['c2_over_c1']) for x in c),
                               'rho_r_max': max((to_float(x['rho_r']) for x in ok), default=float('nan')),
                               'round_off_iterates': len(c) - len(ok), 'dropped_max': max(int(x['dropped']) for x in c)})
        write_csv(root / 'constants' / 'constants.csv', rows_c)
    # M3
    m3 = []
    for folder in ('lilq', 'rows_per_unknown', 'boundary_weight'):
        for d in sorted((root / folder).glob('p*')) if (root / folder).exists() else []:
            if (d / 'run.json').exists():
                m3.append({'run': f'{folder}/{d.name}', **m3_summary(json.loads((d / 'run.json').read_text()))})
    checks['M3'] = {'rows': m3, 'pass': bool(m3) and all(x['pass'] for x in m3)}
    # M6
    m6 = {'area_exact': pb.AREA, 'area_by_rule_70x420': pb.area_by_rule(),
          'area_rel_difference': rel(pb.area_by_rule(), pb.AREA)}
    if expected:
        exp = json.loads(Path(expected).read_text())
        pts = exp['checks']['u_at']
        m6['u_rel_max'] = max(rel(float(pb.u_exact(x, y)), u) for x, y, u, _ in pts)
        m6['rho_rel_max'] = max(rel(float(pb.rho(x, y)), r) for x, y, _, r in pts)
        m6['pass'] = bool(m6['u_rel_max'] < 1e-12 and m6['rho_rel_max'] < 1e-12 and m6['area_rel_difference'] < 1e-3)
    else:                                   # u* and rho at the five points need expected_pb.json: no verdict here
        m6['area_pass'] = bool(m6['area_rel_difference'] < 1e-3)
        m6['pass'] = None
        m6['note'] = 'u* and rho at the five points of expected_pb.json are evaluated at the assembly'
    checks['M6'] = m6
    # M7
    if (root / 'check_M7.json').exists():
        checks['M7'] = json.loads((root / 'check_M7.json').read_text())
    # M8: 2a at p = 14 and 22, seed 0, against expected_pb.json
    if expected:
        exp = {(r['P'], r['seed']): r for r in json.loads(Path(expected).read_text())['lilq']}
        m8 = []
        for r in term:
            if r['seed'] == 0 and r['p'] in (14, 22) and (r['P'], 0) in exp:
                e = exp[(r['P'], 0)]
                m8.append({'p': r['p'], 'rule_step': r['rule_step'], 'rule_step_expected': e['rule'],
                           'stall_rel': rel(r['stall_level'], e['stall']), 'eL2_rel': rel(r['eL2'], e['eL2']),
                           'eH2_rel': rel(r['eH2'], e['eH2'])})
        checks['M8'] = {'rows': m8, 'scipy_of_points': Halton(root / 'points').scipy_version()
                        if (root / 'points' / 'halton_seed0.npz').exists() else None,
                        'pass': bool(m8) and all(x['rule_step'] == x['rule_step_expected'] and
                                                 max(x['stall_rel'], x['eL2_rel'], x['eH2_rel']) <= 0.01 for x in m8)}
    # M9 (item 2a)
    timed_rows = [r for r in term if 'timing_matches_logged' in r]
    checks['M9_item2'] = {'rows': [{k: r[k] for k in ('p', 'seed', 'timed_solves', 'timing_matches_logged',
                                                      'timing_spread', 'median_time_s')} for r in timed_rows],
                          'pass': bool(timed_rows) and all(r['timing_matches_logged'] and r['timing_spread'] < 0.15
                                                           for r in timed_rows)}
    (root / 'checks_item2.json').write_text(json.dumps(checks, indent=2, default=str))
    return term, checks


# ---------------------------------------------------------------- main

STAGES = ('points', 'compare-points', 'lilq', 'square', 'under', 'rows', 'weight', 'constants', 'time', 'm7',
          'solution', 'summarize')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 4, item 2: Poisson-Boltzmann around three inclusions.')
    ap.add_argument('stage', choices=STAGES)
    ap.add_argument('--out', required=True)
    ap.add_argument('--points', help='the folder of the shipped halton_seed*.npz')
    ap.add_argument('--p', type=int, nargs='+')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--expected', help="the advisor's expected_pb.json (checks M6, M8)")
    ap.add_argument('--no-extremes', action='store_true', help='constants: leave out the two extreme 2d sets')
    a = ap.parse_args(argv)
    if a.stage == 'points':
        print(draw_points(a.out, a.points))
        return
    if a.stage == 'summarize':
        _, checks = summarize(a.out, a.expected)
        print(json.dumps({k: v.get('pass') for k, v in checks.items()}, indent=1))
        return
    if a.stage in ('m7', 'solution'):
        print(run_m7(a.out) if a.stage == 'm7' else run_solution(a.out))
        return
    if not a.points:
        ap.error('--points is required for this stage')
    if a.stage == 'compare-points':
        print(json.dumps(compare_points(a.points, a.out), indent=1))
        return
    halton = Halton(a.points)
    stage = {'lilq': lambda: run_lilq(a.out, halton, a.p or SIZES, a.seeds),
             'square': lambda: run_square(a.out, halton, a.p or SQUARE_SIZES, a.seeds),
             'under': lambda: run_under(a.out, halton, a.p or SQUARE_SIZES, a.seeds),
             'rows': lambda: run_rows(a.out, halton),
             'weight': lambda: run_weight(a.out, halton),
             'constants': lambda: run_constants(a.out, halton, a.p or SIZES, a.seeds, extremes=not a.no_extremes),
             'time': lambda: run_timing(a.out, halton, a.p or SIZES, a.seeds)}
    stage[a.stage]()


if __name__ == '__main__':
    main()
