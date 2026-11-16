"""
Package 5, Addendum 2: the plateau run (runs for Section 5.3 of the manuscript)
==============================================================================

The advisor's Addendum 2 of 15 November 2026. Package 2's run
``P2_9_bases/bl_gravity_paper_P1024_ic`` shows the nonlinear residual on a
plateau (0.70 to 0.82 from k = 4 to 8) while the projection residual and the
error keep falling; the rule stops later, at k = 13. The addendum asks for the
same quantities on the manuscript's configuration, for the ingredient that
produces the plateau, and for that run again with every iterate stored.

**Section 3a** (``updates``; post-processing, no runs): from the stored
coefficients of delivered runs, the relative update of the function per
iterate,

    ||u^(k+1) - u^(k)|| / ||u^(k+1)||,  k = 0, 1, ...,

in L2 and in the norm of X, on each item's own quadrature (``lilq.xnorm.relative_updates``):
- the 96 runs of item 1: Bratu in H2, Burgers and Buckley-Leverett in H^{2,1},
  on the Gauss-Legendre rule of item 1 (``n_gl`` of each run). For the lifted
  sine space the lifting is part of u (it cancels in the difference);
- the 27 Poisson-Boltzmann runs with the Chebyshev space (Package 4 item 2a,
  18; Package 5 item 4a, 9) and the 45 random-feature runs of item 4b, in H2,
  on ``problems.poisson_boltzmann.Norms``'s fine quadrature (the one their
  errors used).

Each run's ``updates_by_iterate.csv`` has ``k`` (the new iterate, 1..K),
``update_L2`` and ``update_X``. The Package 5 runs get it in their own folders.
Package 4's 18 runs get it on this branch, under
``P5_A2_initial_profile/updates_package4/<their Package 4 path>/``, so that the
delivered ``package4-results`` branch is not changed.

The advisor's two values agree to every figure he gives: Poisson-Boltzmann
p = 22, seed 0, H2 1.0, 6.3e-2, 2.3e-5, 1.4e-6, then about 1e-6; viscous
Buckley-Leverett, tensor Chebyshev, P = 64, weight 10^4, H^{2,1} 0.99, 2.2,
0.38, 0.25, 0.15, 0.093.

**Sections 2 and 3** (``runs``): the 18 runs, gravity Buckley-Leverett, each
from the least-squares fit of the initial profile held constant in time, K_max
= 60 with the rule read from the log (item 1's convention: the rule's step k_s,
the returned iterate k_s + 1):
- A: item 1's configuration (CGL, Clenshaw-Curtis, N/P = 10, weight 10^4),
  tensor Chebyshev and lifted sine x Chebyshev, P = 256, 576, 1,024;
- B: as A at P = 576 and 1,024 with diffusion 0.05 and 0.02 (Package 2's
  finite-difference references at those diffusions);
- C: as A at P = 1,024 with weight 10;
- D: the paper's run (cosine x Fourier on the random tensor grid) at weights 10
  (the delivered run, repeated with every iterate stored) and 10^4.

Per run: ``iterations.csv``, ``betas.npy``, ``errors_by_iterate.csv`` (eL2
against the finite-difference reference; eX against item 1's p_ref reference
for diffusion 0.1) and ``updates_by_iterate.csv``. Over all runs:
``terminal.csv`` (the rule's step, the returned iterate's errors and their
ratios to delta_P, the longest stretch before the stop over which the nonlinear
residual changes by less than 5% per step, the earliest on a tie),
``stretches.csv`` (every stretch of two or more steps), ``trig_delta_P.json``
(the paper space's delta_P in L2 and H^{2,1} by gelsy and gelsd, with ranks)
and ``checks.json``: K1 (item 1's zero-start runs at P = 1,024, weight 10^4,
rerun: the same rule step, errors within 1e-8) and K2 (part D at weight 10
against the delivered log, every iterate, 1e-8).

Usage::

    python experiments/p5_a2_plateau.py updates --package5 <assembled package5 tree> --package4 <package4 tree>
    python experiments/p5_a2_plateau.py runs --out <out> --reference-dir <Package 2 reference/> \\
        --item1-root <package5 tree with P5_1_weights/bl_reference> --delivered <.../P2_9_bases/bl_gravity_paper_P1024_ic>
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from experiments import p5_1_weights as p51
from lilq.xnorm import relative_updates

ITEM = 'P5_A2_initial_profile'
UPDATE_COLUMNS = ('k', 'update_L2', 'update_X')
CASE = 'gravity'
TABLE3_SPACES = ('cheb', 'sine')
K_MAX = 60
STRETCH_TOL = 0.05             # Section 3: the nonlinear residual changes by less than 5% per step
REF_FACTOR = 3.0               # Section 5: a run's error is reported only above three times the reference's error
C1_TOL = 1e-8                  # item 1's check C1 (experiments/p5_checks.py), for K1 and K2
K2_COLUMNS = ('norm_R_h', 'norm_Rlin_h', 'eps_ref')


# ---------------------------------------------------------------- Section 3a: the updates of delivered runs

def write_updates(path, u2, uX):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(UPDATE_COLUMNS)
        for k, (a, b) in enumerate(zip(u2, uX), start=1):
            w.writerow([k, repr(float(a)), repr(float(b))])


class Item1Norms:
    """Item 1's quadrature and basis matrices per family and size, cached."""

    def __init__(self):
        self._cache = {}

    def get(self, family, P, n_gl):
        key = (family, P, n_gl)
        if key not in self._cache:
            x, y, w = p51.quadrature(family, n_gl)
            p = int(round(math.sqrt(P)))
            if family in p51.SCALAR_SIZES:
                mats, lift = p51.scalar_mats(family, p, x, y), None
            else:
                variant, case = p51.bl_parts(family)
                mats, lift = p51.bl_mats(p51.p32.TrialSpace(variant, p, p51.p32.CASES[case]['T']), x, y)
            comps = p51.H2 if family == 'bratu' else p51.H21
            self._cache.clear()                      # one size at a time: the matrices are large
            self._cache[key] = (w, mats, lift, comps)
        return self._cache[key]


def item1_updates(package5):
    """The 96 runs of item 1, in place. Returns the number written."""
    norms, n = Item1Norms(), 0
    root = Path(package5) / p51.ITEM
    for family in p51.families():
        for run_dir in sorted((root / family).glob('P*_lam*'), key=lambda d: int(d.name[1:].split('_')[0])):
            meta = json.loads((run_dir / 'run.json').read_text())['package5']
            w, mats, lift, comps = norms.get(family, meta['P'], meta['n_gl'])
            betas = np.load(run_dir / 'betas.npy')
            write_updates(run_dir / 'updates_by_iterate.csv', relative_updates(w, mats, betas, ('u',), lift),
                          relative_updates(w, mats, betas, comps, lift))
            n += 1
    return n


def pb_space(run_name):
    """The trial space of a Poisson-Boltzmann run from its folder name: ``p<p>_seed<s>`` (tensor
    Chebyshev) or ``P<P>_sigma<sigma>_draw<d>`` (random features)."""
    from problems import poisson_boltzmann as pb
    m = re.fullmatch(r'p(\d+)_seed\d+', run_name)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r'P(\d+)_sigma([0-9.]+)_draw(\d+)', run_name)
    if m:
        return pb.RandomFeatures(int(m.group(1)), float(m.group(2)), int(m.group(3)))
    raise ValueError(f'not a Poisson-Boltzmann run folder: {run_name}')


def pb_updates(run_dirs, out_of):
    """``out_of(run_dir) -> csv path``. H2 on the fine quadrature of ``Norms``."""
    from problems import poisson_boltzmann as pb
    norms = pb.Norms()
    W = norms.sw ** 2
    n = 0
    for run_dir in run_dirs:
        mats = pb.space_basis(norms.X, norms.Y, pb_space(run_dir.name))
        betas = np.load(run_dir / 'betas.npy')
        write_updates(out_of(run_dir), relative_updates(W, mats, betas, ('v',)), relative_updates(W, mats, betas, pb.KEYS))
        n += 1
    return n


def _runs(folder, pattern):
    return sorted(d for d in Path(folder).glob(pattern) if (d / 'betas.npy').exists())


def all_updates(package5, package4):
    """Section 3a: 96 + 27 + 45 runs. Returns the counts."""
    p5, p4 = Path(package5), Path(package4)
    counts = {'item1': item1_updates(p5)}
    p4_runs = _runs(p4 / 'P4_2_pb_inclusions' / 'lilq', 'p*_seed*')
    counts['package4_2a'] = pb_updates(
        p4_runs, lambda d: p5 / ITEM / 'updates_package4' / 'P4_2_pb_inclusions' / 'lilq' / d.name / 'updates_by_iterate.csv')
    counts['item4a'] = pb_updates(_runs(p5 / 'P5_4_pb' / 'lilq', 'p*_seed*'), lambda d: d / 'updates_by_iterate.csv')
    counts['item4b'] = pb_updates(_runs(p5 / 'P5_4_pb' / 'random_features', 'P*_sigma*_draw*'),
                                  lambda d: d / 'updates_by_iterate.csv')
    return counts


# ---------------------------------------------------------------- Sections 2 and 3: the runs

def runs():
    """``(part, space, P, nu, lam)`` of the 18 runs of Section 2."""
    out = [('A', v, P, 0.1, 1e4) for v in TABLE3_SPACES for P in (256, 576, 1024)]
    out += [('B', v, P, nu, 1e4) for v in TABLE3_SPACES for P in (576, 1024) for nu in (0.05, 0.02)]
    out += [('C', v, 1024, 0.1, 10.0) for v in TABLE3_SPACES]
    out += [('D', 'trig', 1024, 0.1, 10.0), ('D', 'trig', 1024, 0.1, 1e4)]
    return out


def run_name(part, space, P, nu, lam):
    return f'{part}_{space}_P{P}_nu{nu:g}_lam{lam:g}'


def run_dir_of(out_root, part, space, P, nu, lam):
    return Path(out_root) / ITEM / run_name(part, space, P, nu, lam)


def reference_path(reference_dir, nu):
    """The finite-difference reference of the gravity case at diffusion nu (Package 3 for 0.1,
    Package 2's nu-refinement for 0.05 and 0.02)."""
    return Path(reference_dir) / f'bl_fd_ref_gravity_nu{nu:g}.npz'


def load_reference(reference_dir, nu):
    """``((x, t), u, reference_error)``: the reference error is the relative difference of its last
    refinement step (the accuracy his one-third rule compares with)."""
    from lilq.references import load_reference as lr
    axes, u, meta = lr(reference_path(reference_dir, nu))
    return axes, u, float(meta['steps'][-1]['rel_diff_to_previous'])


class Item1:
    """What item 1 supplies for diffusion 0.1: the norm of X against its p_ref reference, delta_P,
    and the gate of its H^{2,1} ratios (the neighbour difference); one Context per trial space."""

    def __init__(self, reference_dir, item1_root):
        self.reference_dir, self.root = reference_dir, item1_root
        self.p_ref = p51.p_ref_of(item1_root, CASE, require_chosen=True)
        self.n_gl = max(p51.N_X, 2 * self.p_ref)
        self.gate = p51.bl_reference_record(CASE, self.p_ref, item1_root)['h21_diff_neighbour']
        self._ctx = {}

    def context(self, family):
        if family not in self._ctx:
            self._ctx[family] = p51.Context(family, self.reference_dir, self.root, require_chosen=True)
        return self._ctx[family]


def table3_run(part, variant, P, nu, lam, out_root, reference_dir, item1, k_max=K_MAX):
    """Parts A to C: item 1's configuration (CGL with Clenshaw-Curtis weights, N/P = 10, the Package 3
    split of the weight ``lam``), the gravity case at diffusion ``nu``, from the least-squares fit of the
    initial profile held constant in time; K_max iterations, the rule read from the log."""
    from experiments.p4_common import logged_run
    from lilq.instrumentation import EPS_MACH
    from lilq.provenance import capture_blas_thread_env
    from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
    space, physics, grid, rows = p51.bl_setup(variant, CASE, P, lam, nu)
    beta0 = p51.p32.initial_guess(space, physics, 'ic')
    (x, t), u, ref_err = load_reference(reference_dir, nu)
    l2 = p51.p32.error_fn(space, ((x, t), u))
    family = f'bl_{variant}_{CASE}'
    if nu == 0.1:
        ctx = item1.context(family)
        mats, lift = ctx.mats(P)
        extra = lambda b: {'eL2': l2(b)['eps_ref'], 'eX': ctx.xnorm.error_of(mats, b, lift)}  # noqa: E731
    else:
        extra = lambda b: {'eL2': l2(b)['eps_ref']}  # noqa: E731
    run_dir = run_dir_of(out_root, part, variant, P, nu, lam)
    _, k_s, _, _, tracker, logger, scaled = logged_run(rows, beta0, run_dir, lambda b: l2(b)['eps_ref'],
                                                       k_max=k_max, extra_fn=extra)
    thread_env = capture_blas_thread_env()
    meta = build_run_metadata(
        N_total=grid['N_points'], N_composition={n: len(b['points']) for n, b in grid['blocks'].items()},
        P_total=P, P_composition={'u': P},
        row_weights=f'Clenshaw-Curtis; boundary and initial lines lambda = {lam:g} x 1D weights x |e|/|dOmega|',
        collocation_construction={'method': 'CGL tensor grid with Clenshaw-Curtis weights (P2-16 / P3-2)',
                                  'ratio_target': p51.RATIO},
        basis_description={'family': 'tensor Chebyshev' if variant == 'cheb' else
                           'lifted plain sine in x x Chebyshev in t: S = (1 - x) + sum beta sin(i pi x) T_j',
                           'modes_per_direction': int(round(math.sqrt(P)))},
        initial_coefficients='ic: least-squares fit of the initial saturation profile held constant in time',
        solver_driver='gelsy', rcond=EPS_MACH,
        stopping_rule={'type': 'none: K_max iterations; the termination rule is read from the log'},
        K_max=k_max, stopping_reason='iteration_cap', first_stall_iteration=first_stall_iteration(logger.rows),
        b2_check=tracker.b2_check, kappa_qr_raw_ratio=None, device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1))
    write_run_json(run_dir / 'run.json', {**meta, 'package5': {
        'item': 'addendum 2', 'part': part, 'space': variant, 'P': P, 'nu': nu, 'lam': lam, 'case': CASE,
        'rule_fires_at_step': k_s, 'identity_scaled_max': scaled, 'n_gl': item1.n_gl,
        'reference': reference_path(reference_dir, nu).name, 'reference_error': ref_err}})
    return run_dir


def trig_run(lam, out_root, reference_dir, item1, k_max=K_MAX):
    """Part D: the paper's run of ``P2_9_bases/bl_gravity_paper_P1024_ic`` (cosine x Fourier on the random
    tensor grid, seed 42, the initial profile fitted and held in time) with the weight ``lam`` on the initial
    and boundary rows (lambda_ic = lambda_bc; 10 is the delivered run), every iterate stored."""
    import dataclasses
    from unittest import mock
    from experiments.run_bl import paper_setup
    from lilq.iteration_log import IterationLogger
    from problems import buckley_leverett as blmod
    config, opt = paper_setup(32, True)
    config = dataclasses.replace(config, initial_guess='ic')
    opt = dataclasses.replace(opt, R_tol=0.0, max_quasi_iters_lil=k_max, lambda_ic=lam, lambda_bc=lam)
    run_dir = run_dir_of(out_root, 'D', 'trig', 32 * 32, 0.1, lam)
    run_dir.mkdir(parents=True, exist_ok=True)
    betas, real = [], blmod.pretrain_lil

    def pretrain(*args, **kwargs):
        coef, loss = real(*args, **kwargs)
        betas.append(np.array(coef, dtype=float))               # iterate 0
        return coef, loss
    logger = IterationLogger()
    with mock.patch.object(blmod, 'pretrain_lil', pretrain):
        basis, _, _, _ = blmod.run_lil_q(
            config, opt, verbose=False, iteration_logger=logger, run_json_path=run_dir / 'run.json',
            diagnostics_callback=lambda A, beta_new, k: betas.append(np.array(beta_new, dtype=float)),
            reference_npz=reference_path(reference_dir, 0.1))
    logger.to_csv(run_dir / 'iterations.csv')
    betas = np.array(betas)
    np.save(run_dir / 'betas.npy', betas)
    from lilq.references import make_eps_ref_fn
    _, _, ref_err = load_reference(reference_dir, 0.1)
    eps = make_eps_ref_fn(basis, reference_path(reference_dir, 0.1))
    ctx = item1.context(f'bl_sine_{CASE}')                       # the quadrature and the p_ref reference of item 1
    mats = trig_mats(basis, ctx.x, ctx.y)
    rows = [{'k': k, 'eL2': eps(b)['eps_ref'], 'eX': ctx.xnorm.error_of(mats, b)} for k, b in enumerate(betas)]
    # the stored coefficients are the logged iterates: same count, and the same L2 error as the log's eps_ref
    log = read_rows(run_dir / 'iterations.csv')
    if len(log) != len(betas):
        raise RuntimeError(f'{len(betas)} coefficient vectors for {len(log)} logged iterates')
    worst = max(rel(r['eL2'], float(lr['eps_ref'])) for r, lr in zip(rows, log))
    if worst > C1_TOL:
        raise RuntimeError(f'stored iterates disagree with the log: eps_ref rel diff {worst:.2e}')
    write_rows(run_dir / 'errors_by_iterate.csv', rows)
    rec = json.loads((run_dir / 'run.json').read_text())
    rec['package5'] = {'item': 'addendum 2', 'part': 'D', 'space': 'trig', 'P': 1024, 'nu': 0.1, 'lam': lam,
                       'case': CASE, 'n_gl': item1.n_gl, 'reference': reference_path(reference_dir, 0.1).name,
                       'reference_error': ref_err, 'lambda_ic': lam, 'lambda_bc': lam,
                       'delivered_run': 'P2_9_bases/bl_gravity_paper_P1024_ic' if lam == 10.0 else None}
    (run_dir / 'run.json').write_text(json.dumps(rec, indent=2, default=str))
    return run_dir, basis


def trig_mats(basis, x, t):
    return {'u': basis.derivative(x, t, 0, 0), 'u_x': basis.derivative(x, t, 1, 0),
            'u_xx': basis.derivative(x, t, 2, 0), 'u_t': basis.derivative(x, t, 0, 1)}


def trig_deltas(basis, reference_dir, item1):
    """delta_P of the trigonometric space in L2 (on the reference's grid) and in H^{2,1} (item 1's
    quadrature and p_ref reference), each by gelsy and by gelsd at rcond = eps, with the rank kept."""
    import scipy.linalg as sla
    (x, t), u, _ = load_reference(reference_dir, 0.1)
    X, T = np.meshgrid(x, t, indexing='ij')
    Phi, f = basis.evaluate(X.ravel(), T.ravel()), u.ravel()
    ctx = item1.context(f'bl_sine_{CASE}')
    mats = trig_mats(basis, ctx.x, ctx.y)
    xn = ctx.xnorm
    A = np.vstack([mats[c] * xn.sw[:, None] for c in xn.components])
    b = np.concatenate([xn.ref[c] * xn.sw for c in xn.components])
    out = {'P': int(Phi.shape[1])}
    eps = np.finfo(float).eps
    for driver in ('gelsy', 'gelsd'):
        c2, _, r2, _ = sla.lstsq(Phi, f, lapack_driver=driver, cond=eps, check_finite=False)
        cX, _, rX, _ = sla.lstsq(A, b, lapack_driver=driver, cond=eps, check_finite=False)
        out[driver] = {'delta_L2': float(np.linalg.norm(Phi @ c2 - f) / np.linalg.norm(f)), 'rank_L2': int(r2),
                       'delta_X': float(np.linalg.norm(A @ cX - b)) / xn.den, 'rank_X': int(rX)}
    out['rank_deficient'] = any(out[d][k] < out['P'] for d in ('gelsy', 'gelsd') for k in ('rank_L2', 'rank_X'))
    return out


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(v):
    return float(v) if v not in ('', None) else None


def stretches(R, last):
    """The stretches of consecutive steps k -> k + 1, k + 1 <= ``last``, over which the nonlinear residual
    ``R`` (by iterate) changes by less than STRETCH_TOL of R_k per step: ``[(first, last)]`` iterates."""
    out, start = [], None
    for k in range(last):
        ok = R[k] is not None and R[k + 1] is not None and abs(R[k + 1] - R[k]) < STRETCH_TOL * abs(R[k])
        if ok and start is None:
            start = k
        if not ok and start is not None:
            out.append((start, k))
            start = None
    if start is not None:
        out.append((start, last))
    return out


def summary_row(run_dir, deltas, gate=None):
    """Section 3's per-run quantities. ``deltas``: {'delta_L2', 'delta_X' (or None)}."""
    from lilq.instrumentation import stall_rule_index
    meta = json.loads((run_dir / 'run.json').read_text())['package5']
    log = read_rows(run_dir / 'iterations.csv')
    ex = read_rows(run_dir / 'errors_by_iterate.csv')
    solve = [r for r in log if r['norm_Rlin_h'] not in ('', None)]
    k_s = stall_rule_index([float(r['chi']) for r in solve], [float(r['norm_Rlin_h']) for r in solve])
    last = len(log) - 1 if k_s is None else k_s + 1                  # the returned iterate, or the last logged
    stop = len(log) - 1 if k_s is None else k_s                      # "before the stop": up to the rule's step
    R = [_f(r['norm_R_h']) for r in log]
    Rlin = [_f(r['norm_Rlin_h']) for r in log]
    eL2 = [_f(r['eL2']) for r in ex]
    eX = [_f(r.get('eX')) for r in ex]
    ref_err = meta['reference_error']
    e_ret = eL2[last]
    reported = e_ret > REF_FACTOR * ref_err                          # his rule: reference error below a third
    dL2, dX = deltas['delta_L2'], deltas.get('delta_X')
    row = {'run': run_dir.name, 'part': meta['part'], 'space': meta['space'], 'P': meta['P'], 'nu': meta['nu'],
           'lam': meta['lam'], 'rule_step': k_s, 'returned_iterate': None if k_s is None else k_s + 1,
           'stops_by_rule': k_s is not None, 'reference_error': ref_err, 'error_reported': reported,
           'eL2': e_ret if reported else None, 'delta_L2': dL2,
           'eL2_over_delta': e_ret / dL2 if reported else None}
    if dX is not None and eX[last] is not None:
        gated = gate is None or dX > p51.GATE_FACTOR * gate
        row.update(eX=eX[last], delta_X=dX, eX_over_delta=eX[last] / dX if gated else None, x_ratio_reported=gated)
    found = stretches(R, stop)
    row['n_stretches'] = len(found)
    if found:
        a, b = max(found, key=lambda s: (s[1] - s[0], -s[0]))       # the longest; the earliest on a tie
        row.update(stretch_steps=b - a, stretch_first=a, stretch_last=b,
                   n_longest=sum(1 for s in found if s[1] - s[0] == b - a),
                   R_first=R[a], R_last=R[b], Rlin_first=Rlin[a], Rlin_last=Rlin[b],
                   eL2_first=eL2[a] if reported else None, eL2_last=eL2[b] if reported else None)
    else:
        row['stretch_steps'] = 0
    return row, [{'run': run_dir.name, 'first': a, 'last': b, 'steps': b - a, 'R_first': R[a], 'R_last': R[b],
                  'Rlin_first': Rlin[a], 'Rlin_last': Rlin[b], 'eL2_first': eL2[a] if reported else None,
                  'eL2_last': eL2[b] if reported else None}
                 for a, b in found if b - a >= 2]


def addendum_updates(run_dir, mats, lift, w):
    betas = np.load(run_dir / 'betas.npy')
    write_updates(run_dir / 'updates_by_iterate.csv', relative_updates(w, mats, betas, ('u',), lift),
                  relative_updates(w, mats, betas, p51.H21, lift))


# ---------------------------------------------------------------- checks

def k1(out_root, reference_dir, item1, item1_root):
    """K1: item 1's zero-start runs at P = 1,024, weight 10^4, both trial spaces, rerun in this job: the same
    rule step, and eL2 and eX at iterates 0..returned within C1_TOL of the delivered runs."""
    from lilq.instrumentation import stall_rule_index
    rows = []
    for variant in TABLE3_SPACES:
        family = f'bl_{variant}_{CASE}'
        new = p51.run(family, 1024, 1e4, Path(out_root) / ITEM / 'K1', reference_dir, ctx=item1.context(family))
        old = p51.run_dir_of(item1_root, family, 1024, 1e4)
        ks = []
        for d in (old, new):
            s = [r for r in read_rows(d / 'iterations.csv') if r['norm_Rlin_h'] not in ('', None)]
            ks.append(stall_rule_index([float(r['chi']) for r in s], [float(r['norm_Rlin_h']) for r in s]))
        last = (ks[0] + 1) if ks[0] is not None else None
        eo, en = read_rows(old / 'errors_by_iterate.csv'), read_rows(new / 'errors_by_iterate.csv')
        lo, ln = read_rows(old / 'iterations.csv'), read_rows(new / 'iterations.csv')
        n = (last + 1) if last is not None else min(len(lo), len(ln))
        diffs = [rel(float(ln[k]['eps_ref']), float(lo[k]['eps_ref'])) for k in range(n)]
        diffs += [rel(float(en[k]['eX']), float(eo[k]['eX'])) for k in range(n)]
        rows.append({'family': family, 'P': 1024, 'lam': 1e4, 'rule_step_delivered': ks[0], 'rule_step_rerun': ks[1],
                     'iterates_compared': n, 'max_rel_diff': max(diffs), 'tolerance': C1_TOL,
                     'pass': ks[0] == ks[1] and max(diffs) <= C1_TOL})
    return {'check': 'K1', 'rows': rows, 'pass': all(r['pass'] for r in rows)}


def k2(run_dir, delivered):
    """K2: the repeated run of part D at weight 10 against the delivered ``bl_gravity_paper_P1024_ic``:
    norm_R_h, norm_Rlin_h and eps_ref at every iterate, to item 1's C1 tolerance. Also the largest
    difference up to the delivered run's first stall (item 1's C1 compared that range)."""
    from lilq.instrumentation import stall_rule_index
    new, old = read_rows(Path(run_dir) / 'iterations.csv'), read_rows(Path(delivered) / 'iterations.csv')
    s = [r for r in old if r['norm_Rlin_h'] not in ('', None)]
    k_s = stall_rule_index([float(r['chi']) for r in s], [float(r['norm_Rlin_h']) for r in s])
    per = []
    for k in range(min(len(new), len(old))):
        d = [rel(_f(new[k][c]), _f(old[k][c])) for c in K2_COLUMNS
             if _f(old[k][c]) is not None and _f(new[k][c]) is not None]
        per.append(max(d))
    to_stall = max(per[:k_s + 2]) if k_s is not None else None
    return {'check': 'K2', 'columns': list(K2_COLUMNS), 'tolerance': C1_TOL, 'iterates': len(per),
            'rows_delivered': len(old), 'rows_rerun': len(new), 'max_rel_diff_every_iterate': max(per),
            'first_iterate_over_tolerance': next((k for k, v in enumerate(per) if v > C1_TOL), None),
            'max_rel_diff_to_first_stall': to_stall, 'delivered_first_stall': k_s,
            'pass': len(new) == len(old) and max(per) <= C1_TOL}


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


# ---------------------------------------------------------------- the whole addendum

def run_some(out_root, reference_dir, item1_root, names=None, k_max=K_MAX):
    """Run the named runs (all 18 by default) and write each one's updates. The thread count is the
    job's (``OMP_NUM_THREADS``): item 1's 4 for parts A to C, the delivered run's 24 for part D."""
    item1 = Item1(reference_dir, item1_root)
    done = []
    for part, space, P, nu, lam in runs():
        name = run_name(part, space, P, nu, lam)
        if names and name not in names:
            continue
        if space == 'trig':
            run_dir, basis = trig_run(lam, out_root, reference_dir, item1, k_max)
            x, t, w = p51.quadrature(f'bl_sine_{CASE}', item1.n_gl)
            addendum_updates(run_dir, trig_mats(basis, x, t), None, w)
        else:
            run_dir = table3_run(part, space, P, nu, lam, out_root, reference_dir, item1, k_max)
            w, mats, lift, _ = Item1Norms().get(f'bl_{space}_{CASE}', P, item1.n_gl)
            addendum_updates(run_dir, mats, lift, w)
        done.append(name)
        print(f'{name}: done', flush=True)
    return done


def paper_basis():
    """The paper's trial space of part D (cosine x Fourier, 32 x 32), as ``run_lil_q`` builds it."""
    from experiments.run_bl import paper_setup
    from lilq.basis import create_basis_2d
    config, _ = paper_setup(32, True)
    return create_basis_2d(config.basis_type, config.N_x, config.N_t, config.x_domain, (0, config.T_final))


def summarize(out_root, reference_dir, item1_root, delivered):
    """terminal.csv, stretches.csv, trig_delta_P.json, K2 (into checks.json, beside K1 if it ran)."""
    from experiments.p2_9_bases import delta_P
    item1 = Item1(reference_dir, item1_root)
    out = Path(out_root) / ITEM
    trig = trig_deltas(paper_basis(), reference_dir, item1)
    terminal, all_stretches = [], []
    for part, space, P, nu, lam in runs():
        run_dir = run_dir_of(out_root, part, space, P, nu, lam)
        if space == 'trig':
            deltas = {'delta_L2': trig['gelsy']['delta_L2'], 'delta_X': trig['gelsy']['delta_X']}
        elif nu == 0.1:
            dL2, dX = item1.context(f'bl_{space}_{CASE}').deltas(P)
            deltas = {'delta_L2': dL2, 'delta_X': dX}
        else:
            (x, t), u, _ = load_reference(reference_dir, nu)
            sp = p51.p32.TrialSpace(space, int(round(math.sqrt(P))), p51.p32.CASES[CASE]['T'])
            deltas = {'delta_L2': delta_P(sp.evaluate, x, t, u, lifted=sp.lifted), 'delta_X': None}
        row, found = summary_row(run_dir, deltas, item1.gate if deltas['delta_X'] is not None else None)
        terminal.append(row)
        all_stretches += found
        print(f"{run_dir.name}: rule step {row['rule_step']}, eL2 {row['eL2']}, delta_L2 {row['delta_L2']:.3g}, "
              f"longest stretch {row['stretch_steps']} steps", flush=True)
    write_rows(out / 'terminal.csv', terminal)
    write_rows(out / 'stretches.csv', all_stretches)
    (out / 'trig_delta_P.json').write_text(json.dumps(trig, indent=2))
    (out / 'item1_references.json').write_text(json.dumps(
        {'p_ref': item1.p_ref, 'n_gl': item1.n_gl, 'h21_diff_neighbour': item1.gate}, indent=2))
    checks = out / 'checks.json'
    rec = json.loads(checks.read_text()) if checks.exists() else {}
    rec['K2'] = k2(run_dir_of(out_root, 'D', 'trig', 1024, 0.1, 10.0), delivered)
    checks.write_text(json.dumps(rec, indent=2, default=str))
    return terminal, rec


def k1_check(out_root, reference_dir, item1_root):
    item1 = Item1(reference_dir, item1_root)
    out = Path(out_root) / ITEM
    out.mkdir(parents=True, exist_ok=True)
    checks = out / 'checks.json'
    rec = json.loads(checks.read_text()) if checks.exists() else {}
    rec['K1'] = k1(out_root, reference_dir, item1, item1_root)
    checks.write_text(json.dumps(rec, indent=2, default=str))
    return rec['K1']


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('updates')
    u.add_argument('--package5', required=True)
    u.add_argument('--package4', required=True)
    for name in ('run', 'k1', 'summarize'):
        s = sub.add_parser(name)
        s.add_argument('--out', required=True)
        s.add_argument('--reference-dir', required=True)
        s.add_argument('--item1-root', required=True, help="item 1's tree (P5_1_weights with bl_reference)")
        if name == 'run':
            s.add_argument('--only', nargs='*', help='run names (default: all 18)')
            s.add_argument('--part', nargs='*', choices=list('ABCD'), help='only these parts')
            s.add_argument('--k-max', type=int, default=K_MAX)
        if name == 'summarize':
            s.add_argument('--delivered', required=True, help='P2_9_bases/bl_gravity_paper_P1024_ic of Package 2')
    args = ap.parse_args(argv)
    if args.cmd == 'updates':
        print(json.dumps(all_updates(args.package5, args.package4)), flush=True)
    elif args.cmd == 'run':
        names = set(args.only or [])
        if args.part:
            names |= {run_name(*r) for r in runs() if r[0] in args.part}
        print(json.dumps(run_some(args.out, args.reference_dir, args.item1_root, names or None, args.k_max)), flush=True)
    elif args.cmd == 'k1':
        print(json.dumps({'K1': k1_check(args.out, args.reference_dir, args.item1_root)['pass']}), flush=True)
    else:
        _, rec = summarize(args.out, args.reference_dir, args.item1_root, args.delivered)
        print(json.dumps({k: v['pass'] for k, v in rec.items()}), flush=True)


if __name__ == '__main__':
    main()
