"""
Package 5, item 1: errors in the norm of X, and the boundary weight
===================================================================

The advisor's instructions of 13 November 2026, Section 3 (scalar problems;
Beltrami is ``p3_1_beltrami_certified.py --lambda-bc``). The Package 2 and 3
configurations on the CGL sets with Clenshaw-Curtis weights at N/P = 10,
unchanged except for the weight ``lam`` of the boundary and initial rows:

- **Bratu** (P2-16): P = 25, 100, 225;
- **Burgers** (P2-16, tensor Chebyshev): P = 25 to 625;
- **Buckley-Leverett** (P3-2): the viscous and gravity cases, the tensor
  Chebyshev space (``cheb``) and the lifted sine x Chebyshev space
  (``sine``), P = 64 to 1,024, from the Package 3 initial guesses.

``lam`` in {10, 10^3, 10^4, 10^6}; K_max = 60 with the termination rule read
from the log (Package 4's ``logged_run``); the coefficients of every iterate
in ``betas.npy``. Untimed (DECISIONS.md, the Package 5 plan, reading 1).

**Errors.**
- **L2:** as delivered, ``eps_ref`` on the paper's 201 x 201 grids, and
  delta_L2 as delivered (P2-16's ``delta_P``; for Buckley-Leverett
  ``p2_9_bases.delta_P``, the lifting subtracted).
- **The norm of X**, on a tensor Gauss-Legendre rule (``lilq.xnorm``):
  - Bratu, H2 (u, u_x, u_y, u_xx, u_xy, u_yy): the p = 48 spectral reference
    of ``bratu_ref_p48.npz``, differentiated from its Chebyshev coefficients;
  - Burgers, H^{2,1} (u, u_x, u_xx, u_t): ``lilq.references.cole_hopf_derivatives``,
    with the check that u_t + u u_x - nu u_xx = 0 at the nodes;
  - Buckley-Leverett, H^{2,1}: a fine LiL-Q solution of the same case
    (lifted sine x Chebyshev, p_ref, ``lam`` = 10^4; ``reference`` stage).
    The record gives its L2 difference from the finite-difference reference
    and its H^{2,1} difference from the solution at the neighbouring size
    ``p_ref - 8``; an H^{2,1} ratio is reported only where delta_X exceeds
    that difference three times (Section 3).

Outputs under ``P5_1_weights/``: per run ``<family>/<run>/{run.json,
iterations.csv, betas.npy, errors_by_iterate.csv}``; the fine references in
``bl_reference/``; ``terminal.csv``, ``references.json``.

Usage::

    python experiments/p5_1_weights.py reference --case viscous --p-ref 88 --out <root> --reference-dir <dir>
    python experiments/p5_1_weights.py run --family bratu --P 100 --lam 1e3 --out <root> --reference-dir <dir>
    python experiments/p5_1_weights.py all --out <root> --reference-dir <dir> [--p-ref 88]
    python experiments/p5_1_weights.py summarize --out <root> --reference-dir <dir>
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from experiments import p2_16_certified as p216
from experiments import p3_2_bl_certified as p32
from experiments.p4_common import logged_run, read_csv, to_float, write_csv
from lilq import certified as cf
from lilq.instrumentation import EPS_MACH, stall_rule_index
from lilq.provenance import capture_blas_thread_env
from lilq.references import cole_hopf_derivatives
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
from lilq.xnorm import XNorm, gauss_legendre_2d

ITEM = 'P5_1_weights'
WEIGHTS = (10.0, 1e3, 1e4, 1e6)
RATIO = 10
K_MAX = 60
N_X = 96                       # Gauss-Legendre nodes per direction for the norm of X (at least)
LAM_REFERENCE = 1e4            # the fine Buckley-Leverett reference's weight (Section 3)
P_REF_DEFAULT = 88            # the fine reference (P = 7,744); its neighbour p = 80 (DECISIONS.md, Package 5 item 1)
REF_NEIGHBOUR_STEP = 8
GATE_FACTOR = 3.0
H2 = ('u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy')
H21 = ('u', 'u_x', 'u_xx', 'u_t')
SCALAR_SIZES = {'bratu': p216.SIZES['bratu'], 'burgers': p216.SIZES['burgers']}
BL_FAMILIES = tuple(f'bl_{v}_{c}' for v in ('cheb', 'sine') for c in ('viscous', 'gravity'))
BL_SIZES = p32.SIZES
PDE_CHECK_TOL = 1e-8


def lam_tag(lam):
    return f'lam{lam:g}'


def families():
    return ('bratu', 'burgers') + BL_FAMILIES


def sizes(family):
    return SCALAR_SIZES[family] if family in SCALAR_SIZES else tuple(p * p for p in BL_SIZES)


def bl_parts(family):
    _, variant, case = family.split('_')
    return variant, case


# ---------------------------------------------------------------- the systems

class ScalarRows:
    """P2-16's rows in the interface of ``logged_run``: ``assemble`` -> (A, f), ``residual``."""

    def __init__(self, bench, P, lam):
        self.bench, self.p = bench, int(round(math.sqrt(P)))
        self.grid = p216.grid(bench, P, RATIO, lam_aux=lam)
        self.rows = p216.Rows(bench, self.p, self.grid['blocks'])

    def assemble(self, beta):
        A, f, _ = p216.assemble(self.rows, beta)
        return A, f

    def residual(self, beta):
        return p216.assemble(self.rows, beta)[2]


def bl_setup(variant, case, P, lam):
    p, T = int(round(math.sqrt(P))), p32.CASES[case]['T']
    _, physics = p32.physics_for(case)
    space = p32.TrialSpace(variant, p, T)
    grid = cf.bl_grid(P, RATIO, T)
    rows = p32.Rows(space, physics, p32.collocation_blocks(grid, lam_aux=lam))
    return space, physics, grid, rows


# ---------------------------------------------------------------- the norm of X: quadratures and references

def quadrature(family, n=N_X):
    if family == 'bratu':
        return gauss_legendre_2d(n, ((0.0, 1.0), (0.0, 1.0)))
    if family == 'burgers':
        return gauss_legendre_2d(n, ((-1.0, 1.0), (0.0, 1.0)))
    _, case = bl_parts(family)
    return gauss_legendre_2d(n, ((0.0, 1.0), (0.0, p32.CASES[case]['T'])))


def scalar_mats(bench, p, x, y):
    """The basis's components at the quadrature, keyed as the norm's components."""
    T = p216.tensor(x, y, p, p216.DOMAIN[bench])
    if bench == 'bratu':
        return {'u': T[(0, 0)], 'u_x': T[(1, 0)], 'u_y': T[(0, 1)], 'u_xx': T[(2, 0)], 'u_xy': T[(1, 1)],
                'u_yy': T[(0, 2)]}
    return {'u': T[(0, 0)], 'u_x': T[(1, 0)], 'u_xx': T[(2, 0)], 'u_t': T[(0, 1)]}


def bl_mats(space, x, t):
    m = space.mats(x, t)
    mats = {'u': m['val'], 'u_x': m['dx'], 'u_xx': m['dxx'], 'u_t': m['dt']}
    lift = {'u': space.lift(x), 'u_x': space.lift(x, 1), 'u_xx': space.lift(x, 2), 'u_t': np.zeros_like(x)}
    return mats, (lift if space.lifted else None)


def bratu_reference_x(reference_dir, x, y):
    """The p = 48 reference's H2 components, from its Chebyshev coefficients."""
    with np.load(Path(reference_dir) / 'bratu_ref_p48.npz') as z:
        coef, p = z['coefficients'], int(z['p'])
    mats = scalar_mats('bratu', p, x, y)
    return {c: mats[c] @ coef.ravel() for c in H2}


def burgers_reference_x(x, t):
    """The Cole-Hopf solution's H^{2,1} components at the nodes (x, t), t > 0, and the PDE check."""
    out = {c: np.empty_like(x) for c in H21}
    worst, scale = 0.0, 0.0
    for tj in np.unique(t):
        m = t == tj
        d = cole_hopf_derivatives(x[m], tj, p216.NU_BURGERS)
        for c in H21:
            out[c][m] = d[c]
        res = d['u_t'] + d['u'] * d['u_x'] - p216.NU_BURGERS * d['u_xx']
        worst = max(worst, float(np.max(np.abs(res))))
        scale = max(scale, float(np.max(np.abs(d['u_t']) + np.abs(d['u'] * d['u_x']) +
                                        np.abs(p216.NU_BURGERS * d['u_xx']))))
    check = {'max_abs_pde_residual': worst, 'max_term_scale': scale, 'relative': worst / scale,
             'tol': PDE_CHECK_TOL, 'passed': worst / scale <= PDE_CHECK_TOL}
    return out, check


def reference_dir_bl(out_root):
    return Path(out_root) / ITEM / 'bl_reference'


def bl_reference_run(case, p_ref, out_root, reference_dir, lam=LAM_REFERENCE):
    """The fine Buckley-Leverett solution: the lifted sine x Chebyshev space at p_ref, weight ``lam``,
    the Package 3 guess, the rule's returned iterate. Logged, coefficients of every iterate stored."""
    P = p_ref * p_ref
    space, physics, grid, rows = bl_setup('sine', case, P, lam)
    run_dir = reference_dir_bl(out_root) / f'{case}_p{p_ref}'
    err = p32.error_fn(space, p32.load_reference(reference_dir, case))
    t0 = time.perf_counter()
    beta, k_s, stall, chi, tracker, logger, scaled = logged_run(
        rows, p32.initial_guess(space, physics, p32.CASES[case]['guess']), run_dir,
        lambda b: err(b)['eps_ref'], k_max=K_MAX, stop_n_s=2)
    write_run_json(run_dir / 'run.json', {
        'package5': {'item': '1', 'role': 'fine Buckley-Leverett reference for the norm of X', 'case': case,
                     'variant': 'sine', 'p': p_ref, 'P': P, 'N': grid['N_points'], 'lam': lam, 'ratio': RATIO,
                     'rule_step': k_s, 'returned_iterate': None if k_s is None else k_s + 1, 'stall_level': stall,
                     'chi_at_rule': chi, 'eps_ref_vs_fd': err(beta)['eps_ref'], 'identity_scaled_max': scaled,
                     'wall_time_s': time.perf_counter() - t0}})
    np.save(run_dir / 'beta_reference.npy', beta)
    return run_dir


def load_bl_reference(case, p_ref, out_root):
    run_dir = reference_dir_bl(out_root) / f'{case}_p{p_ref}'
    meta = json.loads((run_dir / 'run.json').read_text())['package5']
    if meta['rule_step'] is None:
        raise RuntimeError(f'the fine reference {run_dir.name} never met the rule within K_max')
    return np.load(run_dir / 'beta_reference.npy'), meta


def bl_reference_x(case, p_ref, out_root, x, t):
    beta, _ = load_bl_reference(case, p_ref, out_root)
    space = p32.TrialSpace('sine', p_ref, p32.CASES[case]['T'])
    mats, lift = bl_mats(space, x, t)
    return {c: mats[c] @ beta + lift[c] for c in H21}


def bl_reference_record(case, p_ref, out_root, n=None):
    """The fine reference's two accuracy figures (Section 3): its L2 difference from the
    finite-difference reference, and its H^{2,1} difference from the solution at p_ref - 8."""
    family = f'bl_sine_{case}'
    n = n or max(N_X, 2 * p_ref)
    x, t, w = quadrature(family, n)
    xn = XNorm(w, bl_reference_x(case, p_ref, out_root, x, t), H21)
    p_nb = p_ref - REF_NEIGHBOUR_STEP
    _, meta = load_bl_reference(case, p_ref, out_root)
    _, meta_nb = load_bl_reference(case, p_nb, out_root)
    return {'case': case, 'p_ref': p_ref, 'p_neighbour': p_nb, 'n_gl': n,
            'l2_vs_fd': meta['eps_ref_vs_fd'], 'l2_vs_fd_neighbour': meta_nb['eps_ref_vs_fd'],
            'h21_diff_neighbour': xn.error(bl_reference_x(case, p_nb, out_root, x, t)),
            'rule_step': meta['rule_step'], 'rule_step_neighbour': meta_nb['rule_step']}


# ---------------------------------------------------------------- one run

class Context:
    """What every run of a family shares: the norm of X on its quadrature."""

    def __init__(self, family, reference_dir, out_root=None, p_ref=P_REF_DEFAULT):
        self.family = family
        if family in SCALAR_SIZES:
            x, y, w = quadrature(family)
            self.x, self.y = x, y
            if family == 'bratu':
                ref, self.pde_check = bratu_reference_x(reference_dir, x, y), None
            else:
                ref, self.pde_check = burgers_reference_x(x, y)
            self.xnorm = XNorm(w, ref, H2 if family == 'bratu' else H21)
            self.l2_ref = p216.load_reference(family, reference_dir)
        else:
            variant, case = bl_parts(family)
            n = max(N_X, 2 * p_ref)
            x, t, w = quadrature(family, n)
            self.x, self.y, self.p_ref, self.pde_check = x, t, p_ref, None
            self.xnorm = XNorm(w, bl_reference_x(case, p_ref, out_root, x, t), H21)
            self.l2_ref = p32.load_reference(reference_dir, case)

    def mats(self, P):
        p = int(round(math.sqrt(P)))
        if self.family in SCALAR_SIZES:
            return scalar_mats(self.family, p, self.x, self.y), None
        variant, case = bl_parts(self.family)
        return bl_mats(p32.TrialSpace(variant, p, p32.CASES[case]['T']), self.x, self.y)

    def deltas(self, P, driver='gelsy'):
        """delta_L2 as delivered and delta_X."""
        p = int(round(math.sqrt(P)))
        mats, lift = self.mats(P)
        dX = self.xnorm.best(mats, lift, driver)
        if self.family in SCALAR_SIZES:
            return p216.delta_P(self.family, p, self.l2_ref), dX
        from experiments.p2_9_bases import delta_P
        variant, case = bl_parts(self.family)
        space = p32.TrialSpace(variant, p, p32.CASES[case]['T'])
        (x, t), u = self.l2_ref
        return delta_P(space.evaluate, x, t, u, lifted=space.lifted), dX


def run_dir_of(out_root, family, P, lam):
    return Path(out_root) / ITEM / family / f'P{P}_{lam_tag(lam)}'


def run(family, P, lam, out_root, reference_dir, ctx=None, k_max=K_MAX):
    ctx = ctx or Context(family, reference_dir, out_root)
    p = int(round(math.sqrt(P)))
    mats, lift = ctx.mats(P)
    if family in SCALAR_SIZES:
        rows = ScalarRows(family, P, lam)
        beta0 = np.zeros(P)
        l2 = p216.error_fn(family, p, ctx.l2_ref)
        N, comp = rows.grid['N'], {n: len(b[0]) for n, b in rows.grid['blocks'].items()}
        guess = 'zero'
        basis = {'family': 'tensor Chebyshev', 'modes_per_direction': p}
    else:
        variant, case = bl_parts(family)
        space, physics, grid, rows = bl_setup(variant, case, P, lam)
        guess = p32.CASES[case]['guess']
        beta0 = p32.initial_guess(space, physics, guess)
        e = p32.error_fn(space, ctx.l2_ref)
        l2 = lambda b: e(b)['eps_ref']  # noqa: E731
        N, comp = grid['N_points'], {n: len(b['points']) for n, b in grid['blocks'].items()}
        basis = {'family': 'tensor Chebyshev' if variant == 'cheb' else
                 'lifted plain sine in x x Chebyshev in t: S = (1 - x) + sum beta sin(i pi x) T_j',
                 'modes_per_direction': p}
    run_dir = run_dir_of(out_root, family, P, lam)
    t0 = time.perf_counter()
    _, k_s, stall, chi, tracker, logger, scaled = logged_run(
        rows, beta0, run_dir, l2, k_max=k_max, extra_fn=lambda b: {'eX': ctx.xnorm.error_of(mats, b, lift)})
    thread_env = capture_blas_thread_env()
    meta = build_run_metadata(
        N_total=N, N_composition=comp, P_total=P, P_composition={'u': P},
        row_weights=f'Clenshaw-Curtis; boundary and initial lines lambda = {lam:g} x 1D weights x |e|/|dOmega|',
        collocation_construction={'method': 'CGL tensor grid with Clenshaw-Curtis weights (P2-16 / P3-2)',
                                  'ratio_target': RATIO},
        basis_description=basis, initial_coefficients=guess, solver_driver='gelsy', rcond=EPS_MACH,
        stopping_rule={'type': 'none: K_max iterations; the termination rule is read from the log'},
        K_max=k_max, stopping_reason='iteration_cap', first_stall_iteration=first_stall_iteration(logger.rows),
        b2_check=tracker.b2_check, kappa_qr_raw_ratio=None, device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1))
    write_run_json(run_dir / 'run.json', {**meta, 'package5': {
        'item': '1', 'family': family, 'P': P, 'lam': lam, 'rule_fires_at_step': k_s,
        'identity_scaled_max': scaled, 'norm_X': list(ctx.xnorm.components), 'n_gl': int(round(len(ctx.x) ** 0.5)),
        'wall_time_s': time.perf_counter() - t0}})
    return run_dir


# ---------------------------------------------------------------- the summary

def terminal_row(run_dir, dL2, dX, gate=None):
    log = read_csv(run_dir / 'iterations.csv')
    solve = [r for r in log if r['norm_Rlin_h'] not in ('', None)]
    ex = read_csv(run_dir / 'errors_by_iterate.csv')
    meta = json.loads((run_dir / 'run.json').read_text())['package5']
    k_s = stall_rule_index([to_float(r['chi']) for r in solve], [to_float(r['norm_Rlin_h']) for r in solve])
    row = {'family': meta['family'], 'P': meta['P'], 'lam': meta['lam'], 'rule_step': k_s,
           'returned_iterate': None if k_s is None else k_s + 1, 'delta_L2': dL2, 'delta_X': dX}
    if k_s is None:
        return row
    eL2, eX = to_float(log[k_s + 1]['eps_ref']), to_float(ex[k_s + 1]['eX'])
    reported = gate is None or dX > GATE_FACTOR * gate
    row.update({'stall_level': to_float(solve[k_s]['norm_Rlin_h']), 'rank_gelsy': solve[k_s]['num_rank_gelsy'],
                'eL2': eL2, 'eL2_over_delta': eL2 / dL2, 'eX': eX,
                'eX_over_delta': eX / dX if reported else None, 'x_ratio_reported': reported,
                'eL2_last': to_float(log[-1]['eps_ref']), 'eX_last': to_float(ex[-1]['eX']),
                'identity_scaled_max': meta['identity_scaled_max']})
    return row


def summarize(out_root, reference_dir, p_ref=P_REF_DEFAULT, only=None):
    out = Path(out_root) / ITEM
    rows, records = [], {}
    for family in only or families():
        if not (out / family).exists():
            continue
        ctx = Context(family, reference_dir, out_root, p_ref)
        gate = None
        if family in BL_FAMILIES:
            case = bl_parts(family)[1]
            rec = records.setdefault(f'bl_{case}', bl_reference_record(case, p_ref, out_root))
            gate = rec['h21_diff_neighbour']
        if ctx.pde_check is not None:
            records['burgers_pde_check'] = ctx.pde_check
        for P in sizes(family):
            dL2, dX = ctx.deltas(P)
            for lam in WEIGHTS:
                d = run_dir_of(out_root, family, P, lam)
                if (d / 'run.json').exists():
                    rows.append(terminal_row(d, dL2, dX, gate))
    write_csv(out / 'terminal.csv', rows)
    (out / 'references.json').write_text(json.dumps(records, indent=2))
    return rows, records


def run_all(out_root, reference_dir, p_ref=P_REF_DEFAULT, only=None):
    chosen = only or families()
    for case in sorted({bl_parts(f)[1] for f in chosen if f in BL_FAMILIES}):
        for p in (p_ref - REF_NEIGHBOUR_STEP, p_ref):
            if not (reference_dir_bl(out_root) / f'{case}_p{p}' / 'run.json').exists():
                bl_reference_run(case, p, out_root, reference_dir)
    for family in chosen:
        ctx = Context(family, reference_dir, out_root, p_ref)
        for P in sizes(family):
            for lam in WEIGHTS:
                run(family, P, lam, out_root, reference_dir, ctx)
                print(f'  {family} P = {P} lam = {lam:g}', flush=True)
    return summarize(out_root, reference_dir, p_ref, only)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 5, item 1: the norm of X and the boundary weight.')
    ap.add_argument('stage', choices=('all', 'run', 'reference', 'summarize'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir', required=True)
    ap.add_argument('--family', choices=families())
    ap.add_argument('--P', type=int)
    ap.add_argument('--lam', type=float)
    ap.add_argument('--case', choices=tuple(p32.CASES))
    ap.add_argument('--p-ref', type=int, default=P_REF_DEFAULT)
    ap.add_argument('--only', nargs='*', choices=families())
    a = ap.parse_args(argv)
    if a.stage == 'reference':
        print(bl_reference_run(a.case, a.p_ref, a.out, a.reference_dir))
    elif a.stage == 'run':
        print(run(a.family, a.P, a.lam, a.out, a.reference_dir, Context(a.family, a.reference_dir, a.out, a.p_ref)))
    elif a.stage == 'all':
        rows, _ = run_all(a.out, a.reference_dir, a.p_ref, a.only)
        print(f'{len(rows)} runs')
    else:
        rows, rec = summarize(a.out, a.reference_dir, a.p_ref, a.only)
        print(f'{len(rows)} runs; references {json.dumps(rec)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
