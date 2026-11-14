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
  - **The fallback** (the advisor's reply to the cost check, 14 November
    2026, answer 1): p_ref = 88 with 80; if, for a case, the p = 88 solve
    fails or does not stop by the rule, that case falls back to p = 64 with
    56. Each case is decided on its own (``choose`` stage), and
    ``bl_reference/chosen_<case>.json`` records the size and the reason (one file per case:
    the two cases' jobs run side by side). A job that
    stops a solve at its time limit leaves no ``run.json``: that counts as a
    failure. Every later stage reads the size of each case from that record.

Outputs under ``P5_1_weights/``: per run ``<family>/<run>/{run.json,
iterations.csv, betas.npy, errors_by_iterate.csv}``; the fine references in
``bl_reference/``; ``terminal.csv``, ``references.json``.

Usage::

    python experiments/p5_1_weights.py reference --case viscous --p-ref 88 --out <root> --reference-dir <dir>
    python experiments/p5_1_weights.py ref-ok --case viscous --p-ref 80 --out <root>      (exit 0 if it met the rule)
    python experiments/p5_1_weights.py choose --case viscous --out <root> --reference-dir <dir>
    python experiments/p5_1_weights.py run --family bratu --P 100 --lam 1e3 --out <root> --reference-dir <dir>
    python experiments/p5_1_weights.py all --out <root> --reference-dir <dir> [--p-ref 88]
    python experiments/p5_1_weights.py runs --only bratu --out <root> --reference-dir <dir>   (no reference, no summary)
    python experiments/p5_1_weights.py summarize --out <root> --reference-dir <dir> [--expected-bratu expected_weight_bratu.json --expected-bl expected_weight_bl.json]
    python experiments/p5_1_weights.py beltrami --out <root> --package3 <Package 3 root>   (Beltrami beside the delivered rows)
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
P_REF_FALLBACK = 64           # with 56, if a case's p = 88 solve fails or does not stop by the rule (the advisor's reply)
CHOSEN = 'chosen_{case}.json' # bl_reference/chosen_<case>.json: the size of each case's reference, and why
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
    np.save(run_dir / 'beta_reference.npy', beta)          # before run.json, which marks the reference finished
    write_run_json(run_dir / 'run.json', {
        'package5': {'item': '1', 'role': 'fine Buckley-Leverett reference for the norm of X', 'case': case,
                     'variant': 'sine', 'p': p_ref, 'P': P, 'N': grid['N_points'], 'lam': lam, 'ratio': RATIO,
                     'rule_step': k_s, 'returned_iterate': None if k_s is None else k_s + 1, 'stall_level': stall,
                     'chi_at_rule': chi, 'eps_ref_vs_fd': err(beta)['eps_ref'], 'identity_scaled_max': scaled,
                     'wall_time_s': time.perf_counter() - t0}})
    return run_dir


def reference_ok(case, p_ref, out_root):
    """None if the fine solution at p_ref finished and stopped by the rule; otherwise the reason."""
    d = reference_dir_bl(out_root) / f'{case}_p{p_ref}'
    if not d.exists():
        return f'p = {p_ref} was not run'
    if not (d / 'run.json').exists():
        return f'p = {p_ref} did not finish (stopped at its time limit, or failed: see the job log)'
    f = d / 'run.json'
    if json.loads(f.read_text())['package5']['rule_step'] is None:
        return f'p = {p_ref} did not stop by the rule within K_max = {K_MAX}'
    return None


def chosen_record(out_root):
    """Every case's ``choose`` record, keyed by case."""
    d = reference_dir_bl(out_root)
    return {case: json.loads((d / CHOSEN.format(case=case)).read_text())
            for case in sorted(p32.CASES) if (d / CHOSEN.format(case=case)).exists()}


def p_ref_of(out_root, case, p_ref=None, require_chosen=False):
    """The size of a case's fine reference: an explicit p_ref, else the ``choose`` record, else 88.
    A record with no usable size (the fallback failed too) is an error, with its reason; so is a
    missing record when ``require_chosen`` (the Grace jobs, where ``choose`` must have run)."""
    if p_ref is not None:
        return p_ref
    rec = chosen_record(out_root).get(case) if out_root is not None else None
    if rec is None:
        if require_chosen:
            raise RuntimeError(f'no fine reference was chosen for {case}: bl_reference/{CHOSEN.format(case=case)} '
                               'is missing (did the reference job stop before its choose step? see its log)')
        return P_REF_DEFAULT
    if not rec.get('p_ref'):
        raise RuntimeError(f'no usable fine reference for {case}: {rec["reason"]}')
    return rec['p_ref']


def choose(case, out_root, reference_dir, p_ref=P_REF_DEFAULT, fallback=P_REF_FALLBACK):
    """The advisor's fallback, for one case: keep p_ref (with p_ref - 8) if both stopped by the rule;
    otherwise run the fallback pair here and keep it if both stop by the rule. Writes chosen_<case>.json."""
    tried = {p: reference_ok(case, p, out_root) for p in (p_ref, p_ref - REF_NEIGHBOUR_STEP)}
    if not any(tried.values()):
        rec = {'p_ref': p_ref, 'p_neighbour': p_ref - REF_NEIGHBOUR_STEP, 'fallback': False,
               'reason': f'p = {p_ref} and {p_ref - REF_NEIGHBOUR_STEP} stopped by the rule'}
    else:
        for p in (fallback - REF_NEIGHBOUR_STEP, fallback):
            if reference_ok(case, p, out_root) is not None:
                bl_reference_run(case, p, out_root, reference_dir)
        fb = {p: reference_ok(case, p, out_root) for p in (fallback, fallback - REF_NEIGHBOUR_STEP)}
        ok = not any(fb.values())
        why = '; '.join(v for v in tried.values() if v)
        rec = {'p_ref': fallback if ok else None, 'p_neighbour': fallback - REF_NEIGHBOUR_STEP if ok else None,
               'fallback': True,
               'reason': why + (f'; fell back to p = {fallback} with {fallback - REF_NEIGHBOUR_STEP}' if ok else
                                '; the fallback failed too: ' + '; '.join(v for v in fb.values() if v))}
    rec['tried'] = {str(p): v or 'stopped by the rule' for p, v in tried.items()}
    d = reference_dir_bl(out_root)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (CHOSEN.format(case=case) + '.tmp')
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, d / CHOSEN.format(case=case))           # whole or not at all
    return rec


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

    def __init__(self, family, reference_dir, out_root=None, p_ref=None, require_chosen=False):
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
            p_ref = p_ref_of(out_root, case, p_ref, require_chosen)
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


def summarize(out_root, reference_dir, p_ref=None, only=None, expected_bratu=None, expected_bl=None):
    out = Path(out_root) / ITEM
    rows, records = [], {}
    for family in only or families():
        if not (out / family).exists():
            continue
        if family in BL_FAMILIES:
            case = bl_parts(family)[1]
            try:
                p_ref_of(out_root, case, p_ref)
            except RuntimeError as e:                       # no usable reference: say so, go on
                records[f'bl_{case}'] = {'case': case, 'error': str(e), 'chosen': chosen_record(out_root).get(case)}
                print(f'  {family}: left out of the summary: {e}', flush=True)
                continue
        ctx = Context(family, reference_dir, out_root, p_ref)
        gate = None
        if family in BL_FAMILIES:
            case = bl_parts(family)[1]
            if f'bl_{case}' not in records:
                records[f'bl_{case}'] = {**bl_reference_record(case, p_ref_of(out_root, case, p_ref), out_root),
                                         'chosen': chosen_record(out_root).get(case)}
            rec = records[f'bl_{case}']
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
    if expected_bratu or expected_bl:
        write_csv(out / 'compare_expected.csv', compare_expected(rows, expected_bratu, expected_bl))
    return rows, records


# ---------------------------------------------------------------- beside the advisor's values

def compare_expected(rows, bratu_path=None, bl_path=None):
    """Our terminal rows beside the advisor's indicative values (Section 3: his runs on Gauss-Legendre
    grids, "agreement to one figure is expected"): Bratu (``expected_weight_bratu.json``) and viscous
    Buckley-Leverett with the tensor Chebyshev space (``expected_weight_bl.json``), at the sizes and
    weights both ran. His ``solves`` is our returned iterate (the rule's step + 1). An H^{2,1} ratio
    the factor-3 rule leaves out stays empty, with ``x_ratio_reported`` False."""
    exp = []
    if bratu_path:
        exp += [('bratu', e['p'], e['lam'], e['rL2'], e['rH2'], e['solves'], e['dL2'], e['dH2'])
                for e in json.loads(Path(bratu_path).read_text())['bratu']]
    if bl_path:
        exp += [('bl_cheb_viscous', e['p'], e['lam'], e['eL2'] / e['dL2'], e['eH'] / e['dH'], e['solves'],
                 e['dL2'], e['dH']) for e in json.loads(Path(bl_path).read_text())]
    ours = {(r['family'], int(round(math.sqrt(int(r['P'])))), float(r['lam'])): r for r in rows}

    def num(v):                                            # a summary row's number, or None (empty, None, nan)
        x = to_float(None if v in ('None',) else v)
        return None if math.isnan(x) else x
    out, seen = [], set()
    for family, p, lam, rL2, rX, solves, dL2, dX in exp:
        r = ours.get((family, p, float(lam)))
        if r is None or (family, p, float(lam)) in seen:
            continue                                       # his p = 20 and weights we did not run; p^3 = 1e3 at p = 10 once
        seen.add((family, p, float(lam)))
        oL2, oX = num(r.get('eL2_over_delta')), num(r.get('eX_over_delta'))
        out.append({'family': family, 'p': p, 'P': p * p, 'lam': lam,
                    'solves': r.get('returned_iterate'), 'solves_expected': solves,
                    'eL2_over_delta': oL2, 'eL2_over_delta_expected': rL2,
                    'eL2_over_delta_ours_over_his': None if oL2 is None else oL2 / rL2,
                    'eX_over_delta': oX, 'eX_over_delta_expected': rX,
                    'eX_over_delta_ours_over_his': None if oX is None else oX / rX,
                    'x_ratio_reported': str(r.get('x_ratio_reported', True)) == 'True',
                    'delta_L2': to_float(r['delta_L2']), 'delta_L2_expected': dL2,
                    'delta_X': to_float(r['delta_X']), 'delta_X_expected': dX})
    return out


BELTRAMI = 'P3_1_beltrami_certified'
FIELDS = ('u', 'v', 'w', 'p_pin_gauge', 'p')


def beltrami_beside_delivered(p5_root, p3_root):
    """Section 3: Beltrami at B1 and B2, level 1, beside the delivered rows (Package 3,
    ``P3_1_beltrami_certified/terminal.csv``, lambda = 10, pin_scale 1). Per row: the rule's step
    and stall level; the errors of u, v, w and p in the pin gauge (Section 3) and gauge-corrected
    (the delivered ``error_over_delta_P_p``) at the returned iterate, as delivered (the 1e-9
    criterion's iterate, else the last), and their ratios to the DELIVERED delta_P of each field;
    the combined error at the rule's iterate. The stall level is the weighted ||R_lin||, so it
    scales with the weight. Writes ``P3_1_beltrami_certified/beside_delivered.csv``."""
    def rows_of(root):
        return read_csv(Path(root) / BELTRAMI / 'terminal.csv')
    delivered = {}
    for r in rows_of(p3_root):
        key = r['size']
        if r['level'] == '1' and float(r['pin_scale']) == 1.0 and key in ('B1', 'B2') and key not in delivered:
            log = read_csv(Path(p3_root) / BELTRAMI / f'{key}_L1' / 'iterations.csv')
            k = r['k_rule']
            delivered[key] = {**r, 'lambda_bc': 10.0, 'stall_level': log[int(k) - 1]['norm_Rlin_h'] if k != 'never' else ''}
    out = []
    for source, rows in (('delivered (Package 3)', [delivered[s] for s in ('B1', 'B2') if s in delivered]),
                         ('Package 5', [r for r in rows_of(p5_root) if r['level'] == '1'])):
        for r in rows:
            d = delivered[r['size']]
            dp = {f: to_float(d[f'delta_P_{"p" if f == "p_pin_gauge" else f}']) for f in FIELDS}
            row = {'size': r['size'], 'P': int(r['P']), 'lambda': float(r['lambda_bc']), 'source': source,
                   'k_rule': r['k_rule'], 'stall_level': to_float(r['stall_level']),
                   'returned_iterate': r['returned_iterate']}
            for f in FIELDS:
                e = to_float(r[f'rel_l2_{f}_at_returned'])
                row[f'error_{f}'] = e
                row[f'error_{f}_over_delivered_delta_P'] = e / dp[f]
            row['error_combined'] = to_float(r['rel_l2_combined_at_returned'])
            row['error_combined_over_delivered_delta_P'] = row['error_combined'] / to_float(d['delta_P_combined'])
            row['error_combined_at_rule'] = to_float(r['rel_l2_combined_at_rule'])
            row.update({f'delivered_delta_P_{f}': dp[f] for f in ('u', 'v', 'w', 'p')},
                       delivered_delta_P_combined=to_float(d['delta_P_combined']))
            out.append(row)
    out.sort(key=lambda x: (x['size'], x['lambda']))
    write_csv(Path(p5_root) / BELTRAMI / 'beside_delivered.csv', out)
    return out


def run_all(out_root, reference_dir, p_ref=None, only=None, references=True, summary=True):
    """Every run of the chosen families. ``references=False`` never runs a fine reference and
    needs each Buckley-Leverett case's ``choose`` record (the Grace jobs: a case without one
    fails with the reason); ``summary=False`` leaves the summary to a later stage, so that
    several processes can run families side by side."""
    chosen = only or families()
    for case in sorted({bl_parts(f)[1] for f in chosen if f in BL_FAMILIES} if references else ()):
        pr = p_ref_of(out_root, case, p_ref)
        for p in (pr - REF_NEIGHBOUR_STEP, pr):
            if not (reference_dir_bl(out_root) / f'{case}_p{p}' / 'run.json').exists():
                bl_reference_run(case, p, out_root, reference_dir)
    for family in chosen:
        ctx = Context(family, reference_dir, out_root, p_ref, require_chosen=not references)
        for P in sizes(family):
            for lam in WEIGHTS:
                run(family, P, lam, out_root, reference_dir, ctx)
                print(f'  {family} P = {P} lam = {lam:g}', flush=True)
    return summarize(out_root, reference_dir, p_ref, only) if summary else None


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 5, item 1: the norm of X and the boundary weight.')
    ap.add_argument('stage', choices=('all', 'runs', 'run', 'reference', 'ref-ok', 'choose', 'summarize', 'beltrami'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir')
    ap.add_argument('--family', choices=families())
    ap.add_argument('--P', type=int)
    ap.add_argument('--lam', type=float)
    ap.add_argument('--case', choices=tuple(p32.CASES))
    ap.add_argument('--p-ref', type=int, help="the fine reference's size (default: each case's chosen_<case>.json, else 88)")
    ap.add_argument('--only', nargs='*', choices=families())
    ap.add_argument('--expected-bratu', help='summarize: expected_weight_bratu.json')
    ap.add_argument('--expected-bl', help='summarize: expected_weight_bl.json')
    ap.add_argument('--package3', help="beltrami: Package 3's root (the delivered P3_1_beltrami_certified)")
    a = ap.parse_args(argv)
    if a.stage == 'beltrami':
        if not a.package3:
            ap.error('--package3 is required for beltrami')
        print(f'{len(beltrami_beside_delivered(a.out, a.package3))} rows')
        return 0
    if a.stage != 'ref-ok' and not a.reference_dir:
        ap.error('--reference-dir is required')
    if a.stage == 'ref-ok':
        why = reference_ok(a.case, a.p_ref or P_REF_DEFAULT, a.out)
        print(why or f'p = {a.p_ref or P_REF_DEFAULT} stopped by the rule')
        return 0 if why is None else 1
    if a.stage == 'choose':
        rec = choose(a.case, a.out, a.reference_dir, a.p_ref or P_REF_DEFAULT)
        print(json.dumps(rec))
        return 0 if rec['p_ref'] else 1
    if a.stage == 'reference':
        print(bl_reference_run(a.case, a.p_ref or P_REF_DEFAULT, a.out, a.reference_dir))
    elif a.stage == 'run':
        print(run(a.family, a.P, a.lam, a.out, a.reference_dir, Context(a.family, a.reference_dir, a.out, a.p_ref)))
    elif a.stage == 'runs':
        run_all(a.out, a.reference_dir, a.p_ref, a.only, references=False, summary=False)
    elif a.stage == 'all':
        rows, _ = run_all(a.out, a.reference_dir, a.p_ref, a.only)
        print(f'{len(rows)} runs')
    else:
        rows, rec = summarize(a.out, a.reference_dir, a.p_ref, a.only, a.expected_bratu, a.expected_bl)
        print(f'{len(rows)} runs; references {json.dumps(rec)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
