"""
Package 4, item 1: Burgers on Gauss points, with enrichment at stall
===================================================================

The advisor's instructions of 12 November 2026, Section 3. The reference
implementation is ``ref_burgers_enrichment.py``; where the text and that
script differ, the script decides.

**Problem:** u_t + u u_x = nu u_xx on (-1, 1) x (0, 1], u(x, 0) = -sin(pi x),
u(+-1, t) = 0, nu = 0.1.

**Trial space:** sin(n pi (x + 1)/2), n = 1..p, times T_j(2t - 1),
j = 0..p - 1, i.e. the ``sin_cheb`` basis of
``run_burgers_basis_comparison.py`` (``Fourier1D(mode='sin')`` on (-1, 1) times
``Chebyshev1D`` on (0, 1)). P = p^2, coefficients ordered x-major
(beta.reshape(p, p)[n, j]). Every trial function meets the boundary
conditions, so there are no boundary rows.

**Collocation set** (Eq. (2) of the manuscript: each component has a normalized
measure and its own lambda):
- interior: the tensor product of 3p Gauss-Legendre points in x on (-1, 1)
  and 3p in t on (0, 1); squared row weights are the tensor Gauss weights,
  normalized to sum 1 (lambda = 1);
- initial condition: the same x points at t = 0, squared row weights
  lambda_0 = 10 times the x weights normalized to sum 1.
- N = 9p^2 + 3p. These differ from the block weights of Packages 2 and 3,
  which carried |block| / |dOmega|.

**Solve:** gelsy at rcond = eps_mach, from zero coefficients.

**1a** (``logged``): p = 6, 10, ..., 30. K_max = 60 with termination
disabled. Every iterate is logged (``iterations.csv`` with ``eps_ref``,
the error against the Cole-Hopf reference on the 201 x 201 grid; kappa and
the rank by SVD at every iterate) and every coefficient vector is saved
(``betas.npy``).

**1b** (``enrich``): start at p = 6 from zero and iterate until the rule
fires (tau_chi = 0.1, tau_r = 0.01). Stop if the stall level ||r(k)|| is at
most the tolerance; otherwise embed the returned coefficients in the space
with p + 4 (zero padding: the function is unchanged) and continue on the new
level's collocation set. The rule's comparison of ||r(j)|| with ||r(j - 1)||
restarts at every level.
- Variant E1: n_s = 2 at every level.
- Variant E2: n_s = 2 at the first level, n_s = 1 after an embedding.
- Tolerances 1e-3 and 1e-5.
- A run that reaches p = 34 without meeting its tolerance stops there.

**1c** (``constants``): for each p of 1a, at every iterate of the logged
pass, c1 and c2 on the residuals attainable there and the realized ratio
rho_r, by the recipe of Package 3, Section 6
(``experiments.p3_2_bl_certified.section6_row``). The norm of Y is evaluated
by Gauss-Legendre quadrature with twice the points per direction (6p), and
the initial block is its own block; the block-matched weights equal the
collocation set's (interior total 1, initial total lambda_0).

**Timing** (``time``; Section 2.3): an untimed warm-up, then three timings
with the diagnostics off, each the median of five runs if a run is shorter
than 1 s; the reported time is their median. The timed run stops at the
iterate the rule returns. Its timed region covers the construction of the
collocation matrices, the assemblies, the solves and the residual
assemblies the rule needs, at every level for an enrichment run (the
embedding included).

Usage::

    python experiments/p4_1_burgers_gauss.py logged    --out <root> --reference-dir <dir> [--p 6 10 ...]
    python experiments/p4_1_burgers_gauss.py enrich    --out <root> --reference-dir <dir>
    python experiments/p4_1_burgers_gauss.py constants --out <root> [--p ...]
    python experiments/p4_1_burgers_gauss.py time      --out <root> [--which 1a 1b]
    python experiments/p4_1_burgers_gauss.py summarize --out <root> --reference-dir <dir> [--expected expected_burgers.json]
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np
import scipy.linalg as sla

from lilq.basis import Chebyshev1D, Fourier1D
from lilq.instrumentation import EPS_MACH, phase_indicator, roundoff_comparison, stall_flag, stall_rule_index
from lilq.iteration_log import IterationLogger, LilQDiagnosticsTracker
from lilq.provenance import capture_blas_thread_env
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json

ITEMS = {'1a': 'P4_1a_burgers_gauss', '1b': 'P4_1b_burgers_enrichment', '1c': 'P4_1c_burgers_constants'}
NU = 0.1
LAM0 = 10.0
OVER = 3                      # Gauss points per direction: OVER * p
Y_OVER = 6                    # the Y-norm rule of 1c: twice the collocation set's
SIZES = (6, 10, 14, 18, 22, 26, 30)
K_MAX = 60
N_S = 2
P_START, P_STEP, P_LIMIT = 6, 4, 34
VARIANTS = {'E1': 2, 'E2': 1}      # n_s at the levels that start from an embedded iterate
TOLERANCES = (1e-3, 1e-5)
REPEATS = 3
SHORT_RUN_S, SHORT_REPEATS = 1.0, 5
WEIGHTS_NOTE = ("Eq. (2) of the manuscript: each component has normalized measure and its own lambda. Interior: "
                "tensor Gauss-Legendre weights normalized to sum 1, lambda = 1. Initial rows: lambda_0 = 10 times "
                "the x Gauss weights normalized to sum 1. Packages 2 and 3 instead carried the factor |block|/|dOmega|.")


# ---------------------------------------------------------------- the trial space and the point sets

class Space:
    """sin(n pi (x + 1)/2), n = 1..p, times T_j(2t - 1), j < p."""

    def __init__(self, p):
        self.p, self.P = p, p * p
        self.fx = Fourier1D(n_modes=p, domain=(-1.0, 1.0), mode='sin')
        self.ft = Chebyshev1D(order=p - 1, domain=(0.0, 1.0))

    def x_mats(self, x):
        return [self.fx.evaluate(x), self.fx.derivative(x, 1), self.fx.derivative(x, 2)]

    def t_mats(self, t):
        return [self.ft.evaluate(t), self.ft.derivative(t, 1)]

    def values(self, beta, x, t):
        return self.fx.evaluate(x) @ np.asarray(beta).reshape(self.p, self.p) @ self.ft.evaluate(t).T


def gauss(n, a, b):
    """n Gauss-Legendre points on (a, b) and their weights, normalized to sum 1."""
    g, w = np.polynomial.legendre.leggauss(n)
    return a + (b - a) * (g + 1) / 2, w / w.sum()


class Rows:
    """The weighted rows on the tensor Gauss set with ``over * p`` points per
    direction: the interior block (PDE) and the initial block (u = -sin(pi x)
    at t = 0). ``assemble`` gives the quasilinearized system at beta, and
    ``residual`` the weighted nonlinear residual; ``exclude`` is accepted for
    the Section 6 recipe and must be empty (there are no zero rows)."""

    def __init__(self, space, over=OVER):
        p = space.p
        self.space, self.over = space, over
        self.x, self.wx = gauss(over * p, -1.0, 1.0)
        self.t, self.wt = gauss(over * p, 0.0, 1.0)
        self.S = space.x_mats(self.x)
        self.T = space.t_mats(self.t)
        self.W = np.sqrt(np.outer(self.wx, self.wt)).ravel()
        self.w0 = np.sqrt(LAM0 * self.wx)
        self.nx, self.nt = len(self.x), len(self.t)
        self.N = self.nx * self.nt + self.nx
        S0, S1, S2 = self.S
        T0, T1 = self.T
        self.lin = np.einsum('an,bj->abnj', S0, T1) - NU * np.einsum('an,bj->abnj', S2, T0)
        self.b0 = np.einsum('an,bj->abnj', S0, T0)
        self.b1 = np.einsum('an,bj->abnj', S1, T0)
        T00 = space.t_mats(np.array([0.0]))[0][0]
        self.A_ic = (S0[:, :, None] * T00[None, None, :]).reshape(self.nx, -1) * self.w0[:, None]
        self.f_ic = -np.sin(np.pi * self.x) * self.w0

    def _fields(self, beta):
        B = np.asarray(beta).reshape(self.space.p, self.space.p)
        S0, S1, S2 = self.S
        T0, T1 = self.T
        return S0 @ B @ T0.T, S1 @ B @ T0.T, S0 @ B @ T1.T, S2 @ B @ T0.T

    def assemble(self, beta, exclude=()):
        assert not exclude
        U, Ux, _, _ = self._fields(beta)
        A = self.lin + U[:, :, None, None] * self.b1 + Ux[:, :, None, None] * self.b0
        A = A.reshape(-1, self.space.P) * self.W[:, None]
        return np.vstack([A, self.A_ic]), np.concatenate([(U * Ux).ravel() * self.W, self.f_ic])

    def residual(self, beta, exclude=()):
        assert not exclude
        U, Ux, Ut, Uxx = self._fields(beta)
        return np.concatenate([(Ut + U * Ux - NU * Uxx).ravel() * self.W, self.A_ic @ beta - self.f_ic])


def embed(beta, p, p_new):
    """The coefficients of the same function in the space with p_new > p."""
    B = np.zeros((p_new, p_new))
    B[:p, :p] = np.asarray(beta).reshape(p, p)
    return B.ravel()


# ---------------------------------------------------------------- the reference, errors and delta_P

def load_reference(reference_dir):
    from lilq.references import load_reference as load
    (x, t), u, meta = load(Path(reference_dir) / 'burgers_cole_hopf.npz')
    return x, t, u


class Errors:
    """The relative L2 error against the reference on its 201 x 201 grid and
    the best-approximation error delta_P there (by least squares, separable)."""

    def __init__(self, reference):
        self.x, self.t, self.u = reference
        self.norm = float(np.linalg.norm(self.u))
        self._cache = {}

    def _mats(self, space):
        if space.p not in self._cache:
            self._cache[space.p] = (space.fx.evaluate(self.x), space.ft.evaluate(self.t))
        return self._cache[space.p]

    def error(self, space, beta):
        Sx, Tt = self._mats(space)
        return float(np.linalg.norm(Sx @ np.asarray(beta).reshape(space.p, space.p) @ Tt.T - self.u) / self.norm)

    def delta_P(self, space):
        Sx, Tt = self._mats(space)
        Qx, _ = np.linalg.qr(Sx)
        Qt, _ = np.linalg.qr(Tt)
        proj = Qx @ (Qx.T @ self.u @ Qt) @ Qt.T
        return float(np.linalg.norm(self.u - proj) / self.norm)

    def values(self, space, beta):
        Sx, Tt = self._mats(space)
        return Sx @ np.asarray(beta).reshape(space.p, space.p) @ Tt.T


def colexact(x, t, n_nodes=120):
    """Cole-Hopf by Gauss-Hermite quadrature, as ``colexact`` of the reference
    script (vectorized); for check M2."""
    z, w = np.polynomial.hermite.hermgauss(n_nodes)
    X, T = np.meshgrid(x, t, indexing='ij')
    out = -np.sin(np.pi * X)
    pos = T > 0
    Xp, Tp = X[pos][:, None], T[pos][:, None]
    y = Xp - 2 * np.sqrt(NU * Tp) * z[None, :]
    f = np.exp(-np.cos(np.pi * y) / (2 * np.pi * NU))
    out[pos] = -np.sum(w * np.sin(np.pi * y) * f, axis=1) / np.sum(w * f, axis=1)
    return out


# ---------------------------------------------------------------- the iteration with the rule (no diagnostics)

def iterate_to_rule(rows, beta, n_s, k_max=K_MAX):
    """Algorithm 1 without the converged exit (the timed and the enrichment
    runs): solve until the rule fires; returns the returned iterate, the step
    at which it fired (or None), the stall level ||r|| there, and chi there."""
    r_prev, run, rn, chi = None, 0, None, None
    for k in range(k_max):
        A, f = rows.assemble(beta)
        new = sla.lstsq(A, f, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)[0]
        r = A @ new - f
        rn = float(np.linalg.norm(r))
        chi = phase_indicator(rows.residual(new), r)
        run = run + 1 if (r_prev is not None and stall_flag(chi, rn, r_prev)) else 0
        beta, r_prev = new, rn
        if run >= n_s:
            return beta, k, rn, chi
    return beta, None, rn, chi


def enrichment(tol, n_s_after, log_level=None):
    """Item 1b's procedure. ``log_level(level, p, rows, beta_start, n_s)`` runs
    a logged level instead of ``iterate_to_rule`` and returns the same tuple."""
    p, beta, first, levels = P_START, None, True, []
    while True:
        space = Space(p)
        rows = Rows(space)
        if beta is None:
            beta = np.zeros(space.P)
        n_s = N_S if first else n_s_after
        out = (log_level(len(levels), p, rows, beta, n_s) if log_level
               else iterate_to_rule(rows, beta, n_s))
        beta_ret, k_s, stall, chi = out
        levels.append({'p': p, 'P': space.P, 'N': rows.N, 'n_s': n_s, 'k_s': k_s,
                       'solves': None if k_s is None else k_s + 1, 'stall_level': stall, 'chi_at_stop': chi,
                       'beta': beta_ret})
        if k_s is None or stall <= tol or p >= P_LIMIT:
            return levels
        beta, p, first = embed(beta_ret, p, p + P_STEP), p + P_STEP, False


# ---------------------------------------------------------------- the logged passes

def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def _float(v):
    return float(v) if v not in ('', None) else float('nan')


def logged_level(rows, beta, run_dir, errors, k_max=K_MAX, stop_n_s=None):
    """One logged LiL-Q run on ``rows`` from ``beta``. With ``stop_n_s`` None
    termination is disabled (K_max iterations, 1a); otherwise the run stops
    when the rule fires with that n_s (an enrichment level, 1b). Writes
    ``iterations.csv`` and ``betas.npy`` (beta^(0..K)) after every iterate.
    Returns the iterate at the rule (or the last), the step, the stall level,
    chi there, the tracker and the logger."""
    space = rows.space
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = LilQDiagnosticsTracker(test_error_fn=lambda b: {'eps_ref': errors.error(space, b)},
                                     n_s=stop_n_s or N_S)
    logger = IterationLogger()
    betas = [np.asarray(beta, float).copy()]
    k_s = None
    tracker.identity_scaled = []
    for k in range(k_max):
        t0 = time.perf_counter()
        A, f = rows.assemble(beta)
        t1 = time.perf_counter()
        new, _, rank, _ = sla.lstsq(A, f, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)
        t2 = time.perf_counter()
        tracker.identity_scaled.append(identity_scaled(A, f, beta, rows.residual(beta)))
        R_new = rows.residual(new)
        row = tracker.step(k=k, A_stacked=A, b_stacked=f, beta_prev=beta, beta_new=new,
                           total_loss=float(R_new @ R_new), rank_gelsy=int(rank), t_assemble_s=t1 - t0,
                           t_solve_s=t2 - t1, is_final_iterate=False, compute_residual_vector_fn=rows.residual)
        logger.record(**row)
        logger.to_csv(run_dir / 'iterations.csv')
        betas.append(new.copy())
        np.save(run_dir / 'betas.npy', np.array(betas))
        beta = new
        if k_s is None and row['stall_rule_fires']:
            k_s = k
            if stop_n_s is not None:
                break
    logger.record(**tracker.finish(k=len(betas) - 1))
    logger.to_csv(run_dir / 'iterations.csv')
    np.save(run_dir / 'beta_last.npy', beta)
    rows_log = [r for r in logger.rows if r.get('norm_Rlin_h') is not None]
    i = k_s if k_s is not None else len(rows_log) - 1
    return beta, k_s, float(rows_log[i]['norm_Rlin_h']), float(rows_log[i]['chi']), tracker, logger


def identity_scaled(A, f, beta, N_direct):
    """Check M3 without cancellation: ||(A beta - f) - N(beta)|| over
    || |A||beta| + |f| ||, the scale at which the two vectors are formed
    (Section 4.1 of the manuscript). Its round-off bound is about P eps. The
    plain relative difference of ``b2_check`` divides by ||N||, which is
    small near the plateau, so it measures cancellation there."""
    scale = float(np.linalg.norm(np.abs(A) @ np.abs(beta) + np.abs(f)))
    return float(np.linalg.norm(A @ beta - f - N_direct)) / scale if scale > 0 else 0.0


def m3_summary(run_json):
    """M3 for one run: the K4 reading (k = 1) and the scaled maximum."""
    p4 = run_json['package4']
    P = run_json['P_total']
    return {'b2_at_k1': run_json['b2_check']['rel_err'], 'b2_max': run_json['b2_check']['max_rel_err_over_run'],
            'scaled_max': p4['identity_scaled_max'], 'bound_P_eps': P * EPS_MACH,
            'pass': p4['identity_scaled_max'] <= P * EPS_MACH}


def _metadata(rows, tracker, logger, k_max, stopping, reason, initial):
    thread_env = capture_blas_thread_env()
    space = rows.space
    return build_run_metadata(
        N_total=rows.N, N_composition={'interior': rows.nx * rows.nt, 'initial': rows.nx}, P_total=space.P,
        P_composition={'u': space.P}, row_weights=WEIGHTS_NOTE,
        collocation_construction={'method': 'tensor Gauss-Legendre', 'points_per_direction': rows.nx,
                                  'x_interval': [-1.0, 1.0], 't_interval': [0.0, 1.0], 'lambda_interior': 1.0,
                                  'lambda_initial': LAM0},
        basis_description={'family': 'sin(n pi (x + 1)/2), n = 1..p, times T_j(2t - 1), j < p (sin_cheb)',
                           'modes_per_direction': space.p, 'ordering': 'x-major: beta.reshape(p, p)[n, j]'},
        initial_coefficients=initial, solver_driver='gelsy', rcond=EPS_MACH, stopping_rule=stopping, K_max=k_max,
        stopping_reason=reason, first_stall_iteration=first_stall_iteration(logger.rows),
        b2_check=tracker.b2_check, kappa_qr_raw_ratio=None, device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1))


def logged(p, out_root, errors):
    """Item 1a's logged pass at p."""
    space = Space(p)
    rows = Rows(space)
    run_dir = Path(out_root) / ITEMS['1a'] / f'p{p}'
    t0 = time.perf_counter()
    _, k_s, _, _, tracker, logger = logged_level(rows, np.zeros(space.P), run_dir, errors)
    meta = _metadata(rows, tracker, logger, K_MAX,
                     {'type': 'none: K_max iterations; the rule (tau_chi 0.1, tau_r 0.01, n_s 2) is read from the log'},
                     'iteration_cap', 'zero')
    write_run_json(run_dir / 'run.json', {**meta, 'package4': {
        'item': '1a', 'p': p, 'nu': NU, 'lambda_0': LAM0, 'rule_fires_at_step': k_s,
        'delta_P': errors.delta_P(space), 'identity_scaled_max': max(tracker.identity_scaled),
        'wall_time_s': time.perf_counter() - t0}})
    return run_dir


def enrich_logged(variant, tol, out_root, errors):
    """Item 1b's logged pass: one ``iterations.csv`` per level and
    ``levels.csv``; check M4 at every embedding."""
    run_dir = Path(out_root) / ITEMS['1b'] / f'{variant}_tol{tol:g}'
    run_dir.mkdir(parents=True, exist_ok=True)
    m4 = []
    level_dirs = []

    def log_level(i, p, rows, beta_start, n_s):
        d = run_dir / f'level{i}_p{p}'
        level_dirs.append(d)
        if i > 0:                         # check M4: the embedded iterate is the returned one
            prev = levels_so_far[-1]
            before = errors.values(Space(prev['p']), prev['beta'])
            after = errors.values(rows.space, beta_start)
            m4.append({'level': i, 'p_from': prev['p'], 'p_to': p,
                       'max_abs_difference': float(np.abs(after - before).max())})
        beta, k_s, stall, chi, tracker, logger = logged_level(rows, beta_start, d, errors, stop_n_s=n_s)
        meta = _metadata(rows, tracker, logger, K_MAX, {'type': 'termination rule', 'tau_chi': 0.1,
                                                        'tau_r': 0.01, 'n_s': n_s},
                         'rule' if k_s is not None else 'iteration_cap',
                         'zero' if i == 0 else 'the previous level\'s returned iterate, embedded (zero padding)')
        write_run_json(d / 'run.json', {**meta, 'package4': {'item': '1b', 'variant': variant, 'tolerance': tol,
                                                             'level': i, 'p': p, 'n_s': n_s,
                                                             'identity_scaled_max': max(tracker.identity_scaled)}})
        levels_so_far.append({'p': p, 'beta': beta})
        return beta, k_s, stall, chi

    levels_so_far = []
    levels = enrichment(tol, VARIANTS[variant], log_level=log_level)
    rows_out, cum = [], 0
    for lev in levels:
        space = Space(lev['p'])
        cum += lev['solves'] or 0
        rows_out.append({'p': lev['p'], 'P': lev['P'], 'N': lev['N'], 'n_s': lev['n_s'], 'solves': lev['solves'],
                         'stall_level': lev['stall_level'], 'chi_at_stop': lev['chi_at_stop'],
                         'error': errors.error(space, lev['beta']), 'delta_P': errors.delta_P(space),
                         'cumulative_solves': cum})
    _write_csv(run_dir / 'levels.csv', rows_out)
    np.save(run_dir / 'beta_returned.npy', levels[-1]['beta'])
    final = levels[-1]
    status = ('tolerance met' if final['stall_level'] is not None and final['stall_level'] <= tol
              else 'p limit reached' if final['p'] >= P_LIMIT else 'rule did not fire')
    write_run_json(run_dir / 'run.json', {'package4': {'item': '1b', 'variant': variant, 'n_s_after_embedding':
                                                       VARIANTS[variant], 'tolerance': tol, 'status': status,
                                                       'levels': len(levels), 'final_p': final['p'],
                                                       'total_solves': cum, 'check_M4': m4}})
    return run_dir


# ---------------------------------------------------------------- 1c: constants at the iterates

def constants(p, out_root):
    """c1, c2 and rho_r at every iterate of 1a's logged pass at p."""
    from experiments.p3_2_bl_certified import section6_row
    space = Space(p)
    rows_h, rows_y = Rows(space), Rows(space, over=Y_OVER)
    betas = np.load(Path(out_root) / ITEMS['1a'] / f'p{p}' / 'betas.npy')
    out_dir = Path(out_root) / ITEMS['1c'] / f'p{p}'
    out_dir.mkdir(parents=True, exist_ok=True)
    by_iter = []
    for k in range(len(betas) - 1):
        by_iter.append({'k': k, **section6_row(rows_h, rows_y, betas[k], betas[k + 1], ())})
        _write_csv(out_dir / 'constants_by_iterate.csv', by_iter)
    _write_csv(out_dir / 'rho_r.csv', [{key: x[key] for key in ('k', 'rho_r', 'rho_r_numerator', 'rho_r_denominator',
                                                                 'kappa_A_Y', 'round_off')} for x in by_iter])
    return {'p': p, 'P': space.P, 'N': rows_h.N, 'N_Y': rows_y.N, 'iterates': len(by_iter),
            'c1_min': min(x['c1'] for x in by_iter), 'c2_max': max(x['c2'] for x in by_iter),
            'c2_over_c1_max': max(x['c2_over_c1'] for x in by_iter),
            'rho_r_max': max(x['rho_r'] for x in by_iter if not x['round_off']) if any(
                not x['round_off'] for x in by_iter) else float('nan'),
            'dropped_max': max(x['dropped'] for x in by_iter)}


# ---------------------------------------------------------------- timing

def _timed(fn):
    """Section 2.3: an untimed warm-up, then three timings, each the median of
    five runs if the first is shorter than 1 s. ``fn`` returns (seconds,
    solves)."""
    warm_s, warm_solves = fn()
    timings, solves = [], set()
    for _ in range(REPEATS):
        s, n = fn()
        solves.add(n)
        if s < SHORT_RUN_S:
            reps = [fn() for _ in range(SHORT_REPEATS)]
            solves.update(r[1] for r in reps)
            timings.append({'time_s': statistics.median(r[0] for r in reps), 'runs_s': [r[0] for r in reps]})
        else:
            timings.append({'time_s': s, 'runs_s': [s]})
    return {'warmup_s': warm_s, 'warmup_solves': warm_solves, 'timings': timings,
            'median_s': statistics.median(t['time_s'] for t in timings), 'solves': sorted(solves)}


def time_single(p):
    def once():
        t0 = time.perf_counter()
        space = Space(p)
        rows = Rows(space)
        _, k_s, _, _ = iterate_to_rule(rows, np.zeros(space.P), N_S)
        return time.perf_counter() - t0, None if k_s is None else k_s + 1
    return _timed(once)


def time_enrichment(variant, tol):
    def once():
        t0 = time.perf_counter()
        levels = enrichment(tol, VARIANTS[variant])
        return time.perf_counter() - t0, sum(lev['solves'] or 0 for lev in levels)
    return _timed(once)


def timing(out_root, which=('1a', '1b'), sizes=SIZES):
    thread_env = capture_blas_thread_env()
    if '1a' in which:
        for p in sizes:
            run_dir = Path(out_root) / ITEMS['1a'] / f'p{p}'
            run_dir.mkdir(parents=True, exist_ok=True)
            out = time_single(p)
            (run_dir / 'timing.json').write_text(json.dumps({**out, 'p': p, 'thread_env': thread_env}, indent=2))
            print(f"1a p={p}: median {out['median_s']:.4g} s, solves {out['solves']}", flush=True)
    if '1b' in which:
        for variant in VARIANTS:
            for tol in TOLERANCES:
                run_dir = Path(out_root) / ITEMS['1b'] / f'{variant}_tol{tol:g}'
                run_dir.mkdir(parents=True, exist_ok=True)
                out = time_enrichment(variant, tol)
                (run_dir / 'timing.json').write_text(json.dumps({**out, 'variant': variant, 'tolerance': tol,
                                                                 'thread_env': thread_env}, indent=2))
                print(f"1b {variant} tol={tol:g}: median {out['median_s']:.4g} s, solves {out['solves']}", flush=True)


# ---------------------------------------------------------------- the summary and the checks

def _log(run_dir):
    rows = _read_csv(Path(run_dir) / 'iterations.csv')
    solve = [r for r in rows if r['norm_Rlin_h'] not in ('', None)]
    return rows, solve


def terminal_row(run_dir, delta):
    rows, solve = _log(run_dir)
    chi = [_float(r['chi']) for r in solve]
    rlin = [_float(r['norm_Rlin_h']) for r in solve]
    k_s = stall_rule_index(chi, rlin)
    run = json.loads((Path(run_dir) / 'run.json').read_text())
    timing_path = Path(run_dir) / 'timing.json'
    t = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    out = {'p': run['package4']['p'], 'P': run['P_total'], 'N': run['N_total'], 'rule_step': k_s,
           'returned_iterate': None if k_s is None else k_s + 1, 'delta_P': delta,
           'median_time_s': t.get('median_s'), 'timed_solves': t.get('solves')}
    if k_s is None:
        return out
    r = solve[k_s]
    ret = rows[k_s + 1]
    kr = _float(r['kappa_retained'])
    ratio, level = roundoff_comparison(_float(r['norm_Rlin_h']), _float(r['norm_f_h']), kr)
    err = _float(ret['eps_ref'])
    last = solve[-1]
    out.update({'class': 'C' if ratio < 10 * level else 'A', 'stall_level': _float(r['norm_Rlin_h']),
                'chi_at_rule': _float(r['chi']), 'norm_N_returned': _float(ret['norm_R_h']), 'error': err,
                'error_over_delta_P': err / delta, 'rank_gelsy': r['num_rank_gelsy'], 'rank_svd': r['num_rank_svd'],
                'kappa_retained_at_rule': kr, 'kappa_raw_at_rule': _float(r['kappa_raw']),
                'kappa_retained_last': _float(last['kappa_retained']), 'kappa_raw_last': _float(last['kappa_raw']),
                'rank_gelsy_last': last['num_rank_gelsy'], 'rank_svd_last': last['num_rank_svd'],
                'solves_to_stop': k_s + 1, 'timing_matches_logged': (t.get('solves') == [k_s + 1]) if t else None,
                'b2_max_over_run': run['b2_check']['max_rel_err_over_run'], 'b2_at_k1': run['b2_check']['rel_err']})
    return out


def _rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


def summarize(out_root, reference_dir, expected=None, sizes=SIZES):
    errors = Errors(load_reference(reference_dir))
    root = Path(out_root)
    term = [terminal_row(root / ITEMS['1a'] / f'p{p}', errors.delta_P(Space(p))) for p in sizes
            if (root / ITEMS['1a'] / f'p{p}' / 'iterations.csv').exists()]
    _write_csv(root / ITEMS['1a'] / 'terminal.csv', term)
    checks = {}
    by_p = {r['p']: r for r in term}
    # M1: 1a at p = 10 and 22 against the advisor's expected values
    if expected:
        exp = json.loads(Path(expected).read_text())['direct']
        m1 = []
        for p in (10, 22):
            if p in by_p and str(p) in exp:
                e, r = exp[str(p)], by_p[p]
                m1.append({'p': p, 'solves': r.get('solves_to_stop'), 'solves_expected': e['it'],
                           'stall_rel': _rel(r['stall_level'], e['stall']), 'error_rel': _rel(r['error'], e['err']),
                           'delta_P_rel': _rel(r['delta_P'], e['dP'])})
        checks['M1'] = {'rows': m1, 'pass': bool(m1) and all(
            x['solves'] == x['solves_expected'] and max(x['stall_rel'], x['error_rel'], x['delta_P_rel']) <= 1e-6
            for x in m1)}
    # M2: the Cole-Hopf reference against the Gauss-Hermite evaluation
    x, t, u = load_reference(reference_dir)
    d = float(np.abs(colexact(x, t) - u).max())
    checks['M2'] = {'max_abs_difference': d, 'pass': d < 1e-10}
    # M3: the identity of Remark 1 (as K4 of Package 3: k = 1, run max reported)
    m3 = [{'run': f'1a p{p}', **m3_summary(json.loads((root / ITEMS['1a'] / f'p{p}' / 'run.json').read_text()))}
          for p in sizes if (root / ITEMS['1a'] / f'p{p}' / 'run.json').exists()]
    # 1b: levels, M4 and M5
    enrich_rows = []
    m4, m5 = [], []
    for variant in VARIANTS:
        for tol in TOLERANCES:
            d1b = root / ITEMS['1b'] / f'{variant}_tol{tol:g}'
            if not (d1b / 'levels.csv').exists():
                continue
            levels = _read_csv(d1b / 'levels.csv')
            info = json.loads((d1b / 'run.json').read_text())['package4']
            m4 += [dict(x, variant=variant, tolerance=tol) for x in info['check_M4']]
            for lev in levels:
                p = int(lev['p'])
                if p in by_p and by_p[p].get('stall_level') is not None:
                    m5.append({'variant': variant, 'tolerance': tol, 'p': p,
                               'stall_rel': _rel(_float(lev['stall_level']), by_p[p]['stall_level']),
                               'error_rel': _rel(_float(lev['error']), by_p[p]['error'])})
            for lev_dir in sorted(d1b.glob('level*_p*')):
                m3.append({'run': f'1b {variant} tol {tol:g} {lev_dir.name}',
                           **m3_summary(json.loads((lev_dir / 'run.json').read_text()))})
            t = json.loads((d1b / 'timing.json').read_text()) if (d1b / 'timing.json').exists() else {}
            final_p = int(levels[-1]['p'])
            single = by_p.get(final_p, {}).get('median_time_s')
            enrich_rows.append({'variant': variant, 'tolerance': tol, 'final_p': final_p,
                                'final_P': final_p ** 2, 'levels': len(levels),
                                'total_solves': int(levels[-1]['cumulative_solves']), 'status': info['status'],
                                'median_time_s': t.get('median_s'), 'timed_solves': t.get('solves'),
                                'single_solve_time_at_final_P_s': single})
    if enrich_rows:
        _write_csv(root / ITEMS['1b'] / 'summary.csv', enrich_rows)
    checks['M3'] = {'rows': m3, 'pass': all(x['pass'] for x in m3),
                    'note': ('judged by the scaled difference ||(A beta - f) - N|| / || |A||beta| + |f| || against '
                             'P eps at every iterate (no cancellation); b2_at_k1 is the K4 reading of Package 3, '
                             'reported. A level of 1b starts on the plateau, where ||N|| is small and the plain '
                             'relative difference measures cancellation (DECISIONS.md, Package 4).')}
    checks['M4'] = {'rows': m4, 'pass': all(x['max_abs_difference'] < 1e-12 for x in m4)}
    checks['M5'] = {'rows': m5, 'pass': all(x['stall_rel'] <= 1e-6 for x in m5),
                    'error_rel_max': max((x['error_rel'] for x in m5), default=None),
                    'note': ("judged on the stall levels. The errors of a level's returned iterate are reported: a "
                             "level stops at a different step of the plateau than the single solve (E2: n_s = 1), "
                             "and the reference script gives the same differences (up to 1.6e-4 at p = 10, E2)")}
    # 1c summary
    cdir = root / ITEMS['1c']
    if cdir.exists():
        rows_c = []
        for p in sizes:
            f = cdir / f'p{p}' / 'constants_by_iterate.csv'
            if f.exists():
                c = _read_csv(f)
                ok = [x for x in c if x['round_off'] == 'False']
                rows_c.append({'p': p, 'P': p * p, 'iterates': len(c), 'c1_min': min(_float(x['c1']) for x in c),
                               'c2_max': max(_float(x['c2']) for x in c),
                               'c2_over_c1_max': max(_float(x['c2_over_c1']) for x in c),
                               'rho_r_max': max((_float(x['rho_r']) for x in ok), default=float('nan')),
                               'round_off_iterates': len(c) - len(ok),
                               'dropped_max': max(int(x['dropped']) for x in c)})
        _write_csv(cdir / 'constants.csv', rows_c)
    (root / 'checks_item1.json').write_text(json.dumps(checks, indent=2))
    return term, checks


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 4, item 1: Burgers on Gauss points.')
    ap.add_argument('stage', choices=['logged', 'enrich', 'constants', 'time', 'summarize', 'all'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir')
    ap.add_argument('--p', type=int, nargs='+', default=list(SIZES))
    ap.add_argument('--which', nargs='+', default=['1a', '1b'], choices=['1a', '1b'])
    ap.add_argument('--expected', help="the advisor's expected_burgers.json (check M1)")
    a = ap.parse_args(argv)
    if a.stage in ('logged', 'enrich', 'all', 'summarize') and not a.reference_dir:
        ap.error('--reference-dir is required for this stage')
    if a.stage in ('logged', 'all'):
        errors = Errors(load_reference(a.reference_dir))
        for p in a.p:
            print('logged p', p, logged(p, a.out, errors), flush=True)
    if a.stage in ('enrich', 'all'):
        errors = Errors(load_reference(a.reference_dir))
        for variant in VARIANTS:
            for tol in TOLERANCES:
                print('enrich', variant, tol, enrich_logged(variant, tol, a.out, errors), flush=True)
    if a.stage in ('constants', 'all'):
        for p in a.p:
            print('constants', constants(p, a.out), flush=True)
    if a.stage in ('time', 'all'):
        timing(a.out, a.which, a.p)
    if a.stage in ('summarize', 'all'):
        term, checks = summarize(a.out, a.reference_dir, a.expected, a.p)
        print(json.dumps({k: v['pass'] for k, v in checks.items()}, indent=1))


if __name__ == '__main__':
    main()
