"""
Package 2, item 3 (P2-16): the certified-grid scalar runs
=========================================================

The advisor's instructions of 4 October 2026, Section 6.3, and his Stage-1
reply (Section 4, item 2). LiL-Q on Bratu and Burgers run fully inside the
hypotheses of Section 5.6, with the constants computed and the terminal
error compared with delta_P. Untimed.

**Trial spaces:** tensor Chebyshev. Bratu: T_i(2x-1) T_j(2y-1), i, j < p_d,
P = p_d^2 in {25, 100, 225}. Burgers: T_i(x) T_j(2t-1), P = p_d^2 in
{25, ..., 625}. Boundary and initial conditions are weighted rows,
lambda = 10; there is no lifting.

**Collocation sets** (``grid``). N counts every row. The proportions are
the paper's: Bratu 85% interior and 15% boundary; Burgers 90% interior, 5%
initial and 5% boundary. For a target r = N/P:
- **Bratu:** an M x M Chebyshev-Gauss-Lobatto (CGL) tensor grid,
  M = ceil(sqrt(0.85 r P)), and ceil(0.15 r P / 4) CGL points per edge.
- **Burgers:** an M x M interior grid, M = ceil(sqrt(0.9 r P)),
  ceil(0.05 r P) CGL points on the initial line, and ceil(0.025 r P) on
  each lateral line (CGL in t on (0, 1]: the n+1-point rule without t = 0).

Weights:
- **Interior rows:** the tensor Clenshaw-Curtis (CC) weights of the M-point
  rules, normalized to the measure |Omega| = 1.
- **Auxiliary rows:** lambda times the 1D CC weights times |e|/|dOmega|. For
  Burgers, dOmega is the whole boundary of the space-time rectangle
  (|dOmega| = 6); the line t = 1 carries no rows.

The nodes and weights are B10's (``lilq.collocation.points_1d(..., 'cgl')``
and ``clenshaw_curtis_weights``), endpoints included. r in {5, 10, 20}; K_max = 60 with the full ``iterations.csv`` and
``eps_ref``. The paper's loss target is logged but does not stop the run.

**Constants** (``constants``). c1^2 and c2^2 are the extreme eigenvalues of
the CC-weighted Gram matrix, on the interior points, of an orthonormal
(tensor Legendre) basis of the ambient space Q_q, q = m p_d per direction:
- m = 2 at every r, and m = 3 at r = 20;
- reported as not computable when dim Q_q >= N_interior (check C6).

**Realized ratio** rho_r at every iterate k:
- the numerator is the Y-norm of the linearized residual (at beta^(k)) of
  the collocation minimizer beta^(k+1);
- the denominator is the smallest attainable over the trial space, from a
  separate least-squares solve;
- both are on a 96 x 96 Gauss-Legendre quadrature of Omega and 96 points per
  boundary line, weighted as the collocation norm (S8.5 note).

**delta_P:** the relative discrete L2 distance of the reference solution to
the trial space (a least-squares fit on the test grid). The reference errors
are reported:
- at the iterate returned by the termination rule (Section 5.7, defaults);
- at k = 60;
- at the first iterate meeting the paper's loss target (the advisor's
  Stage-1 reply).

Each is also given as a ratio to delta_P.

**Check C4** (the CC part). Integrals of tensor polynomials of degree
<= M - 1 per direction (relative to the polynomial's L2 norm), and squared
L2 norms of degree <= floor((M - 1)/2), are exact to 1e-12. Literally, norms of degree <= M - 1 would need degree
2M - 2 exactness, which an M-point rule does not have.

**Outputs** under ``P2_16_certified/``:
- per run, ``<benchmark>_P<P>_NP<r>/{run.json, iterations.csv, rho_r.csv,
  collocation.npz}``;
- ``constants.csv`` and ``terminal.csv``;
- ``check_c4_c6.json``;
- ``figures/``.

Usage::

    python experiments/p2_16_certified.py --out <stage root> --reference-dir <dir with the Bratu and Burgers references>
"""

import argparse
import csv
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
import scipy.linalg as sla
from numpy.polynomial import chebyshev as C
from numpy.polynomial import legendre as L

from lilq.collocation import clenshaw_curtis_weights as cc_weights
from lilq.collocation import points_1d
from lilq.instrumentation import EPS_MACH, stall_rule_index
from lilq.iteration_log import IterationLogger, LilQDiagnosticsTracker
from lilq.provenance import capture_blas_thread_env, save_provenance
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
from lilq.source_lock import current_commit

RATIOS = (5, 10, 20)
SIZES = {'bratu': (25, 100, 225), 'burgers': (25, 100, 225, 400, 625)}
DOMAIN = {'bratu': ((0.0, 1.0), (0.0, 1.0)), 'burgers': ((-1.0, 1.0), (0.0, 1.0))}
LAM_AUX = 10.0
K_MAX = 60
N_QUAD = 96
LAM_BRATU, NU_BURGERS = 6.2, 0.1
TARGETS = {'bratu': {25: 2.5e-1, 100: 1e-4, 225: 2.5e-7},
           'burgers': {25: 6e-2, 100: 1e-3, 225: 2e-5, 400: 1e-7, 625: 5e-9}}


# ---------------------------------------------------------------- the trial space

def cheb(z, n, a, b):
    """T_j(xi(z)), j < n, and its first and second z-derivatives (xi maps [a, b] to [-1, 1])."""
    t = 2 * (np.asarray(z, float) - a) / (b - a) - 1
    s = 2 / (b - a)
    eye = np.eye(n)
    V = C.chebvander(t, n - 1)
    V1 = np.stack([s * C.chebval(t, C.chebder(eye[j], 1)) for j in range(n)], 1)
    V2 = np.stack([s * s * C.chebval(t, C.chebder(eye[j], 2)) for j in range(n)], 1)
    return V, V1, V2


def tensor(x, y, p, dom):
    """``{(dx, dy): matrix}`` of the p x p tensor basis at the points, x-index outer."""
    X, Y = cheb(x, p, *dom[0]), cheb(y, p, *dom[1])
    k = lambda A, B: (A[:, :, None] * B[:, None, :]).reshape(len(x), -1)  # noqa: E731
    return {(i, j): k(X[i], Y[j]) for i, j in ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2))}


# ---------------------------------------------------------------- collocation sets

def _cgl(m, a, b):
    return points_1d(a, b, m, 'cgl'), cc_weights(m)      # B10's nodes and weights; the weights sum to 1


def grid(bench, P, r):
    """The certified collocation set: ``{block: (x, y, w2)}`` with the squared
    row weights, and the counts."""
    (ax, bx), (ay, by) = DOMAIN[bench]
    blocks = {}
    if bench == 'bratu':
        M = math.ceil(math.sqrt(0.85 * r * P))
        n_e = math.ceil(0.15 * r * P / 4)
        gx, wx = _cgl(M, ax, bx)
        gy, wy = _cgl(M, ay, by)
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        blocks['interior'] = (X.ravel(), Y.ravel(), np.outer(wx, wy).ravel())
        t, w = _cgl(n_e, 0.0, 1.0)
        share = 1.0 / 4.0                                 # |e| / |dOmega| for the unit square's edges
        for name, x, y in (('bottom', t, 0 * t), ('top', t, 0 * t + 1), ('left', 0 * t, t), ('right', 0 * t + 1, t)):
            blocks[name] = (x, y, LAM_AUX * share * w)
    else:
        M = math.ceil(math.sqrt(0.9 * r * P))
        n_i, n_b = math.ceil(0.05 * r * P), math.ceil(0.025 * r * P)
        gx, wx = _cgl(M, ax, bx)
        gt, wt = _cgl(M, ay, by)
        X, T = np.meshgrid(gx, gt, indexing='ij')
        blocks['interior'] = (X.ravel(), T.ravel(), np.outer(wx, wt).ravel())
        perim = 2 * (bx - ax) + 2 * (by - ay)             # 6: the whole boundary of the space-time rectangle
        xi, wi = _cgl(n_i, ax, bx)
        blocks['initial'] = (xi, 0 * xi, LAM_AUX * (bx - ax) / perim * wi)
        tb, wb = _cgl(n_b + 1, ay, by)                    # (0, 1]: the (n_b + 1)-point rule without t = 0
        tb, wb = tb[1:], wb[1:]
        for name, xe in (('left', ax), ('right', bx)):
            blocks[name] = (0 * tb + xe, tb, LAM_AUX * (by - ay) / perim * wb)
    N = sum(len(b[0]) for b in blocks.values())
    return {'blocks': blocks, 'M': M, 'N': N, 'N_interior': len(blocks['interior'][0]), 'ratio_actual': N / P}


def quadrature(bench, n=N_QUAD):
    """The Y-norm's quadrature, weighted as the collocation norm: Gauss-Legendre
    on Omega (normalized) and on each boundary line carrying rows."""
    (ax, bx), (ay, by) = DOMAIN[bench]
    t, w = L.leggauss(n)
    gl = lambda a, b: (a + (b - a) * (t + 1) / 2, w / 2)  # noqa: E731  (weights sum to 1)
    gx, wx = gl(ax, bx)
    gy, wy = gl(ay, by)
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    blocks = {'interior': (X.ravel(), Y.ravel(), np.outer(wx, wy).ravel())}
    if bench == 'bratu':
        e, we = gl(0.0, 1.0)
        for name, x, y in (('bottom', e, 0 * e), ('top', e, 0 * e + 1), ('left', 0 * e, e), ('right', 0 * e + 1, e)):
            blocks[name] = (x, y, LAM_AUX / 4 * we)
    else:
        perim = 2 * (bx - ax) + 2 * (by - ay)
        blocks['initial'] = (gx, 0 * gx, LAM_AUX * (bx - ax) / perim * wx)
        for name, xe in (('left', ax), ('right', bx)):
            blocks[name] = (0 * gy + xe, gy, LAM_AUX * (by - ay) / perim * wy)
    return blocks


# ---------------------------------------------------------------- the linearized system

class Rows:
    """The basis at a set of blocks, evaluated once."""

    def __init__(self, bench, p, blocks):
        self.bench, self.p, self.blocks = bench, p, blocks
        dom = DOMAIN[bench]
        self.B = {name: (tensor(x, y, p, dom), np.sqrt(w2)) for name, (x, y, w2) in blocks.items()}


def assemble(rows: Rows, beta):
    """``(A, f, R)``: the quasilinearized system at beta (A beta_new = f, weighted)
    and the weighted nonlinear residual at beta."""
    A, f, R = [], [], []
    for name, (T, w) in rows.B.items():
        val = T[(0, 0)]
        if name == 'interior':
            u = val @ beta
            if rows.bench == 'bratu':
                e = LAM_BRATU * np.exp(u)
                lap = T[(2, 0)] + T[(0, 2)]
                A.append(w[:, None] * (lap + e[:, None] * val))
                f.append(w * e * (u - 1))
                R.append(w * (lap @ beta + e))
            else:
                ux = T[(1, 0)] @ beta
                op = T[(0, 1)] + u[:, None] * T[(1, 0)] + ux[:, None] * val - NU_BURGERS * T[(2, 0)]
                A.append(w[:, None] * op)
                f.append(w * u * ux)
                R.append(w * (T[(0, 1)] @ beta + u * ux - NU_BURGERS * (T[(2, 0)] @ beta)))
        else:
            data = -np.sin(np.pi * rows.blocks[name][0]) if name == 'initial' else 0 * w
            A.append(w[:, None] * val)
            f.append(w * data)
            R.append(w * (val @ beta - data))
    return np.vstack(A), np.concatenate(f), np.concatenate(R)


def rho_r(quad: Rows, beta_k, beta_new):
    """The realized ratio at the linearization beta_k: the Y-residual of the
    collocation minimizer over the smallest attainable (least squares on the quadrature)."""
    A, f, _ = assemble(quad, beta_k)
    best = np.linalg.norm(A @ sla.lstsq(A, f, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)[0] - f)
    got = np.linalg.norm(A @ beta_new - f)
    return float(got / best), float(got), float(best)


# ---------------------------------------------------------------- references, errors, delta_P

def load_reference(bench, reference_dir):
    from lilq.references import load_reference as lr
    from lilq.references import reference_path
    axes, u, _ = lr(reference_path(reference_dir, bench))
    return axes, u


def error_fn(bench, p, ref):
    (x, y), u_ref = ref
    dom = DOMAIN[bench]
    Vx, Vy = cheb(x, p, *dom[0])[0], cheb(y, p, *dom[1])[0]
    nref = np.linalg.norm(u_ref)
    return lambda beta: float(np.linalg.norm(Vx @ beta.reshape(p, p) @ Vy.T - u_ref) / nref)


def delta_P(bench, p, ref):
    """The reference's relative discrete L2 distance to the trial space on the test grid."""
    (x, y), u_ref = ref
    dom = DOMAIN[bench]
    Vx, Vy = cheb(x, p, *dom[0])[0], cheb(y, p, *dom[1])[0]
    fit = Vx @ np.linalg.pinv(Vx) @ u_ref @ np.linalg.pinv(Vy).T @ Vy.T   # separable least squares on the tensor grid
    return float(np.linalg.norm(fit - u_ref) / np.linalg.norm(u_ref))


# ---------------------------------------------------------------- one run

def run(bench, P, r, ref, out_dir, k_max=K_MAX):
    p = int(round(math.sqrt(P)))
    g = grid(bench, P, r)
    rows, quad = Rows(bench, p, g['blocks']), Rows(bench, p, quadrature(bench))
    err = error_fn(bench, p, ref)
    n_int = g['N_interior']
    tracker = LilQDiagnosticsTracker(test_error_fn=lambda b: {'eps_ref': err(b)})
    logger = IterationLogger()
    beta = np.zeros(p * p)
    rho = []
    for k in range(k_max):
        t0 = time.perf_counter()
        A, f, R = assemble(rows, beta)
        t1 = time.perf_counter()
        new, _, rank, _ = sla.lstsq(A, f, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)
        t2 = time.perf_counter()
        ratio, got, best = rho_r(quad, beta, new)
        rho.append({'k': k, 'rho_r': ratio, 'Y_residual_collocation': got, 'Y_residual_best': best})
        logger.record(**tracker.step(
            k=k, A_stacked=A, b_stacked=f, beta_prev=beta, beta_new=new,
            total_loss=float(np.sum(assemble(rows, new)[2] ** 2)), rank_gelsy=int(rank),
            t_assemble_s=t1 - t0, t_solve_s=t2 - t1, is_final_iterate=(k == k_max - 1),
            compute_residual_vector_fn=lambda b: assemble(rows, b)[2]))
        beta = new
    logger.record(**tracker.finish(k=k_max))
    run_dir = Path(out_dir) / f'{bench}_P{P}_NP{r}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.to_csv(run_dir / 'iterations.csv')
    with open(run_dir / 'rho_r.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rho[0]))
        w.writeheader()
        w.writerows(rho)
    np.savez_compressed(run_dir / 'collocation.npz', **{f'{n}_{c}': v for n, b in g['blocks'].items()
                                                         for c, v in zip(('x', 'y', 'weight_squared'), b)})
    thread_env = capture_blas_thread_env()
    write_run_json(run_dir / 'run.json', build_run_metadata(
        N_total=g['N'], N_composition={n: len(b[0]) for n, b in g['blocks'].items()}, P_total=p * p,
        P_composition={'u': p * p}, row_weights='Clenshaw-Curtis: interior tensor weights (sum 1); each auxiliary line '
        f'{LAM_AUX} x 1D weights x |e|/|dOmega| (collocation.npz)',
        collocation_construction={'method': 'CGL tensor grid with Clenshaw-Curtis weights (Package 2, Section 6.3)',
                                  'ratio_target': r, 'ratio_actual': g['ratio_actual'], 'M': g['M'],
                                  'N_interior': n_int},
        basis_description={'family': 'tensor Chebyshev', 'modes_per_direction': p},
        initial_coefficients='zero', solver_driver='gelsy', rcond=EPS_MACH,
        stopping_rule={'type': 'none: K_max iterations; the termination rule and the loss target are evaluated '
                               'on the log'}, K_max=k_max, stopping_reason='iteration_cap',
        first_stall_iteration=first_stall_iteration(logger.rows), b2_check=tracker.b2_check,
        kappa_qr_raw_ratio=tracker.kappa_qr_raw_ratio, device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1)))
    return g, logger.rows, rho


def terminal_row(bench, P, r, g, log, rho, dP):
    """The stopping points: the termination rule's returned iterate, k = 60,
    and the first iterate meeting the paper's loss target."""
    chi = np.array([np.nan if x['chi'] in ('', None) else float(x['chi']) for x in log])
    rlin = np.array([np.nan if x['norm_Rlin_h'] in ('', None) else float(x['norm_Rlin_h']) for x in log])
    K = len(log) - 1
    i = stall_rule_index(chi[:K], rlin[:K])
    eps = [float(x['eps_ref']) for x in log]
    k_rule = i + 1 if i is not None else None
    target = TARGETS[bench][P]
    k_target = next((int(x['k']) for x in log if float(x['norm_R_h']) ** 2 <= target), None)
    return {'benchmark': bench, 'P': P, 'NP': r, 'NP_actual': g['ratio_actual'], 'N': g['N'],
            'k_rule': k_rule if k_rule is not None else '',
            'eps_ref_rule': eps[k_rule] if k_rule is not None else '',
            'eps_ref_60': eps[K], 'delta_P': dP,
            'ratio_rule': eps[k_rule] / dP if k_rule is not None else '', 'ratio_60': eps[K] / dP,
            'loss_60': float(log[K]['norm_R_h']) ** 2,
            'k_target': k_target if k_target is not None else '', 'target_mse': target,
            'eps_ref_target': eps[k_target] if k_target is not None else '',
            'ratio_target': eps[k_target] / dP if k_target is not None else '',
            'rho_r_final': rho[-1]['rho_r'], 'rho_r_max': max(x['rho_r'] for x in rho)}


# ---------------------------------------------------------------- constants and checks

def legendre_orth(z, a, b, q):
    t = 2 * (np.asarray(z, float) - a) / (b - a) - 1
    return np.stack([np.sqrt(2 * k + 1) * L.legval(t, np.eye(q + 1)[k]) for k in range(q + 1)], 1)


def constants(bench, P, r, g, m):
    """c1, c2 on the ambient space Q_{m p_d} at the interior points with their weights."""
    p = int(round(math.sqrt(P)))
    q = m * p
    x, y, w2 = g['blocks']['interior']
    dim = (q + 1) ** 2
    row = {'benchmark': bench, 'P': P, 'NP': r, 'm': m, 'dim_ambient': dim, 'N_interior': len(x)}
    if dim >= len(x):
        return {**row, 'c1': '', 'c2': '', 'c2_over_c1': '', 'computable': False}
    (ax, bx), (ay, by) = DOMAIN[bench]
    Bx, By = legendre_orth(x, ax, bx, q), legendre_orth(y, ay, by, q)
    Phi = (Bx[:, :, None] * By[:, None, :]).reshape(len(x), -1)
    s = np.linalg.svd(np.sqrt(w2)[:, None] * Phi, compute_uv=False)
    return {**row, 'c1': float(s[-1]), 'c2': float(s[0]), 'c2_over_c1': float(s[0] / s[-1]), 'computable': True}


def check_cc_exactness(bench, g, rng=None):
    """Check C4 (CC exactness) on the run's interior grid: integrals of degree <=
    M - 1 and squared norms of degree <= floor((M - 1)/2), per direction."""
    rng = np.random.default_rng(0) if rng is None else rng
    x, y, w2 = g['blocks']['interior']
    (ax, bx), (ay, by) = DOMAIN[bench]
    M = g['M']
    tq, wq = L.leggauss(M + 4)

    def tensor_poly(deg):
        c = rng.standard_normal((deg + 1, deg + 1))
        f = lambda X, Y: np.einsum('ij,ni,nj->n', c, legendre_orth(X, ax, bx, deg), legendre_orth(Y, ay, by, deg))  # noqa: E731
        return f, np.linalg.norm(c)                       # the L2 norm on the normalized measure
    gx, gy = ax + (bx - ax) * (tq + 1) / 2, ay + (by - ay) * (tq + 1) / 2
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    W = np.outer(wq, wq).ravel() / 4                      # Gauss on Omega, normalized to |Omega| = 1
    f, f_norm = tensor_poly(M - 1)
    # relative to ||f||_L2: the integral of a random polynomial is its constant coefficient
    # alone, which can be far smaller than the function, so dividing by it measures cancellation
    integral = abs(np.sum(w2 * f(x, y)) - np.sum(W * f(GX.ravel(), GY.ravel()))) / f_norm
    h, _ = tensor_poly((M - 1) // 2)
    norm = abs(np.sum(w2 * h(x, y) ** 2) - np.sum(W * h(GX.ravel(), GY.ravel()) ** 2)) / np.sum(W * h(GX.ravel(), GY.ravel()) ** 2)
    return {'M': M, 'integral_degree_M_minus_1_err_over_L2_norm': float(integral),
            'norm_degree_half_rel_err': float(norm), 'passed': bool(integral < 1e-12 and norm < 1e-12)}


# ---------------------------------------------------------------- the stage

def run_all(out_root, reference_dir, sizes=SIZES, ratios=RATIOS, k_max=K_MAX):
    out = Path(out_root) / 'P2_16_certified'
    terminal, consts, checks = [], [], []
    for bench, Ps in sizes.items():
        ref = load_reference(bench, reference_dir)
        for P in Ps:
            dP = delta_P(bench, int(round(math.sqrt(P))), ref)
            for r in ratios:
                g, log, rho = run(bench, P, r, ref, out, k_max)
                t = terminal_row(bench, P, r, g, log, rho, dP)
                terminal.append(t)
                for m in ((2, 3) if r == 20 else (2,)):
                    consts.append(constants(bench, P, r, g, m))
                checks.append({'benchmark': bench, 'P': P, 'NP': r, **check_cc_exactness(bench, g)})
                print(f"  {bench:7s} P = {P:3d} N/P = {r:2d} ({g['ratio_actual']:.2f}): eps_ref rule {t['eps_ref_rule'] if t['eps_ref_rule'] == '' else format(t['eps_ref_rule'], '.2e')} "
                      f"/ k=60 {t['eps_ref_60']:.2e}, delta_P {dP:.2e}, rho_r final {t['rho_r_final']:.4f}")
    for name, rows in (('terminal.csv', terminal), ('constants.csv', consts)):
        with open(out / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    c6 = bool(all(c['dim_ambient'] < c['N_interior'] for c in consts if c['computable']))
    (out / 'check_c4_c6.json').write_text(json.dumps({
        'C4_cc_exactness': {'passed': bool(all(c['passed'] for c in checks)), 'runs': checks,
                            'note': 'CC on M points integrates degree <= M - 1 exactly; squared norms are exact for '
                                    'degree <= floor((M - 1)/2) per direction (a norm of degree M - 1 needs '
                                    'degree 2M - 2)'},
        'C6_computable_only_when_dim_below_N_interior': c6, 'commit': current_commit()}, indent=2))
    save_provenance(out)
    figures(out, terminal)
    return terminal, consts, checks


def figures(out, terminal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for bench, mk in (('bratu', 'o'), ('burgers', 's')):
        for r in RATIOS:
            t = [x for x in terminal if x['benchmark'] == bench and x['NP'] == r]
            P = [x['P'] for x in t]
            axes[0].loglog(P, [x['eps_ref_60'] for x in t], mk + '-', ms=4, label=f'{bench}, N/P = {r}')
            axes[1].loglog(P, [x['ratio_60'] for x in t], mk + '-', ms=4)
        P = sorted({x['P'] for x in terminal if x['benchmark'] == bench})
        axes[0].loglog(P, [next(x['delta_P'] for x in terminal if x['benchmark'] == bench and x['P'] == p) for p in P],
                       'k' + mk + ':', ms=4, label=f'{bench}, delta_P')
    for run_dir in sorted(out.glob('*_NP*')):
        with open(run_dir / 'rho_r.csv') as fh:
            rr = [float(x['rho_r']) for x in csv.DictReader(fh)]
        axes[2].semilogy(rr, lw=0.8)
    axes[0].set(xlabel='P', ylabel='eps_ref at k = 60', title='error and delta_P')
    axes[1].set(xlabel='P', ylabel='eps_ref / delta_P (k = 60)', title='ratio to the best approximation')
    axes[2].set(xlabel='k', ylabel='rho_r', title='realized ratio per iterate (every run)')
    axes[0].legend(fontsize=6)
    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'certified.pdf')
    fig.savefig(out / 'figures' / 'certified.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 2, item 3: the certified-grid scalar runs.")
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--reference-dir', required=True, help='the folder with the Bratu and Burgers references')
    args = ap.parse_args(argv)
    run_all(args.out, args.reference_dir)


if __name__ == '__main__':
    main()
