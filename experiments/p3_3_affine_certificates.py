"""
Package 3, item 3: a posteriori certificates for the affine problems
====================================================================

The advisor's instructions of 8 October 2026, Sections 5 and 6. For an affine
operator the residual space is fixed: W = span{L phi_p} + span{f}, of
dimension at most P + 1. Its constants on a collocation set certify
Assumption A6 there, with C_r = c2/c1. They are computed on the paper's own
collocation sets and row weights, with one re-solve of each configuration
for beta_h.

**The re-solve is the paper's own solver.** ``solve_elasticity`` and
``solve_lilq_darcy`` run unchanged. The system they pass to
``scipy.linalg.lstsq`` and its solution are captured (``captured_solve``),
so the collocation side (A_h, b_h, beta_h) is the paper's run itself. Its
residual ||A_h beta_h - b_h|| is compared with the logged one (k = 0,
``norm_Rlin_h``):
- Package 1 for the paper's elasticity and for Darcy, run at 48 threads;
- P2-10 for the two manufactured elasticity solutions, run at 24 threads.

"To all digits" needs Grace's stack and the same thread count, so each
configuration runs in its own process with its original thread count
(``THREADS``).

**Configurations:**
- **Elasticity:** the paper's solution, P2-10's ``compatible`` and
  ``specified``, at P = 50, 200, 450, 800, 1,250 (N = 5 .. 25). The blocks
  are the code's ten, in its order: PDE-x and PDE-y (squared weight
  lambda_pde / n_pde); then bottom u_x, bottom u_y, top u_x, top sigma_yy,
  left sigma_xx, left u_y, right sigma_xx, right u_y (lambda_bc / n_bc each).
- **Darcy:** S1, S2, S3 and SPE10 with the paper's LiL configuration
  (P = 3,169). The code's blocks agree with the instructions' reading:
  Darcy-x, Darcy-y and continuity at the 60 x 220 cell centres, weight 1, no
  boundary rows. The sqrt(K*) factors are part of each row's operator.

**Zero rows** (Section 6) are left out of both systems. A block is a zero
block when its rows are zero to round-off (|A| <= 1e-10 max|A_h|) and its
data are zero. In elasticity that is every u_x and u_y Dirichlet block on an
edge where the basis vanishes (sin(0), sin(n pi)), and the lateral sigma_xx
blocks when the lateral traction is zero (paper, compatible). For
``specified`` the lateral traction is not zero, so those blocks, zero in A
and not in b, stay: they are the inconsistent rows of P2-10's stall.

**The Y-norm, block-matched** (Section 6). Each block's continuous mean
square over its set carries the block's total squared weight in the
collocation set:
- elasticity: 96 x 96 Gauss-Legendre inside and 96 per edge;
- Darcy: n_q x n_q Gauss-Legendre per cell (n_q = 4), with that cell's K*,
  so each cell's quadrature weights sum to its centre row's 1.

**Section 6 from the R factor.** The Y-matrix B_Y = [A_Y | b_Y] enters only
through its R factor, accumulated a chunk of cells at a time
(``lilq.certified.tsqr``). The column norms, the pivoted QR of the
unit-scaled B_Y, the least-squares minimum, the residual at beta_h and
kappa(A_Y) are all invariant under B_Y's orthogonal factor
(``section6_from_R``, ``rho_from_R``). Darcy's B_Y is 633,600 x 3,170 at
n_q = 4 and 1.43 M x 3,170 at n_q = 6 (K5), and is never held whole.

Outputs, in ``P3_3_affine_certificates/``: one JSON per configuration in
``runs/``; ``elasticity.csv`` and ``darcy.csv``; ``checks_item3.json`` (the
re-solves, K5).

Usage::

    python experiments/p3_3_affine_certificates.py all --out <root> --package1 <package1> --p2-10 <P2_10_elasticity_manufactured>
    python experiments/p3_3_affine_certificates.py run --problem darcy --case SPE10 --out <root> [--n-quad 6 --tag _k5]
"""

import argparse
import contextlib
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
import scipy.linalg

from lilq import certified as cf
from lilq.provenance import capture_blas_thread_env
from lilq.source_lock import current_commit

ITEM = 'P3_3_affine_certificates'
ELAST_N = (5, 10, 15, 20, 25)
SOLUTIONS = ('paper', 'compatible', 'specified')
FIELDS = ('S1', 'S2', 'S3', 'SPE10')
N_QUAD = {'elasticity': 96, 'darcy': 4}
K5_QUAD = {'elasticity': 144, 'darcy': 6}
THREADS = {('elasticity', 'paper'): 48, ('elasticity', 'compatible'): 24, ('elasticity', 'specified'): 24,
           ('darcy', None): 48}                         # the original runs' (Package 1 wave 2; P2-10 on Grace)
ELAST_BLOCKS = ('pde_x', 'pde_y', 'bot_ux', 'bot_uy', 'top_ux', 'top_syy', 'left_sxx', 'left_uy', 'right_sxx',
                'right_uy')
ZERO_A, ZERO_B = 1e-10, 1e-14


@contextlib.contextmanager
def captured_solve():
    """Record the first ``scipy.linalg.lstsq`` call's (A, b) and solution,
    calling the real one unchanged."""
    real = scipy.linalg.lstsq
    box = {}

    def spy(a, b, *args, **kwargs):
        out = real(a, b, *args, **kwargs)
        if 'A' not in box:
            box.update(A=np.array(a, copy=True), b=np.array(b, copy=True), x=np.array(out[0], copy=True))
        return out
    scipy.linalg.lstsq = spy
    try:
        yield box
    finally:
        scipy.linalg.lstsq = real


# ---------------------------------------------------------------- elasticity

def elasticity_setup(solution, N):
    from problems.elasticity import (CompatibleManufacturedElasticityPhysics, ElasticityConfig, ElasticityPhysics,
                                     ManufacturedElasticityPhysics)
    config = ElasticityConfig(N_x=N, N_y=N, k_ratio=10)
    cls = {'paper': ElasticityPhysics, 'compatible': CompatibleManufacturedElasticityPhysics,
           'specified': ManufacturedElasticityPhysics}[solution]
    return config, cls(config)


def elasticity_bases(config):
    from lilq.basis import create_basis_2d
    return (create_basis_2d(config.basis_u, config.N_x, config.N_y, config.x_domain, config.y_domain),
            create_basis_2d(config.basis_v, config.N_x, config.N_y, config.x_domain, config.y_domain))


def elasticity_rows(bu, bv, physics, points, w2):
    """``{block: (A, b)}`` at ``points[block] = (x, y)``, rows times sqrt(w2[block]):
    ``solve_elasticity``'s rows (bc_mode 'paper'), operation for operation."""
    C11, C12, mu = physics.C11, physics.C12, physics.mu
    Cx = physics.lam + mu
    Pu, Pv = bu.n_basis, bv.n_basis
    out = {}
    for name in ELAST_BLOCKS:
        x, y = points[name]
        n = len(x)
        if name == 'pde_x':
            A = np.hstack([C11 * bu.derivative(x, y, dx=2, dy=0) + mu * bu.derivative(x, y, dx=0, dy=2),
                           Cx * bv.derivative(x, y, dx=1, dy=1)])
            b = physics.body_force_x(x, y)
        elif name == 'pde_y':
            A = np.hstack([Cx * bu.derivative(x, y, dx=1, dy=1),
                           mu * bv.derivative(x, y, dx=2, dy=0) + C11 * bv.derivative(x, y, dx=0, dy=2)])
            b = physics.body_force_y(x, y)
        elif name.endswith('_ux'):
            A, b = np.hstack([bu.evaluate(x, y), np.zeros((n, Pv))]), np.zeros(n)
        elif name.endswith('_uy'):
            A, b = np.hstack([np.zeros((n, Pu)), bv.evaluate(x, y)]), np.zeros(n)
        elif name == 'top_syy':
            A = np.hstack([C12 * bu.derivative(x, y, dx=1, dy=0), C11 * bv.derivative(x, y, dx=0, dy=1)])
            b = physics.traction_top_syy(x)
        else:                                            # left_sxx, right_sxx
            A = np.hstack([C11 * bu.derivative(x, y, dx=1, dy=0), C12 * bv.derivative(x, y, dx=0, dy=1)])
            b = physics.traction_lateral_sxx(x, y)
        s = np.sqrt(w2[name])
        out[name] = (s[:, None] * A, s * b)
    return out


def elasticity_collocation(config, physics):
    """The paper's points (``problems.elasticity._generate_collocation``) and squared weights."""
    from problems.elasticity import _generate_collocation
    bu, bv = elasticity_bases(config)
    pts = _generate_collocation(config, physics, bu.n_basis + bv.n_basis)
    edge = {'bot': ('x_bot', 'y_bot'), 'top': ('x_top', 'y_top'), 'left': ('x_left', 'y_left'),
            'right': ('x_right', 'y_right')}
    points, w2 = {}, {}
    for name in ELAST_BLOCKS:
        if name.startswith('pde'):
            points[name] = (pts['x_pde'], pts['y_pde'])
            w2[name] = np.full(pts['n_pde'], config.lambda_pde / pts['n_pde'])
        else:
            xk, yk = edge[name.split('_')[0]]
            points[name] = (pts[xk], pts[yk])
            w2[name] = np.full(len(pts[xk]), config.lambda_bc / len(pts[xk]))
    return points, w2


def elasticity_quadrature(config, w2_colloc, n_gl):
    """Gauss-Legendre: n_gl x n_gl inside and n_gl per edge, each block
    carrying its total squared weight in the collocation set."""
    g, gw = np.polynomial.legendre.leggauss(n_gl)
    t, wt = (g + 1) / 2, gw / 2
    X, Y = np.meshgrid(t, t, indexing='ij')
    W = np.outer(wt, wt).ravel()
    points, w2 = {}, {}
    for name in ELAST_BLOCKS:
        total = float(np.sum(w2_colloc[name]))
        if name.startswith('pde'):
            points[name], w2[name] = (X.ravel(), Y.ravel()), total * W
        else:
            side = name.split('_')[0]
            fixed = {'bot': ('x', 0.0), 'top': ('x', 1.0), 'left': ('y', 0.0), 'right': ('y', 1.0)}[side]
            points[name] = ((t, 0 * t + fixed[1]) if fixed[0] == 'x' else (0 * t + fixed[1], t))
            w2[name] = total * wt
    return points, w2


def zero_blocks(blocks):
    """The blocks whose rows are zero to round-off and whose data are zero."""
    amax = max(np.abs(A).max() for A, _ in blocks.values())
    bmax = max(np.abs(b).max() for _, b in blocks.values())
    return [n for n, (A, b) in blocks.items() if np.abs(A).max() <= ZERO_A * amax and np.abs(b).max() <= ZERO_B * bmax]


def run_elasticity(solution, N, n_gl=None):
    from problems.elasticity import solve_elasticity
    n_gl = n_gl or N_QUAD['elasticity']
    config, physics = elasticity_setup(solution, N)
    t0 = time.perf_counter()
    with captured_solve() as cap:
        solve_elasticity(config, verbose=False, physics=physics)
    A_h, b_h, beta = cap['A'], cap['b'], cap['x']
    bu, bv = elasticity_bases(config)
    points, w2 = elasticity_collocation(config, physics)
    mine = elasticity_rows(bu, bv, physics, points, w2)
    A_mine = np.vstack([mine[n][0] for n in ELAST_BLOCKS])
    b_mine = np.concatenate([mine[n][1] for n in ELAST_BLOCKS])
    zeros = zero_blocks(mine)
    keep = [n for n in ELAST_BLOCKS if n not in zeros]
    sizes = [len(mine[n][1]) for n in ELAST_BLOCKS]
    starts = dict(zip(ELAST_BLOCKS, np.cumsum([0] + sizes[:-1])))
    rows = np.concatenate([np.arange(starts[n], starts[n] + len(mine[n][1])) for n in keep])
    B_h = np.column_stack([A_h[rows], b_h[rows]])
    qp, qw = elasticity_quadrature(config, w2, n_gl)
    quad = elasticity_rows(bu, bv, physics, qp, qw)
    R_Y = cf.tsqr([np.column_stack([quad[n][0], quad[n][1]]) for n in keep])
    n_rows_Y = sum(len(quad[n][1]) for n in keep)
    return _record('elasticity', solution, config, A_h, b_h, beta, B_h, R_Y, n_rows_Y, zeros,
                   {'assembly_matches_paper_bitwise': bool(np.array_equal(A_mine, A_h) and np.array_equal(b_mine, b_h)),
                    'assembly_max_abs_diff': float(max(np.abs(A_mine - A_h).max(), np.abs(b_mine - b_h).max()))},
                   f'Gauss-Legendre {n_gl} x {n_gl} inside, {n_gl} per edge', t0, P=bu.n_basis + bv.n_basis)


# ---------------------------------------------------------------- Darcy

def darcy_setup(field, orders=None):
    """The paper's LiL configuration; ``orders`` (tests only) shrinks the bases."""
    from problems.darcy import DarcyConfig, DarcyPhysics, _create_basis_h_tilde, _create_basis_u, _create_basis_v
    config = DarcyConfig(perm_file=f'perm_field_{field}.txt')
    if orders is not None:
        import dataclasses
        config = dataclasses.replace(config, ORDER_H=orders, ORDER_U=orders, ORDER_V=orders)
    physics = DarcyPhysics(config, verbose=False)
    bases = (_create_basis_h_tilde(config.ORDER_H), _create_basis_u(config.ORDER_U), _create_basis_v(config.ORDER_V))
    return config, physics, bases


def darcy_rows(bases, R, x, y, sqrt_K, w=None):
    """``[A | b]`` of the three blocks (Darcy-x, Darcy-y, continuity) at the
    points, as ``solve_lilq_darcy`` builds them (operation for operation);
    each row times sqrt(w) when w is given."""
    bh, bu, bv = bases
    n_h, n_u, n_v = bh.n_basis, bu.n_basis, bv.n_basis
    n = len(x)
    A_Dx = np.hstack([(sqrt_K * R)[:, None] * bh.derivative(x, y, dx=1, dy=0), bu.evaluate(x, y) / sqrt_K[:, None],
                      np.zeros((n, n_v))])
    A_Dy = np.hstack([sqrt_K[:, None] * bh.derivative(x, y, dx=0, dy=1), np.zeros((n, n_u)),
                      bv.evaluate(x, y) / sqrt_K[:, None]])
    A_CE = np.hstack([np.zeros((n, n_h)), R * bu.derivative(x, y, dx=1, dy=0), bv.derivative(x, y, dx=0, dy=1)])
    B = np.column_stack([np.vstack([A_Dx, A_Dy, A_CE]), np.concatenate([np.zeros(n), -sqrt_K, np.zeros(n)])])
    if w is not None:
        B *= np.sqrt(np.concatenate([w, w, w]))[:, None]
    return B


def darcy_quadrature_chunks(config, physics, bases, n_q, x_cells_per_chunk=5):
    """n_q x n_q Gauss-Legendre per cell with that cell's K*; each cell's
    weights sum to 1 (its centre row's squared weight). Yields the weighted
    rows a few columns of cells at a time."""
    g, gw = np.polynomial.legendre.leggauss(n_q)
    t, wt = (g + 1) / 2, gw / 2
    nx, ny = config.NX_CELLS, config.NY_CELLS
    for i0 in range(0, nx, x_cells_per_chunk):
        ii = np.arange(i0, min(nx, i0 + x_cells_per_chunk))
        I, J, A, B = np.meshgrid(ii, np.arange(ny), np.arange(n_q), np.arange(n_q), indexing='ij')
        x = (I + t[A]) / nx
        y = (J + t[B]) / ny
        w = (wt[A] * wt[B]).ravel()
        sk = physics.sqrt_K_star[I.ravel(), J.ravel()]
        yield darcy_rows(bases, physics.R, x.ravel(), y.ravel(), sk, w)


def run_darcy(field, n_q=None, orders=None):
    from problems.darcy import solve_lilq_darcy
    n_q = n_q or N_QUAD['darcy']
    config, physics, bases = darcy_setup(field, orders)
    t0 = time.perf_counter()
    with captured_solve() as cap:
        solve_lilq_darcy(config, physics, verbose=False)
    A_h, b_h, beta = cap['A'], cap['b'], cap['x']
    xc = (np.arange(config.NX_CELLS) + 0.5) / config.NX_CELLS
    yc = (np.arange(config.NY_CELLS) + 0.5) / config.NY_CELLS
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    mine = darcy_rows(bases, physics.R, X.ravel(), Y.ravel(), np.sqrt(physics.K_star.ravel()))
    B_h = np.column_stack([A_h, b_h])
    R_Y = cf.tsqr(darcy_quadrature_chunks(config, physics, bases, n_q))
    n_rows_Y = 3 * config.NX_CELLS * config.NY_CELLS * n_q * n_q
    return _record('darcy', field, config, A_h, b_h, beta, B_h, R_Y, n_rows_Y, [],
                   {'assembly_matches_paper_bitwise': bool(np.array_equal(mine, B_h)),
                    'assembly_max_abs_diff': float(np.abs(mine - B_h).max())},
                   f'Gauss-Legendre {n_q} x {n_q} per cell, the cell\'s K*', t0, P=A_h.shape[1])


# ---------------------------------------------------------------- one record

def _record(problem, case, config, A_h, b_h, beta, B_h, R_Y, n_rows_Y, zeros, assembly, quad, t0, P):
    W = cf.section6_from_R(B_h, R_Y)
    Aonly = cf.section6_from_R(B_h[:, :-1], R_Y[:-1, :-1])
    rho = cf.rho_from_R(R_Y, beta, n_rows_Y)
    thread_env = capture_blas_thread_env()
    return {'problem': problem, 'case': case, 'P': int(P), 'N': int(A_h.shape[0]), 'N_used': int(B_h.shape[0]),
            'c1': W['c1'], 'c2': W['c2'], 'c2_over_c1': W['c2_over_c1'], 'dropped': W['dropped'],
            'c1_A_only': Aonly['c1'], 'c2_A_only': Aonly['c2'], 'c2_over_c1_A_only': Aonly['c2_over_c1'],
            'dropped_A_only': Aonly['dropped'], **rho, 'quadrature': quad, 'N_Y': int(n_rows_Y),
            'zero_blocks_excluded': zeros, 'resolve_residual': float(np.linalg.norm(A_h @ beta - b_h)),
            **assembly, 'threads': thread_env.get('OMP_NUM_THREADS'), 'seconds': time.perf_counter() - t0,
            'commit': current_commit()}


# ---------------------------------------------------------------- logged residuals, the sweep, the summary

def logged_residual(problem, case, P, package1=None, p2_10=None):
    """The logged k = 0 ``norm_Rlin_h``: Package 1 (paper elasticity, Darcy)
    or P2-10 (the manufactured solutions); None if not given."""
    if problem == 'darcy':
        path = Path(package1) / 'B_instrumentation' / f'darcy_{case}_cpu_paper' / 'iterations.csv' if package1 else None
    elif case == 'paper':
        path = Path(package1) / 'B_instrumentation' / f'elasticity_P{P}_cpu_paper' / 'iterations.csv' if package1 else None
    else:
        path = Path(p2_10) / case / f'P{P}' / 'iterations.csv' if p2_10 else None
    if path is None or not path.exists():
        return None, None
    with open(path) as fh:
        row = next(r for r in csv.DictReader(fh) if r['k'] == '0')
    return float(row['norm_Rlin_h']), str(path)


def run_one(problem, case, N, out_root, n_quad=None, tag=''):
    rec = run_elasticity(case, N, n_quad) if problem == 'elasticity' else run_darcy(case, n_quad)
    rec['tag'] = tag
    d = Path(out_root) / ITEM / 'runs'
    d.mkdir(parents=True, exist_ok=True)
    name = f"{problem}_{case}" + (f"_P{rec['P']}" if problem == 'elasticity' else '') + tag
    (d / f'{name}.json').write_text(json.dumps(rec, indent=2))
    return d / f'{name}.json'


def _run_process(problem, case, N, out_root, threads, n_quad=None, tag=''):
    import subprocess
    env = {**os.environ, **{v: str(threads) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    cmd = [sys.executable, os.path.abspath(__file__), 'run', '--problem', problem, '--case', case, '--out',
           str(Path(out_root).resolve())] + (['--N', str(N)] if N else []) + \
        (['--n-quad', str(n_quad)] if n_quad else []) + (['--tag', tag] if tag else [])
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, cwd=_proj)


def run_all(out_root, threads=None, with_k5=True):
    """Every configuration in its own process at its original thread count
    (``THREADS``; ``threads`` overrides, e.g. on a laptop), then K5."""
    for solution in SOLUTIONS:
        for N in ELAST_N:
            _run_process('elasticity', solution, N, out_root, threads or THREADS[('elasticity', solution)])
    for field in FIELDS:
        _run_process('darcy', field, None, out_root, threads or THREADS[('darcy', None)])
    if with_k5:
        for solution in SOLUTIONS:
            _run_process('elasticity', solution, 25, out_root, threads or THREADS[('elasticity', solution)],
                         n_quad=K5_QUAD['elasticity'], tag='_k5')
        _run_process('darcy', 'SPE10', None, out_root, threads or THREADS[('darcy', None)],
                     n_quad=K5_QUAD['darcy'], tag='_k5')


def _write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def summarize(out_root, package1=None, p2_10=None):
    d = Path(out_root) / ITEM
    recs = [json.loads(p.read_text()) for p in sorted((d / 'runs').glob('*.json'))]
    main = [r for r in recs if not r.get('tag')]
    for r in main:
        logged, src = logged_residual(r['problem'], r['case'], r['P'], package1, p2_10)
        r['logged_residual'], r['logged_source'] = logged, src
        r['resolve_identical'] = None if logged is None else r['resolve_residual'] == logged
        r['resolve_rel_diff'] = None if logged is None else abs(r['resolve_residual'] - logged) / max(abs(logged), 1e-300)
    cols = ['problem', 'case', 'P', 'N', 'N_used', 'c1', 'c2', 'c2_over_c1', 'c1_A_only', 'c2_A_only',
            'c2_over_c1_A_only', 'dropped', 'dropped_A_only', 'rho_r', 'rho_r_numerator', 'rho_r_denominator',
            'round_off', 'kappa_A_Y', 'quadrature', 'N_Y', 'zero_blocks_excluded', 'resolve_residual',
            'logged_residual', 'resolve_identical', 'resolve_rel_diff', 'assembly_matches_paper_bitwise', 'threads']
    for problem, name in (('elasticity', 'elasticity.csv'), ('darcy', 'darcy.csv')):
        rows = [{c: r.get(c) for c in cols} for r in main if r['problem'] == problem]
        if rows:
            _write_csv(d / name, rows)
    checks = {'assembly': {'all_bitwise': all(r['assembly_matches_paper_bitwise'] for r in main)},
              'resolves': {f"{r['problem']} {r['case']} P={r['P']}": {'resolve': r['resolve_residual'],
                                                                     'logged': r['logged_residual'],
                                                                     'identical': r['resolve_identical']}
                           for r in main}}
    k5 = {}
    for r in (x for x in recs if x.get('tag') == '_k5'):
        base = next((m for m in main if m['problem'] == r['problem'] and m['case'] == r['case'] and m['P'] == r['P']), None)
        if base is None:
            continue
        ch = {c: abs(r[c] - base[c]) / abs(base[c]) for c in ('c1', 'c2')}
        if not (r['round_off'] or base['round_off']):
            ch['rho_r'] = abs(r['rho_r'] - base['rho_r']) / abs(base['rho_r'])
        k5[f"{r['problem']} {r['case']} P={r['P']}"] = {'max_rel_change': ch, 'quadrature': r['quadrature'],
                                                       'rho_r_round_off': bool(r['round_off'] or base['round_off']),
                                                       'passed': max(ch.values()) < 1e-3}
    checks['K5'] = {'runs': k5, 'passed': bool(k5) and all(v['passed'] for v in k5.values())}
    (d / 'checks_item3.json').write_text(json.dumps(checks, indent=2))
    return main, checks


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 3, item 3: certificates for the affine problems.')
    ap.add_argument('stage', choices=('run', 'all', 'summarize'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--problem', choices=('elasticity', 'darcy'))
    ap.add_argument('--case', help='paper | compatible | specified (elasticity), S1 | S2 | S3 | SPE10 (Darcy)')
    ap.add_argument('--N', type=int, choices=ELAST_N)
    ap.add_argument('--n-quad', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--threads', type=int, default=None, help="all: override the original runs' thread counts")
    ap.add_argument('--package1')
    ap.add_argument('--p2-10', dest='p2_10')
    args = ap.parse_args(argv)
    if args.stage == 'run':
        print(run_one(args.problem, args.case, args.N, args.out, args.n_quad, args.tag))
        return
    if args.stage == 'all':
        run_all(args.out, args.threads)
    main_, checks = summarize(args.out, args.package1, args.p2_10)
    for r in main_:
        print(f"  {r['problem']:10s} {r['case']:10s} P={r['P']:5d}: c2/c1 {r['c2_over_c1']:.4f}, dropped {r['dropped']}, "
              f"rho_r {r['rho_r']:.6g}{' (round-off)' if r['round_off'] else ''}, re-solve "
              f"{'identical' if r['resolve_identical'] else 'differs' if r['resolve_identical'] is False else 'not compared'}"
              + (f" ({r['resolve_rel_diff']:.1e})" if r['resolve_rel_diff'] is not None else ''))
    print(f"  assembly bitwise: {checks['assembly']['all_bitwise']}; K5: "
          f"{'passed' if checks['K5']['passed'] else 'FAILED or not run'}")


if __name__ == '__main__':
    main()
