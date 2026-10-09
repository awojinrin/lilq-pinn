"""
Package 3, item 2: Buckley-Leverett on CC-weighted CGL grids
============================================================

The advisor's instructions of 8 October 2026, Sections 4 and 6. The paper's
two cases (Section 6.4, nu_a = 0.1, ``BLConfig.D_coef = -0.1``) and its
quasilinearization (``problems.buckley_leverett._make_lil_q_system_fn``),
on ``lilq.certified.bl_grid``: the P2-16 Burgers rule on x in [0, 1],
t in [0, T]. The cases are viscous (T = 0.4) and gravity (T = 0.175).

**Trial spaces:**
- **2a**, ``cheb``: T_i(2x - 1) T_j(2t/T - 1), i, j < p. The boundary and
  initial conditions are weighted rows.
- **2b**, ``sine``: S = (1 - x) + sum beta_ij sin(i pi x) T_j(2t/T - 1),
  i = 1..p, j < p (P2-9's ``lifted_sine_plain`` with Chebyshev in t). The
  lateral rows are zero rows, kept in the solve as in P2-9; the initial
  condition is a weighted row. The lifting's terms go to the right-hand
  side.

**Weights:** each row's squared weight is lambda x share x its CC weight.
lambda is 1 inside (as P2-16's Burgers) and 10 on the initial and lateral
lines. The shares are |line| / |dOmega|, with |dOmega| = 2 (1 + T).

**Runs:** p = 8, 16, 24, 32 (P = 64 .. 1,024), N/P in {5, 10, 20}, K_max = 60,
all logged. The initial guesses are the paper's:
- viscous: the initial profile extended in time, least-squares fitted
  (``pretrain_lil``; 2b fits IC - (1 - x));
- gravity: zero.

Section 8.2: a run whose termination rule never fires within 60 iterates is
run once more from the other guess (viscous zero; gravity the extended
profile), and both are reported.

**Per iterate k**, at the linearization beta^(k) (Section 6), with zero rows
(2b's lateral lines) excluded:
- the collocation system (A_h, b_h) and the Y-system (A_Y, b_Y), the latter
  on Gauss-Legendre n_GL x n_GL inside and n_GL per line,
  n_GL = max(96, 4p), block-matched: each block's continuous mean square
  times its total squared weight in the collocation set;
- c1 and c2 of W = span{A} + span{b}: the extreme singular values of
  B_h R^-1, from the column-pivoted QR of the unit-scaled B_Y = [A_Y | b_Y];
  the same on span{A} alone; the columns dropped (|R_ii| < 1e-10 |R_11|);
- rho_r: ||A_Y beta_h - b_Y|| at the collocation solution, over
  min ||A_Y beta - b_Y||, marked round-off below 100 eps kappa(A_Y) ||b_Y||.

**2a also** gets the a priori interior constants (m = 2; m = 3 at N/P = 20)
and the block constants of its three lines (Section 2.3(b)), in
``constants.csv``.

Errors: ``eps_ref`` against the refined references
``bl_fd_ref_<case>_nu0.1.npz`` (P2-2) on their 201 x 201 grid. delta_P: the
reference's least-squares distance to the trial space there (2b: the
lifting subtracted), as ``p2_9_bases.delta_P``. The loss target
(``run_bl.TARGET_LOSSES`` / ``GRAVITY_TARGET_LOSSES``) is compared with the
CC-weighted ||R||_h^2, and logged, not used to stop.

Usage::

    python experiments/p3_2_bl_certified.py all --out <root> --reference-dir <dir>
    python experiments/p3_2_bl_certified.py run --variant cheb --case viscous --p 16 --ratio 10 --out <root> --reference-dir <dir>
    python experiments/p3_2_bl_certified.py summarize --out <root> --reference-dir <dir>
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
import torch

from lilq import certified as cf
from lilq.instrumentation import EPS_MACH, conditioning_via_svd, roundoff_comparison, stall_rule_index
from lilq.iteration_log import IterationLogger, LilQDiagnosticsTracker
from lilq.provenance import capture_blas_thread_env
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json

ITEMS = {'cheb': 'P3_2a_bl_chebyshev_certified', 'sine': 'P3_2b_bl_lifted_sine_cheb'}
CASES = {'viscous': {'gravity': False, 'T': 0.4, 'guess': 'ic'},
         'gravity': {'gravity': True, 'T': 0.175, 'guess': 'zero'}}
OTHER_GUESS = {'ic': 'zero', 'zero': 'ic'}
SIZES = (8, 16, 24, 32)
RATIOS = (5, 10, 20)
K_MAX = 60
LAM_INT, LAM_AUX = 1.0, 10.0
B2_TOL = 1e-10
TIMING_COLUMNS = ('t_assemble_s', 't_solve_s', 't_cum_s')
NOTE_B2 = ("Package 1's convention (DECISIONS.md, B2 in real runs): the identity is judged at k = 1; near convergence R is a small difference of O(1) quantities and cancellation dominates the relative difference, so the run maximum is reported, not tested")


def physics_for(case):
    from problems.buckley_leverett import BLConfig, BLPhysics
    config = BLConfig.with_gravity() if CASES[case]['gravity'] else BLConfig()
    assert config.T_final == CASES[case]['T'] and config.D_coef == -0.1 and (config.S_left, config.S_right) == (1.0, 0.0)
    return config, BLPhysics(config)


# ---------------------------------------------------------------- the trial spaces

def _cheb(z, n, a, b):
    from experiments.p2_16_certified import cheb
    return cheb(z, n, a, b)


def _sine(x, n):
    """sin(i pi x), i = 1..n, and its first and second x-derivatives."""
    k = np.pi * np.arange(1, n + 1)
    arg = np.asarray(x, float)[:, None] * k[None, :]
    s, c = np.sin(arg), np.cos(arg)
    return s, k * c, -(k * k) * s


class TrialSpace:
    """``cheb`` (2a): T_i(2x - 1) T_j(2t/T - 1); ``sine`` (2b): sin(i pi x)
    T_j(2t/T - 1) with the lifting 1 - x. Coefficients x-index outer."""

    def __init__(self, variant, p, T):
        self.variant, self.p, self.T = variant, p, T
        self.n_basis = p * p
        self.lifted = variant == 'sine'

    def _x(self, x):
        return _sine(x, self.p) if self.lifted else _cheb(x, self.p, 0.0, 1.0)

    def mats(self, x, t):
        """``{'val', 'dx', 'dt', 'dxx'}`` at the points."""
        X, Tt = self._x(x), _cheb(t, self.p, 0.0, self.T)
        k = lambda A, B: (A[:, :, None] * B[:, None, :]).reshape(len(A), -1)  # noqa: E731
        return {'val': k(X[0], Tt[0]), 'dx': k(X[1], Tt[0]), 'dt': k(X[0], Tt[1]), 'dxx': k(X[2], Tt[0])}

    def derivative(self, x, t, dx=0, dy=0):
        """The basis's mixed derivative (the paper's basis interface)."""
        X, Tt = self._x(np.ravel(x))[dx], _cheb(np.ravel(t), self.p, 0.0, self.T)[dy]
        return (X[:, :, None] * Tt[:, None, :]).reshape(len(X), -1)

    def evaluate(self, x, t):
        return self.derivative(x, t)

    def lift(self, x, order=0):
        x = np.asarray(x, float)
        if not self.lifted:
            return np.zeros_like(x)
        return 1.0 - x if order == 0 else (-np.ones_like(x) if order == 1 else np.zeros_like(x))


# ---------------------------------------------------------------- the systems

class Rows:
    """The basis, the lifting and the data at a set of blocks
    ``{name: (x, t, w2)}`` (w2 the squared row weights), evaluated once."""

    def __init__(self, space, physics, blocks):
        self.space, self.physics = space, physics
        self.blocks = {}
        for name, (x, t, w2) in blocks.items():
            b = {'s': np.sqrt(w2), 'x': x, 't': t}
            if name == 'interior':
                b.update(space.mats(x, t), L=space.lift(x), Lx=space.lift(x, 1), Lxx=space.lift(x, 2))
            else:
                b['val'] = space.evaluate(x, t)
                b['L'] = space.lift(x)
                b['data'] = (physics.initial_condition(x) if name == 'initial' else
                             physics.bc_left(t) if name == 'left' else physics.bc_right(t)).astype(float)
            self.blocks[name] = b

    def _flux(self, S):
        St = torch.tensor(S, dtype=torch.float64)
        return self.physics.flux_derivative(St).numpy(), self.physics.flux_second_derivative(St).numpy()

    def assemble(self, beta, exclude=()):
        """``(A, b)``: the quasilinearized system at beta, rows weighted (the
        paper's Bellman-Kalaba form; a lifting's terms on the right)."""
        A, b = [], []
        D = self.physics.D
        for name, B in self.blocks.items():
            if name in exclude:
                continue
            s = B['s']
            if name == 'interior':
                S = B['L'] + B['val'] @ beta
                Sx = B['Lx'] + B['dx'] @ beta
                fp, fpp = self._flux(S)
                op = B['dt'] + fp[:, None] * B['dx'] + D * B['dxx'] + (fpp * Sx)[:, None] * B['val']
                rhs = fpp * S * Sx - (fp * B['Lx'] + D * B['Lxx'] + fpp * Sx * B['L'])
            else:
                op, rhs = B['val'], B['data'] - B['L']
            A.append(s[:, None] * op)
            b.append(s * rhs)
        return np.vstack(A), np.concatenate(b)

    def residual(self, beta, exclude=()):
        """The weighted nonlinear residual (S_t + f'(S) S_x + D S_xx inside), in
        :meth:`assemble`'s row order, written independently of it."""
        R = []
        for name, B in self.blocks.items():
            if name in exclude:
                continue
            if name == 'interior':
                S = B['L'] + B['val'] @ beta
                Sx = B['Lx'] + B['dx'] @ beta
                fp, _ = self._flux(S)
                R.append(B['s'] * (B['dt'] @ beta + fp * Sx + self.physics.D * (B['Lxx'] + B['dxx'] @ beta)))
            else:
                R.append(B['s'] * (B['L'] + B['val'] @ beta - B['data']))
        return np.concatenate(R)


def collocation_blocks(grid):
    """``{name: (x, t, w2)}``: lambda x share x the CC weights (Section 2.2)."""
    out = {}
    for name, b in grid['blocks'].items():
        lam = LAM_INT if name == 'interior' else LAM_AUX
        out[name] = (b['points'][:, 0], b['points'][:, 1], lam * b['share'] * b['w'])
    return out


def quadrature_blocks(colloc, T, n_gl):
    """The block-matched Y-norm (Section 6): Gauss-Legendre over each block's
    set, carrying the block's total squared weight in the collocation set."""
    total = {name: float(np.sum(w2)) for name, (_, _, w2) in colloc.items()}
    g, gw = np.polynomial.legendre.leggauss(n_gl)
    on = lambda a, b: (a + (b - a) * (g + 1) / 2, gw / 2)  # noqa: E731  (weights sum to 1)
    xq, wx = on(0.0, 1.0)
    tq, wt = on(0.0, T)
    X, Tq = np.meshgrid(xq, tq, indexing='ij')
    out = {'interior': (X.ravel(), Tq.ravel(), total['interior'] * np.outer(wx, wt).ravel())}
    out['initial'] = (xq, 0 * xq, total['initial'] * wx)
    for name, xe in (('left', 0.0), ('right', 1.0)):
        out[name] = (0 * tq + xe, tq, total[name] * wt)
    return out


# ---------------------------------------------------------------- constants (Section 6) and rho_r

def section6_constants(B_h, B_Y, tol=cf.COLUMN_DROP_TOL):
    """c1, c2 of the span of B's columns: B_Y's columns scaled to unit norm, the
    same scaling on B_h, a column-pivoted QR of B_Y, columns with
    |R_ii| < tol |R_11| dropped from both, then the extreme singular values of
    B_h R^-1."""
    scale = np.linalg.norm(B_Y, axis=0)
    scale[scale == 0] = 1.0
    _, R, piv = sla.qr(B_Y / scale, mode='economic', pivoting=True, check_finite=False)
    d = np.abs(np.diag(R))
    k = int((d > tol * d[0]).sum())
    X = sla.solve_triangular(R[:k, :k], (B_h / scale)[:, piv[:k]].T, trans='T', check_finite=False).T
    s = sla.svdvals(X, check_finite=False)
    return {'c1': float(s[-1]), 'c2': float(s[0]), 'c2_over_c1': float(s[0] / s[-1]), 'dropped': B_Y.shape[1] - k}


def section6_row(rows_h, rows_y, beta_k, beta_h, exclude):
    A_h, b_h = rows_h.assemble(beta_k, exclude)
    A_Y, b_Y = rows_y.assemble(beta_k, exclude)
    W = section6_constants(np.column_stack([A_h, b_h]), np.column_stack([A_Y, b_Y]))
    Aonly = section6_constants(A_h, A_Y)
    best_coef = sla.lstsq(A_Y, b_Y, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)[0]
    best = float(np.linalg.norm(A_Y @ best_coef - b_Y))
    got = float(np.linalg.norm(A_Y @ beta_h - b_Y))
    s = sla.svdvals(A_Y, check_finite=False)
    rank = int((s > max(A_Y.shape) * EPS_MACH * s[0]).sum())
    kappa = float(s[0] / s[rank - 1])
    floor = 100 * EPS_MACH * kappa * float(np.linalg.norm(b_Y))
    return {'c1': W['c1'], 'c2': W['c2'], 'c2_over_c1': W['c2_over_c1'], 'dropped': W['dropped'],
            'c1_A_only': Aonly['c1'], 'c2_A_only': Aonly['c2'], 'c2_over_c1_A_only': Aonly['c2_over_c1'],
            'dropped_A_only': Aonly['dropped'], 'rho_r': got / best if best > 0 else float('inf'),
            'rho_r_numerator': got, 'rho_r_denominator': best, 'kappa_A_Y': kappa, 'rank_A_Y': rank,
            'round_off': best < floor}


def apriori_constants(case, P, r):
    """2a's a priori constants (Section 2.3): the interior on Q_q, q = m p + 1
    (m = 2; m = 3 at N/P = 20), and each line's block constants."""
    p, T = int(round(math.sqrt(P))), CASES[case]['T']
    _, physics = physics_for(case)
    g = cf.bl_grid(P, r, T)
    rows = []
    for m in ((2, 3) if r == 20 else (2,)):
        k = cf.interior_constants_kronecker(g['M'], m * p + 1, 2)
        rows.append({'case': case, 'P': P, 'NP': r, 'block': 'interior', 'm': m, 'c1': k['c1'], 'c2': k['c2'],
                     'c2_over_c1': k['c2_over_c1'], 'computable': k['computable'], 'columns': k['dim_ambient'],
                     'dropped': 0})
    for name, b in g['blocks'].items():
        if name == 'interior':
            continue
        if name == 'initial':
            z, iv, data = b['points'][:, 0], [(0.0, 1.0)], lambda q: physics.initial_condition(q[:, 0])
        else:
            z, iv = b['points'][:, 1], [(0.0, T)]
            data = (lambda q: physics.bc_left(q[:, 0])) if name == 'left' else (lambda q: physics.bc_right(q[:, 0]))
        c = cf.block_constants(z, b['w'], iv, p, data=data)
        rows.append({'case': case, 'P': P, 'NP': r, 'block': name, 'm': '', 'c1': c['c1'], 'c2': c['c2'],
                     'c2_over_c1': c['c2_over_c1'], 'computable': True, 'columns': c['columns'],
                     'dropped': c['dropped']})
    comp = [x for x in rows if x['computable']]
    m2 = [x for x in comp if x['block'] != 'interior' or x['m'] == 2]
    if any(x['block'] == 'interior' for x in m2):
        c1, c2 = min(x['c1'] for x in m2), max(x['c2'] for x in m2)
        rows.append({'case': case, 'P': P, 'NP': r, 'block': 'overall (m = 2)', 'm': 2, 'c1': c1, 'c2': c2,
                     'c2_over_c1': c2 / c1, 'computable': True, 'columns': '', 'dropped': ''})
    else:
        rows.append({'case': case, 'P': P, 'NP': r, 'block': 'overall (m = 2)', 'm': 2, 'c1': '', 'c2': '',
                     'c2_over_c1': '', 'computable': False, 'columns': '', 'dropped': ''})
    return rows


# ---------------------------------------------------------------- one run

def _write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_name(case, P, r, guess, tag=''):
    return f'{case}_P{P}_NP{r}_{guess}{tag}'


def initial_guess(space, physics, guess):
    if guess == 'zero':
        return np.zeros(space.n_basis)
    from lilq.pretraining import pretrain_lil
    coef, _ = pretrain_lil(space, lambda x, t: physics.initial_condition(x) - space.lift(x), (0.0, 1.0),
                           (0.0, space.T), n_grid=50, verbose=False)
    return coef


def error_fn(space, reference):
    (x, t), u_ref = reference
    X, Tt = np.meshgrid(x, t, indexing='ij')
    Phi, L = space.evaluate(X.ravel(), Tt.ravel()), space.lift(X.ravel())
    u = u_ref.ravel()
    nu = np.linalg.norm(u)
    return lambda beta: {'eps_ref': float(np.linalg.norm(L + Phi @ beta - u) / nu)}


def load_reference(reference_dir, case):
    from lilq.references import load_reference as lr
    axes, u, _ = lr(Path(reference_dir) / f'bl_fd_ref_{case}_nu0.1.npz')
    return axes, u


def run(variant, case, P, r, out_root, reference_dir, guess=None, tag='', quad_scale=1.0, k_max=K_MAX):
    p, T = int(round(math.sqrt(P))), CASES[case]['T']
    guess = guess or CASES[case]['guess']
    config, physics = physics_for(case)
    space = TrialSpace(variant, p, T)
    g = cf.bl_grid(P, r, T)
    colloc = collocation_blocks(g)
    n_gl = int(round(max(96, 4 * p) * quad_scale))
    rows_h, rows_y = Rows(space, physics, colloc), Rows(space, physics, quadrature_blocks(colloc, T, n_gl))
    exclude = ('left', 'right') if variant == 'sine' else ()            # 2b's zero rows (Section 6)
    ref = load_reference(reference_dir, case)
    run_dir = Path(out_root) / ITEMS[variant] / run_name(case, P, r, guess, tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    base = {'package3': {'item': '2a' if variant == 'cheb' else '2b', 'variant': variant, 'case': case, 'P': P,
                         'NP': r, 'guess': guess, 'M': g['M'], 'N': g['N_points'], 'n_gl': n_gl,
                         'quad_scale': quad_scale, 'status': 'running',
                         'zero_rows_excluded_from_section6': list(exclude),
                         'weights_note': 'Clenshaw-Curtis squared row weights (Section 2.2): interior 1 x tensor CC '
                         'weight; each line 10 x CC weight x |line|/|dOmega|, |dOmega| = 2(1 + T). The paper\'s runs '
                         'put lambda on each line instead; accuracy differences from its tables are partly due to this.'}}
    (run_dir / 'run.json').write_text(json.dumps(base, indent=2))
    t_start = time.perf_counter()
    tracker = LilQDiagnosticsTracker(conditioning_svd_threshold=-1, test_error_fn=error_fn(space, ref))
    logger = IterationLogger()
    beta = initial_guess(space, physics, guess)
    by_iter, ranks = [], []
    for k in range(k_max):
        t0 = time.perf_counter()
        A, b = rows_h.assemble(beta)
        t1 = time.perf_counter()
        new, _, rank, _ = sla.lstsq(A, b, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)
        t2 = time.perf_counter()
        ranks.append(int(rank))
        R_new = rows_h.residual(new)
        row = tracker.step(k=k, A_stacked=A, b_stacked=b, beta_prev=beta, beta_new=new,
                           total_loss=float(R_new @ R_new), rank_gelsy=int(rank), t_assemble_s=t1 - t0,
                           t_solve_s=t2 - t1, is_final_iterate=False, compute_residual_vector_fn=rows_h.residual)
        if k == k_max - 1:                              # kappa at the last iterate, by SVD (P < 3,200)
            cond = conditioning_via_svd(A)
            row.update(kappa=cond['kappa'], kappa_method=cond['kappa_method'], num_rank_svd=cond['num_rank_svd'],
                       kappa_raw=cond.get('kappa_raw'), kappa_retained=cond.get('kappa_retained'),
                       kappa_eps=roundoff_comparison(row['norm_Rlin_h'], row['norm_f_h'], cond['kappa'])[1])
        logger.record(**row)
        logger.to_csv(run_dir / 'iterations.csv')
        np.save(run_dir / 'beta_last.npy', new)
        by_iter.append({'k': k, **section6_row(rows_h, rows_y, beta, new, exclude)})
        _write_csv(run_dir / 'constants_by_iterate.csv', by_iter)
        beta = new
    logger.record(**tracker.finish(k=k_max))
    logger.to_csv(run_dir / 'iterations.csv')
    _write_csv(run_dir / 'rho_r.csv', [{key: x[key] for key in ('k', 'rho_r', 'rho_r_numerator', 'rho_r_denominator',
                                                                 'kappa_A_Y', 'round_off')} for x in by_iter])
    thread_env = capture_blas_thread_env()
    meta = build_run_metadata(
        N_total=g['N_points'], N_composition={n: len(b['points']) for n, b in g['blocks'].items()}, P_total=P,
        P_composition={'S': P}, row_weights='Clenshaw-Curtis (Package 3, Section 2.2)',
        collocation_construction={'method': 'CGL tensor grid with Clenshaw-Curtis weights (P2-16 Burgers rule)',
                                  'M': g['M'], 'n_initial': g['n_initial'], 'n_lateral': g['n_lateral'],
                                  'ratio_target': r, 'boundary_measure': g['boundary_measure']},
        basis_description={'family': 'tensor Chebyshev' if variant == 'cheb' else
                           'lifted plain sine in x x Chebyshev in t: S = (1 - x) + sum beta sin(i pi x) T_j',
                           'modes_per_direction': p},
        initial_coefficients=('zero' if guess == 'zero' else
                              'initial profile extended in time, least-squares fitted (pretrain_lil)'
                              + (' after subtracting the lifting' if variant == 'sine' else '')),
        solver_driver='gelsy', rcond=EPS_MACH,
        stopping_rule={'type': 'none: K_max iterations; the target and the termination rule are read from the log'},
        K_max=k_max, stopping_reason='iteration_cap', first_stall_iteration=first_stall_iteration(logger.rows),
        b2_check=tracker.b2_check, kappa_qr_raw_ratio=None, device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1))
    base['package3'].update(status='complete', wall_time_s=time.perf_counter() - t_start, ranks=ranks)
    write_run_json(run_dir / 'run.json', {**meta, **base})
    return run_dir


# ---------------------------------------------------------------- the summary

def _floats(rows, col):
    return np.array([np.nan if r.get(col) in ('', None) else float(r[col]) for r in rows])


def target_loss(case, p):
    from experiments.run_bl import GRAVITY_TARGET_LOSSES, TARGET_LOSSES
    return (GRAVITY_TARGET_LOSSES if CASES[case]['gravity'] else TARGET_LOSSES).get(p)


def rule_fires(run_dir):
    with open(run_dir / 'iterations.csv') as fh:
        log = list(csv.DictReader(fh))
    K = sum(1 for x in log if x.get('rel_dbeta') not in ('', None))
    return stall_rule_index(_floats(log, 'chi')[:K], _floats(log, 'norm_Rlin_h')[:K]) is not None


def terminal_row(run_dir, delta):
    meta = json.loads((run_dir / 'run.json').read_text())
    p3 = meta['package3']
    with open(run_dir / 'iterations.csv') as fh:
        log = list(csv.DictReader(fh))
    with open(run_dir / 'constants_by_iterate.csv') as fh:
        c6 = list(csv.DictReader(fh))
    K = sum(1 for x in log if x.get('rel_dbeta') not in ('', None))
    eps = _floats(log, 'eps_ref')
    i = stall_rule_index(_floats(log, 'chi')[:K], _floats(log, 'norm_Rlin_h')[:K])
    k_rule = i + 1 if i is not None else None
    last = log[K - 1]
    kret = float(last['kappa_retained'] or last['kappa'] or 'nan') if last.get('kappa') else float('nan')
    cls = ('C' if float(log[i]['roundoff_ratio']) < 10 * kret * EPS_MACH else 'A') if i is not None and np.isfinite(kret) else ''
    p = int(round(math.sqrt(p3['P'])))
    target = target_loss(p3['case'], p)
    k_target = next((int(x['k']) for x in log if x.get('norm_R_h') not in ('', None)
                     and float(x['norm_R_h']) ** 2 <= target), None) if target else None
    at = lambda k: float(eps[k]) if k is not None else ''  # noqa: E731
    ratio = _floats(c6, 'c2_over_c1')
    k6 = min(k_rule, len(c6) - 1) if k_rule is not None else len(c6) - 1
    dP = delta[(p3['variant'], p3['case'], p3['P'])]
    return {'variant': p3['variant'], 'case': p3['case'], 'P': p3['P'], 'NP': p3['NP'], 'guess': p3['guess'],
            'M': p3['M'], 'N': p3['N'], 'k_target': k_target if k_target is not None else '', 'target_loss': target,
            'eps_ref_target': at(k_target), 'k_rule': k_rule if k_rule is not None else 'never', 'rule_class': cls,
            'eps_ref_rule': at(k_rule), 'eps_ref_60': float(eps[K]), 'delta_P': dP,
            'ratio_target': at(k_target) / dP if k_target is not None else '',
            'ratio_rule': at(k_rule) / dP if k_rule is not None else '', 'ratio_60': float(eps[K]) / dP,
            'loss_60': float(log[K]['norm_R_h']) ** 2, 'rank_min': min(p3.get('ranks') or [0]),
            'kappa_last': last.get('kappa'), 'c1_at_rule': c6[k6]['c1'], 'c2_at_rule': c6[k6]['c2'],
            'c2_over_c1_at_rule': c6[k6]['c2_over_c1'], 'section6_iterate': k6,
            'c2_over_c1_max': float(np.nanmax(ratio)), 'rho_r_at_rule': c6[k6]['rho_r'],
            'rho_r_max': float(np.nanmax(_floats(c6, 'rho_r'))),
            'round_off_any': any(x['round_off'] == 'True' for x in c6),
            'b2_rel_err_k1': (meta.get('b2_check') or {}).get('rel_err'),
            'b2_max_rel_err': (meta.get('b2_check') or {}).get('max_rel_err_over_run'),
            'wall_time_s': p3.get('wall_time_s'), 'status': p3['status']}


def delta_table(reference_dir, variants=('cheb', 'sine'), cases=tuple(CASES), sizes=SIZES):
    from experiments.p2_9_bases import delta_P
    out = {}
    for case in cases:
        (x, t), u = load_reference(reference_dir, case)
        for variant in variants:
            for p in sizes:
                space = TrialSpace(variant, p, CASES[case]['T'])
                out[(variant, case, p * p)] = delta_P(space.evaluate, x, t, u, lifted=space.lifted)
    return out


def compare_logs(a, b):
    """K6: iterations.csv in every column but the timings, and the last coefficients."""
    def load(d):
        with open(d / 'iterations.csv') as fh:
            return list(csv.DictReader(fh))
    la, lb = load(a), load(b)
    diff = [(i, c) for i, (ra, rb) in enumerate(zip(la, lb)) for c in ra if c not in TIMING_COLUMNS and ra[c] != rb[c]]
    same_beta = np.array_equal(np.load(a / 'beta_last.npy'), np.load(b / 'beta_last.npy'))
    return {'identical_except_timings': not diff and len(la) == len(lb), 'differing': diff[:20],
            'beta_identical': same_beta, 'passed': not diff and len(la) == len(lb) and same_beta,
            'excluded_columns': list(TIMING_COLUMNS)}


def summarize(out_root, reference_dir):
    delta = delta_table(reference_dir)
    checks = {}
    for variant, item in ITEMS.items():
        d = Path(out_root) / item
        if not d.is_dir():
            continue
        runs = sorted(x for x in d.iterdir() if x.is_dir() and (x / 'constants_by_iterate.csv').exists()
                      and not x.name.endswith(('_k5', '_k6')))
        terminal = [terminal_row(x, delta) for x in runs]
        if terminal:
            _write_csv(d / 'terminal.csv', terminal)
        b2 = [float(t['b2_rel_err_k1']) for t in terminal if t['b2_rel_err_k1'] is not None]
        checks[f'K4 ({variant})'] = {'max_rel_err_at_k1': max(b2) if b2 else None, 'tolerance': B2_TOL,
                                     'max_rel_err_over_runs': max((float(t['b2_max_rel_err']) for t in terminal
                                                                   if t['b2_max_rel_err'] is not None), default=None),
                                     'passed': bool(b2) and max(b2) < B2_TOL, 'rule': NOTE_B2}
        checks[f'K7 ({variant})'] = {'not_full_rank': [t for t in terminal if int(t['rank_min']) < int(t['P'])],
                                     'passed': all(int(t['rank_min']) == int(t['P']) for t in terminal)}
    k5 = Path(out_root) / ITEMS['cheb'] / (run_name('viscous', 1024, 10, 'ic') + '_k5')
    base = Path(out_root) / ITEMS['cheb'] / run_name('viscous', 1024, 10, 'ic')
    if k5.exists() and base.exists():
        def load(x):
            with open(x / 'constants_by_iterate.csv') as fh:
                return list(csv.DictReader(fh))
        a, b = load(base), load(k5)
        worst = {c: max(abs(float(x[c]) - float(y[c])) / abs(float(x[c])) for x, y in zip(a, b))
                 for c in ('c1', 'c2', 'rho_r')}
        checks['K5 (BL viscous P = 1,024, N/P = 10, n_GL x 1.5)'] = {'max_rel_change': worst,
                                                                    'passed': max(worst.values()) < 1e-3}
    k6 = Path(out_root) / ITEMS['cheb'] / (run_name('viscous', 256, 10, 'ic') + '_k6')
    if k6.exists():
        checks['K6 (BL viscous P = 256, N/P = 10)'] = compare_logs(
            Path(out_root) / ITEMS['cheb'] / run_name('viscous', 256, 10, 'ic'), k6)
    (Path(out_root) / 'checks_item2.json').write_text(json.dumps(checks, indent=2, default=str))
    return checks


def _run_process(variant, case, P, r, out_root, reference_dir, guess, threads, tag='', quad_scale=1.0):
    """One run in its own process, with its BLAS thread count fixed (set
    before numpy loads), so a run's numbers depend on its thread count only,
    never on what else runs beside it."""
    import subprocess
    env = {**os.environ, **{v: str(threads) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    cmd = [sys.executable, os.path.abspath(__file__), 'run', '--variant', variant, '--case', case,
           '--p', str(int(round(math.sqrt(P)))), '--ratio', str(r), '--guess', guess, '--out', str(out_root),
           '--reference-dir', str(reference_dir), '--quad-scale', repr(quad_scale)] + (['--tag', tag] if tag else [])
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    return Path(out_root) / ITEMS[variant] / run_name(case, P, r, guess, tag)


def run_all(out_root, reference_dir, variants=('cheb', 'sine'), cases=tuple(CASES), sizes=SIZES, ratios=RATIOS,
            workers=1, threads=None):
    """Every run, then the other guess where the rule never fires (Section
    8.2); then 2a's a priori constants. ``workers`` runs side by side, each in
    its own process with ``threads`` BLAS threads, the largest first. Returns
    the run folders' names."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out_root, reference_dir = Path(out_root).resolve(), Path(reference_dir).resolve()
    threads = threads or max(1, (os.cpu_count() or 1) // workers)
    tasks = sorted(((v, c, p * p, r) for v in variants for c in cases for p in sizes for r in ratios),
                   key=lambda t: -t[2] * t[3])
    done = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_run_process, v, c, P, r, out_root, reference_dir, CASES[c]['guess'], threads):
                   (v, c, P, r, False) for v, c, P, r in tasks}
        while pending:
            fut = next(as_completed(pending))
            v, c, P, r, is_other = pending.pop(fut)
            d = fut.result()
            done.append(d.name)
            if not is_other and not rule_fires(d):           # Section 8.2: once more from the other guess
                other = OTHER_GUESS[CASES[c]['guess']]
                pending[pool.submit(_run_process, v, c, P, r, out_root, reference_dir, other, threads)] = (
                    v, c, P, r, True)
    if 'cheb' in variants:
        consts = [row for case in cases for p in sizes for r in ratios for row in apriori_constants(case, p * p, r)]
        _write_csv(out_root / ITEMS['cheb'] / 'constants.csv', consts)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 3, item 2: Buckley-Leverett on CC-CGL grids.')
    ap.add_argument('stage', choices=('all', 'run', 'checks', 'summarize'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir', required=True)
    ap.add_argument('--variant', choices=tuple(ITEMS))
    ap.add_argument('--case', choices=tuple(CASES))
    ap.add_argument('--p', type=int, choices=SIZES)
    ap.add_argument('--ratio', type=int, choices=RATIOS)
    ap.add_argument('--guess', choices=('zero', 'ic'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--quad-scale', type=float, default=1.0, help='K5: n_GL times this')
    ap.add_argument('--workers', type=int, default=1, help='all: runs side by side')
    ap.add_argument('--threads', type=int, default=None, help='all, checks: BLAS threads per run')
    args = ap.parse_args(argv)
    if args.stage == 'run':
        print(run(args.variant, args.case, args.p ** 2, args.ratio, args.out, args.reference_dir, guess=args.guess,
                  tag=args.tag, quad_scale=args.quad_scale))
        return
    if args.stage == 'all':
        names = run_all(args.out, args.reference_dir, workers=args.workers, threads=args.threads)
        print(f'{len(names)} runs')
    if args.stage == 'checks':                           # K5 and K6 (Section 7), at the sweep's thread count
        th = args.threads or os.cpu_count() or 1
        out, ref = Path(args.out).resolve(), Path(args.reference_dir).resolve()
        _run_process('cheb', 'viscous', 1024, 10, out, ref, 'ic', th, tag='_k5', quad_scale=1.5)
        _run_process('cheb', 'viscous', 256, 10, out, ref, 'ic', th, tag='_k6')
    checks = summarize(args.out, args.reference_dir)
    for name, c in checks.items():
        print(f"  {name}: {'passed' if c.get('passed') else 'FAILED'}")


if __name__ == '__main__':
    main()
