"""
Package 3, item 1: Beltrami on CC-weighted CGL grids
====================================================

The advisor's instructions of 8 October 2026, Section 3. The paper's Beltrami
trial spaces, linearization, nu, exact solution, lambdas and zero initial
coefficients, on a new collocation set: ``lilq.certified.beltrami_grid(M)``.
- **Interior:** the M^4 tensor CGL grid, with the three momentum equations
  and continuity at every point.
- **Faces:** the M^3 grid on each of the six, with the three velocity
  Dirichlet rows.
- **Initial slab:** the M^3 grid in (x, y, z), with three velocity rows.

**Squared row weights (Section 2.2):**
- interior: lambda x the tensor CC weight;
- each face and the slab: lambda x its CC weight x |block| / |dOmega|
  (4/40 and 8/40). The paper's runs put lambda on each face instead.
- the N_p pressure pins at (-1, -1, -1) keep the paper's lambda_bc / N_p,
  times ``pin_scale`` (check K3). They are gauge rows, outside every constant
  and rho_r.

**Sizes** (Section 3.1; q = 2 N_vel + 1 for the interior constants):

| Size | N_vel | N_p | P | M (levels 1, 2) |
|---|---|---|---|---|
| B1 | 4 | 5 | 1,393 | 13, 16 |
| B2 | 5 | 6 | 3,171 | 15, 18 |
| B3 | 6 | 8 | 7,984 | 16, 19 |

**Iteration:** ``tol = 0``, 8 iterates, all logged. ``iterations.csv`` and
beta^(k+1) are written after every iterate, so a run stopped by its wall cap
still reports what it did.
- The paper's criterion would return the first iterate whose relative
  coefficient change is below 1e-9; it is read from the log.
- The termination rule (n_s = 2, tau_chi = 0.1, tau_r = 0.01) is too, with
  its class, A or C (``experiments.stopping_rule_table.classify``).
- kappa is computed at the last iterate only (Section 2.4): by SVD for
  P < 3,200, otherwise by the pivoted QR of the paper's Beltrami runs. The
  class uses it.
- The tracker is not given ``n_interior_rows`` or ``interior_weight``, which
  assume equal weights (Section 2.4, as P2-16).

**Memory** (Section 3.4, optional): u, v and w share one set of basis
matrices (one basis, one set of points), and A is preallocated and reused,
so no list of row blocks is stacked.

**Per run** (Section 3.5), in ``P3_1_beltrami_certified/<size>_L<level>/``:
- ``run.json`` and ``iterations.csv``;
- ``errors.csv``: the paper's errors (21^3 x 11, the pressure shifted per
  time level), with w, the pin gauge, t = 1 and the combined error, at every
  iterate;
- ``coefficients/beta_<k>.npy``;
- ``rho_r.csv`` for B1: the Y-norm is tensor Gauss-Legendre with 12 points
  per direction, block-matched (Section 6), pins excluded.

The per-grid constants (interior a priori, the 21 face and slab blocks, and
the overall line) go to ``constants_<size>_L<level>.csv``. ``summarize``
gathers them, with ``terminal.csv`` and checks K3, K4, K6 and K7.

Usage::

    python experiments/p3_1_beltrami_certified.py run --size B1 --level 1 --out <root> [--rho] [--pin-scale 1e-3 --tag _pin1e-3]
    python experiments/p3_1_beltrami_certified.py constants --size B1 --level 1 --out <root>
    python experiments/p3_1_beltrami_certified.py summarize --out <root>
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

from lilq import certified as cf
from lilq.basis import create_basis_nd
from lilq.instrumentation import (EPS_MACH, conditioning_via_pivoted_qr, conditioning_via_svd, roundoff_comparison,
                                  stall_rule_index)
from lilq.iteration_log import IterationLogger, LilQDiagnosticsTracker
from lilq.provenance import capture_blas_thread_env
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
from lilq.test_errors import tensor_grid_values
from problems.beltrami import (BeltramiConfig, BeltramiPhysics, _cgl_temporal_pin_nodes, _lstsq,
                               compute_time_snapshot_errors, make_test_error_fn)

ITEM = 'P3_1_beltrami_certified'
INTERVALS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))
SIZES = {'B1': (4, 5, (13, 16)), 'B2': (5, 6, (15, 18)), 'B3': (6, 8, (16, 19))}
K_ITERS = 8
PAPER_TOL = 1e-9
SVD_KAPPA_BELOW = 3200
GL_RHO = 12
EVAL = (21, 11)                                           # the paper's 21^3 x 11 error grid
B2_TOL = 1e-10
TIMING_COLUMNS = ('t_assemble_s', 't_solve_s', 't_cum_s')
NOTE_B2 = ("Package 1's convention (DECISIONS.md, B2 in real runs): the identity is judged at k = 1; near convergence R is a small difference of O(1) quantities and cancellation dominates the relative difference, so the run maximum is reported, not tested")


def config_for(size):
    nv, npr, _ = SIZES[size]
    return BeltramiConfig(N_vel=nv, N_p=npr, basis_type='chebyshev', n_pressure_pin_levels=npr,
                          max_iter=K_ITERS, tol=0.0)


def peak_host_bytes():
    from experiments.p2_1_scaling import peak_host_bytes as peak
    return peak()


# ---------------------------------------------------------------- the system

class System:
    """The linearized Beltrami system on one set of blocks: the CC-CGL
    collocation set, or a Gauss-Legendre Y-quadrature (``pins=False``), with
    each row's square-root weight."""

    def __init__(self, config, blocks, pins=True, pin_scale=1.0):
        self.config = config
        self.physics = phys = BeltramiPhysics(config)
        dom = [config.x_domain, config.y_domain, config.z_domain, config.t_domain]
        self.bv = create_basis_nd(config.basis_type, config.N_vel, dom)     # u, v and w: one basis
        self.bp = create_basis_nd(config.basis_type, config.N_p, dom)
        self.Pv, self.Pp = self.bv.n_basis, self.bp.n_basis
        self.P = 3 * self.Pv + self.Pp
        nu = config.nu
        ib = blocks['interior']
        x, y, z, t = ib['points'].T
        d = lambda bas, *o: bas.derivative(x, y, z, t, orders=list(o))  # noqa: E731
        self.V = self.bv.evaluate(x, y, z, t)
        self.Dx, self.Dy, self.Dz = d(self.bv, 1, 0, 0, 0), d(self.bv, 0, 1, 0, 0), d(self.bv, 0, 0, 1, 0)
        self.C = d(self.bv, 0, 0, 0, 1) - nu * (d(self.bv, 2, 0, 0, 0) + d(self.bv, 0, 2, 0, 0) + d(self.bv, 0, 0, 2, 0))
        self.Px, self.Py, self.Pz = d(self.bp, 1, 0, 0, 0), d(self.bp, 0, 1, 0, 0), d(self.bp, 0, 0, 1, 0)
        s = np.sqrt(ib['share'] * ib['w'])
        self.s_mom, self.s_cont = np.sqrt(config.lambda_mom) * s, np.sqrt(config.lambda_cont) * s
        self.n_int = len(x)
        self.aux = []
        for name, b in blocks.items():
            if name == 'interior':
                continue
            xb, yb, zb, tb = b['points'].T
            lam = config.lambda_ic if name == 'initial' else config.lambda_bc
            self.aux.append({'name': name, 'Phi': self.bv.evaluate(xb, yb, zb, tb),
                             'data': (phys.exact_u(xb, yb, zb, tb), phys.exact_v(xb, yb, zb, tb),
                                      phys.exact_w(xb, yb, zb, tb)),
                             's': np.sqrt(lam * b['share'] * b['w'])})
        self.n_pin = 0
        if pins:
            n = config.n_pressure_pin_levels
            tp = _cgl_temporal_pin_nodes(n, config.t_domain)
            x0, y0, z0 = (np.full(n, dm[0]) for dm in dom[:3])
            self.Phi_pin = self.bp.evaluate(x0, y0, z0, tp)
            self.p_pin = phys.exact_p(x0, y0, z0, tp)
            self.s_pin = np.sqrt(pin_scale * config.lambda_bc / n)
            self.n_pin = n
        self.N = 4 * self.n_int + 3 * sum(len(a['s']) for a in self.aux) + self.n_pin
        self.composition = {'momentum (each of 3)': self.n_int, 'continuity': self.n_int,
                            **{f"{a['name']} (each of u, v, w)": len(a['s']) for a in self.aux},
                            'pressure_pin': self.n_pin}

    def split(self, theta):
        P = self.Pv
        return theta[:P], theta[P:2 * P], theta[2 * P:3 * P], theta[3 * P:]

    def _fields(self, tu, tv, tw):
        f = {}
        for name, th in (('u', tu), ('v', tv), ('w', tw)):
            f[name], f[name + 'x'], f[name + 'y'], f[name + 'z'] = (self.V @ th, self.Dx @ th, self.Dy @ th,
                                                                    self.Dz @ th)
        return f

    def assemble(self, theta, out=None):
        """``(A, b)``: the quasilinearized system at theta, rows weighted, into
        ``out`` when given (N x P, reused across iterations)."""
        N, P, Pv = self.N, self.P, self.Pv
        A = np.zeros((N, P)) if out is None else out
        if out is not None:
            A.fill(0.0)
        b = np.zeros(N)
        tu, tv, tw, _ = self.split(theta)
        f = self._fields(tu, tv, tw)
        u, v, w = f['u'], f['v'], f['w']
        conv = u[:, None] * self.Dx + v[:, None] * self.Dy + w[:, None] * self.Dz + self.C
        n, sm, sc = self.n_int, self.s_mom[:, None], self.s_cont[:, None]
        V = self.V
        cu, cv, cw = slice(0, Pv), slice(Pv, 2 * Pv), slice(2 * Pv, 3 * Pv)
        cp = slice(3 * Pv, P)
        for e, (diag, (gx, gy, gz), Pd) in enumerate((
                ('u', ('ux', 'uy', 'uz'), self.Px), ('v', ('vx', 'vy', 'vz'), self.Py),
                ('w', ('wx', 'wy', 'wz'), self.Pz))):
            r = slice(e * n, (e + 1) * n)
            g = (f[gx], f[gy], f[gz])                     # the gradient of this momentum's velocity
            for c, col in enumerate((cu, cv, cw)):
                blk = g[c][:, None] * V
                if 'uvw'[c] == diag:
                    blk += conv
                A[r, col] = sm * blk
            A[r, cp] = sm * Pd
            b[r] = self.s_mom * (u * g[0] + v * g[1] + w * g[2])
        r = slice(3 * n, 4 * n)
        A[r, cu], A[r, cv], A[r, cw] = sc * self.Dx, sc * self.Dy, sc * self.Dz
        i = 4 * n
        for a in self.aux:
            m, s = len(a['s']), a['s'][:, None]
            for c, col in enumerate((cu, cv, cw)):
                A[i:i + m, col] = s * a['Phi']
                b[i:i + m] = a['s'] * a['data'][c]
                i += m
        if self.n_pin:
            A[i:, cp] = self.s_pin * self.Phi_pin
            b[i:] = self.s_pin * self.p_pin
        return A, b

    def residual(self, theta):
        """The weighted nonlinear residual at theta, written independently of
        :meth:`assemble`, in its row order. Check K4 (B2) compares the two."""
        tu, tv, tw, tp = self.split(theta)
        f = self._fields(tu, tv, tw)
        u, v, w = f['u'], f['v'], f['w']
        r1 = self.C @ tu + u * f['ux'] + v * f['uy'] + w * f['uz'] + self.Px @ tp
        r2 = self.C @ tv + u * f['vx'] + v * f['vy'] + w * f['vz'] + self.Py @ tp
        r3 = self.C @ tw + u * f['wx'] + v * f['wy'] + w * f['wz'] + self.Pz @ tp
        r4 = f['ux'] + f['vy'] + f['wz']
        out = [self.s_mom * r1, self.s_mom * r2, self.s_mom * r3, self.s_cont * r4]
        for a in self.aux:
            for th, data in zip((tu, tv, tw), a['data']):
                out.append(a['s'] * (a['Phi'] @ th - data))
        if self.n_pin:
            out.append(self.s_pin * (self.Phi_pin @ tp - self.p_pin))
        return np.concatenate(out)


# ---------------------------------------------------------------- errors and delta_P

def eval_axes(physics):
    n_s, n_t = EVAL
    return [np.linspace(*physics.x_domain, n_s), np.linspace(*physics.y_domain, n_s),
            np.linspace(*physics.z_domain, n_s), np.linspace(*physics.t_domain, n_t)]


def exact_on_grid(physics):
    X, Y, Z, T = np.meshgrid(*eval_axes(physics), indexing='ij')
    return {'u': physics.exact_u(X, Y, Z, T), 'v': physics.exact_v(X, Y, Z, T),
            'w': physics.exact_w(X, Y, Z, T), 'p': physics.exact_p(X, Y, Z, T)}


def field_errors(sys_, theta, exact):
    """The paper's errors at theta (``problems.beltrami.compute_errors``'s
    metric) with the absolute norms for the combined error, the pin gauge and
    the t = 1 values."""
    axes = eval_axes(sys_.physics)
    tu, tv, tw, tp = sys_.split(theta)
    got = {'u': tensor_grid_values(sys_.bv, tu, axes), 'v': tensor_grid_values(sys_.bv, tv, axes),
           'w': tensor_grid_values(sys_.bv, tw, axes), 'p': tensor_grid_values(sys_.bp, tp, axes)}
    pin_gauge = np.linalg.norm(got['p'] - exact['p']) / np.linalg.norm(exact['p'])
    got['p'] = got['p'] - got['p'].mean(axis=(0, 1, 2)) + exact['p'].mean(axis=(0, 1, 2))
    err = {f: float(np.linalg.norm(got[f] - exact[f])) for f in 'uvwp'}
    norm = {f: float(np.linalg.norm(exact[f])) for f in 'uvwp'}
    row = {f'rel_l2_{f}': err[f] / norm[f] for f in 'uvwp'}
    row['rel_l2_p_pin_gauge'] = float(pin_gauge)
    row['rel_l2_combined'] = math.sqrt(sum(e * e for e in err.values())) / math.sqrt(sum(n * n for n in norm.values()))
    snap = compute_time_snapshot_errors(sys_.physics, sys_.bv, sys_.bv, sys_.bv, sys_.bp, tu, tv, tw, tp,
                                        t_vals=(1.0,))[0]
    row.update({f't1_{f}': snap[f] for f in ('u', 'v', 'w', 'p', 'p_pin_gauge')})
    return row, err, norm


def delta_P(sys_, exact):
    """The relative L2 distance of each exact field to the span of its basis,
    by least squares on the error grid. For p, 11 columns constant on each
    time level are added, matching the per-level pressure shift."""
    axes = eval_axes(sys_.physics)
    X, Y, Z, T = (g.ravel() for g in np.meshgrid(*axes, indexing='ij'))
    out, res = {}, {}
    Phi = sys_.bv.evaluate(X, Y, Z, T)
    F = np.column_stack([exact[f].ravel() for f in 'uvw'])
    coef = sla.lstsq(Phi, F, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)[0]
    for j, f in enumerate('uvw'):
        res[f] = float(np.linalg.norm(Phi @ coef[:, j] - F[:, j]))
    del Phi, F
    levels = (T[:, None] == axes[3][None, :]).astype(float)       # 11 columns, constant on each time level
    Phi = np.hstack([sys_.bp.evaluate(X, Y, Z, T), levels])
    fp = exact['p'].ravel()
    c = sla.lstsq(Phi, fp, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)[0]
    res['p'] = float(np.linalg.norm(Phi @ c - fp))
    norm = {f: float(np.linalg.norm(exact[f])) for f in 'uvwp'}
    out = {f'delta_P_{f}': res[f] / norm[f] for f in 'uvwp'}
    out['delta_P_combined'] = math.sqrt(sum(r * r for r in res.values())) / math.sqrt(sum(n * n for n in norm.values()))
    return out, res


# ---------------------------------------------------------------- constants

def constants(size, level):
    """The interior a priori constants (Kronecker, Q_q with q = 2 N_vel + 1)
    and the 21 face and slab blocks (Section 2.3(b)); the overall line is
    min c1 and max c2 over them."""
    nv, _, Ms = SIZES[size]
    M = Ms[level - 1]
    g = cf.beltrami_grid(M, INTERVALS)
    phys = BeltramiPhysics(config_for(size))
    k = cf.interior_constants_kronecker(M, 2 * nv + 1, 4)
    rows = [{'size': size, 'level': level, 'M': M, 'block': 'interior', 'component': 'all', 'c1': k['c1'],
             'c2': k['c2'], 'c2_over_c1': k['c2_over_c1'], 'space': f'Q_q, q = {2 * nv + 1} (a priori, Kronecker)',
             'columns': k['dim_ambient'], 'dropped': 0, 'n_gl': ''}]
    for name, b in g['blocks'].items():
        if name == 'interior':
            continue
        free, fixed = list(b['free']), [i for i in range(4) if i not in b['free']][0]
        value = b['points'][0, fixed]
        for comp, ex in (('u', phys.exact_u), ('v', phys.exact_v), ('w', phys.exact_w)):
            def datum(zf, ex=ex):
                pts = np.empty((len(zf), 4))
                pts[:, free] = zf
                pts[:, fixed] = value
                return ex(*pts.T)
            r = cf.block_constants(b['points'][:, free], b['w'], [INTERVALS[i] for i in free], nv, data=datum)
            rows.append({'size': size, 'level': level, 'M': M, 'block': name, 'component': comp, 'c1': r['c1'],
                         'c2': r['c2'], 'c2_over_c1': r['c2_over_c1'],
                         'space': f'traces ({nv}^3 Chebyshev) + datum', 'columns': r['columns'],
                         'dropped': r['dropped'], 'n_gl': r['n_gl']})
    c1, c2 = min(r['c1'] for r in rows), max(r['c2'] for r in rows)
    rows.append({'size': size, 'level': level, 'M': M, 'block': 'overall', 'component': 'all', 'c1': c1, 'c2': c2,
                 'c2_over_c1': c2 / c1, 'space': 'min c1, max c2 over the interior and the 21 blocks',
                 'columns': '', 'dropped': sum(r['dropped'] for r in rows), 'n_gl': ''})
    return rows


# ---------------------------------------------------------------- one run

def _write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rho_r(sys_y, theta_k, theta_new):
    """rho_r at the linearization theta_k (Section 6): the Y-residual of the
    collocation minimizer over the smallest attainable, both on the
    block-matched Gauss-Legendre quadrature without pins."""
    A, b = sys_y.assemble(theta_k)
    best_coef, _, rank, _ = sla.lstsq(A, b, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)
    best = float(np.linalg.norm(A @ best_coef - b))
    got = float(np.linalg.norm(A @ theta_new - b))
    cond = conditioning_via_svd(A)
    kappa = cond['kappa_retained']
    floor = 100 * EPS_MACH * kappa * float(np.linalg.norm(b))
    return {'rho_r': got / best if best > 0 else float('inf'), 'Y_residual_collocation': got, 'Y_residual_best': best,
            'kappa_retained_A_Y': kappa, 'rank_A_Y': int(rank), 'roundoff_floor': floor, 'round_off': best < floor}


def run(size, level, out_root, pin_scale=1.0, tag='', with_rho=None, k_iters=K_ITERS):
    nv, npr, Ms = SIZES[size]
    M = Ms[level - 1]
    with_rho = (size == 'B1') if with_rho is None else with_rho
    config = config_for(size)
    t_start = time.perf_counter()
    g = cf.beltrami_grid(M, INTERVALS)
    sys_ = System(config, g['blocks'], pins=True, pin_scale=pin_scale)
    sys_y = System(config, cf.beltrami_grid(GL_RHO, INTERVALS, rule='gauss')['blocks'], pins=False) if with_rho else None
    run_dir = Path(out_root) / ITEM / f'{size}_L{level}{tag}'
    (run_dir / 'coefficients').mkdir(parents=True, exist_ok=True)
    base = {'package3': {'item': 1, 'size': size, 'level': level, 'M': M, 'N_vel': nv, 'N_p': npr,
                         'pin_scale': pin_scale, 'status': 'running', 'N': sys_.N, 'P': sys_.P,
                         'weights_note': 'Clenshaw-Curtis squared row weights (Section 2.2): interior lambda x tensor CC '
                         'weight; each face and the initial slab lambda x CC weight x |block|/|dOmega| (4/40, 8/40). The '
                         "paper's runs put lambda on each face instead; accuracy differences from its tables are "
                         'partly due to this.'}}
    (run_dir / 'run.json').write_text(json.dumps(base, indent=2))
    exact = exact_on_grid(sys_.physics)
    tracker = LilQDiagnosticsTracker(conditioning_svd_threshold=-1,     # kappa at the last iterate only, below
                                     test_error_fn=make_test_error_fn(sys_.physics, sys_.bv, sys_.bv, sys_.bv, sys_.bp))
    logger = IterationLogger()
    theta = np.zeros(sys_.P)
    A_buf = np.empty((sys_.N, sys_.P))
    errors, rho, kappa_s, ranks = [], [], None, []
    for k in range(k_iters):
        t0 = time.perf_counter()
        A, b = sys_.assemble(theta, out=A_buf)
        t1 = time.perf_counter()
        new, rank, _ = _lstsq(A, b)
        t2 = time.perf_counter()
        ranks.append(int(rank))
        R_new = sys_.residual(new)
        row = tracker.step(k=k, A_stacked=A, b_stacked=b, beta_prev=theta, beta_new=new,
                           total_loss=float(R_new @ R_new), rank_gelsy=rank, t_assemble_s=t1 - t0, t_solve_s=t2 - t1,
                           is_final_iterate=False, compute_residual_vector_fn=sys_.residual)
        if k == k_iters - 1:                              # kappa at the last iterate (Section 2.4)
            tk = time.perf_counter()
            cond = conditioning_via_svd(A) if sys_.P < SVD_KAPPA_BELOW else conditioning_via_pivoted_qr(A)
            kappa_s = time.perf_counter() - tk
            row.update(kappa=cond['kappa'], kappa_method=cond['kappa_method'], num_rank_svd=cond.get('num_rank_svd'),
                       kappa_raw=cond.get('kappa_raw'), kappa_retained=cond.get('kappa_retained'),
                       num_rank_qr=cond.get('num_rank_qr'),
                       kappa_eps=roundoff_comparison(row['norm_Rlin_h'], row['norm_f_h'], cond['kappa'])[1])
        logger.record(**row)
        logger.to_csv(run_dir / 'iterations.csv')
        np.save(run_dir / 'coefficients' / f'beta_{k + 1}.npy', new)
        e, _, _ = field_errors(sys_, new, exact)
        errors.append({'iterate': k + 1, **e})
        _write_csv(run_dir / 'errors.csv', errors)
        if sys_y is not None:
            rho.append({'k': k, **rho_r(sys_y, theta, new)})
            _write_csv(run_dir / 'rho_r.csv', rho)
        theta = new
    logger.record(**tracker.finish(k=k_iters))
    logger.to_csv(run_dir / 'iterations.csv')
    wall = time.perf_counter() - t_start
    del A_buf
    thread_env = capture_blas_thread_env()
    meta = build_run_metadata(
        N_total=sys_.N, N_composition=sys_.composition, P_total=sys_.P,
        P_composition={'u': sys_.Pv, 'v': sys_.Pv, 'w': sys_.Pv, 'p': sys_.Pp},
        row_weights='Clenshaw-Curtis (Package 3, Section 2.2); pins sqrt(pin_scale x lambda_bc / N_p)',
        collocation_construction={'method': 'tensor CGL with Clenshaw-Curtis weights (Package 3, Section 3.2)',
                                  'M': M, 'blocks': {n: len(b['points']) for n, b in g['blocks'].items()},
                                  'boundary_measure': g['boundary_measure']},
        basis_description={'family': 'tensor Chebyshev', 'N_vel': nv, 'N_p': npr},
        initial_coefficients='zero', solver_driver='gelsy', rcond=EPS_MACH,
        stopping_rule={'type': 'none: 8 iterates (tol = 0); the 1e-9 criterion and the termination rule are read '
                               'from the log'}, K_max=k_iters, stopping_reason='iteration_cap',
        first_stall_iteration=first_stall_iteration(logger.rows), b2_check=tracker.b2_check,
        kappa_qr_raw_ratio=None, device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1))
    base['package3'].update(status='complete', wall_time_s=wall, kappa_time_s=kappa_s,
                            seconds_per_iteration=float(np.mean([r['t_assemble_s'] + r['t_solve_s']
                                                                 for r in logger.rows[:k_iters]])),
                            peak_host_bytes=peak_host_bytes(), ranks=ranks,
                            b2_rel_err_by_k={int(kk): v for kk, v in tracker._b2_rel_err_by_k.items()})
    write_run_json(run_dir / 'run.json', {**meta, **base})
    return run_dir


# ---------------------------------------------------------------- the summary

def _floats(rows, col):
    return np.array([np.nan if r.get(col) in ('', None) else float(r[col]) for r in rows])


def terminal_row(run_dir, exact_cache={}):
    meta = json.loads((run_dir / 'run.json').read_text())
    p3 = meta['package3']
    size, level = p3['size'], p3['level']
    with open(run_dir / 'iterations.csv') as fh:
        log = list(csv.DictReader(fh))
    with open(run_dir / 'errors.csv') as fh:
        errs = {int(r['iterate']): {k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)}
    K = sum(1 for r in log if r.get('rel_dbeta') not in ('', None))           # solves done
    K = min(K, max(errs, default=0))       # a run stopped by its cap (Section 8.2) may lack its last iterate's errors
    rel = _floats(log, 'rel_dbeta')[:K]
    k_paper = next((k + 1 for k in range(K) if rel[k] < PAPER_TOL), None)
    i = stall_rule_index(_floats(log, 'chi')[:K], _floats(log, 'norm_Rlin_h')[:K])
    last = log[K - 1]
    kappa_ret = float(last['kappa_retained'] or last['kappa'] or 'nan') if K == K_ITERS else float('nan')
    cls = ''
    if i is not None and np.isfinite(kappa_ret):
        cls = 'C' if float(log[i]['roundoff_ratio']) < 10 * kappa_ret * EPS_MACH else 'A'
    returned = k_paper if k_paper is not None else K
    config = config_for(size)
    key = (size,)
    if key not in exact_cache:
        sys_ = System(config, cf.beltrami_grid(3, INTERVALS)['blocks'], pins=False)   # bases only
        ex = exact_on_grid(sys_.physics)
        exact_cache[key] = (sys_, ex, delta_P(sys_, ex))
    sys_, ex, (dp, dp_res) = exact_cache[key]
    e_ret = errs[returned]
    ranks = p3.get('ranks') or [int(float(r['num_rank_gelsy'])) for r in log[:K]   # complete runs record them;
                                if r.get('num_rank_gelsy') not in ('', None)]     # a capped one, its log
    row = {'size': size, 'level': level, 'M': p3['M'], 'pin_scale': p3['pin_scale'], 'N': p3['N'], 'P': p3['P'],
           'N_over_P': p3['N'] / p3['P'], 'iterates': K,
           'k_paper_1e9': k_paper if k_paper is not None else 'never',
           'k_rule': i + 1 if i is not None else 'never', 'rule_class': cls,
           'rank_last': last['num_rank_gelsy'], 'rank_min': min(ranks),
           'kappa_last': last['kappa'], 'kappa_retained_last': last['kappa_retained'],
           'kappa_method': last['kappa_method'],
           **{f'{k}_at_returned': v for k, v in e_ret.items() if k != 'iterate'}, 'returned_iterate': returned,
           'rel_l2_combined_at_rule': errs[i + 1]['rel_l2_combined'] if i is not None else '',
           'rel_l2_combined_at_last': errs[K]['rel_l2_combined'] if K in errs else '', **dp,
           **{f'error_over_delta_P_{f}': e_ret[f'rel_l2_{f}'] / dp[f'delta_P_{f}'] for f in 'uvwp'},
           'error_over_delta_P_combined': e_ret['rel_l2_combined'] / dp['delta_P_combined'],
           'b2_rel_err_k1': (meta.get('b2_check') or {}).get('rel_err'),
           'b2_max_rel_err': (meta.get('b2_check') or {}).get('max_rel_err_over_run'),
           'peak_host_GB': (p3.get('peak_host_bytes') or 0) / 1e9, 'wall_time_s': p3.get('wall_time_s'),
           'seconds_per_iteration': p3.get('seconds_per_iteration'), 'kappa_time_s': p3.get('kappa_time_s'),
           'status': p3['status']}
    return row


def summarize(out_root):
    out = Path(out_root) / ITEM
    runs = sorted(d for d in out.iterdir() if d.is_dir() and (d / 'iterations.csv').exists()
                  and (d / 'errors.csv').exists())        # written after an iterate's log row: one iterate at least
    terminal = [terminal_row(d) for d in runs]
    _write_csv(out / 'terminal.csv', terminal)
    consts = []
    for f in sorted(out.glob('constants_*.csv')):
        with open(f) as fh:
            consts += list(csv.DictReader(fh))
    if consts:
        _write_csv(out / 'constants.csv', consts)
    checks = {}
    by_name = {d.name: r for d, r in zip(runs, terminal)}
    base = by_name.get('B1_L1')                           # K3's reference: B1 level 1 at pin_scale 1
    k3 = []
    for d, r in zip(runs, terminal):
        if d.name.startswith('B1_L1_pin') and base is not None:
            diff = max(abs(float(r[f'rel_l2_{f}_at_returned']) - float(base[f'rel_l2_{f}_at_returned']))
                       / float(base[f'rel_l2_{f}_at_returned']) for f in 'uvwp')
            k3.append({'run': d.name, 'pin_scale': r['pin_scale'], 'max_rel_diff_errors': diff, 'passed': diff < 1e-6})
    checks['K3'] = {'runs': k3, 'passed': all(x['passed'] for x in k3) if k3 else None,      # None: not run
                    'note': 'velocity and gauge-corrected pressure errors at the returned iterate, against pin_scale 1'}
    k1 = [float(r['b2_rel_err_k1']) for r in terminal if r['b2_rel_err_k1'] is not None]
    checks['K4'] = {'max_rel_err_at_k1': max(k1) if k1 else None, 'tolerance': B2_TOL,
                    'max_rel_err_over_runs': max((float(r['b2_max_rel_err']) for r in terminal
                                                  if r['b2_max_rel_err'] is not None), default=None),
                    'passed': bool(k1) and max(k1) < B2_TOL, 'rule': NOTE_B2}
    checks['K7'] = {'runs': {d.name: {'rank_min': r['rank_min'], 'P': r['P']} for d, r in zip(runs, terminal)}}
    checks['K7']['passed'] = all(int(v['rank_min']) == int(v['P']) for v in checks['K7']['runs'].values())
    rerun = out / 'B1_L1_rerun'
    if rerun.exists() and (out / 'B1_L1').exists():
        checks['K6'] = compare_logs(out / 'B1_L1', rerun)
    (out / 'checks_item1.json').write_text(json.dumps(checks, indent=2))
    return terminal, consts, checks


def compare_logs(a, b):
    """Check K6: the two runs' iterations.csv agree in every column but the
    timings, which no rerun can reproduce (and errors.csv entirely)."""
    def load(d, name):
        with open(d / name) as fh:
            return list(csv.DictReader(fh))
    la, lb = load(a, 'iterations.csv'), load(b, 'iterations.csv')
    diff = [(i, c) for i, (ra, rb) in enumerate(zip(la, lb)) for c in ra if c not in TIMING_COLUMNS and ra[c] != rb[c]]
    same_errors = load(a, 'errors.csv') == load(b, 'errors.csv')
    coef = all(np.array_equal(np.load(p), np.load(b / 'coefficients' / p.name))
               for p in sorted((a / 'coefficients').glob('beta_*.npy')))
    return {'identical_except_timings': not diff and len(la) == len(lb), 'differing': diff[:20],
            'errors_identical': same_errors, 'coefficients_identical': coef,
            'passed': not diff and len(la) == len(lb) and same_errors and coef,
            'excluded_columns': list(TIMING_COLUMNS)}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 3, item 1: Beltrami on CC-CGL grids.')
    ap.add_argument('stage', choices=('run', 'constants', 'summarize'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', choices=tuple(SIZES))
    ap.add_argument('--level', type=int, choices=(1, 2))
    ap.add_argument('--pin-scale', type=float, default=1.0, help='K3: the pins squared weight x this')
    ap.add_argument('--tag', default='', help='a suffix for the run folder (K3, K6)')
    ap.add_argument('--rho', action='store_true', default=None, help='rho_r at every iterate (default: B1 only)')
    args = ap.parse_args(argv)
    if args.stage == 'run':
        d = run(args.size, args.level, args.out, pin_scale=args.pin_scale, tag=args.tag, with_rho=args.rho)
        print(f'wrote {d}')
    elif args.stage == 'constants':
        rows = constants(args.size, args.level)
        out = Path(args.out) / ITEM
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / f'constants_{args.size}_L{args.level}.csv', rows)
        o = rows[-1]
        print(f"{args.size} L{args.level}: overall c1 {o['c1']:.4f}, c2 {o['c2']:.4f}, ratio {o['c2_over_c1']:.4f}")
    else:
        terminal, _, checks = summarize(args.out)
        for r in terminal:
            print(f"  {r['size']} L{r['level']} pin x{r['pin_scale']} ({r['status']}, {r['iterates']} iterates): "
                  f"N/P {r['N_over_P']:.1f}, 1e-9 at "
                  f"{r['k_paper_1e9']}, rule {r['k_rule']} ({r['rule_class']}), combined error "
                  f"{r['rel_l2_combined_at_returned']:.2e}, error/delta_P {r['error_over_delta_P_combined']:.2f}")
        for name, c in checks.items():
            print(f"  {name}: {'not run' if c.get('passed') is None else 'passed' if c['passed'] else 'FAILED'}")


if __name__ == '__main__':
    main()
