"""
Beltrami 3D Flow Problem (LiL-Q Only)
========================================

3D unsteady incompressible Navier-Stokes, Ethier-Steinman benchmark.

Equations:
    u_t + u*u_x + v*u_y + w*u_z + p_x - nu*(u_xx+u_yy+u_zz) = 0
    v_t + u*v_x + v*v_y + w*v_z + p_y - nu*(v_xx+v_yy+v_zz) = 0
    w_t + u*w_x + v*w_y + w*w_z + p_z - nu*(w_xx+w_yy+w_zz) = 0
    u_x + v_y + w_z = 0

Exact solution (Ethier & Steinman):
    u = -a*(exp(a*x)*sin(a*y+d*z) + exp(a*z)*cos(a*x+d*y))*exp(-d^2*t)
    v = -a*(exp(a*y)*sin(a*z+d*x) + exp(a*x)*cos(a*y+d*z))*exp(-d^2*t)
    w = -a*(exp(a*z)*sin(a*x+d*y) + exp(a*y)*cos(a*z+d*x))*exp(-d^2*t)

Multi-field LiL-Q with 4D tensor product bases.
"""

import numpy as np
import os
import scipy.linalg
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List

from lilq.basis import TensorProductBasisND, create_basis_nd, Chebyshev1D, Fourier1D
from lilq.instrumentation import EPS_MACH
from lilq.iteration_log import IterationLogger, LilQDiagnosticsTracker
from lilq.provenance import capture_blas_thread_env
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
from lilq.test_errors import max_abs, rel_l2, tensor_grid_values

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeltramiConfig:
    # Physics
    a: float = 1.0
    d: float = 1.0
    nu: float = 1.0
    x_domain: Tuple[float, float] = (-1.0, 1.0)
    y_domain: Tuple[float, float] = (-1.0, 1.0)
    z_domain: Tuple[float, float] = (-1.0, 1.0)
    t_domain: Tuple[float, float] = (0.0, 1.0)
    # Discretization — N_vel for u,v,w; N_p for pressure (can be larger)
    N_vel: int = 6
    N_p: int = 8
    # Collocation (per-dim interior, boundary, IC counts)
    N_x: int = 8
    N_y: int = 8
    N_z: int = 8
    N_t: int = 8
    N_bc: int = 6
    N_t_bc: int = 6
    N_ic: int = 8
    seed: int = 42
    basis_type: str = 'chebyshev'
    # Section 3.7: number of temporal levels at which pressure is pinned
    # (Chebyshev-Gauss-Lobatto nodes in t_domain). 1 (default) reproduces
    # the original single-pin-at-t=0 behavior exactly -- see
    # _cgl_temporal_pin_nodes and DECISIONS.md.
    n_pressure_pin_levels: int = 1
    # Solver
    max_iter: int = 20
    tol: float = 1e-9
    lambda_mom: float = 1.0
    lambda_cont: float = 1.0
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0
    use_gpu: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

class BeltramiPhysics:
    def __init__(self, config: BeltramiConfig):
        self.a, self.d, self.nu = config.a, config.d, config.nu
        self.x_domain = config.x_domain
        self.y_domain = config.y_domain
        self.z_domain = config.z_domain
        self.t_domain = config.t_domain

    def exact_u(self, x, y, z, t):
        a, d = self.a, self.d
        return -a * (np.exp(a*x)*np.sin(a*y+d*z) + np.exp(a*z)*np.cos(a*x+d*y)) * np.exp(-d**2*t)

    def exact_v(self, x, y, z, t):
        a, d = self.a, self.d
        return -a * (np.exp(a*y)*np.sin(a*z+d*x) + np.exp(a*x)*np.cos(a*y+d*z)) * np.exp(-d**2*t)

    def exact_w(self, x, y, z, t):
        a, d = self.a, self.d
        return -a * (np.exp(a*z)*np.sin(a*x+d*y) + np.exp(a*y)*np.cos(a*z+d*x)) * np.exp(-d**2*t)

    def exact_p(self, x, y, z, t):
        a, d = self.a, self.d
        return (-0.5*a**2
                * (np.exp(2*a*x) + np.exp(2*a*y) + np.exp(2*a*z)
                   + 2*np.sin(a*x+d*y)*np.cos(a*z+d*x)*np.exp(a*(y+z))
                   + 2*np.sin(a*y+d*z)*np.cos(a*x+d*y)*np.exp(a*(z+x))
                   + 2*np.sin(a*z+d*x)*np.cos(a*y+d*z)*np.exp(a*(x+y)))
                * np.exp(-2*d**2*t))


# ─────────────────────────────────────────────────────────────────────────────
# Collocation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_collocation(config: BeltramiConfig, physics: BeltramiPhysics, P_total):
    eps = 1e-6
    xd, yd, zd, td = physics.x_domain, physics.y_domain, physics.z_domain, physics.t_domain

    xs = np.linspace(xd[0]+eps, xd[1]-eps, config.N_x)
    ys = np.linspace(yd[0]+eps, yd[1]-eps, config.N_y)
    zs = np.linspace(zd[0]+eps, zd[1]-eps, config.N_z)
    ts = np.linspace(td[0]+eps, td[1]-eps, config.N_t)
    Xg, Yg, Zg, Tg = np.meshgrid(xs, ys, zs, ts, indexing='ij')
    x_pde = Xg.ravel(); y_pde = Yg.ravel(); z_pde = Zg.ravel(); t_pde = Tg.ravel()
    n_pde = len(x_pde)

    def _face(free_doms):
        d0 = np.linspace(free_doms[0][0], free_doms[0][1], config.N_bc)
        d1 = np.linspace(free_doms[1][0], free_doms[1][1], config.N_bc)
        tt = np.linspace(td[0], td[1], config.N_t_bc)
        D0, D1, TT = np.meshgrid(d0, d1, tt, indexing='ij')
        return D0.ravel(), D1.ravel(), TT.ravel()

    bc = {}
    for label, val in [('x_lo', xd[0]), ('x_hi', xd[1])]:
        d0, d1, tt = _face([yd, zd])
        bc[label] = (np.full_like(d0, val), d0, d1, tt)
    for label, val in [('y_lo', yd[0]), ('y_hi', yd[1])]:
        d0, d1, tt = _face([xd, zd])
        bc[label] = (d0, np.full_like(d0, val), d1, tt)
    for label, val in [('z_lo', zd[0]), ('z_hi', zd[1])]:
        d0, d1, tt = _face([xd, yd])
        bc[label] = (d0, d1, np.full_like(d0, val), tt)
    n_bc = sum(len(v[0]) for v in bc.values())

    xi = np.linspace(xd[0], xd[1], config.N_ic)
    yi = np.linspace(yd[0], yd[1], config.N_ic)
    zi = np.linspace(zd[0], zd[1], config.N_ic)
    Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing='ij')
    x_ic, y_ic, z_ic = Xi.ravel(), Yi.ravel(), Zi.ravel()
    t_ic = np.zeros_like(x_ic)

    return {
        'x_pde': x_pde, 'y_pde': y_pde, 'z_pde': z_pde, 't_pde': t_pde,
        'n_pde': n_pde, 'bc': bc, 'n_bc_total': n_bc,
        'x_ic': x_ic, 'y_ic': y_ic, 'z_ic': z_ic, 't_ic': t_ic, 'n_ic': len(x_ic),
    }


def _cgl_temporal_pin_nodes(n_pin, t_domain):
    """Section 3.7: ``n_pin`` pressure-pin times in ``t_domain``.

    ``n_pin == 1`` (the pre-3.7 default) returns exactly ``[t_domain[0]]``,
    bit-identical to the original single-pin-at-t0 behavior -- no
    Chebyshev machinery involved, by construction. ``n_pin >= 2`` returns
    the standard Chebyshev-Gauss-Lobatto nodes ``cos(j*pi/(n_pin-1))``,
    ``j=0..n_pin-1``, affinely mapped from ``[-1,1]`` onto ``t_domain``.
    """
    if n_pin == 1:
        return np.array([t_domain[0]])
    j = np.arange(n_pin)
    xi = np.cos(j * np.pi / (n_pin - 1))
    a, b = t_domain
    return 0.5 * (b - a) * (xi + 1.0) + a


def _lstsq(A, b, use_gpu=False):
    if use_gpu and HAS_TORCH_CUDA:
        At = torch.as_tensor(A, dtype=torch.float64, device='cuda')
        bt = torch.as_tensor(b, dtype=torch.float64, device='cuda').unsqueeze(1)
        result = torch.linalg.lstsq(At, bt, driver='gelsd')
        x = result.solution.squeeze(1).cpu().numpy()
        rank = int(result.rank) if result.rank is not None else At.shape[1]
        return x, rank
    else:
        x, _, rank, _ = scipy.linalg.lstsq(A, b, cond=EPS_MACH, lapack_driver='gelsy')
        return x, rank


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def _make_beltrami_nonlinear_loss_fn(
    Mu, Mv, Mw, Mp, bc_blocks, ic_block, Phi_p_pin, p_pin_val,
    nu, Pu, Pv, Pw, Pp, lambda_mom, lambda_cont, lambda_bc, lambda_ic,
):
    """Scalar total-loss evaluator for Beltrami's Section 3.1
    instrumentation -- the MSE-based analogue of Kovasznay's
    ``_make_kovasznay_nonlinear_loss_fn`` (``problems/kovasznay.py``),
    generalized to Beltrami's 4-field (u,v,w,p), 4D (x,y,z,t)
    momentum/continuity/BC/IC/pin block structure. Not used by
    ``solve_beltrami``'s own convergence check (``rel_delta`` on the
    coefficients, unchanged) -- only feeds ``iterations.csv``'s
    ``norm_R_h`` when ``iteration_logger`` is given.

    Returns ``(theta) -> total_loss`` where ``theta`` is the concatenated
    ``[theta_u; theta_v; theta_w; theta_p]`` coefficient vector.
    """
    def compute_loss(theta):
        tu = theta[:Pu]
        tv = theta[Pu:Pu + Pv]
        tw = theta[Pu + Pv:Pu + Pv + Pw]
        tp = theta[Pu + Pv + Pw:]

        u = Mu['val'] @ tu; ux = Mu['dx'] @ tu; uy = Mu['dy'] @ tu; uz = Mu['dz'] @ tu
        v = Mv['val'] @ tv; vx = Mv['dx'] @ tv; vy = Mv['dy'] @ tv; vz = Mv['dz'] @ tv
        w = Mw['val'] @ tw; wx = Mw['dx'] @ tw; wy = Mw['dy'] @ tw; wz = Mw['dz'] @ tw
        ut = Mu['dt'] @ tu; vt = Mv['dt'] @ tv; wt = Mw['dt'] @ tw
        px = Mp['dx'] @ tp; py = Mp['dy'] @ tp; pz = Mp['dz'] @ tp
        lap_u = (Mu['dxx'] + Mu['dyy'] + Mu['dzz']) @ tu
        lap_v = (Mv['dxx'] + Mv['dyy'] + Mv['dzz']) @ tv
        lap_w = (Mw['dxx'] + Mw['dyy'] + Mw['dzz']) @ tw

        r1 = ut + u * ux + v * uy + w * uz + px - nu * lap_u
        r2 = vt + u * vx + v * vy + w * vz + py - nu * lap_v
        r3 = wt + u * wx + v * wy + w * wz + pz - nu * lap_w
        r4 = ux + vy + wz

        bc_mse = 0.0
        for blk in bc_blocks.values():
            bc_mse += float(np.mean((blk['Phi_u'] @ tu - blk['u_ex']) ** 2))
            bc_mse += float(np.mean((blk['Phi_v'] @ tv - blk['v_ex']) ** 2))
            bc_mse += float(np.mean((blk['Phi_w'] @ tw - blk['w_ex']) ** 2))

        ic_mse = (
            float(np.mean((ic_block['Phi_u'] @ tu - ic_block['u_ex']) ** 2))
            + float(np.mean((ic_block['Phi_v'] @ tv - ic_block['v_ex']) ** 2))
            + float(np.mean((ic_block['Phi_w'] @ tw - ic_block['w_ex']) ** 2))
        )

        # Mean-of-squares over all n_pin pin rows (Section 3.7 generalizes
        # a single pin to n_pin>=1 Chebyshev-Gauss-Lobatto temporal
        # levels). At n_pin=1 this is exactly pin_res**2 for the one row,
        # bit-identical to the pre-3.7 scalar form.
        pin_mse = float(np.mean((Phi_p_pin @ tp - p_pin_val) ** 2))

        total = (
            lambda_mom * (float(np.mean(r1 ** 2)) + float(np.mean(r2 ** 2)) + float(np.mean(r3 ** 2)))
            + lambda_cont * float(np.mean(r4 ** 2))
            + lambda_bc * bc_mse
            + lambda_ic * ic_mse
            + lambda_bc * pin_mse
        )
        return total

    return compute_loss


def _make_beltrami_residual_vector_fn(
    Mu, Mv, Mw, Mp, bc_blocks, ic_block, Phi_p_pin, p_pin_val,
    nu, Pu, Pv, Pw, Pp, n_pde, lambda_mom, lambda_cont, lambda_bc, lambda_ic,
):
    """Weighted nonlinear residual **vector** for Section 3.1's phase
    indicator -- the vector form of
    :func:`_make_beltrami_nonlinear_loss_fn`, stacked and weighted
    identically to ``solve_beltrami``'s own ``A_sys``/``b_sys`` assembly
    (x/y/z-momentum, continuity, then BC-u/v/w per face, then IC-u/v/w,
    then the pressure pin), by construction: ``norm(vector)**2 == total``
    from the loss function above -- independently re-derived here (not
    shared code with the assembly loop or the loss function above) so
    check B2 is a real cross-check, not a tautology.

    Returns ``(theta) -> weighted_residual_vector``.
    """
    w_mom = np.sqrt(lambda_mom / n_pde)
    w_cont = np.sqrt(lambda_cont / n_pde)
    # "One-scalar-per-block" convention generalized to n_pin>=1 pin rows
    # (sqrt(lambda_bc/n_pin)) -- reduces to sqrt(lambda_bc) at n_pin=1.
    w_pin = np.sqrt(lambda_bc / Phi_p_pin.shape[0])

    def compute_residual_vector(theta):
        tu = theta[:Pu]
        tv = theta[Pu:Pu + Pv]
        tw = theta[Pu + Pv:Pu + Pv + Pw]
        tp = theta[Pu + Pv + Pw:]

        u = Mu['val'] @ tu; ux = Mu['dx'] @ tu; uy = Mu['dy'] @ tu; uz = Mu['dz'] @ tu
        v = Mv['val'] @ tv; vx = Mv['dx'] @ tv; vy = Mv['dy'] @ tv; vz = Mv['dz'] @ tv
        w = Mw['val'] @ tw; wx = Mw['dx'] @ tw; wy = Mw['dy'] @ tw; wz = Mw['dz'] @ tw
        ut = Mu['dt'] @ tu; vt = Mv['dt'] @ tv; wt = Mw['dt'] @ tw
        px = Mp['dx'] @ tp; py = Mp['dy'] @ tp; pz = Mp['dz'] @ tp
        lap_u = (Mu['dxx'] + Mu['dyy'] + Mu['dzz']) @ tu
        lap_v = (Mv['dxx'] + Mv['dyy'] + Mv['dzz']) @ tv
        lap_w = (Mw['dxx'] + Mw['dyy'] + Mw['dzz']) @ tw

        r1 = ut + u * ux + v * uy + w * uz + px - nu * lap_u
        r2 = vt + u * vx + v * vy + w * vz + py - nu * lap_v
        r3 = wt + u * wx + v * wy + w * wz + pz - nu * lap_w
        r4 = ux + vy + wz

        blocks = [w_mom * r1, w_mom * r2, w_mom * r3, w_cont * r4]
        for blk in bc_blocks.values():
            ne = blk['n']
            w_bc = np.sqrt(lambda_bc / ne)
            blocks.append(w_bc * (blk['Phi_u'] @ tu - blk['u_ex']))
            blocks.append(w_bc * (blk['Phi_v'] @ tv - blk['v_ex']))
            blocks.append(w_bc * (blk['Phi_w'] @ tw - blk['w_ex']))

        ni = ic_block['n']
        w_ic = np.sqrt(lambda_ic / ni)
        blocks.append(w_ic * (ic_block['Phi_u'] @ tu - ic_block['u_ex']))
        blocks.append(w_ic * (ic_block['Phi_v'] @ tv - ic_block['v_ex']))
        blocks.append(w_ic * (ic_block['Phi_w'] @ tw - ic_block['w_ex']))

        blocks.append(w_pin * (Phi_p_pin @ tp - p_pin_val))

        return np.concatenate(blocks)

    return compute_residual_vector


def make_test_error_fn(physics, basis_u, basis_v, basis_w, basis_p, n_s=21, n_t=11):
    """``beta -> eps_u/eps_v/eps_p/maxerr_*`` with the paper's own metric
    (:func:`compute_errors`: an n_s^3 x n_t uniform space-time grid, the
    pressure shifted to the exact mean at every time level) -- Section 2
    says "Beltrami as in the paper". The schema has no ``eps_w``; w's
    error equals u's by the flow's symmetry. ``beta`` is
    ``[theta_u; theta_v; theta_w; theta_p]``."""
    axes = [np.linspace(*physics.x_domain, n_s), np.linspace(*physics.y_domain, n_s),
            np.linspace(*physics.z_domain, n_s), np.linspace(*physics.t_domain, n_t)]
    X, Y, Z, T = np.meshgrid(*axes, indexing='ij')
    ue, ve, pe = physics.exact_u(X, Y, Z, T), physics.exact_v(X, Y, Z, T), physics.exact_p(X, Y, Z, T)
    pe_mean_t = pe.mean(axis=(0, 1, 2))
    Pu, Pv, Pw = basis_u.n_basis, basis_v.n_basis, basis_w.n_basis

    def test_errors(beta):
        u = tensor_grid_values(basis_u, beta[:Pu], axes)
        v = tensor_grid_values(basis_v, beta[Pu:Pu + Pv], axes)
        p = tensor_grid_values(basis_p, beta[Pu + Pv + Pw:], axes)
        p = p - p.mean(axis=(0, 1, 2)) + pe_mean_t
        return {
            'eps_u': rel_l2(u, ue), 'eps_v': rel_l2(v, ve), 'eps_p': rel_l2(p, pe),
            'maxerr_u': max_abs(u, ue), 'maxerr_v': max_abs(v, ve), 'maxerr_p': max_abs(p, pe),
        }

    return test_errors


def solve_beltrami(config: BeltramiConfig, verbose=True,
                    analyze_conditioning: bool = False,
                    iteration_logger=None,
                    run_json_path=None) -> Dict:
    """Solve 3D Beltrami flow via multi-field LiL-Q.

    Parameters
    ----------
    analyze_conditioning : bool
        If True, compute and log the system matrix's condition number
        (``np.linalg.cond``, a full SVD) every outer iteration. Defaults
        to False: at this problem's scale (P_total ~ 8000), a full SVD
        per iteration is expensive enough to roughly double total solve
        time for a diagnostic nothing currently consumes -- confirmed via
        a same-machine, same-moment comparison against the pre-GitHub
        codebase, which gates this identically (``analyze_svd=False`` by
        default) for the same reason. See DECISIONS.md. Independent of
        ``iteration_logger`` below, which is the real Section 3.1 item 8
        replacement this flag anticipated -- ``LilQDiagnosticsTracker``
        already does final-iterate-only pivoted-QR (not a full SVD) for
        P_total this large, so it doesn't pay this flag's cost.
    iteration_logger : ``lilq.iteration_log.IterationLogger``, optional
        When given, a full Section 3.1 ``iterations.csv`` row is recorded
        every outer iteration, mirroring ``problems.kovasznay``'s manual
        ``LilQDiagnosticsTracker`` wiring -- Beltrami has its own
        self-contained quasilinearization loop, same as Kovasznay.
        Interior-row unweighting is only populated when
        ``config.lambda_mom == config.lambda_cont`` (true by default),
        same reasoning as Kovasznay: momentum (3*n_pde rows) and
        continuity (n_pde rows) are independently weighted and only
        collapse to the tracker's required single leading scalar weight
        when the two match. **CPU only** (``config.use_gpu=False``) --
        raises ``NotImplementedError`` otherwise, since the GPU path here
        uses a different LAPACK driver (``gelsd`` via
        ``torch.linalg.lstsq``, not ``gelsy``) and a full GPU
        instrumentation path is Section 3.2's concern, Kovasznay-only per
        the spec, not attempted here. Omitted (``None``, the default),
        behavior -- including ``history`` -- is unchanged from before
        this parameter existed.
    run_json_path : str or Path, optional
        When given, writes the Section 3.1 "once per run" ``run.json``
        metadata file for this solve (see ``lilq.run_metadata``) after
        the solve completes. Requires ``iteration_logger`` (its rows
        supply ``first_stall_iteration``) -- raises ``ValueError`` if
        given without it.
    """
    if iteration_logger is not None and config.use_gpu:
        raise NotImplementedError(
            "iteration_logger is not supported with config.use_gpu=True: "
            "the GPU lstsq path uses a different LAPACK driver (gelsd, not "
            "gelsy) and a GPU instrumentation path is out of scope here "
            "(Section 3.2 of the package spec is Kovasznay-only). Use the "
            "CPU path (use_gpu=False) when iteration_logger is given."
        )
    if run_json_path is not None and iteration_logger is None:
        raise ValueError("run_json_path requires iteration_logger (for first_stall_iteration).")
    physics = BeltramiPhysics(config)
    nu = physics.nu

    domains = [config.x_domain, config.y_domain, config.z_domain, config.t_domain]
    basis_u = create_basis_nd(config.basis_type, config.N_vel, domains)
    basis_v = create_basis_nd(config.basis_type, config.N_vel, domains)
    basis_w = create_basis_nd(config.basis_type, config.N_vel, domains)
    basis_p = create_basis_nd(config.basis_type, config.N_p, domains)

    Pu, Pv, Pw, Pp = basis_u.n_basis, basis_v.n_basis, basis_w.n_basis, basis_p.n_basis
    P_total = Pu + Pv + Pw + Pp

    if verbose:
        print("=" * 70)
        print("LiL-Q SOLVE: 3D Beltrami Flow")
        print(f"  P_u={Pu}, P_v={Pv}, P_w={Pw}, P_p={Pp}, P_total={P_total}")
        print("=" * 70)

    t_start = time.time()
    t_diag = 0.0  # time in the passive diagnostics, excluded from total_time
    pts = _generate_collocation(config, physics, P_total)
    xp, yp, zp, tp = pts['x_pde'], pts['y_pde'], pts['z_pde'], pts['t_pde']
    n_pde = pts['n_pde']

    # Precompute basis matrices
    if verbose: print("  Precomputing basis matrices...", end=' ', flush=True)
    t0 = time.time()

    def _mats(bas, x, y, z, t):
        ev = lambda *o: bas.derivative(x, y, z, t, orders=list(o))
        return {'val': bas.evaluate(x, y, z, t),
                'dx': ev(1,0,0,0), 'dy': ev(0,1,0,0), 'dz': ev(0,0,1,0), 'dt': ev(0,0,0,1),
                'dxx': ev(2,0,0,0), 'dyy': ev(0,2,0,0), 'dzz': ev(0,0,2,0)}

    Mu = _mats(basis_u, xp, yp, zp, tp)
    Mv = _mats(basis_v, xp, yp, zp, tp)
    Mw = _mats(basis_w, xp, yp, zp, tp)
    Mp = {'val': basis_p.evaluate(xp, yp, zp, tp),
          'dx': basis_p.derivative(xp, yp, zp, tp, orders=[1,0,0,0]),
          'dy': basis_p.derivative(xp, yp, zp, tp, orders=[0,1,0,0]),
          'dz': basis_p.derivative(xp, yp, zp, tp, orders=[0,0,1,0])}

    Diff_u = -nu * (Mu['dxx'] + Mu['dyy'] + Mu['dzz'])
    Diff_v = -nu * (Mv['dxx'] + Mv['dyy'] + Mv['dzz'])
    Diff_w = -nu * (Mw['dxx'] + Mw['dyy'] + Mw['dzz'])
    if verbose: print(f"{time.time()-t0:.2f}s")

    # BC matrices
    bc_blocks = {}
    for fname, (xb, yb, zb, tb) in pts['bc'].items():
        bc_blocks[fname] = {
            'Phi_u': basis_u.evaluate(xb, yb, zb, tb),
            'Phi_v': basis_v.evaluate(xb, yb, zb, tb),
            'Phi_w': basis_w.evaluate(xb, yb, zb, tb),
            'u_ex': physics.exact_u(xb, yb, zb, tb),
            'v_ex': physics.exact_v(xb, yb, zb, tb),
            'w_ex': physics.exact_w(xb, yb, zb, tb),
            'n': len(xb),
        }

    # IC matrices
    xi, yi, zi, ti = pts['x_ic'], pts['y_ic'], pts['z_ic'], pts['t_ic']
    ic_block = {
        'Phi_u': basis_u.evaluate(xi, yi, zi, ti),
        'Phi_v': basis_v.evaluate(xi, yi, zi, ti),
        'Phi_w': basis_w.evaluate(xi, yi, zi, ti),
        'u_ex': physics.exact_u(xi, yi, zi, ti),
        'v_ex': physics.exact_v(xi, yi, zi, ti),
        'w_ex': physics.exact_w(xi, yi, zi, ti),
        'n': len(xi),
    }

    # Pressure pin (Section 3.7: n_pressure_pin_levels temporal levels at
    # the fixed spatial corner (x0,y0,z0), n_pressure_pin_levels=1 gives
    # back exactly the original single pin at t=t_domain[0])
    n_pin = config.n_pressure_pin_levels
    t_pin = _cgl_temporal_pin_nodes(n_pin, physics.t_domain)
    x0 = np.full(n_pin, physics.x_domain[0])
    y0 = np.full(n_pin, physics.y_domain[0])
    z0 = np.full(n_pin, physics.z_domain[0])
    Phi_p_pin = basis_p.evaluate(x0, y0, z0, t_pin)
    p_pin_val = physics.exact_p(x0, y0, z0, t_pin)

    # Initialize
    theta_u = np.zeros(Pu); theta_v = np.zeros(Pv)
    theta_w = np.zeros(Pw); theta_p = np.zeros(Pp)

    history = {k: [] for k in ['iteration', 'coeff_change', 'pde_residual',
                                'continuity_residual', 'solve_time', 'cond_number']}

    tracker = None
    loss_fn = None
    residual_vector_fn = None
    if iteration_logger is not None:
        loss_fn = _make_beltrami_nonlinear_loss_fn(
            Mu, Mv, Mw, Mp, bc_blocks, ic_block, Phi_p_pin, p_pin_val,
            nu, Pu, Pv, Pw, Pp,
            config.lambda_mom, config.lambda_cont, config.lambda_bc, config.lambda_ic,
        )
        residual_vector_fn = _make_beltrami_residual_vector_fn(
            Mu, Mv, Mw, Mp, bc_blocks, ic_block, Phi_p_pin, p_pin_val,
            nu, Pu, Pv, Pw, Pp, n_pde,
            config.lambda_mom, config.lambda_cont, config.lambda_bc, config.lambda_ic,
        )

        tracker_kwargs = {}
        if config.lambda_mom == config.lambda_cont:
            tracker_kwargs["n_interior_rows"] = 4 * n_pde
            tracker_kwargs["interior_weight"] = float(np.sqrt(config.lambda_mom / n_pde))
        # P_total ~ 8000 for the paper's Beltrami config -- above
        # DEFAULT_SVD_CONDITIONING_THRESHOLD (3200), so the tracker
        # already does pivoted-QR-at-final-iterate-only here, matching
        # Section 3.1 item 8's Beltrami-specific conditioning method
        # (not a full per-iteration SVD -- see analyze_conditioning above).

        tracker = LilQDiagnosticsTracker(
            test_error_fn=make_test_error_fn(physics, basis_u, basis_v, basis_w, basis_p),
            **tracker_kwargs,
        )

    # ── Quasilinearization loop ──
    for k in range(config.max_iter):
        t_iter = time.time()
        t0 = time.perf_counter()

        uk = Mu['val'] @ theta_u; uk_x = Mu['dx'] @ theta_u
        uk_y = Mu['dy'] @ theta_u; uk_z = Mu['dz'] @ theta_u
        vk = Mv['val'] @ theta_v; vk_x = Mv['dx'] @ theta_v
        vk_y = Mv['dy'] @ theta_v; vk_z = Mv['dz'] @ theta_v
        wk = Mw['val'] @ theta_w; wk_x = Mw['dx'] @ theta_w
        wk_y = Mw['dy'] @ theta_w; wk_z = Mw['dz'] @ theta_w

        wm = np.sqrt(config.lambda_mom / n_pde)
        wc = np.sqrt(config.lambda_cont / n_pde)
        Z_p = np.zeros((n_pde, Pp))

        # x-momentum
        A_mom1_u = uk[:,None]*Mu['dx'] + uk_x[:,None]*Mu['val'] + vk[:,None]*Mu['dy'] + wk[:,None]*Mu['dz'] + Mu['dt'] + Diff_u
        A_mom1_v = uk_y[:,None]*Mv['val']
        A_mom1_w = uk_z[:,None]*Mw['val']

        # y-momentum
        A_mom2_u = vk_x[:,None]*Mu['val']
        A_mom2_v = uk[:,None]*Mv['dx'] + vk_y[:,None]*Mv['val'] + vk[:,None]*Mv['dy'] + wk[:,None]*Mv['dz'] + Mv['dt'] + Diff_v
        A_mom2_w = vk_z[:,None]*Mw['val']

        # z-momentum
        A_mom3_u = wk_x[:,None]*Mu['val']
        A_mom3_v = wk_y[:,None]*Mv['val']
        A_mom3_w = uk[:,None]*Mw['dx'] + vk[:,None]*Mw['dy'] + wk_z[:,None]*Mw['val'] + wk[:,None]*Mw['dz'] + Mw['dt'] + Diff_w

        A_rows = [
            wm * np.hstack([A_mom1_u, A_mom1_v, A_mom1_w, Mp['dx']]),
            wm * np.hstack([A_mom2_u, A_mom2_v, A_mom2_w, Mp['dy']]),
            wm * np.hstack([A_mom3_u, A_mom3_v, A_mom3_w, Mp['dz']]),
            wc * np.hstack([Mu['dx'], Mv['dy'], Mw['dz'], Z_p]),
        ]
        b_rows = [
            wm * (uk*uk_x + vk*uk_y + wk*uk_z),
            wm * (uk*vk_x + vk*vk_y + wk*vk_z),
            wm * (uk*wk_x + vk*wk_y + wk*wk_z),
            wc * np.zeros(n_pde),
        ]

        # BC rows
        for fname, blk in bc_blocks.items():
            ne = blk['n']
            wb = np.sqrt(config.lambda_bc / ne)
            zu = np.zeros((ne, Pu)); zv = np.zeros((ne, Pv))
            zw = np.zeros((ne, Pw)); zp = np.zeros((ne, Pp))
            A_rows.extend([
                wb * np.hstack([blk['Phi_u'], zv, zw, zp]),
                wb * np.hstack([zu, blk['Phi_v'], zw, zp]),
                wb * np.hstack([zu, zv, blk['Phi_w'], zp]),
            ])
            b_rows.extend([wb*blk['u_ex'], wb*blk['v_ex'], wb*blk['w_ex']])

        # IC rows
        ni = ic_block['n']
        wi = np.sqrt(config.lambda_ic / ni)
        zu = np.zeros((ni, Pu)); zv = np.zeros((ni, Pv))
        zw = np.zeros((ni, Pw)); zp = np.zeros((ni, Pp))
        A_rows.extend([
            wi * np.hstack([ic_block['Phi_u'], zv, zw, zp]),
            wi * np.hstack([zu, ic_block['Phi_v'], zw, zp]),
            wi * np.hstack([zu, zv, ic_block['Phi_w'], zp]),
        ])
        b_rows.extend([wi*ic_block['u_ex'], wi*ic_block['v_ex'], wi*ic_block['w_ex']])

        # Pressure pin (n_pin rows -- see Section 3.7 note above)
        wp = np.sqrt(config.lambda_bc / n_pin)
        pin = np.zeros((n_pin, P_total))
        pin[:, Pu+Pv+Pw:] = Phi_p_pin
        A_rows.append(wp * pin)
        b_rows.append(wp * p_pin_val)

        A_sys = np.vstack(A_rows)
        b_sys = np.concatenate(b_rows)
        t_assemble_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        theta_new, rank = _lstsq(A_sys, b_sys, use_gpu=config.use_gpu)
        t_solve_s = time.perf_counter() - t0
        dt_iter = time.time() - t_iter

        tu = theta_new[:Pu]
        tv = theta_new[Pu:Pu+Pv]
        tw = theta_new[Pu+Pv:Pu+Pv+Pw]
        tp_ = theta_new[Pu+Pv+Pw:]

        theta_old = np.concatenate([theta_u, theta_v, theta_w, theta_p])
        rel_delta = np.linalg.norm(theta_new - theta_old) / (np.linalg.norm(theta_new) + 1e-30)

        # Nonlinear residual
        u_n = Mu['val']@tu; ux = Mu['dx']@tu; uy = Mu['dy']@tu; uz = Mu['dz']@tu
        v_n = Mv['val']@tv; vx = Mv['dx']@tv; vy = Mv['dy']@tv; vz = Mv['dz']@tv
        w_n = Mw['val']@tw; wx = Mw['dx']@tw; wy = Mw['dy']@tw; wz = Mw['dz']@tw
        ut = Mu['dt']@tu; vt = Mv['dt']@tv; wt = Mw['dt']@tw
        px = Mp['dx']@tp_; py = Mp['dy']@tp_; pz = Mp['dz']@tp_

        r1 = ut + u_n*ux + v_n*uy + w_n*uz + px - nu*((Mu['dxx']+Mu['dyy']+Mu['dzz'])@tu)
        r2 = vt + u_n*vx + v_n*vy + w_n*vz + py - nu*((Mv['dxx']+Mv['dyy']+Mv['dzz'])@tv)
        r3 = wt + u_n*wx + v_n*wy + w_n*wz + pz - nu*((Mw['dxx']+Mw['dyy']+Mw['dzz'])@tw)
        r4 = ux + vy + wz

        pde_res = (np.mean(r1**2) + np.mean(r2**2) + np.mean(r3**2)) / 3
        cont_res = np.mean(r4**2)

        history['iteration'].append(k)
        history['coeff_change'].append(rel_delta)
        history['pde_residual'].append(pde_res)
        history['continuity_residual'].append(cont_res)
        history['solve_time'].append(dt_iter)
        history['cond_number'].append(
            float(np.linalg.cond(A_sys)) if analyze_conditioning else float('nan')
        )

        if verbose:
            print(f"  Iter {k:3d}: delta={rel_delta:.3e}  "
                  f"PDE={pde_res:.3e}  div={cont_res:.3e}  "
                  f"QR={dt_iter:.3f}s  rank={rank}/{P_total}")

        if tracker is not None:
            t_diag0 = time.perf_counter()
            is_final_iterate = (rel_delta < config.tol) or (k == config.max_iter - 1)
            total_loss = loss_fn(theta_new)
            row = tracker.step(
                k=k,
                A_stacked=A_sys, b_stacked=b_sys,
                beta_prev=theta_old, beta_new=theta_new,
                total_loss=total_loss, rank_gelsy=rank,
                t_assemble_s=t_assemble_s, t_solve_s=t_solve_s,
                is_final_iterate=is_final_iterate,
                compute_residual_vector_fn=residual_vector_fn,
            )
            iteration_logger.record(**row)
            t_diag += time.perf_counter() - t_diag0

        theta_u, theta_v, theta_w, theta_p = tu, tv, tw, tp_

        if rel_delta < config.tol:
            if verbose: print(f"  Converged at iteration {k}.")
            break

    if tracker is not None:
        t_diag0 = time.perf_counter()
        iteration_logger.record(**tracker.finish(k=k + 1))
        t_diag += time.perf_counter() - t_diag0

    # The Section 3.1 diagnostics are passive: off the method's clock.
    total_time = time.time() - t_start - t_diag
    rel_l2 = compute_errors(physics, basis_u, basis_v, basis_w, basis_p,
                            theta_u, theta_v, theta_w, theta_p)
    snap = compute_time_snapshot_errors(physics, basis_u, basis_v, basis_w, basis_p,
                                        theta_u, theta_v, theta_w, theta_p)

    if verbose:
        print(f"\n  Total time: {total_time:.3f}s,  Iters: {k+1}")
        for key in ['rel_l2_u', 'rel_l2_v', 'rel_l2_w', 'rel_l2_p']:
            print(f"  {key}: {rel_l2[key]:.3e}")

    if run_json_path is not None:
        n_bc_edge_total = sum(blk['n'] for blk in bc_blocks.values())
        n_ic_total = ic_block['n']
        wm = float(np.sqrt(config.lambda_mom / n_pde))
        wc = float(np.sqrt(config.lambda_cont / n_pde))
        wi = float(np.sqrt(config.lambda_ic / n_ic_total))
        wp_pin = float(np.sqrt(config.lambda_bc / n_pin))
        thread_env = capture_blas_thread_env()
        final_rel_delta = history['coeff_change'][-1]
        metadata = build_run_metadata(
            N_total=int(A_sys.shape[0]),
            N_composition={
                'x_momentum': n_pde, 'y_momentum': n_pde, 'z_momentum': n_pde,
                'continuity': n_pde,
                'bc_u': n_bc_edge_total, 'bc_v': n_bc_edge_total, 'bc_w': n_bc_edge_total,
                'ic_u': n_ic_total, 'ic_v': n_ic_total, 'ic_w': n_ic_total,
                'pressure_pin': n_pin,
            },
            P_total=int(P_total),
            P_composition={'u': int(Pu), 'v': int(Pv), 'w': int(Pw), 'p': int(Pp)},
            row_weights={
                'momentum': wm, 'continuity': wc,
                'bc': {face: float(np.sqrt(config.lambda_bc / blk['n']))
                       for face, blk in bc_blocks.items()},
                'ic': wi, 'pressure_pin': wp_pin,
            },
            collocation_construction={
                'method': 'equispaced tensor grid',
                'N_x': config.N_x, 'N_y': config.N_y, 'N_z': config.N_z, 'N_t': config.N_t,
                'N_bc': config.N_bc, 'N_t_bc': config.N_t_bc, 'N_ic': config.N_ic,
            },
            basis_description={
                'family': config.basis_type,
                'u': {'modes': config.N_vel, 'dims': 4},
                'v': {'modes': config.N_vel, 'dims': 4},
                'w': {'modes': config.N_vel, 'dims': 4},
                'p': {'modes': config.N_p, 'dims': 4},
            },
            initial_coefficients='zero',
            solver_driver='gelsy', rcond=EPS_MACH,
            stopping_rule={'type': 'rel_coeff_change', 'tolerance': config.tol},
            K_max=config.max_iter,
            stopping_reason='target' if final_rel_delta < config.tol else 'iteration_cap',
            first_stall_iteration=first_stall_iteration(iteration_logger.rows),
            b2_check=tracker.b2_check,
            kappa_qr_raw_ratio=tracker.kappa_qr_raw_ratio,
            device='cpu',
            thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1),
        )
        write_run_json(run_json_path, metadata)

    return {
        'theta_u': theta_u, 'theta_v': theta_v, 'theta_w': theta_w, 'theta_p': theta_p,
        'basis_u': basis_u, 'basis_v': basis_v, 'basis_w': basis_w, 'basis_p': basis_p,
        'n_params': P_total, 'n_outer_iters': k + 1,
        'solve_time_total': total_time,
        'diagnostics_time': t_diag,
        **rel_l2, 'pde_mse': pde_res, 'cont_mse': cont_res,
        'history': history, 'snapshots': snap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Error Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _rel_l2(pred, exact):
    return np.sqrt(np.mean((pred - exact)**2)) / max(np.sqrt(np.mean(exact**2)), 1e-15)


def compute_errors(phys, bu, bv, bw, bp, tu, tv, tw, tp, n_s=21, n_t=11):
    """Global relative L2 errors with pressure gauge correction.

    Fields are evaluated on the tensor grid through the 1D factors
    (:func:`lilq.test_errors.tensor_grid_values`) rather than full basis
    matrices, which at the paper's P = 7,984 needed a 3.8 GB peak; the
    errors agree with the full-matrix evaluation to round-off.
    """
    axes = [np.linspace(*phys.x_domain, n_s), np.linspace(*phys.y_domain, n_s),
            np.linspace(*phys.z_domain, n_s), np.linspace(*phys.t_domain, n_t)]
    X, Y, Z, T = np.meshgrid(*axes, indexing='ij')

    up = tensor_grid_values(bu, tu, axes)
    vp = tensor_grid_values(bv, tv, axes)
    wp = tensor_grid_values(bw, tw, axes)
    pp = tensor_grid_values(bp, tp, axes)
    ue = phys.exact_u(X, Y, Z, T)
    ve = phys.exact_v(X, Y, Z, T)
    we = phys.exact_w(X, Y, Z, T)
    pe = phys.exact_p(X, Y, Z, T)

    # Per-time-step pressure shift
    pp = pp - pp.mean(axis=(0, 1, 2)) + pe.mean(axis=(0, 1, 2))

    return {'rel_l2_u': _rel_l2(up, ue), 'rel_l2_v': _rel_l2(vp, ve),
            'rel_l2_w': _rel_l2(wp, we), 'rel_l2_p': _rel_l2(pp, pe)}


def compute_time_snapshot_errors(phys, bu, bv, bw, bp, tu, tv, tw, tp,
                                  t_vals=(0.0, 0.25, 0.5, 0.75, 1.0), n_s=21):
    """Per-time-step errors, matching NSFnets Table 4 format."""
    snapshots = []
    for t_val in t_vals:
        axes = [np.linspace(*phys.x_domain, n_s), np.linspace(*phys.y_domain, n_s),
                np.linspace(*phys.z_domain, n_s), np.array([t_val], dtype=np.float64)]
        X, Y, Z, T = np.meshgrid(*axes, indexing='ij')

        up = tensor_grid_values(bu, tu, axes)
        vp = tensor_grid_values(bv, tv, axes)
        wp = tensor_grid_values(bw, tw, axes)
        pp = tensor_grid_values(bp, tp, axes)
        ue = phys.exact_u(X, Y, Z, T)
        ve = phys.exact_v(X, Y, Z, T)
        we = phys.exact_w(X, Y, Z, T)
        pe = phys.exact_p(X, Y, Z, T)

        pp = pp - np.mean(pp) + np.mean(pe)
        snapshots.append({'t': t_val, 'u': _rel_l2(up, ue), 'v': _rel_l2(vp, ve),
                          'w': _rel_l2(wp, we), 'p': _rel_l2(pp, pe)})
    return snapshots


def evaluate_fields_at_slice(result, physics, z_val=0.0, t_val=1.0, n_eval=51):
    """Evaluate fields at a 2D slice (fixed z, t)."""
    bu, bv, bw, bp = result['basis_u'], result['basis_v'], result['basis_w'], result['basis_p']
    tu, tv, tw, tp = result['theta_u'], result['theta_v'], result['theta_w'], result['theta_p']

    xs = np.linspace(*physics.x_domain, n_eval)
    ys = np.linspace(*physics.y_domain, n_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xf, yf = X.ravel(), Y.ravel()
    zf = np.full_like(xf, z_val); tf = np.full_like(xf, t_val)

    pp_raw = bp.evaluate(xf, yf, zf, tf) @ tp
    pe = physics.exact_p(xf, yf, zf, tf)
    pp = pp_raw - np.mean(pp_raw) + np.mean(pe)

    pred = {'X': X, 'Y': Y,
            'u': (bu.evaluate(xf, yf, zf, tf) @ tu).reshape(X.shape),
            'v': (bv.evaluate(xf, yf, zf, tf) @ tv).reshape(X.shape),
            'w': (bw.evaluate(xf, yf, zf, tf) @ tw).reshape(X.shape),
            'p': pp.reshape(X.shape)}
    exact = {'X': X, 'Y': Y,
             'u': physics.exact_u(xf, yf, zf, tf).reshape(X.shape),
             'v': physics.exact_v(xf, yf, zf, tf).reshape(X.shape),
             'w': physics.exact_w(xf, yf, zf, tf).reshape(X.shape),
             'p': pe.reshape(X.shape)}
    return exact, pred


def evaluate_fields_3d(result, physics, t_val=1.0, n_eval=21):
    """Evaluate fields on a full 3D grid at fixed t."""
    bu, bv, bw, bp = result['basis_u'], result['basis_v'], result['basis_w'], result['basis_p']
    tu, tv, tw, tp = result['theta_u'], result['theta_v'], result['theta_w'], result['theta_p']

    xs = np.linspace(*physics.x_domain, n_eval)
    ys = np.linspace(*physics.y_domain, n_eval)
    zs = np.linspace(*physics.z_domain, n_eval)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    xf, yf, zf = X.ravel(), Y.ravel(), Z.ravel()
    tf = np.full_like(xf, t_val)

    pp_raw = bp.evaluate(xf, yf, zf, tf) @ tp
    pe = physics.exact_p(xf, yf, zf, tf)
    pp = pp_raw - np.mean(pp_raw) + np.mean(pe)

    pred = {'X': X, 'Y': Y, 'Z': Z,
            'u': (bu.evaluate(xf, yf, zf, tf) @ tu).reshape(X.shape),
            'v': (bv.evaluate(xf, yf, zf, tf) @ tv).reshape(X.shape),
            'w': (bw.evaluate(xf, yf, zf, tf) @ tw).reshape(X.shape),
            'p': pp.reshape(X.shape)}
    exact = {'X': X, 'Y': Y, 'Z': Z,
             'u': physics.exact_u(xf, yf, zf, tf).reshape(X.shape),
             'v': physics.exact_v(xf, yf, zf, tf).reshape(X.shape),
             'w': physics.exact_w(xf, yf, zf, tf).reshape(X.shape),
             'p': pe.reshape(X.shape)}

    for d in [pred, exact]:
        d['vel_mag'] = np.sqrt(d['u']**2 + d['v']**2 + d['w']**2)

    return exact, pred
