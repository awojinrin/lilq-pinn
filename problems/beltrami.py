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
import scipy.linalg
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List

from lilq.basis import TensorProductBasisND, create_basis_nd, Chebyshev1D, Fourier1D

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


def _lstsq(A, b, use_gpu=False):
    if use_gpu and HAS_TORCH_CUDA:
        At = torch.as_tensor(A, dtype=torch.float64, device='cuda')
        bt = torch.as_tensor(b, dtype=torch.float64, device='cuda').unsqueeze(1)
        result = torch.linalg.lstsq(At, bt, driver='gelsd')
        x = result.solution.squeeze(1).cpu().numpy()
        rank = int(result.rank) if result.rank is not None else At.shape[1]
        return x, rank
    else:
        x, _, rank, _ = scipy.linalg.lstsq(A, b, lapack_driver='gelsy')
        return x, rank


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_beltrami(config: BeltramiConfig, verbose=True) -> Dict:
    """Solve 3D Beltrami flow via multi-field LiL-Q."""
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

    # Pressure pin
    x0 = np.array([physics.x_domain[0]])
    y0 = np.array([physics.y_domain[0]])
    z0 = np.array([physics.z_domain[0]])
    t0_ = np.array([physics.t_domain[0]])
    Phi_p_pin = basis_p.evaluate(x0, y0, z0, t0_)
    p_pin_val = physics.exact_p(x0[0], y0[0], z0[0], t0_[0])

    # Initialize
    theta_u = np.zeros(Pu); theta_v = np.zeros(Pv)
    theta_w = np.zeros(Pw); theta_p = np.zeros(Pp)

    history = {k: [] for k in ['iteration', 'coeff_change', 'pde_residual',
                                'continuity_residual', 'solve_time', 'cond_number']}

    # ── Quasilinearization loop ──
    for k in range(config.max_iter):
        t_iter = time.time()

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

        # Pressure pin
        wp = np.sqrt(config.lambda_bc)
        pin = np.zeros((1, P_total))
        pin[0, Pu+Pv+Pw:] = Phi_p_pin
        A_rows.append(wp * pin)
        b_rows.append(wp * np.array([p_pin_val]))

        A_sys = np.vstack(A_rows)
        b_sys = np.concatenate(b_rows)

        theta_new, rank = _lstsq(A_sys, b_sys, use_gpu=config.use_gpu)
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
        history['cond_number'].append(float(np.linalg.cond(A_sys)))

        if verbose:
            print(f"  Iter {k:3d}: delta={rel_delta:.3e}  "
                  f"PDE={pde_res:.3e}  div={cont_res:.3e}  "
                  f"QR={dt_iter:.3f}s  rank={rank}/{P_total}")

        theta_u, theta_v, theta_w, theta_p = tu, tv, tw, tp_

        if rel_delta < config.tol:
            if verbose: print(f"  Converged at iteration {k}.")
            break

    total_time = time.time() - t_start
    rel_l2 = compute_errors(physics, basis_u, basis_v, basis_w, basis_p,
                            theta_u, theta_v, theta_w, theta_p)
    snap = compute_time_snapshot_errors(physics, basis_u, basis_v, basis_w, basis_p,
                                        theta_u, theta_v, theta_w, theta_p)

    if verbose:
        print(f"\n  Total time: {total_time:.3f}s,  Iters: {k+1}")
        for key in ['rel_l2_u', 'rel_l2_v', 'rel_l2_w', 'rel_l2_p']:
            print(f"  {key}: {rel_l2[key]:.3e}")

    return {
        'theta_u': theta_u, 'theta_v': theta_v, 'theta_w': theta_w, 'theta_p': theta_p,
        'basis_u': basis_u, 'basis_v': basis_v, 'basis_w': basis_w, 'basis_p': basis_p,
        'n_params': P_total, 'n_outer_iters': k + 1,
        'solve_time_total': total_time,
        **rel_l2, 'pde_mse': pde_res, 'cont_mse': cont_res,
        'history': history, 'snapshots': snap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Error Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _rel_l2(pred, exact):
    return np.sqrt(np.mean((pred - exact)**2)) / max(np.sqrt(np.mean(exact**2)), 1e-15)


def compute_errors(phys, bu, bv, bw, bp, tu, tv, tw, tp, n_s=21, n_t=11):
    """Global relative L2 errors with pressure gauge correction."""
    xs = np.linspace(*phys.x_domain, n_s)
    ys = np.linspace(*phys.y_domain, n_s)
    zs = np.linspace(*phys.z_domain, n_s)
    ts = np.linspace(*phys.t_domain, n_t)
    X, Y, Z, T = np.meshgrid(xs, ys, zs, ts, indexing='ij')
    xf, yf, zf, tf = X.ravel(), Y.ravel(), Z.ravel(), T.ravel()

    up = bu.evaluate(xf, yf, zf, tf) @ tu
    vp = bv.evaluate(xf, yf, zf, tf) @ tv
    wp = bw.evaluate(xf, yf, zf, tf) @ tw
    pp = bp.evaluate(xf, yf, zf, tf) @ tp
    ue = phys.exact_u(xf, yf, zf, tf)
    ve = phys.exact_v(xf, yf, zf, tf)
    we = phys.exact_w(xf, yf, zf, tf)
    pe = phys.exact_p(xf, yf, zf, tf)

    # Per-time-step pressure shift
    n_spatial = n_s ** 3
    pp_4d = pp.reshape(n_s, n_s, n_s, n_t)
    pe_4d = pe.reshape(n_s, n_s, n_s, n_t)
    for it in range(n_t):
        pp_4d[:,:,:,it] -= np.mean(pp_4d[:,:,:,it])
        pp_4d[:,:,:,it] += np.mean(pe_4d[:,:,:,it])
    pp = pp_4d.ravel()

    return {'rel_l2_u': _rel_l2(up, ue), 'rel_l2_v': _rel_l2(vp, ve),
            'rel_l2_w': _rel_l2(wp, we), 'rel_l2_p': _rel_l2(pp, pe)}


def compute_time_snapshot_errors(phys, bu, bv, bw, bp, tu, tv, tw, tp,
                                  t_vals=(0.0, 0.25, 0.5, 0.75, 1.0), n_s=21):
    """Per-time-step errors, matching NSFnets Table 4 format."""
    snapshots = []
    for t_val in t_vals:
        xs = np.linspace(*phys.x_domain, n_s)
        ys = np.linspace(*phys.y_domain, n_s)
        zs = np.linspace(*phys.z_domain, n_s)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        xf, yf, zf = X.ravel(), Y.ravel(), Z.ravel()
        tf = np.full_like(xf, t_val)

        up = bu.evaluate(xf, yf, zf, tf) @ tu
        vp = bv.evaluate(xf, yf, zf, tf) @ tv
        wp = bw.evaluate(xf, yf, zf, tf) @ tw
        pp = bp.evaluate(xf, yf, zf, tf) @ tp
        ue = phys.exact_u(xf, yf, zf, tf)
        ve = phys.exact_v(xf, yf, zf, tf)
        we = phys.exact_w(xf, yf, zf, tf)
        pe = phys.exact_p(xf, yf, zf, tf)

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
