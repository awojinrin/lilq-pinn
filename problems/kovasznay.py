"""
Kovasznay Flow Problem (LiL-Q Only)
=====================================

Steady 2D incompressible Navier-Stokes with the Kovasznay analytical
solution as benchmark.

Equations (velocity-pressure formulation):
    u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0   (x-momentum)
    u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0   (y-momentum)
    u_x + v_y = 0                                   (continuity)

Exact solution:
    u(x,y) = 1 - exp(lam*x)*cos(2*pi*y)
    v(x,y) = (lam/(2*pi))*exp(lam*x)*sin(2*pi*y)
    p(x,y) = 0.5*(1 - exp(2*lam*x))
    lam    = Re/2 - sqrt(Re^2/4 + 4*pi^2)

Multi-field LiL-Q: Three separate bases (u, v, p), assembled into
a block system and solved via QR at each quasilinear iteration.
"""

import numpy as np
import scipy.linalg
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from lilq.basis import (
    Chebyshev1D, Fourier1D, TensorProductBasis2D, AugmentedBasis1D,
    create_chebyshev_basis_2d, create_basis_2d,
)
from lilq.analysis import svd_analysis

pi = np.pi


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KovasznayConfig:
    Re: float = 40.0
    x_domain: Tuple[float, float] = (-0.5, 1.0)
    y_domain: Tuple[float, float] = (-0.5, 1.5)
    N_x: int = 15
    N_y: int = 15
    k_ratio: float = 10
    collocation_ratios: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    seed: int = 42
    basis_type: str = 'chebyshev'
    max_iter: int = 30
    tol: float = 1e-12
    lambda_mom: float = 1.0
    lambda_cont: float = 1.0
    lambda_bc: float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

class KovasznayPhysics:
    """Kovasznay flow analytical solution."""

    def __init__(self, config: KovasznayConfig):
        self.Re = config.Re
        self.nu = 1.0 / config.Re
        self.x_domain = config.x_domain
        self.y_domain = config.y_domain
        self.lam = config.Re / 2.0 - np.sqrt(config.Re**2 / 4.0 + 4.0 * pi**2)

    def exact_u(self, x, y):
        return 1.0 - np.exp(self.lam * x) * np.cos(2.0 * pi * y)

    def exact_v(self, x, y):
        return (self.lam / (2.0 * pi)) * np.exp(self.lam * x) * np.sin(2.0 * pi * y)

    def exact_p(self, x, y):
        return 0.5 * (1.0 - np.exp(2.0 * self.lam * x))


# ─────────────────────────────────────────────────────────────────────────────
# Collocation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_collocation(config: KovasznayConfig, P_total):
    """Generate interior and boundary collocation points."""
    np.random.seed(config.seed)
    x_min, x_max = config.x_domain
    y_min, y_max = config.y_domain

    ratios = config.collocation_ratios
    norm_r = [r / sum(ratios) for r in ratios]

    n_pde = config.k_ratio * norm_r[0] * P_total / 3
    n_dim = max(math.ceil(np.sqrt(n_pde)), 10)

    xp = np.linspace(x_min + 1e-6, x_max - 1e-6, n_dim, dtype=np.float64)
    yp = np.linspace(y_min + 1e-6, y_max - 1e-6, n_dim, dtype=np.float64)
    xx, yy = np.meshgrid(xp, yp)
    x_pde, y_pde = xx.ravel(), yy.ravel()

    n_bc = max(math.ceil(config.k_ratio * norm_r[2] * P_total / (3 * 4)), 10)
    tx = np.linspace(x_min, x_max, n_bc, dtype=np.float64)
    ty = np.linspace(y_min, y_max, n_bc, dtype=np.float64)

    return {
        'x_pde': x_pde, 'y_pde': y_pde, 'n_pde': len(x_pde),
        'x_bot': tx, 'y_bot': np.full_like(tx, y_min),
        'x_top': tx, 'y_top': np.full_like(tx, y_max),
        'x_left': np.full_like(ty, x_min), 'y_left': ty,
        'x_right': np.full_like(ty, x_max), 'y_right': ty,
        'n_bc_edge': n_bc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LiL-Q Solver (Multi-field Quasilinearization)
# ─────────────────────────────────────────────────────────────────────────────

def solve_kovasznay(config: KovasznayConfig, verbose=True) -> Dict:
    """Solve Kovasznay flow via multi-field LiL-Q.

    Returns a dict containing coefficients, errors, and iteration history.
    """
    physics = KovasznayPhysics(config)
    nu = physics.nu

    # Create bases (same for u, v, p)
    basis_u = create_basis_2d(config.basis_type, config.N_x, config.N_y,
                              config.x_domain, config.y_domain)
    basis_v = create_basis_2d(config.basis_type, config.N_x, config.N_y,
                              config.x_domain, config.y_domain)
    basis_p = create_basis_2d(config.basis_type, config.N_x, config.N_y,
                              config.x_domain, config.y_domain)

    Pu, Pv, Pp = basis_u.n_basis, basis_v.n_basis, basis_p.n_basis
    P_total = Pu + Pv + Pp

    if verbose:
        print("=" * 70)
        print("LiL-Q SOLVE: Kovasznay Flow")
        print(f"  Re={physics.Re}, nu={nu:.4f}, lam={physics.lam:.4f}")
        print(f"  P_u={Pu}, P_v={Pv}, P_p={Pp}, P_total={P_total}")
        print("=" * 70)

    t_start = time.time()

    # Collocation
    pts = _generate_collocation(config, P_total)
    xp, yp = pts['x_pde'], pts['y_pde']
    n_pde = pts['n_pde']

    # Precompute basis matrices
    Phi_u = basis_u.evaluate(xp, yp)
    Phi_u_x = basis_u.derivative(xp, yp, dx=1, dy=0)
    Phi_u_y = basis_u.derivative(xp, yp, dx=0, dy=1)
    Phi_u_xx = basis_u.derivative(xp, yp, dx=2, dy=0)
    Phi_u_yy = basis_u.derivative(xp, yp, dx=0, dy=2)

    Phi_v = basis_v.evaluate(xp, yp)
    Phi_v_x = basis_v.derivative(xp, yp, dx=1, dy=0)
    Phi_v_y = basis_v.derivative(xp, yp, dx=0, dy=1)
    Phi_v_xx = basis_v.derivative(xp, yp, dx=2, dy=0)
    Phi_v_yy = basis_v.derivative(xp, yp, dx=0, dy=2)

    Phi_p = basis_p.evaluate(xp, yp)
    Phi_p_x = basis_p.derivative(xp, yp, dx=1, dy=0)
    Phi_p_y = basis_p.derivative(xp, yp, dx=0, dy=1)

    Diff_u = -nu * (Phi_u_xx + Phi_u_yy)
    Diff_v = -nu * (Phi_v_xx + Phi_v_yy)

    # BC basis matrices
    bc_blocks = {}
    for edge in ['bot', 'top', 'left', 'right']:
        xe, ye = pts[f'x_{edge}'], pts[f'y_{edge}']
        bc_blocks[edge] = {
            'Phi_u': basis_u.evaluate(xe, ye),
            'Phi_v': basis_v.evaluate(xe, ye),
            'Phi_p': basis_p.evaluate(xe, ye),
            'u_exact': physics.exact_u(xe, ye),
            'v_exact': physics.exact_v(xe, ye),
            'n': len(xe),
        }

    # Pressure pin
    x_pin = np.array([config.x_domain[0]], dtype=np.float64)
    y_pin = np.array([config.y_domain[0]], dtype=np.float64)
    Phi_p_pin = basis_p.evaluate(x_pin, y_pin)
    p_pin_val = physics.exact_p(x_pin[0], y_pin[0])

    # Initialize
    theta_u = np.zeros(Pu, dtype=np.float64)
    theta_v = np.zeros(Pv, dtype=np.float64)
    theta_p = np.zeros(Pp, dtype=np.float64)

    history = {
        'iteration': [], 'coeff_change': [],
        'pde_residual': [], 'continuity_residual': [],
        'solve_time': [], 'cond_number': [],
    }

    # ── Quasilinearization loop ──
    for k in range(config.max_iter):
        t_iter = time.time()

        uk = Phi_u @ theta_u
        uk_x = Phi_u_x @ theta_u
        uk_y = Phi_u_y @ theta_u
        vk = Phi_v @ theta_v
        vk_x = Phi_v_x @ theta_v
        vk_y = Phi_v_y @ theta_v

        # x-momentum
        A_mom1_u = uk[:, None] * Phi_u_x + uk_x[:, None] * Phi_u + vk[:, None] * Phi_u_y + Diff_u
        A_mom1_v = uk_y[:, None] * Phi_v
        A_mom1_p = Phi_p_x
        b_mom1 = uk * uk_x + vk * uk_y

        # y-momentum
        A_mom2_u = vk_x[:, None] * Phi_u
        A_mom2_v = uk[:, None] * Phi_v_x + vk_y[:, None] * Phi_v + vk[:, None] * Phi_v_y + Diff_v
        A_mom2_p = Phi_p_y
        b_mom2 = uk * vk_x + vk * vk_y

        # Continuity
        A_cont_u = Phi_u_x
        A_cont_v = Phi_v_y
        Z_cont_p = np.zeros((n_pde, Pp))

        w_mom = np.sqrt(config.lambda_mom / n_pde)
        w_cont = np.sqrt(config.lambda_cont / n_pde)

        A_rows = [
            w_mom * np.hstack([A_mom1_u, A_mom1_v, A_mom1_p]),
            w_mom * np.hstack([A_mom2_u, A_mom2_v, A_mom2_p]),
            w_cont * np.hstack([A_cont_u, A_cont_v, Z_cont_p]),
        ]
        b_rows = [w_mom * b_mom1, w_mom * b_mom2, w_cont * np.zeros(n_pde)]

        # BC rows
        for edge in ['bot', 'top', 'left', 'right']:
            blk = bc_blocks[edge]
            ne = blk['n']
            w_bc = np.sqrt(config.lambda_bc / ne)

            A_rows.append(w_bc * np.hstack([blk['Phi_u'], np.zeros((ne, Pv)), np.zeros((ne, Pp))]))
            b_rows.append(w_bc * blk['u_exact'])

            A_rows.append(w_bc * np.hstack([np.zeros((ne, Pu)), blk['Phi_v'], np.zeros((ne, Pp))]))
            b_rows.append(w_bc * blk['v_exact'])

        # Pressure pin
        w_pin = np.sqrt(config.lambda_bc)
        A_rows.append(w_pin * np.hstack([np.zeros((1, Pu)), np.zeros((1, Pv)), Phi_p_pin]))
        b_rows.append(w_pin * np.array([p_pin_val]))

        A_sys = np.vstack(A_rows)
        b_sys = np.concatenate(b_rows)

        theta_new = scipy.linalg.lstsq(A_sys, b_sys, lapack_driver='gelsy')[0]
        dt = time.time() - t_iter

        theta_u_new = theta_new[:Pu]
        theta_v_new = theta_new[Pu:Pu+Pv]
        theta_p_new = theta_new[Pu+Pv:]

        # Convergence
        theta_old = np.concatenate([theta_u, theta_v, theta_p])
        rel_delta = np.linalg.norm(theta_new - theta_old) / (np.linalg.norm(theta_new) + 1e-30)

        # Nonlinear PDE residual
        u_new = Phi_u @ theta_u_new
        u_new_x = Phi_u_x @ theta_u_new
        u_new_y = Phi_u_y @ theta_u_new
        v_new = Phi_v @ theta_v_new
        v_new_x = Phi_v_x @ theta_v_new
        v_new_y = Phi_v_y @ theta_v_new
        p_new_x = Phi_p_x @ theta_p_new
        p_new_y = Phi_p_y @ theta_p_new
        lap_u = (Phi_u_xx + Phi_u_yy) @ theta_u_new
        lap_v = (Phi_v_xx + Phi_v_yy) @ theta_v_new

        res_xmom = u_new*u_new_x + v_new*u_new_y + p_new_x - nu*lap_u
        res_ymom = u_new*v_new_x + v_new*v_new_y + p_new_y - nu*lap_v
        res_cont = u_new_x + v_new_y

        pde_res = 0.5*(np.mean(res_xmom**2) + np.mean(res_ymom**2))
        cont_res = np.mean(res_cont**2)

        history['iteration'].append(k)
        history['coeff_change'].append(rel_delta)
        history['pde_residual'].append(pde_res)
        history['continuity_residual'].append(cont_res)
        history['solve_time'].append(dt)
        history['cond_number'].append(float(np.linalg.cond(A_sys)))

        if verbose:
            print(f"  Iter {k:3d}: d_theta={rel_delta:.3e}  "
                  f"PDE={pde_res:.3e}  cont={cont_res:.3e}  QR={dt:.4f}s")

        theta_u, theta_v, theta_p = theta_u_new, theta_v_new, theta_p_new

        if rel_delta < config.tol:
            if verbose:
                print(f"  Converged at iteration {k}.")
            break

    total_time = time.time() - t_start

    # Final errors vs. exact solution
    n_ev = 200
    x_ev = np.linspace(config.x_domain[0], config.x_domain[1], n_ev, dtype=np.float64)
    y_ev = np.linspace(config.y_domain[0], config.y_domain[1], n_ev, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()

    u_pred = basis_u.evaluate(xf, yf) @ theta_u
    v_pred = basis_v.evaluate(xf, yf) @ theta_v
    p_pred = basis_p.evaluate(xf, yf) @ theta_p
    u_ex = physics.exact_u(xf, yf)
    v_ex = physics.exact_v(xf, yf)
    p_ex = physics.exact_p(xf, yf)

    rel_l2_u = np.sqrt(np.mean((u_pred - u_ex)**2)) / max(np.sqrt(np.mean(u_ex**2)), 1e-15)
    rel_l2_v = np.sqrt(np.mean((v_pred - v_ex)**2)) / max(np.sqrt(np.mean(v_ex**2)), 1e-15)
    rel_l2_p = np.sqrt(np.mean((p_pred - p_ex)**2)) / max(np.sqrt(np.mean(p_ex**2)), 1e-15)

    if verbose:
        print(f"\n  Total time: {total_time:.4f}s")
        print(f"  Final rel L2 errors:  u={rel_l2_u:.3e}  v={rel_l2_v:.3e}  p={rel_l2_p:.3e}")

    return {
        'basis_u': basis_u, 'basis_v': basis_v, 'basis_p': basis_p,
        'theta_u': theta_u, 'theta_v': theta_v, 'theta_p': theta_p,
        'n_params': P_total,
        'n_outer_iters': k + 1,
        'solve_time_total': total_time,
        'rel_l2_u': rel_l2_u, 'rel_l2_v': rel_l2_v, 'rel_l2_p': rel_l2_p,
        'pde_mse': pde_res, 'cont_mse': cont_res,
        'history': history,
    }


def evaluate_all_fields(result, physics, n_eval=200):
    """Evaluate predicted and exact fields on a grid."""
    x_ev = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval, dtype=np.float64)
    y_ev = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()

    predicted = {
        'X': XX, 'Y': YY,
        'u': (result['basis_u'].evaluate(xf, yf) @ result['theta_u']).reshape(XX.shape),
        'v': (result['basis_v'].evaluate(xf, yf) @ result['theta_v']).reshape(XX.shape),
        'p': (result['basis_p'].evaluate(xf, yf) @ result['theta_p']).reshape(XX.shape),
    }
    exact = {
        'X': XX, 'Y': YY,
        'u': physics.exact_u(xf, yf).reshape(XX.shape),
        'v': physics.exact_v(xf, yf).reshape(XX.shape),
        'p': physics.exact_p(xf, yf).reshape(XX.shape),
    }
    return predicted, exact
