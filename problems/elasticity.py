"""
Linear Elasticity Problem (LiL-Q Only)
========================================

2D plane-strain linear elasticity from
Haghighat, Raissi, Moure, Gomez & Juanes (CMAME 2021).

PDE (displacement formulation):
    (lam+2*mu)*u_xx + mu*u_yy + (lam+mu)*v_xy = -f_x
    mu*v_xx + (lam+2*mu)*v_yy + (lam+mu)*u_xy = -f_y

Exact solution:
    u_x(x,y) = cos(2*pi*x)*sin(pi*y)
    u_y(x,y) = sin(pi*x)*Q*y^4/4

Parameters: lambda=1, mu=0.5, Q=4

This is a LINEAR PDE, so LiL-Q reduces to a single QR solve (no
quasilinearization iterations needed).
"""

import numpy as np
import scipy.linalg
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict

from lilq.basis import create_basis_2d, TensorProductBasis2D

pi = np.pi


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ElasticityConfig:
    # Material
    lam: float = 1.0
    mu: float = 0.5
    Q: float = 4.0
    x_domain: Tuple[float, float] = (0.0, 1.0)
    y_domain: Tuple[float, float] = (0.0, 1.0)
    # Discretization
    N_x: int = 15
    N_y: int = 15
    k_ratio: int = 10
    collocation_ratios: Tuple[float, float] = (0.85, 0.15)
    seed: int = 42
    basis_u: str = 'cos_sin'
    basis_v: str = 'sin_cheb'
    # Solver
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    bc_mode: str = 'paper'  # 'paper' or 'exact'


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

class ElasticityPhysics:
    def __init__(self, config: ElasticityConfig):
        self.lam = config.lam
        self.mu = config.mu
        self.Q = config.Q
        self.x_domain = config.x_domain
        self.y_domain = config.y_domain
        self.C11 = config.lam + 2.0 * config.mu
        self.C12 = config.lam
        self.C33 = 2.0 * config.mu

    # Exact displacements
    def exact_ux(self, x, y):
        return np.cos(2*pi*x) * np.sin(pi*y)

    def exact_uy(self, x, y):
        return np.sin(pi*x) * self.Q * y**4 / 4.0

    # Exact strains
    def exact_exx(self, x, y):
        return -2*pi*np.sin(2*pi*x)*np.sin(pi*y)

    def exact_eyy(self, x, y):
        return np.sin(pi*x)*self.Q*y**3

    def exact_exy(self, x, y):
        return 0.5*(pi*np.cos(2*pi*x)*np.cos(pi*y) + pi*np.cos(pi*x)*self.Q*y**4/4.0)

    # Exact stresses
    def exact_sxx(self, x, y):
        return self.C11*self.exact_exx(x,y) + self.C12*self.exact_eyy(x,y)

    def exact_syy(self, x, y):
        return self.C11*self.exact_eyy(x,y) + self.C12*self.exact_exx(x,y)

    def exact_sxy(self, x, y):
        return self.C33*self.exact_exy(x,y)

    # Body forces
    def body_force_x(self, x, y):
        lam, mu, Q = self.lam, self.mu, self.Q
        return (-lam*(4*pi**2*np.cos(2*pi*x)*np.sin(pi*y) - Q*y**3*pi*np.cos(pi*x))
                -mu*(pi**2*np.cos(2*pi*x)*np.sin(pi*y) - Q*y**3*pi*np.cos(pi*x))
                -8*mu*pi**2*np.cos(2*pi*x)*np.sin(pi*y))

    def body_force_y(self, x, y):
        lam, mu, Q = self.lam, self.mu, self.Q
        return (lam*(3*Q*y**2*np.sin(pi*x) - 2*pi**2*np.cos(pi*y)*np.sin(2*pi*x))
                -mu*(2*pi**2*np.cos(pi*y)*np.sin(2*pi*x) + Q*y**4*pi**2*np.sin(pi*x)/4.0)
                +6*Q*mu*y**2*np.sin(pi*x))


# ─────────────────────────────────────────────────────────────────────────────
# Collocation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_collocation(config: ElasticityConfig, physics: ElasticityPhysics, P_total):
    np.random.seed(config.seed)
    ratios = config.collocation_ratios
    norm_r = [r / sum(ratios) for r in ratios]
    x_min, x_max = physics.x_domain
    y_min, y_max = physics.y_domain

    n_pde = config.k_ratio * norm_r[0] * P_total / 2
    n_dim = max(math.ceil(np.sqrt(n_pde)), 10)

    x_pde = np.linspace(x_min + 1e-6, x_max - 1e-6, n_dim, dtype=np.float64)
    y_pde = np.linspace(y_min + 1e-6, y_max - 1e-6, n_dim, dtype=np.float64)
    xx, yy = np.meshgrid(x_pde, y_pde)

    n_bc = max(math.ceil(config.k_ratio * norm_r[1] * P_total / (2*4)), 10)
    tx = np.linspace(x_min, x_max, n_bc, dtype=np.float64)
    ty = np.linspace(y_min, y_max, n_bc, dtype=np.float64)

    return {
        'x_pde': xx.ravel(), 'y_pde': yy.ravel(), 'n_pde': xx.size,
        'x_bot': tx, 'y_bot': np.full_like(tx, y_min),
        'x_top': tx, 'y_top': np.full_like(tx, y_max),
        'x_left': np.full_like(ty, x_min), 'y_left': ty,
        'x_right': np.full_like(ty, x_max), 'y_right': ty,
        'n_bc_edge': len(tx),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Solver (single QR solve — linear PDE)
# ─────────────────────────────────────────────────────────────────────────────

def solve_elasticity(config: ElasticityConfig, verbose=True) -> Dict:
    """Solve linear elasticity via a single LiL-Q QR solve."""
    physics = ElasticityPhysics(config)
    lam, mu = physics.lam, physics.mu
    C11, C12 = physics.C11, physics.C12
    C_cross = lam + mu

    basis_u = create_basis_2d(config.basis_u, config.N_x, config.N_y,
                              config.x_domain, config.y_domain)
    basis_v = create_basis_2d(config.basis_v, config.N_x, config.N_y,
                              config.x_domain, config.y_domain)

    Pu, Pv = basis_u.n_basis, basis_v.n_basis
    P_total = Pu + Pv

    if verbose:
        print("=" * 70)
        print("LiL DIRECT SOLVE: Linear Elasticity")
        print(f"  P_u={Pu}, P_v={Pv}, P_total={P_total}")
        print(f"  BC mode: {config.bc_mode}")
        print("=" * 70)

    t_start = time.time()
    pts = _generate_collocation(config, physics, P_total)
    x_pde, y_pde = pts['x_pde'], pts['y_pde']
    n_pde = pts['n_pde']

    # PDE basis matrices
    Phi_u_xx = basis_u.derivative(x_pde, y_pde, dx=2, dy=0)
    Phi_u_yy = basis_u.derivative(x_pde, y_pde, dx=0, dy=2)
    Phi_u_xy = basis_u.derivative(x_pde, y_pde, dx=1, dy=1)
    Phi_v_xx = basis_v.derivative(x_pde, y_pde, dx=2, dy=0)
    Phi_v_yy = basis_v.derivative(x_pde, y_pde, dx=0, dy=2)
    Phi_v_xy = basis_v.derivative(x_pde, y_pde, dx=1, dy=1)

    A_pde_uu = C11 * Phi_u_xx + mu * Phi_u_yy
    A_pde_uv = C_cross * Phi_v_xy
    A_pde_vu = C_cross * Phi_u_xy
    A_pde_vv = mu * Phi_v_xx + C11 * Phi_v_yy

    fx = physics.body_force_x(x_pde, y_pde)
    fy = physics.body_force_y(x_pde, y_pde)
    w_pde = np.sqrt(config.lambda_pde / n_pde)

    A_rows = [
        w_pde * np.hstack([A_pde_uu, A_pde_uv]),
        w_pde * np.hstack([A_pde_vu, A_pde_vv]),
    ]
    b_rows = [w_pde * fx, w_pde * fy]

    # BCs
    n_bc_total = 0
    lb = config.lambda_bc

    if config.bc_mode == 'paper':
        # Bottom: u_x=0, u_y=0
        xb, yb = pts['x_bot'], pts['y_bot']
        ne = len(xb); wb = np.sqrt(lb / ne)
        A_rows.append(wb * np.hstack([basis_u.evaluate(xb, yb), np.zeros((ne, Pv))]))
        b_rows.append(wb * np.zeros(ne))
        A_rows.append(wb * np.hstack([np.zeros((ne, Pu)), basis_v.evaluate(xb, yb)]))
        b_rows.append(wb * np.zeros(ne))
        n_bc_total += 2 * ne

        # Top: u_x=0, sigma_yy = C11*Q*sin(pi*x)
        xt, yt = pts['x_top'], pts['y_top']
        ne = len(xt); wb = np.sqrt(lb / ne)
        A_rows.append(wb * np.hstack([basis_u.evaluate(xt, yt), np.zeros((ne, Pv))]))
        b_rows.append(wb * np.zeros(ne))
        # sigma_yy = C11*v_y + C12*u_x
        A_syy = np.hstack([C12 * basis_u.derivative(xt, yt, dx=1, dy=0),
                           C11 * basis_v.derivative(xt, yt, dx=0, dy=1)])
        b_syy = C11 * physics.Q * np.sin(pi * xt)
        A_rows.append(wb * A_syy)
        b_rows.append(wb * b_syy)
        n_bc_total += 2 * ne

        # Left/Right: sigma_xx=0, u_y=0
        for xe, ye in [(pts['x_left'], pts['y_left']), (pts['x_right'], pts['y_right'])]:
            ne = len(xe); wb = np.sqrt(lb / ne)
            A_sxx = np.hstack([C11 * basis_u.derivative(xe, ye, dx=1, dy=0),
                               C12 * basis_v.derivative(xe, ye, dx=0, dy=1)])
            A_rows.append(wb * A_sxx)
            b_rows.append(wb * np.zeros(ne))
            A_rows.append(wb * np.hstack([np.zeros((ne, Pu)), basis_v.evaluate(xe, ye)]))
            b_rows.append(wb * np.zeros(ne))
            n_bc_total += 2 * ne

    elif config.bc_mode == 'exact':
        for edge in ['bot', 'top', 'left', 'right']:
            xe, ye = pts[f'x_{edge}'], pts[f'y_{edge}']
            ne = len(xe); wb = np.sqrt(lb / ne)
            A_rows.append(wb * np.hstack([basis_u.evaluate(xe, ye), np.zeros((ne, Pv))]))
            b_rows.append(wb * physics.exact_ux(xe, ye))
            A_rows.append(wb * np.hstack([np.zeros((ne, Pu)), basis_v.evaluate(xe, ye)]))
            b_rows.append(wb * physics.exact_uy(xe, ye))
            n_bc_total += 2 * ne

    A_sys = np.vstack(A_rows)
    b_sys = np.concatenate(b_rows)

    t_solve = time.time()
    theta, _, rank, _ = scipy.linalg.lstsq(A_sys, b_sys, lapack_driver='gelsy')
    solve_time = time.time() - t_solve

    theta_u = theta[:Pu]
    theta_v = theta[Pu:]

    # PDE residuals
    pde_res_x = A_pde_uu @ theta_u + A_pde_uv @ theta_v - fx
    pde_res_y = A_pde_vu @ theta_u + A_pde_vv @ theta_v - fy
    pde_mse = 0.5 * (float(np.mean(pde_res_x**2)) + float(np.mean(pde_res_y**2)))

    # Errors on fine grid
    n_ev = 200
    x_ev = np.linspace(*config.x_domain, n_ev, dtype=np.float64)
    y_ev = np.linspace(*config.y_domain, n_ev, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()

    ux_pred = basis_u.evaluate(xf, yf) @ theta_u
    uy_pred = basis_v.evaluate(xf, yf) @ theta_v
    ux_exact = physics.exact_ux(xf, yf)
    uy_exact = physics.exact_uy(xf, yf)

    rel_l2_ux = np.sqrt(np.mean((ux_pred-ux_exact)**2)) / max(np.sqrt(np.mean(ux_exact**2)), 1e-15)
    rel_l2_uy = np.sqrt(np.mean((uy_pred-uy_exact)**2)) / max(np.sqrt(np.mean(uy_exact**2)), 1e-15)

    # Stress errors
    exx_pred = basis_u.derivative(xf, yf, dx=1, dy=0) @ theta_u
    eyy_pred = basis_v.derivative(xf, yf, dx=0, dy=1) @ theta_v
    exy_pred = 0.5*(basis_u.derivative(xf, yf, dx=0, dy=1) @ theta_u
                     + basis_v.derivative(xf, yf, dx=1, dy=0) @ theta_v)

    sxx_pred = C11*exx_pred + C12*eyy_pred
    syy_pred = C11*eyy_pred + C12*exx_pred
    sxy_pred = physics.C33*exy_pred

    sxx_exact = physics.exact_sxx(xf, yf)
    syy_exact = physics.exact_syy(xf, yf)
    sxy_exact = physics.exact_sxy(xf, yf)

    rel_l2_sxx = np.sqrt(np.mean((sxx_pred-sxx_exact)**2)) / max(np.sqrt(np.mean(sxx_exact**2)), 1e-15)
    rel_l2_syy = np.sqrt(np.mean((syy_pred-syy_exact)**2)) / max(np.sqrt(np.mean(syy_exact**2)), 1e-15)
    rel_l2_sxy = np.sqrt(np.mean((sxy_pred-sxy_exact)**2)) / max(np.sqrt(np.mean(sxy_exact**2)), 1e-15)

    total_time = time.time() - t_start

    if verbose:
        print(f"\n  QR solve time: {solve_time:.4f}s")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  PDE residual MSE: {pde_mse:.6e}")
        print(f"  Displacement errors (rel L2): u_x={rel_l2_ux:.6e}, u_y={rel_l2_uy:.6e}")
        print(f"  Stress errors (rel L2): sxx={rel_l2_sxx:.6e}, syy={rel_l2_syy:.6e}, sxy={rel_l2_sxy:.6e}")
        print("=" * 70)

    return {
        'basis_u': basis_u, 'basis_v': basis_v,
        'theta_u': theta_u, 'theta_v': theta_v,
        'n_params': P_total,
        'solve_time_qr': solve_time, 'solve_time_total': total_time,
        'pde_mse': pde_mse,
        'rel_l2_ux': rel_l2_ux, 'rel_l2_uy': rel_l2_uy,
        'rel_l2_sxx': rel_l2_sxx, 'rel_l2_syy': rel_l2_syy, 'rel_l2_sxy': rel_l2_sxy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_fields(result, physics, n_eval=200):
    """Evaluate displacements, strains, stresses on a grid."""
    basis_u, basis_v = result['basis_u'], result['basis_v']
    theta_u, theta_v = result['theta_u'], result['theta_v']

    x_ev = np.linspace(*physics.x_domain, n_eval, dtype=np.float64)
    y_ev = np.linspace(*physics.y_domain, n_eval, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()

    ux = (basis_u.evaluate(xf, yf) @ theta_u).reshape(XX.shape)
    uy = (basis_v.evaluate(xf, yf) @ theta_v).reshape(XX.shape)

    exx = (basis_u.derivative(xf, yf, dx=1, dy=0) @ theta_u).reshape(XX.shape)
    eyy = (basis_v.derivative(xf, yf, dx=0, dy=1) @ theta_v).reshape(XX.shape)
    exy = (0.5*(basis_u.derivative(xf, yf, dx=0, dy=1) @ theta_u
                + basis_v.derivative(xf, yf, dx=1, dy=0) @ theta_v)).reshape(XX.shape)

    predicted = {'X': XX, 'Y': YY, 'ux': ux, 'uy': uy,
                 'exx': exx, 'eyy': eyy, 'exy': exy,
                 'sxx': physics.C11*exx + physics.C12*eyy,
                 'syy': physics.C11*eyy + physics.C12*exx,
                 'sxy': physics.C33*exy}

    exact = {'X': XX, 'Y': YY,
             'ux': physics.exact_ux(xf, yf).reshape(XX.shape),
             'uy': physics.exact_uy(xf, yf).reshape(XX.shape),
             'sxx': physics.exact_sxx(xf, yf).reshape(XX.shape),
             'syy': physics.exact_syy(xf, yf).reshape(XX.shape),
             'sxy': physics.exact_sxy(xf, yf).reshape(XX.shape)}

    return predicted, exact
