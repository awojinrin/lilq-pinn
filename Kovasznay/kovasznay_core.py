"""
Kovasznay Flow Core Module — LiL-Q Forward Solver
===================================================

Solves the steady 2D incompressible Navier-Stokes equations
using the Kovasznay flow analytical solution as a benchmark.

Equations (velocity-pressure formulation):
    u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0   (x-momentum)
    u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0   (y-momentum)
    u_x + v_y = 0                                   (continuity)

Exact solution:
    u(x,y) = 1 - exp(lam*x)*cos(2*pi*y)
    v(x,y) = (lam/(2*pi))*exp(lam*x)*sin(2*pi*y)
    p(x,y) = 0.5*(1 - exp(2*lam*x))
    lam    = Re/2 - sqrt(Re^2/4 + 4*pi^2)

Quasilinearization: The quadratic convective terms u*u_x, v*u_y,
u*v_x, v*v_y are linearized at each outer iteration via
Bellman-Kalaba (Newton-Kantorovich), producing a sequence of
linear least-squares problems solved by QR.

Reusable files (place in same directory):
    - lil_basis.py
    - pub_style.py

Author: Gbenga / Claude collaboration
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from lilq_shared.basis import (
    Chebyshev1D, Fourier1D, TensorProductBasis2D,
    AugmentedBasis1D,
    create_chebyshev_basis_2d, create_fourier_basis_2d, create_mixed_basis_2d
)

pi = np.pi


# =============================================================================
# PHYSICS
# =============================================================================

@dataclass
class KovasznayPhysics:
    """Kovasznay flow: exact solution to steady incompressible Navier-Stokes."""
    Re: float = 20.0
    x_domain: Tuple[float, float] = (0.0, 1.0)
    y_domain: Tuple[float, float] = (0.0, 1.0)

    @property
    def nu(self):
        return 1.0 / self.Re

    @property
    def lam(self):
        return self.Re / 2.0 - np.sqrt(self.Re**2 / 4.0 + 4.0 * pi**2)

    # --- Exact fields ---
    def exact_u(self, x, y):
        return 1.0 - np.exp(self.lam * x) * np.cos(2.0 * pi * y)

    def exact_v(self, x, y):
        return (self.lam / (2.0 * pi)) * np.exp(self.lam * x) * np.sin(2.0 * pi * y)

    def exact_p(self, x, y):
        return 0.5 * (1.0 - np.exp(2.0 * self.lam * x))

    # --- Exact derivatives (for verification and initial guess) ---
    def exact_u_x(self, x, y):
        return -self.lam * np.exp(self.lam * x) * np.cos(2.0 * pi * y)

    def exact_u_y(self, x, y):
        return 2.0 * pi * np.exp(self.lam * x) * np.sin(2.0 * pi * y)

    def exact_v_x(self, x, y):
        return (self.lam**2 / (2.0 * pi)) * np.exp(self.lam * x) * np.sin(2.0 * pi * y)

    def exact_v_y(self, x, y):
        return self.lam * np.exp(self.lam * x) * np.cos(2.0 * pi * y)


# =============================================================================
# DISCRETISATION
# =============================================================================

@dataclass
class KovasznayDiscretization:
    N_x: int = 15
    N_y: int = 15
    k_ratio: float = 10
    collocation_ratios: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    # (PDE interior, continuity, BCs)
    domain_sampling: str = "uniform"
    boundary_sampling: str = "uniform"
    seed: int = 42


# =============================================================================
# COLLOCATION POINTS
# =============================================================================

def generate_collocation_points(physics, disc, P_total):
    """Generate interior and boundary collocation points."""
    np.random.seed(disc.seed)
    x_min, x_max = physics.x_domain
    y_min, y_max = physics.y_domain

    ratios = disc.collocation_ratios
    norm_r = [r / sum(ratios) for r in ratios]

    # Interior PDE points
    n_pde = disc.k_ratio * norm_r[0] * P_total / 3
    n_dim = max(math.ceil(np.sqrt(n_pde)), 10)

    if disc.domain_sampling == "random":
        xp = np.sort(np.random.uniform(x_min + 1e-6, x_max - 1e-6, n_dim)).astype(np.float64)
        yp = np.sort(np.random.uniform(y_min + 1e-6, y_max - 1e-6, n_dim)).astype(np.float64)
    else:
        xp = np.linspace(x_min + 1e-6, x_max - 1e-6, n_dim, dtype=np.float64)
        yp = np.linspace(y_min + 1e-6, y_max - 1e-6, n_dim, dtype=np.float64)
    xx, yy = np.meshgrid(xp, yp)
    x_pde, y_pde = xx.ravel(), yy.ravel()

    # Boundary points per edge
    n_bc = max(math.ceil(disc.k_ratio * norm_r[2] * P_total / (3 * 4)), 10)
    if disc.boundary_sampling == "random":
        tx = np.sort(np.random.uniform(x_min, x_max, n_bc)).astype(np.float64)
        ty = np.sort(np.random.uniform(y_min, y_max, n_bc)).astype(np.float64)
    else:
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


# =============================================================================
# BASIS CREATION
# =============================================================================

def create_kovasznay_basis(basis_type, N_x, N_y, x_domain, y_domain):
    """Create a TensorProductBasis2D for a single field."""
    bt = basis_type.lower()
    if bt == 'chebyshev':
        return create_chebyshev_basis_2d(N_x - 1, N_y - 1, x_domain, y_domain)
    elif bt == 'augsin_cheb':
        bx = AugmentedBasis1D(Fourier1D(N_x, x_domain, mode='sin'),
                               include_constant=True, include_linear=True)
        by = Chebyshev1D(N_y - 1, y_domain)
        return TensorProductBasis2D(bx, by)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


# =============================================================================
# LiL-Q SOLVER (QUASILINEARIZATION + QR)
# =============================================================================
def solve_lilq_kovasznay(
        physics: KovasznayPhysics,
        disc: KovasznayDiscretization,
        basis_u, basis_v, basis_p=None,
        lambda_mom: float = 1.0,
        lambda_cont: float = 1.0,
        lambda_bc: float = 10.0,
        max_iter: int = 30,
        tol: float = 1e-12,
        verbose: bool = True,
        analyze_svd: bool = False
) -> Dict:
    """
    Solve the Kovasznay flow via quasilinearized LiL-Q.

    At each outer iteration k:
      1. Evaluate u^(k), v^(k) and their derivatives at collocation points.
      2. Assemble the linearized 3-block system.
      3. Solve for theta^(k+1) = [theta_u; theta_v; theta_p] via QR.
      4. Check convergence.

    Parameters
    ----------
    basis_u, basis_v : TensorProductBasis2D
        Bases for velocity components.
    basis_p : TensorProductBasis2D or None
        Basis for pressure. If None, uses basis_u.
    max_iter : int
        Maximum quasilinearization iterations.
    tol : float
        Convergence tolerance on relative coefficient change.
    """
    if basis_p is None:
        basis_p = basis_u

    t_start = time.time()

    Pu = basis_u.n_basis
    Pv = basis_v.n_basis
    Pp = basis_p.n_basis
    P_total = Pu + Pv + Pp
    nu = physics.nu

    if verbose:
        print("=" * 70)
        print("LiL-Q SOLVE: Kovasznay Flow (Quasilinearized Navier-Stokes)")
        print(f"  Re = {physics.Re},  nu = {nu:.4f},  lam = {physics.lam:.4f}")
        print(f"  P_u={Pu}, P_v={Pv}, P_p={Pp}, P_total={P_total}")
        print("=" * 70)

    # --- Collocation points ---
    pts = generate_collocation_points(physics, disc, P_total)
    xp, yp = pts['x_pde'], pts['y_pde']
    n_pde = pts['n_pde']

    # --- Precompute ALL basis matrices at interior points ---
    Phi_u    = basis_u.evaluate(xp, yp)
    Phi_u_x  = basis_u.derivative(xp, yp, dx=1, dy=0)
    Phi_u_y  = basis_u.derivative(xp, yp, dx=0, dy=1)
    Phi_u_xx = basis_u.derivative(xp, yp, dx=2, dy=0)
    Phi_u_yy = basis_u.derivative(xp, yp, dx=0, dy=2)

    Phi_v    = basis_v.evaluate(xp, yp)
    Phi_v_x  = basis_v.derivative(xp, yp, dx=1, dy=0)
    Phi_v_y  = basis_v.derivative(xp, yp, dx=0, dy=1)
    Phi_v_xx = basis_v.derivative(xp, yp, dx=2, dy=0)
    Phi_v_yy = basis_v.derivative(xp, yp, dx=0, dy=2)

    Phi_p    = basis_p.evaluate(xp, yp)
    Phi_p_x  = basis_p.derivative(xp, yp, dx=1, dy=0)
    Phi_p_y  = basis_p.derivative(xp, yp, dx=0, dy=1)

    # Diffusion matrices (constant across iterations)
    Diff_u = -nu * (Phi_u_xx + Phi_u_yy)   # (n_pde, Pu)
    Diff_v = -nu * (Phi_v_xx + Phi_v_yy)   # (n_pde, Pv)

    # --- Precompute BC basis matrices ---
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
    n_bc_total = sum(b['n'] for b in bc_blocks.values())

    # --- Pressure pin: fix p at one corner ---
    x_pin = np.array([physics.x_domain[0]], dtype=np.float64)
    y_pin = np.array([physics.y_domain[0]], dtype=np.float64)
    Phi_p_pin = basis_p.evaluate(x_pin, y_pin)
    p_pin_val = physics.exact_p(x_pin[0], y_pin[0])

    if verbose:
        print(f"  Interior points: {n_pde}")
        print(f"  BC points per edge: {pts['n_bc_edge']}")
        print(f"  Pressure pinned at ({x_pin[0]}, {y_pin[0]}), p = {p_pin_val:.6f}")

    # --- Initialize coefficients to zero (first iteration = Stokes) ---
    theta_u = np.zeros(Pu, dtype=np.float64)
    theta_v = np.zeros(Pv, dtype=np.float64)
    theta_p = np.zeros(Pp, dtype=np.float64)

    # --- Iteration history ---
    history = {
        'iteration': [],
        'coeff_change': [],
        'pde_residual': [],
        'continuity_residual': [],
        'solve_time': [],
        'cond_number': [],
    }

    # =================================================================
    # QUASILINEARIZATION LOOP
    # =================================================================
    for k in range(max_iter):
        t_iter = time.time()

        # --- Evaluate current fields at interior points ---
        uk   = Phi_u   @ theta_u       # (n_pde,)
        uk_x = Phi_u_x @ theta_u
        uk_y = Phi_u_y @ theta_u
        vk   = Phi_v   @ theta_v
        vk_x = Phi_v_x @ theta_v
        vk_y = Phi_v_y @ theta_v

        # =============================================================
        # ASSEMBLE LINEARIZED SYSTEM
        # =============================================================
        # x-momentum (linearized):
        #   u^k * u^{k+1}_x + u^{k+1} * u^k_x
        #   + v^k * u^{k+1}_y + v^{k+1} * u^k_y
        #   + p^{k+1}_x - nu * laplacian(u^{k+1})
        #   = u^k * u^k_x + v^k * u^k_y
        #
        # Blocks for theta_u:
        A_mom1_u = (uk[:, None] * Phi_u_x
                    + uk_x[:, None] * Phi_u
                    + vk[:, None] * Phi_u_y
                    + Diff_u)                       # (n_pde, Pu)
        # Block for theta_v:
        A_mom1_v = uk_y[:, None] * Phi_v            # (n_pde, Pv)
        # Block for theta_p:
        A_mom1_p = Phi_p_x                          # (n_pde, Pp)
        # RHS:
        b_mom1 = uk * uk_x + vk * uk_y             # (n_pde,)

        # y-momentum (linearized):
        #   u^k * v^{k+1}_x + u^{k+1} * v^k_x
        #   + v^k * v^{k+1}_y + v^{k+1} * v^k_y
        #   + p^{k+1}_y - nu * laplacian(v^{k+1})
        #   = u^k * v^k_x + v^k * v^k_y
        A_mom2_u = vk_x[:, None] * Phi_u            # (n_pde, Pu)
        A_mom2_v = (uk[:, None] * Phi_v_x
                    + vk_y[:, None] * Phi_v
                    + vk[:, None] * Phi_v_y
                    + Diff_v)                       # (n_pde, Pv)
        A_mom2_p = Phi_p_y                          # (n_pde, Pp)
        b_mom2 = uk * vk_x + vk * vk_y             # (n_pde,)

        # Continuity (linear, no linearization needed):
        #   u^{k+1}_x + v^{k+1}_y = 0
        A_cont_u = Phi_u_x                          # (n_pde, Pu)
        A_cont_v = Phi_v_y                          # (n_pde, Pv)
        Z_cont_p = np.zeros((n_pde, Pp))

        # --- Weights ---
        w_mom  = np.sqrt(lambda_mom / n_pde)
        w_cont = np.sqrt(lambda_cont / n_pde)

        # --- Stack PDE rows ---
        A_rows = [
            w_mom * np.hstack([A_mom1_u, A_mom1_v, A_mom1_p]),
            w_mom * np.hstack([A_mom2_u, A_mom2_v, A_mom2_p]),
            w_cont * np.hstack([A_cont_u, A_cont_v, Z_cont_p]),
        ]
        b_rows = [
            w_mom * b_mom1,
            w_mom * b_mom2,
            w_cont * np.zeros(n_pde),
        ]

        # --- BC rows (Dirichlet for u, v from exact solution) ---
        for edge in ['bot', 'top', 'left', 'right']:
            blk = bc_blocks[edge]
            ne = blk['n']
            w_bc = np.sqrt(lambda_bc / ne)

            # u BC
            A_rows.append(w_bc * np.hstack([
                blk['Phi_u'], np.zeros((ne, Pv)), np.zeros((ne, Pp))]))
            b_rows.append(w_bc * blk['u_exact'])

            # v BC
            A_rows.append(w_bc * np.hstack([
                np.zeros((ne, Pu)), blk['Phi_v'], np.zeros((ne, Pp))]))
            b_rows.append(w_bc * blk['v_exact'])

        # --- Pressure pin ---
        w_pin = np.sqrt(lambda_bc)
        A_rows.append(w_pin * np.hstack([
            np.zeros((1, Pu)), np.zeros((1, Pv)), Phi_p_pin]))
        b_rows.append(w_pin * np.array([p_pin_val]))

        # --- Solve ---
        A_sys = np.vstack(A_rows)
        b_sys = np.concatenate(b_rows)

        theta_new, _, rank, _ = scipy.linalg.lstsq(A_sys, b_sys,
                                                     lapack_driver='gelsy')
        dt = time.time() - t_iter
        
        A_cond = 0.0
        if analyze_svd:
            A_cond = np.linalg.cond(A_sys)

            # 1. Compute the SVD to get the singular values (sigma)
            U, sigma, Vh = np.linalg.svd(A_sys, full_matrices=False)
            
            # 2. Compute the condition number kappa
            # \kappa := \sigma_max / \sigma_min
            kappa = sigma[0] / sigma[-1]
            
            # 3. Compute the quartiles (Percentiles: 100%, 75%, 50%, 25%, 0%)
            # q4 corresponds to maximum (sigma[0]) and q0 corresponds to minimum (sigma[-1])
            q4, q3, q2, q1, q0 = np.percentile(sigma, [100, 75, 50, 25, 0])
            
            # 4. Define the tolerance and compute the numerical rank
            # Machine epsilon for float64
            epsilon = np.finfo(A_sys.dtype).eps 
            tolerance = max(A_sys.shape) * sigma[0] * epsilon
            
            # Count singular values strictly greater than the tolerance
            numerical_rank = np.sum(sigma > tolerance)
            
            # Print clean summary containing the quartiles
            print(f"  SVD analysis:\n"
                f"   sigma_max = {q4:.4e}, Q3 = {q3:.4e}, median = {q2:.4e}, Q1 = {q1:.4e}, sigma_min = {q0:.4e}\n"
                f"   kappa = {kappa:.4e}, epsilon = {epsilon:.4e}, \n"
                f"   tolerance = {tolerance:.4e}, numerical rank = {numerical_rank}/{A_sys.shape[1]}")
        
        theta_u_new = theta_new[:Pu]
        theta_v_new = theta_new[Pu:Pu+Pv]
        theta_p_new = theta_new[Pu+Pv:]

        # --- Convergence check ---
        theta_old = np.concatenate([theta_u, theta_v, theta_p])
        delta = np.linalg.norm(theta_new - theta_old)
        rel_delta = delta / (np.linalg.norm(theta_new) + 1e-30)

        # PDE residual (evaluate full nonlinear NS)
        u_new   = Phi_u   @ theta_u_new
        u_new_x = Phi_u_x @ theta_u_new
        u_new_y = Phi_u_y @ theta_u_new
        v_new   = Phi_v   @ theta_v_new
        v_new_x = Phi_v_x @ theta_v_new
        v_new_y = Phi_v_y @ theta_v_new
        p_new_x = Phi_p_x @ theta_p_new
        p_new_y = Phi_p_y @ theta_p_new
        lap_u   = (Phi_u_xx + Phi_u_yy) @ theta_u_new
        lap_v   = (Phi_v_xx + Phi_v_yy) @ theta_v_new

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
        history['cond_number'].append(A_cond)

        if verbose:
            print(f"  Iter {k:3d}: coeff_change={rel_delta:.3e}  "
                  f"PDE_res={pde_res:.3e}  cont_res={cont_res:.3e}  "
                  f"QR={dt:.4f}s  rank={rank}")

        # Update
        theta_u = theta_u_new
        theta_v = theta_v_new
        theta_p = theta_p_new

        if rel_delta < tol:
            if verbose:
                print(f"  Converged at iteration {k}.")
            break

    total_time = time.time() - t_start

    # =================================================================
    # FINAL ERRORS
    # =================================================================
    n_ev = 200
    x_ev = np.linspace(physics.x_domain[0], physics.x_domain[1], n_ev, dtype=np.float64)
    y_ev = np.linspace(physics.y_domain[0], physics.y_domain[1], n_ev, dtype=np.float64)
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
        print(f"  Outer iterations: {k+1}")
        print(f"  Final rel L2 errors:  u={rel_l2_u:.3e}  v={rel_l2_v:.3e}  p={rel_l2_p:.3e}")
        print("=" * 70)

    return {
        'basis_u': basis_u, 'basis_v': basis_v, 'basis_p': basis_p,
        'theta_u': theta_u, 'theta_v': theta_v, 'theta_p': theta_p,
        'n_params': P_total, 'n_params_u': Pu, 'n_params_v': Pv, 'n_params_p': Pp,
        'n_outer_iters': k + 1,
        'solve_time_total': total_time,
        'rel_l2_u': rel_l2_u, 'rel_l2_v': rel_l2_v, 'rel_l2_p': rel_l2_p,
        'pde_mse': pde_res, 'cont_mse': cont_res,
        'history': history,
    }


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_all_fields(basis_u, basis_v, basis_p, theta_u, theta_v, theta_p,
                         physics, n_eval=200):
    x_ev = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval, dtype=np.float64)
    y_ev = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()
    return {
        'X': XX, 'Y': YY,
        'u': (basis_u.evaluate(xf, yf) @ theta_u).reshape(XX.shape),
        'v': (basis_v.evaluate(xf, yf) @ theta_v).reshape(XX.shape),
        'p': (basis_p.evaluate(xf, yf) @ theta_p).reshape(XX.shape),
    }


def evaluate_exact_fields(physics, n_eval=200):
    x_ev = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval, dtype=np.float64)
    y_ev = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()
    return {
        'X': XX, 'Y': YY,
        'u': physics.exact_u(xf, yf).reshape(XX.shape),
        'v': physics.exact_v(xf, yf).reshape(XX.shape),
        'p': physics.exact_p(xf, yf).reshape(XX.shape),
    }
