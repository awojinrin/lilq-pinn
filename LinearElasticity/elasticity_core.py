"""
Linear Elasticity Core Module — LiL Forward Solver (v2)
========================================================

Solves the 2D plane-strain linear elasticity problem from
Haghighat, Raissi, Moure, Gomez & Juanes (CMAME 2021)
using the LiL (Linear-in-Learnables) direct solve.

Problem:
    Displacement formulation on [0,1] x [0,1]:

    (lam+2*mu)*u_xx + mu*u_yy + (lam+mu)*v_xy = -f_x
    mu*v_xx + (lam+2*mu)*v_yy + (lam+mu)*u_xy = -f_y

Boundary conditions (matching Fig. 1 of the paper):
    Bottom (y=0):  u_x = 0,  u_y = 0
    Top    (y=1):  u_x = 0,  sigma_yy = (lam+2*mu)*Q*sin(pi*x)
    Left   (x=0):  sigma_xx = 0,  u_y = 0
    Right  (x=1):  sigma_xx = 0,  u_y = 0

The Neumann (traction) BCs are expressed in terms of displacements:
    sigma_xx = (lam+2*mu)*u_{x,x} + lam*u_{y,y}
    sigma_yy = (lam+2*mu)*u_{y,y} + lam*u_{x,x}

Exact solution (Eqs. 6-7 of the paper):
    u_x(x,y) = cos(2*pi*x) * sin(pi*y)
    u_y(x,y) = sin(pi*x) * Q * y^4 / 4

Parameters: lambda=1, mu=0.5, Q=4

Changes from v1:
    - Proper mixed Dirichlet/Neumann BCs matching the paper
    - Independent basis for u_x and u_y fields
    - Optional all-Dirichlet mode for comparison

Author: Gbenga / Claude collaboration
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
import time
import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional

from lilq_shared.basis import (
    Chebyshev1D, Fourier1D, TensorProductBasis2D,
    AugmentedBasis1D,
    create_chebyshev_basis_2d, create_fourier_basis_2d, create_mixed_basis_2d
)

np.set_printoptions(precision=16)
pi = np.pi


# =============================================================================
# PHYSICS: Exact solutions, body forces, derived quantities
# =============================================================================

@dataclass
class ElasticityPhysics:
    """Physics configuration for the linear elasticity benchmark.

    PDE (displacement formulation, plane strain):
        (lam+2*mu)*u_xx + mu*u_yy + (lam+mu)*v_xy + f_x = 0
        mu*v_xx + (lam+2*mu)*v_yy + (lam+mu)*u_xy + f_y = 0

    Domain: [0,1] x [0,1]
    """
    lam: float = 1.0
    mu: float = 0.5
    Q: float = 4.0
    x_domain: Tuple[float, float] = (0.0, 1.0)
    y_domain: Tuple[float, float] = (0.0, 1.0)

    @property
    def C11(self):
        return self.lam + 2.0 * self.mu

    @property
    def C12(self):
        return self.lam

    @property
    def C33(self):
        return 2.0 * self.mu

    # --- Exact displacement fields ---
    def exact_ux(self, x, y):
        return np.cos(2*pi*x) * np.sin(pi*y)

    def exact_uy(self, x, y):
        return np.sin(pi*x) * self.Q * y**4 / 4.0

    # --- Exact strain fields ---
    def exact_exx(self, x, y):
        return -2*pi*np.sin(2*pi*x)*np.sin(pi*y)

    def exact_eyy(self, x, y):
        return np.sin(pi*x)*self.Q*y**3

    def exact_exy(self, x, y):
        return 0.5*(pi*np.cos(2*pi*x)*np.cos(pi*y)
                     + pi*np.cos(pi*x)*self.Q*y**4/4.0)

    # --- Exact stress fields ---
    def exact_sxx(self, x, y):
        return self.C11*self.exact_exx(x, y) + self.C12*self.exact_eyy(x, y)

    def exact_syy(self, x, y):
        return self.C11*self.exact_eyy(x, y) + self.C12*self.exact_exx(x, y)

    def exact_sxy(self, x, y):
        return self.C33*self.exact_exy(x, y)

    # --- Body force fields (Eq. 5 of the paper) ---
    def body_force_x(self, x, y):
        lam, mu, Q = self.lam, self.mu, self.Q
        fx = (- lam*(4*pi**2*np.cos(2*pi*x)*np.sin(pi*y)
                      - Q*y**3*pi*np.cos(pi*x))
              - mu*(pi**2*np.cos(2*pi*x)*np.sin(pi*y)
                    - Q*y**3*pi*np.cos(pi*x))
              - 8*mu*pi**2*np.cos(2*pi*x)*np.sin(pi*y))
        return fx

    def body_force_y(self, x, y):
        lam, mu, Q = self.lam, self.mu, self.Q
        fy = (lam*(3*Q*y**2*np.sin(pi*x)
                    - 2*pi**2*np.cos(pi*y)*np.sin(2*pi*x))
              - mu*(2*pi**2*np.cos(pi*y)*np.sin(2*pi*x)
                    + (Q*y**4*pi**2*np.sin(pi*x))/4.0)
              + 6*Q*mu*y**2*np.sin(pi*x))
        return fy


# =============================================================================
# DISCRETISATION
# =============================================================================

@dataclass
class ElasticityDiscretization:
    """Discretization parameters."""
    N_x: int = 15
    N_y: int = 15
    k_ratio: int = 10
    collocation_ratios: Tuple[float, float] = (0.85, 0.15)
    domain_sampling: str = "uniform"
    boundary_sampling: str = "uniform"
    seed: int = 42


# =============================================================================
# COLLOCATION POINT GENERATION
# =============================================================================

def generate_collocation_points(physics: ElasticityPhysics,
                                 disc: ElasticityDiscretization,
                                 P_total: int) -> Dict:
    """Generate interior PDE and per-edge boundary collocation points.

    Returns separate arrays for each boundary edge so that
    different BC types can be applied per edge.
    """
    np.random.seed(disc.seed)

    ratios = disc.collocation_ratios
    norm_ratios = [r / sum(ratios) for r in ratios]

    x_min, x_max = physics.x_domain
    y_min, y_max = physics.y_domain

    # Interior points
    n_pde = disc.k_ratio * norm_ratios[0] * P_total / 2
    n_pde_dim = max(math.ceil(np.sqrt(n_pde)), 10)

    if disc.domain_sampling == "random":
        x_pde = np.sort(np.random.uniform(x_min + 1e-6, x_max - 1e-6, n_pde_dim)).astype(np.float64)
        y_pde = np.sort(np.random.uniform(y_min + 1e-6, y_max - 1e-6, n_pde_dim)).astype(np.float64)
    else:
        x_pde = np.linspace(x_min + 1e-6, x_max - 1e-6, n_pde_dim, dtype=np.float64)
        y_pde = np.linspace(y_min + 1e-6, y_max - 1e-6, n_pde_dim, dtype=np.float64)

    xx_pde, yy_pde = np.meshgrid(x_pde, y_pde)
    pts_pde_x = xx_pde.ravel()
    pts_pde_y = yy_pde.ravel()

    # Boundary points per edge
    n_bc_edge = max(math.ceil(disc.k_ratio * norm_ratios[1] * P_total / (2 * 4)), 10)

    if disc.boundary_sampling == "random":
        t_x = np.sort(np.random.uniform(x_min, x_max, n_bc_edge)).astype(np.float64)
        t_y = np.sort(np.random.uniform(y_min, y_max, n_bc_edge)).astype(np.float64)
    else:
        t_x = np.linspace(x_min, x_max, n_bc_edge, dtype=np.float64)
        t_y = np.linspace(y_min, y_max, n_bc_edge, dtype=np.float64)

    return {
        'x_pde': pts_pde_x, 'y_pde': pts_pde_y,
        'n_pde': len(pts_pde_x),
        # Bottom edge (y = y_min)
        'x_bot': t_x, 'y_bot': np.full_like(t_x, y_min),
        # Top edge (y = y_max)
        'x_top': t_x, 'y_top': np.full_like(t_x, y_max),
        # Left edge (x = x_min)
        'x_left': np.full_like(t_y, x_min), 'y_left': t_y,
        # Right edge (x = x_max)
        'x_right': np.full_like(t_y, x_max), 'y_right': t_y,
        'n_bc_edge': len(t_x),
    }


# =============================================================================
# BASIS CREATION
# =============================================================================

def create_elasticity_basis(basis_type: str, N_x: int, N_y: int,
                             x_domain=(0.0, 1.0), y_domain=(0.0, 1.0)):
    """Create a TensorProductBasis2D for a single field.

    Parameters
    ----------
    basis_type : str
        'chebyshev', 'fourier', 'sin_sin', 'sin_cheb',
        'cheb_sin', 'augsin_cheb', 'augsin_augsin', etc.
    N_x, N_y : int
        Basis dimension parameters in each direction.
    """
    bt = basis_type.lower()

    if bt == 'chebyshev':
        return create_chebyshev_basis_2d(N_x - 1, N_y - 1, x_domain, y_domain)
    elif bt == 'fourier':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain,
                                        mode_x='both', mode_y='both')
    elif bt == 'sin_sin':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain,
                                        mode_x='sin', mode_y='sin')
    elif bt == 'cos_sin':
        return create_mixed_basis_2d('cos', 'sin', N_x, N_y, x_domain, y_domain)
    elif bt == 'sin_cheb':
        return create_mixed_basis_2d('sin', 'chebyshev', N_x, N_y, x_domain, y_domain)
    elif bt == 'cheb_sin':
        return create_mixed_basis_2d('chebyshev', 'sin', N_x, N_y, x_domain, y_domain)
    elif bt == 'cheb_cos':
        return create_mixed_basis_2d('chebyshev', 'cos', N_x, N_y, x_domain, y_domain)
    elif bt == 'cos_cheb':
        return create_mixed_basis_2d('cos', 'chebyshev', N_x, N_y, x_domain, y_domain)
    elif bt == 'augsin_cheb':
        basis_x = AugmentedBasis1D(
            Fourier1D(N_x, x_domain, mode='sin'),
            include_constant=True, include_linear=True)
        basis_y = Chebyshev1D(N_y - 1, y_domain)
        return TensorProductBasis2D(basis_x, basis_y)
    elif bt == 'augsin_augsin':
        basis_x = AugmentedBasis1D(
            Fourier1D(N_x, x_domain, mode='sin'),
            include_constant=True, include_linear=True)
        basis_y = AugmentedBasis1D(
            Fourier1D(N_y, y_domain, mode='sin'),
            include_constant=True, include_linear=True)
        return TensorProductBasis2D(basis_x, basis_y)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


# =============================================================================
# LiL DIRECT SOLVER
# =============================================================================

def solve_lil_elasticity(
        physics: ElasticityPhysics,
        disc: ElasticityDiscretization,
        basis_u,
        basis_v=None,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        bc_mode: str = "paper",
        verbose: bool = True,
        analyze_svd: bool = False
) -> Dict:
    """
    Solve the coupled linear elasticity system via a single LiL QR solve.

    Parameters
    ----------
    physics : ElasticityPhysics
    disc : ElasticityDiscretization
    basis_u : TensorProductBasis2D
        Basis for the u_x displacement field.
    basis_v : TensorProductBasis2D or None
        Basis for the u_y displacement field.
        If None, uses the same basis as basis_u.
    lambda_pde, lambda_bc : float
        Weights for PDE and BC residuals in the least-squares system.
    bc_mode : str
        'paper'  — Mixed Dirichlet/Neumann BCs matching Fig. 1 of the paper.
        'exact'  — All-Dirichlet BCs from the exact solution on every edge.
    verbose : bool

    Returns
    -------
    result : dict
    """
    if basis_v is None:
        basis_v = basis_u

    t_start = time.time()

    Pu = basis_u.n_basis   # Basis dimension for u_x
    Pv = basis_v.n_basis   # Basis dimension for u_y
    P_total = Pu + Pv

    if verbose:
        print("=" * 70)
        print("LiL DIRECT SOLVE: Linear Elasticity (Coupled System)")
        print(f"  Basis u_x: {basis_u}")
        print(f"  Basis u_y: {basis_v}")
        print(f"  P_u = {Pu},  P_v = {Pv},  P_total = {P_total}")
        print(f"  BC mode: {bc_mode}")
        print("=" * 70)

    # --- Material constants ---
    lam = physics.lam
    mu  = physics.mu
    C11 = physics.C11       # lam + 2*mu
    C12 = physics.C12       # lam
    C_cross = lam + mu

    # --- Collocation points ---
    pts = generate_collocation_points(physics, disc, P_total)
    x_pde, y_pde = pts['x_pde'], pts['y_pde']
    n_pde = pts['n_pde']

    if verbose:
        print(f"  Interior points: {n_pde}")
        print(f"  BC points per edge: {pts['n_bc_edge']}")

    # =====================================================================
    # INTERIOR PDE ROWS
    # =====================================================================
    # Basis matrices for u_x at interior points
    Phi_u      = basis_u.evaluate(x_pde, y_pde)                    # (n_pde, Pu)
    Phi_u_xx   = basis_u.derivative(x_pde, y_pde, dx=2, dy=0)
    Phi_u_yy   = basis_u.derivative(x_pde, y_pde, dx=0, dy=2)
    Phi_u_xy   = basis_u.derivative(x_pde, y_pde, dx=1, dy=1)

    # Basis matrices for u_y at interior points
    Phi_v      = basis_v.evaluate(x_pde, y_pde)                    # (n_pde, Pv)
    Phi_v_xx   = basis_v.derivative(x_pde, y_pde, dx=2, dy=0)
    Phi_v_yy   = basis_v.derivative(x_pde, y_pde, dx=0, dy=2)
    Phi_v_xy   = basis_v.derivative(x_pde, y_pde, dx=1, dy=1)

    # PDE block matrices
    #   x-momentum: C11*u_xx + mu*u_yy + C_cross*v_xy = -f_x
    #   y-momentum: mu*v_xx + C11*v_yy + C_cross*u_xy = -f_y
    A_pde_uu = C11 * Phi_u_xx + mu * Phi_u_yy      # (n_pde, Pu)
    A_pde_uv = C_cross * Phi_v_xy                   # (n_pde, Pv)
    A_pde_vu = C_cross * Phi_u_xy                   # (n_pde, Pu)
    A_pde_vv = mu * Phi_v_xx + C11 * Phi_v_yy       # (n_pde, Pv)

    fx = physics.body_force_x(x_pde, y_pde)
    fy = physics.body_force_y(x_pde, y_pde)

    w_pde = np.sqrt(lambda_pde / n_pde)

    A_rows = [
        w_pde * np.hstack([A_pde_uu, A_pde_uv]),     # x-momentum
        w_pde * np.hstack([A_pde_vu, A_pde_vv]),     # y-momentum
    ]
    b_rows = [
        w_pde * fx,
        w_pde * fy,
    ]

    # =====================================================================
    # BOUNDARY CONDITION ROWS
    # =====================================================================
    n_bc_total = 0

    if bc_mode == "paper":
        # ---------------------------------------------------------------
        # Fig. 1 BCs:
        #   Bottom (y=0):  u_x = 0,  u_y = 0
        #   Top    (y=1):  u_x = 0,  sigma_yy = (lam+2*mu)*Q*sin(pi*x)
        #   Left   (x=0):  sigma_xx = 0,  u_y = 0
        #   Right  (x=1):  sigma_xx = 0,  u_y = 0
        #
        # sigma_xx = C11*u_{x,x} + C12*u_{y,y}
        # sigma_yy = C11*u_{y,y} + C12*u_{x,x}
        # ---------------------------------------------------------------

        def _dirichlet_rows_u(x_bc, y_bc, val):
            """Dirichlet row for u_x: [Phi_u | 0] @ theta = val"""
            Phi = basis_u.evaluate(x_bc, y_bc)
            n = len(x_bc)
            A = np.hstack([Phi, np.zeros((n, Pv))])
            return A, val

        def _dirichlet_rows_v(x_bc, y_bc, val):
            """Dirichlet row for u_y: [0 | Phi_v] @ theta = val"""
            Phi = basis_v.evaluate(x_bc, y_bc)
            n = len(x_bc)
            A = np.hstack([np.zeros((n, Pu)), Phi])
            return A, val

        def _neumann_sxx_rows(x_bc, y_bc, val):
            """sigma_xx = C11*u_{x,x} + C12*u_{y,y} = val"""
            dPhiu_dx = basis_u.derivative(x_bc, y_bc, dx=1, dy=0)
            dPhiv_dy = basis_v.derivative(x_bc, y_bc, dx=0, dy=1)
            n = len(x_bc)
            A = np.hstack([C11 * dPhiu_dx, C12 * dPhiv_dy])
            return A, val

        def _neumann_syy_rows(x_bc, y_bc, val):
            """sigma_yy = C11*u_{y,y} + C12*u_{x,x} = val"""
            dPhiu_dx = basis_u.derivative(x_bc, y_bc, dx=1, dy=0)
            dPhiv_dy = basis_v.derivative(x_bc, y_bc, dx=0, dy=1)
            n = len(x_bc)
            A = np.hstack([C12 * dPhiu_dx, C11 * dPhiv_dy])
            return A, val

        # --- Bottom (y=0): u_x=0, u_y=0 ---
        xb, yb = pts['x_bot'], pts['y_bot']
        A_bc, b_bc = _dirichlet_rows_u(xb, yb, np.zeros(len(xb)))
        A_rows.append(np.sqrt(lambda_bc / len(xb)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xb)) * b_bc)

        A_bc, b_bc = _dirichlet_rows_v(xb, yb, np.zeros(len(xb)))
        A_rows.append(np.sqrt(lambda_bc / len(xb)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xb)) * b_bc)
        n_bc_total += 2 * len(xb)

        # --- Top (y=1): u_x=0, sigma_yy = (lam+2*mu)*Q*sin(pi*x) ---
        xt, yt = pts['x_top'], pts['y_top']
        A_bc, b_bc = _dirichlet_rows_u(xt, yt, np.zeros(len(xt)))
        A_rows.append(np.sqrt(lambda_bc / len(xt)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xt)) * b_bc)

        syy_top = C11 * physics.Q * np.sin(pi * xt)
        A_bc, b_bc = _neumann_syy_rows(xt, yt, syy_top)
        A_rows.append(np.sqrt(lambda_bc / len(xt)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xt)) * b_bc)
        n_bc_total += 2 * len(xt)

        # --- Left (x=0): sigma_xx=0, u_y=0 ---
        xl, yl = pts['x_left'], pts['y_left']
        A_bc, b_bc = _neumann_sxx_rows(xl, yl, np.zeros(len(xl)))
        A_rows.append(np.sqrt(lambda_bc / len(xl)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xl)) * b_bc)

        A_bc, b_bc = _dirichlet_rows_v(xl, yl, np.zeros(len(xl)))
        A_rows.append(np.sqrt(lambda_bc / len(xl)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xl)) * b_bc)
        n_bc_total += 2 * len(xl)

        # --- Right (x=1): sigma_xx=0, u_y=0 ---
        xr, yr = pts['x_right'], pts['y_right']
        A_bc, b_bc = _neumann_sxx_rows(xr, yr, np.zeros(len(xr)))
        A_rows.append(np.sqrt(lambda_bc / len(xr)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xr)) * b_bc)

        A_bc, b_bc = _dirichlet_rows_v(xr, yr, np.zeros(len(xr)))
        A_rows.append(np.sqrt(lambda_bc / len(xr)) * A_bc)
        b_rows.append(np.sqrt(lambda_bc / len(xr)) * b_bc)
        n_bc_total += 2 * len(xr)

    elif bc_mode == "exact":
        # All-Dirichlet from the exact solution on every edge
        for edge in ['bot', 'top', 'left', 'right']:
            xe = pts[f'x_{edge}']
            ye = pts[f'y_{edge}']
            ne = len(xe)
            w = np.sqrt(lambda_bc / ne)

            Phi_bc_u = basis_u.evaluate(xe, ye)
            Phi_bc_v = basis_v.evaluate(xe, ye)

            # u_x BC
            A_rows.append(w * np.hstack([Phi_bc_u, np.zeros((ne, Pv))]))
            b_rows.append(w * physics.exact_ux(xe, ye))

            # u_y BC
            A_rows.append(w * np.hstack([np.zeros((ne, Pu)), Phi_bc_v]))
            b_rows.append(w * physics.exact_uy(xe, ye))

            n_bc_total += 2 * ne
    else:
        raise ValueError(f"Unknown bc_mode: {bc_mode}")

    # =====================================================================
    # STACK AND SOLVE
    # =====================================================================
    A_sys = np.vstack(A_rows)
    b_sys = np.concatenate(b_rows)

    if verbose:
        print(f"  Total BC rows: {n_bc_total}")
        print(f"  System matrix shape: {A_sys.shape}")
        print(f"  Oversampling ratio: {A_sys.shape[0] / A_sys.shape[1]:.1f}")

    t_solve = time.time()
    theta, _, rank, _ = scipy.linalg.lstsq(A_sys, b_sys, lapack_driver='gelsy')
    solve_time = time.time() - t_solve

    kappa = 0.0
    if analyze_svd:
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
        

    theta_u = theta[:Pu]
    theta_v = theta[Pu:]

    # Determine how many nonzero values are in theta_u and theta_v (for sparsity insight)
    # nonzero_theta_u = np.sum(np.abs(theta_u) > 1e-6)
    # nonzero_theta_v = np.sum(np.abs(theta_v) > 1e-6)
    # print(f"  Nonzero coefficients (|theta| > 1e-6): u_x: {nonzero_theta_u}/{Pu}, u_y: {nonzero_theta_v}/{Pv}")

    # =====================================================================
    # COMPUTE RESIDUALS AND ERRORS
    # =====================================================================
    # PDE residuals (unweighted)
    pde_res_x = A_pde_uu @ theta_u + A_pde_uv @ theta_v - fx
    pde_res_y = A_pde_vu @ theta_u + A_pde_vv @ theta_v - fy
    pde_mse_x = float(np.mean(pde_res_x**2))
    pde_mse_y = float(np.mean(pde_res_y**2))
    pde_mse = 0.5 * (pde_mse_x + pde_mse_y)

    # Errors against exact solution (fine grid)
    n_eval = 200
    x_ev = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval, dtype=np.float64)
    y_ev = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()

    ux_pred  = basis_u.evaluate(xf, yf) @ theta_u
    uy_pred  = basis_v.evaluate(xf, yf) @ theta_v
    ux_exact = physics.exact_ux(xf, yf)
    uy_exact = physics.exact_uy(xf, yf)

    l2_err_ux  = np.sqrt(np.mean((ux_pred - ux_exact)**2))
    l2_err_uy  = np.sqrt(np.mean((uy_pred - uy_exact)**2))
    linf_err_ux = np.max(np.abs(ux_pred - ux_exact))
    linf_err_uy = np.max(np.abs(uy_pred - uy_exact))

    l2_exact_ux = np.sqrt(np.mean(ux_exact**2))
    l2_exact_uy = np.sqrt(np.mean(uy_exact**2))
    rel_l2_ux = l2_err_ux / max(l2_exact_ux, 1e-15)
    rel_l2_uy = l2_err_uy / max(l2_exact_uy, 1e-15)

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
        print(f"\n  QR solve time:   {solve_time:.4f} s")
        print(f"  Total wall time: {total_time:.4f} s")
        print(f"  Matrix rank: {rank}")
        print(f"\n  PDE residual MSE:  {pde_mse:.6e}")
        print(f"    x-momentum:      {pde_mse_x:.6e}")
        print(f"    y-momentum:      {pde_mse_y:.6e}")
        print(f"\n  Displacement errors (L2 relative):")
        print(f"    u_x: {rel_l2_ux:.6e}    (Linf: {linf_err_ux:.6e})")
        print(f"    u_y: {rel_l2_uy:.6e}    (Linf: {linf_err_uy:.6e})")
        print(f"  Stress errors (L2 relative):")
        print(f"    sigma_xx: {rel_l2_sxx:.6e}")
        print(f"    sigma_yy: {rel_l2_syy:.6e}")
        print(f"    sigma_xy: {rel_l2_sxy:.6e}")
        print("=" * 70)

    return {
        'basis_u': basis_u,
        'basis_v': basis_v,
        'theta_u': theta_u,
        'theta_v': theta_v,
        'theta_combined': theta,
        'n_params': P_total,
        'n_params_u': Pu,
        'n_params_v': Pv,
        'solve_time_qr': solve_time,
        'solve_time_total': total_time,
        'matrix_rank': rank,
        'pde_mse': pde_mse,
        'pde_mse_x': pde_mse_x,
        'pde_mse_y': pde_mse_y,
        'l2_err_ux': l2_err_ux,
        'l2_err_uy': l2_err_uy,
        'linf_err_ux': linf_err_ux,
        'linf_err_uy': linf_err_uy,
        'rel_l2_ux': rel_l2_ux,
        'rel_l2_uy': rel_l2_uy,
        'rel_l2_sxx': rel_l2_sxx,
        'rel_l2_syy': rel_l2_syy,
        'rel_l2_sxy': rel_l2_sxy,
        'n_colloc_pde': n_pde,
        'n_colloc_bc': n_bc_total,
        'bc_mode': bc_mode,
        'cond_number': kappa,
    }


# =============================================================================
# EVALUATION FUNCTIONS (for plotting)
# =============================================================================

def evaluate_all_fields(basis_u, basis_v, theta_u, theta_v, physics, n_eval=200):
    """Evaluate displacements, strains, stresses on a uniform grid."""
    x_ev = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval, dtype=np.float64)
    y_ev = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()

    ux = (basis_u.evaluate(xf, yf) @ theta_u).reshape(XX.shape)
    uy = (basis_v.evaluate(xf, yf) @ theta_v).reshape(XX.shape)

    exx = (basis_u.derivative(xf, yf, dx=1, dy=0) @ theta_u).reshape(XX.shape)
    eyy = (basis_v.derivative(xf, yf, dx=0, dy=1) @ theta_v).reshape(XX.shape)
    exy = (0.5*(basis_u.derivative(xf, yf, dx=0, dy=1) @ theta_u
                + basis_v.derivative(xf, yf, dx=1, dy=0) @ theta_v)).reshape(XX.shape)

    C11, C12, C33 = physics.C11, physics.C12, physics.C33
    sxx = C11*exx + C12*eyy
    syy = C11*eyy + C12*exx
    sxy = C33*exy

    return {'X': XX, 'Y': YY,
            'ux': ux, 'uy': uy,
            'exx': exx, 'eyy': eyy, 'exy': exy,
            'sxx': sxx, 'syy': syy, 'sxy': sxy}


def evaluate_exact_fields(physics, n_eval=200):
    """Evaluate exact solution fields on a uniform grid."""
    x_ev = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval, dtype=np.float64)
    y_ev = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval, dtype=np.float64)
    XX, YY = np.meshgrid(x_ev, y_ev)
    xf, yf = XX.ravel(), YY.ravel()

    return {'X': XX, 'Y': YY,
            'ux': physics.exact_ux(xf, yf).reshape(XX.shape),
            'uy': physics.exact_uy(xf, yf).reshape(XX.shape),
            'exx': physics.exact_exx(xf, yf).reshape(XX.shape),
            'eyy': physics.exact_eyy(xf, yf).reshape(XX.shape),
            'exy': physics.exact_exy(xf, yf).reshape(XX.shape),
            'sxx': physics.exact_sxx(xf, yf).reshape(XX.shape),
            'syy': physics.exact_syy(xf, yf).reshape(XX.shape),
            'sxy': physics.exact_sxy(xf, yf).reshape(XX.shape)}
