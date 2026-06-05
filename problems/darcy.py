"""
SPE10 / Darcy Flow Problem
===========================

Single-phase Darcy flow through heterogeneous permeability (SPE10 benchmark).

Three solver approaches:
1. **FVM** (Finite Volume Method) -- reference solution with harmonic-mean
   transmissibilities and sparse direct solve.
2. **LiL-Q** -- Linear-in-Learnables with lifting function
   h*(x*,y*) = y* + h_tilde*(x*,y*) and sqrt(K*) row-equilibration.
3. **NiL-N** (PINN stub) -- placeholder for future neural-network solver.

Governing equations (dimensionless, sqrt(K*) normalization)::

    Darcy-x:    u*/sqrt(K*) + sqrt(K*) * R * dh_tilde*/dx*  = 0
    Darcy-y:    v*/sqrt(K*) + sqrt(K*) *     dh_tilde*/dy*  = -sqrt(K*)
    Continuity: R * du*/dx* + dv*/dy*                       = 0

Boundary conditions (satisfied exactly by basis construction):
    - h*(y*=0) = 0,  h*(y*=1) = 1   (Dirichlet, via sin(y*) vanishing)
    - dh*/dx* = 0 at x*=0,1          (Neumann, via cos(x*) derivative)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from pathlib import Path
import time

from lilq.basis import Fourier1D, TensorProductBasis2D, AugmentedBasis1D


# ── Data directory ───────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'spe10'


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DarcyConfig:
    """Configuration for the SPE10 Darcy flow problem."""
    # Grid
    NX_CELLS: int = 60
    NY_CELLS: int = 220
    DX: float = 20.0       # ft per cell in x
    DY: float = 10.0       # ft per cell in y

    # Boundary conditions
    P_TOP: float = 6000.0       # psi (injection)
    P_BOTTOM: float = 3000.0    # psi (production)

    # Permeability
    perm_file: str = 'perm_field_S3.txt'
    K_MIN_CLIP: float = 0.01    # mD floor (matches PINN)

    # Basis orders
    ORDER_H: int = 32
    ORDER_U: int = 32
    ORDER_V: int = 32

    # Solver
    solver_method: str = 'qr'   # 'qr' -> gelsy, 'lstsq' -> numpy lstsq

    # Output
    output_dir: str = 'darcy_results'


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

class DarcyPhysics:
    """Pre-computed physical quantities for the Darcy/SPE10 problem.

    Loads the permeability field, computes geometric-mean normalisation,
    and derives all scaling factors needed by both FVM and LiL-Q.

    Parameters
    ----------
    config : DarcyConfig
        Problem configuration.
    verbose : bool
        Print diagnostics on load.
    """

    def __init__(self, config: DarcyConfig, verbose: bool = True):
        self.config = config

        # Domain geometry
        self.LX = config.NX_CELLS * config.DX
        self.LY = config.NY_CELLS * config.DY
        self.R = self.LY / self.LX
        self.DELTA_P = config.P_TOP - config.P_BOTTOM

        # Load permeability
        self.K, self.K0 = self._load_permeability(config, verbose)

        # Normalised permeability
        self.K_star = self.K / self.K0
        self.sqrt_K_star = np.sqrt(self.K_star)

        # Scaling factors (matching PINN)
        self.dP_scale = self.DELTA_P / self.LY
        self.V_SCALE = self.K0 * self.dP_scale
        self.CE_SCALE = self.V_SCALE / self.LY

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _load_permeability(config: DarcyConfig,
                           verbose: bool) -> Tuple[np.ndarray, float]:
        """Load and clip SPE10 permeability field.

        Returns
        -------
        K : ndarray of shape (NX_CELLS, NY_CELLS)
            Clipped permeability in mD.
        K0 : float
            Geometric mean of K (log-space mean).
        """
        perm_path = _DATA_DIR / config.perm_file
        K_raw = np.loadtxt(perm_path)
        K = K_raw.reshape(config.NX_CELLS, config.NY_CELLS)
        K = np.maximum(K, config.K_MIN_CLIP)

        # Geometric mean (matches PINN: K0 = exp(mean(log(K))))
        K0 = float(np.exp(np.mean(np.log(K))))

        if verbose:
            sqrt_K_star = np.sqrt(K / K0)
            print("Loaded permeability: %s" % config.perm_file)
            print("  Shape: %s" % (K.shape,))
            print("  K range (after clip): [%.4f, %.2f] mD" % (K.min(), K.max()))
            print("  K ratio: %.0fx" % (K.max() / K.min()))
            print("  K0 (geometric mean): %.4f mD" % K0)
            print("  sqrt(K*) range: [%.4f, %.2f]" % (sqrt_K_star.min(), sqrt_K_star.max()))
            print("  sqrt(K*) ratio: %.0fx" % (sqrt_K_star.max() / sqrt_K_star.min()))

        return K, K0


# ─────────────────────────────────────────────────────────────────────────────
# FVM Reference Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_fvm(physics: DarcyPhysics) -> np.ndarray:
    """Finite-volume reference solver for single-phase Darcy flow.

    Uses harmonic-mean transmissibilities and a sparse direct solve,
    matching the PINN reference implementation exactly.

    Parameters
    ----------
    physics : DarcyPhysics
        Pre-computed physical quantities.

    Returns
    -------
    P : ndarray of shape (NX_CELLS, NY_CELLS)
        Cell-centre pressures in psi.
    """
    cfg = physics.config
    Nx, Ny = cfg.NX_CELLS, cfg.NY_CELLS
    dx = physics.LX / Nx
    dy = physics.LY / Ny

    # K_field shape: (NX, NY) -> transpose to (NY, NX) for FVM indexing
    Kx_mD = physics.K.T   # (NY, NX)
    Ky_mD = physics.K.T

    N = Nx * Ny
    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N)

    def gi(i, j):
        return j * Nx + i

    def hm(a, b_):
        return 2 * a * b_ / (a + b_) if (a + b_) > 1e-30 else 0.0

    for j in range(Ny):
        for i in range(Nx):
            k = gi(i, j)
            d = 0.0
            Kx = Kx_mD[j, i]
            Ky = Ky_mD[j, i]

            # x-neighbours
            if i > 0:
                T = hm(Kx, Kx_mD[j, i - 1]) * dy / dx
                A[k, gi(i - 1, j)] = T
                d -= T
            if i < Nx - 1:
                T = hm(Kx, Kx_mD[j, i + 1]) * dy / dx
                A[k, gi(i + 1, j)] = T
                d -= T

            # y-neighbours
            if j > 0:
                T = hm(Ky, Ky_mD[j - 1, i]) * dx / dy
                A[k, gi(i, j - 1)] = T
                d -= T
            else:
                # Bottom boundary (Dirichlet P_BOTTOM)
                Tb = 2.0 * Ky * dx / dy
                d -= Tb
                b[k] -= Tb * cfg.P_BOTTOM

            if j < Ny - 1:
                T = hm(Ky, Ky_mD[j + 1, i]) * dx / dy
                A[k, gi(i, j + 1)] = T
                d -= T
            else:
                # Top boundary (Dirichlet P_TOP)
                Tb = 2.0 * Ky * dx / dy
                d -= Tb
                b[k] -= Tb * cfg.P_TOP

            A[k, k] = d

    P = spsolve(A.tocsr(), b).reshape(Ny, Nx)
    return P.T  # Return shape (NX, NY)


# ─────────────────────────────────────────────────────────────────────────────
# Basis Construction
# ─────────────────────────────────────────────────────────────────────────────

def _create_basis_h_tilde(order: int) -> TensorProductBasis2D:
    """Basis for h_tilde* (lifting-function correction).

    h*(x*,y*) = y* + h_tilde*(x*,y*)

    Uses Cos(x) x Sin(y) on [0,1]^2.
    Sin(y) vanishes at y*=0 and y*=1 -> exact Dirichlet BCs.
    """
    basis_x = Fourier1D(n_modes=order, domain=(0, 1), mode='cos')
    basis_y = Fourier1D(n_modes=order, domain=(0, 1), mode='sin')
    return TensorProductBasis2D(basis_x, basis_y)


def _create_basis_u(order: int) -> TensorProductBasis2D:
    """Basis for u* (x-velocity).

    Sin(x) x AugCos(y).  Sin(x) vanishes at x*=0,1 (no-flow BC).
    """
    basis_x = Fourier1D(n_modes=order, domain=(0, 1), mode='sin')
    basis_y = AugmentedBasis1D(
        Fourier1D(n_modes=order, domain=(0, 1), mode='cos'),
        include_constant=False, include_linear=True,
    )
    return TensorProductBasis2D(basis_x, basis_y)


def _create_basis_v(order: int) -> TensorProductBasis2D:
    """Basis for v* (y-velocity).

    AugCos(x) x AugCos(y).
    """
    basis_x = AugmentedBasis1D(
        Fourier1D(n_modes=order, domain=(0, 1), mode='cos'),
        include_constant=False, include_linear=True,
    )
    basis_y = AugmentedBasis1D(
        Fourier1D(n_modes=order, domain=(0, 1), mode='cos'),
        include_constant=False, include_linear=True,
    )
    return TensorProductBasis2D(basis_x, basis_y)


# ─────────────────────────────────────────────────────────────────────────────
# LiL-Q Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_lilq_darcy(config: DarcyConfig,
                     physics: DarcyPhysics,
                     verbose: bool = True) -> Dict:
    """LiL-Q solver for Darcy flow with lifting function.

    Builds and solves the overdetermined collocation system

        [Darcy-x ]         [  0         ]
        [Darcy-y ] c   =   [ -sqrt(K*)  ]
        [Contin. ]         [  0         ]

    using ``scipy.linalg.lstsq`` with ``lapack_driver='gelsy'``.

    The system uses sqrt(K*) normalization for row equilibration matching
    the PINN formulation exactly.

    Parameters
    ----------
    config : DarcyConfig
    physics : DarcyPhysics
    verbose : bool

    Returns
    -------
    result : dict
        Keys: 'c_h_tilde', 'c_u', 'c_v', 'metrics',
              'basis_h_tilde', 'basis_u', 'basis_v', 'physics'.
    """
    cfg = config
    R = physics.R
    DELTA_P = physics.DELTA_P

    # ── Basis functions ──────────────────────────────────────────────────
    basis_h_tilde = _create_basis_h_tilde(cfg.ORDER_H)
    basis_u = _create_basis_u(cfg.ORDER_U)
    basis_v = _create_basis_v(cfg.ORDER_V)

    n_h = basis_h_tilde.n_basis
    n_u = basis_u.n_basis
    n_v = basis_v.n_basis
    n_total = n_h + n_u + n_v

    # ── Collocation (cell centres, normalised to [0,1]^2) ────────────────
    x_c = (np.arange(cfg.NX_CELLS) + 0.5) / cfg.NX_CELLS
    y_c = (np.arange(cfg.NY_CELLS) + 0.5) / cfg.NY_CELLS
    Xg, Yg = np.meshgrid(x_c, y_c, indexing='ij')
    x_pde = Xg.ravel()
    y_pde = Yg.ravel()
    n_pde = len(x_pde)

    # K* at collocation points (same ordering as permeability array)
    sqrt_K = np.sqrt(physics.K_star.ravel())

    if verbose:
        print("=" * 70)
        print("LiL-Q SOLVER -- DARCY / SPE10 (sqrt(K*) normalization)")
        print("=" * 70)
        print("Domain: %.0f x %.0f ft,  R = %.4f" % (physics.LX, physics.LY, R))
        print("Pressure: P_bottom=%.0f, P_top=%.0f psi" % (cfg.P_BOTTOM, cfg.P_TOP))
        print("K0 = %.4f mD (geometric mean)" % physics.K0)
        print("sqrt(K*) range: [%.4f, %.2f]"
              % (physics.sqrt_K_star.min(), physics.sqrt_K_star.max()))
        print("")
        print("Lifting function: h* = y* + h_tilde*")
        print("  (BCs satisfied EXACTLY by construction)")
        print("")
        print("Basis (on [0,1]^2):")
        print("  h_tilde*: %d (Cos x Sin -- vanishes at y*=0,1)" % n_h)
        print("  u*:       %d (Sin x Aug.Cos)" % n_u)
        print("  v*:       %d (Aug.Cos x Aug.Cos)" % n_v)
        print("  Total:    %d DOFs" % n_total)
        print("")
        print("Collocation: %d PDE points" % n_pde)
        print("-" * 70)

    # ── Build system ─────────────────────────────────────────────────────
    t_start = time.time()

    # Basis evaluations
    dh_dx = basis_h_tilde.derivative(x_pde, y_pde, dx=1, dy=0)
    dh_dy = basis_h_tilde.derivative(x_pde, y_pde, dx=0, dy=1)
    Phi_u = basis_u.evaluate(x_pde, y_pde)
    du_dx = basis_u.derivative(x_pde, y_pde, dx=1, dy=0)
    Phi_v = basis_v.evaluate(x_pde, y_pde)
    dv_dy = basis_v.derivative(x_pde, y_pde, dx=0, dy=1)

    # Darcy-x: u*/sqrt(K*) + sqrt(K*)*R*dh_tilde*/dx* = 0
    A_Dx = np.hstack([
        (sqrt_K * R)[:, None] * dh_dx,
        Phi_u / sqrt_K[:, None],
        np.zeros((n_pde, n_v)),
    ])
    b_Dx = np.zeros(n_pde)

    # Darcy-y: v*/sqrt(K*) + sqrt(K*)*dh_tilde*/dy* = -sqrt(K*)
    A_Dy = np.hstack([
        sqrt_K[:, None] * dh_dy,
        np.zeros((n_pde, n_u)),
        Phi_v / sqrt_K[:, None],
    ])
    b_Dy = -sqrt_K   # lifting function contribution

    # Continuity: R*du*/dx* + dv*/dy* = 0  (no K -- matches PINN)
    A_CE = np.hstack([
        np.zeros((n_pde, n_h)),
        R * du_dx,
        dv_dy,
    ])
    b_CE = np.zeros(n_pde)

    # Stack (no BC rows -- satisfied exactly by construction)
    A = np.vstack([A_Dx, A_Dy, A_CE])
    b = np.concatenate([b_Dx, b_Dy, b_CE])

    t_build = time.time() - t_start

    if verbose:
        print("System: A shape = %s, build time = %.4fs" % (A.shape, t_build))

    # ── Solve ────────────────────────────────────────────────────────────
    t_solve_start = time.time()
    if cfg.solver_method == 'lstsq':
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    else:
        coeffs = linalg.lstsq(A, b, lapack_driver='gelsy')[0]
    t_solve = time.time() - t_solve_start

    c_h_tilde = coeffs[:n_h]
    c_u = coeffs[n_h:n_h + n_u]
    c_v = coeffs[n_h + n_u:]

    # ── FVM reference ────────────────────────────────────────────────────
    if verbose:
        print("Solving FVM reference...")
    t_fvm_start = time.time()
    P_fvm = solve_fvm(physics)
    t_fvm = time.time() - t_fvm_start

    # ── Residuals ────────────────────────────────────────────────────────
    residual = A @ coeffs - b

    res_Dx_nd = residual[:n_pde]
    res_Dy_nd = residual[n_pde:2 * n_pde]
    res_CE_nd = residual[2 * n_pde:]

    # Physical residuals (matching PINN scaling)
    sqrt_K0 = np.sqrt(physics.K0)
    res_Dx_phys = res_Dx_nd * sqrt_K0 * physics.dP_scale
    res_Dy_phys = res_Dy_nd * sqrt_K0 * physics.dP_scale
    res_CE_phys = res_CE_nd * physics.CE_SCALE

    # Evaluate LiL pressure at cell centres
    h_tilde_vals = (basis_h_tilde.evaluate(Xg.ravel(), Yg.ravel())
                    @ c_h_tilde).reshape(cfg.NX_CELLS, cfg.NY_CELLS)
    h_lil = Yg + h_tilde_vals
    P_lil = h_lil * DELTA_P + cfg.P_BOTTOM

    # BC verification (should be ~0 by construction)
    h_tilde_bot = basis_h_tilde.evaluate(x_c, np.zeros(cfg.NX_CELLS)) @ c_h_tilde
    h_tilde_top = basis_h_tilde.evaluate(x_c, np.ones(cfg.NX_CELLS)) @ c_h_tilde
    bc_err_bot = np.abs(h_tilde_bot).max() * DELTA_P
    bc_err_top = np.abs(h_tilde_top).max() * DELTA_P

    # Error vs FVM
    err = P_lil - P_fvm
    rmse = np.sqrt(np.mean(err**2))
    rel_L2 = np.linalg.norm(err) / np.linalg.norm(P_fvm)

    t_total = time.time() - t_start

    metrics = {
        'build_time': t_build,
        'solve_time': t_solve,
        'fvm_time': t_fvm,
        'total_time': t_total,
        'n_coefficients': n_total,
        # Dimensionless MSE
        'mse_Dx_nd': float(np.mean(res_Dx_nd**2)),
        'mse_Dy_nd': float(np.mean(res_Dy_nd**2)),
        'mse_CE_nd': float(np.mean(res_CE_nd**2)),
        # Physical MSE
        'mse_Dx_phys': float(np.mean(res_Dx_phys**2)),
        'mse_Dy_phys': float(np.mean(res_Dy_phys**2)),
        'mse_CE_phys': float(np.mean(res_CE_phys**2)),
        # Residual norms
        'residual_norm': float(np.linalg.norm(residual)),
        'res_darcy_x_mse': float(np.mean(res_Dx_nd**2)),
        'res_darcy_y_mse': float(np.mean(res_Dy_nd**2)),
        'res_continuity_mse': float(np.mean(res_CE_nd**2)),
        # BC errors
        'bc_error_bottom_psi': float(bc_err_bot),
        'bc_error_top_psi': float(bc_err_top),
        # FVM comparison
        'fvm_rmse_psi': float(rmse),
        'fvm_rel_L2': float(rel_L2),
        'fvm_max_err_psi': float(np.abs(err).max()),
    }

    if verbose:
        print("LiL solve: %.4fs, FVM solve: %.4fs" % (t_solve, t_fvm))
        print("-" * 70)
        print("Dimensionless MSE Residuals:")
        print("  Darcy-x:    %.6e" % metrics['mse_Dx_nd'])
        print("  Darcy-y:    %.6e" % metrics['mse_Dy_nd'])
        print("  Continuity: %.6e" % metrics['mse_CE_nd'])
        print("-" * 70)
        print("Physical MSE Residuals:")
        print("  Darcy-x:    %.6e (mD*psi/ft)^2" % metrics['mse_Dx_phys'])
        print("  Darcy-y:    %.6e (mD*psi/ft)^2" % metrics['mse_Dy_phys'])
        print("  Continuity: %.6e (mD*psi/ft^2)^2" % metrics['mse_CE_phys'])
        print("-" * 70)
        print("BC Verification (should be ~0 by construction):")
        print("  P at y=0:  max|error| = %.6e psi" % bc_err_bot)
        print("  P at y=Ly: max|error| = %.6e psi" % bc_err_top)
        print("-" * 70)
        print("LiL vs FVM:")
        print("  RMSE:      %.2f psi" % rmse)
        print("  Rel L2:    %.6f" % rel_L2)
        print("  Max Error: %.2f psi" % np.abs(err).max())
        print("=" * 70)

    return {
        'c_h_tilde': c_h_tilde,
        'c_u': c_u,
        'c_v': c_v,
        'metrics': metrics,
        'P_fvm': P_fvm,
        'basis_h_tilde': basis_h_tilde,
        'basis_u': basis_u,
        'basis_v': basis_v,
        'physics': physics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Field Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fields(result: Dict,
                    physics: DarcyPhysics,
                    n_eval_x: int = 180,
                    n_eval_y: int = 660) -> Dict:
    """Evaluate pressure and velocity fields on a uniform evaluation grid.

    Parameters
    ----------
    result : dict
        Output from :func:`solve_lilq_darcy`.
    physics : DarcyPhysics
    n_eval_x, n_eval_y : int
        Number of evaluation points per direction.

    Returns
    -------
    fields : dict
        Keys: 'X', 'Y', 'P', 'ux', 'uy' -- all shape (n_eval_x, n_eval_y).
    """
    cfg = physics.config
    LX, LY = physics.LX, physics.LY
    DELTA_P = physics.DELTA_P

    basis_h_tilde = result['basis_h_tilde']
    basis_u = result['basis_u']
    basis_v = result['basis_v']
    c_h_tilde = result['c_h_tilde']
    c_u = result['c_u']
    c_v = result['c_v']

    x = np.linspace(0, LX, n_eval_x)
    y = np.linspace(0, LY, n_eval_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x_flat, y_flat = X.ravel(), Y.ravel()

    # Normalised coordinates
    x_star = x_flat / LX
    y_star = y_flat / LY

    # Pressure: P = (y* + h_tilde*) * DELTA_P + P_BOTTOM
    h_tilde = basis_h_tilde.evaluate(x_star, y_star) @ c_h_tilde
    h_star = y_star + h_tilde
    P = (h_star * DELTA_P + cfg.P_BOTTOM).reshape(X.shape)

    # Velocities (physical units)
    u_star = basis_u.evaluate(x_star, y_star) @ c_u
    v_star = basis_v.evaluate(x_star, y_star) @ c_v
    ux = (u_star * physics.V_SCALE).reshape(X.shape)
    uy = (v_star * physics.V_SCALE).reshape(X.shape)

    return {'X': X, 'Y': Y, 'P': P, 'ux': ux, 'uy': uy}


# ─────────────────────────────────────────────────────────────────────────────
# Point Evaluation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pressure(result: Dict, physics: DarcyPhysics,
                      x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate physical pressure at arbitrary physical coordinates.

    Parameters
    ----------
    result : dict
        Output from :func:`solve_lilq_darcy`.
    physics : DarcyPhysics
    x, y : ndarray
        Physical coordinates (ft).

    Returns
    -------
    P : ndarray
        Pressure in psi.
    """
    x_star = np.asarray(x) / physics.LX
    y_star = np.asarray(y) / physics.LY
    h_tilde = result['basis_h_tilde'].evaluate(x_star, y_star) @ result['c_h_tilde']
    h_star = y_star + h_tilde
    return h_star * physics.DELTA_P + physics.config.P_BOTTOM


def evaluate_velocity_x(result: Dict, physics: DarcyPhysics,
                        x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate physical x-velocity at physical coordinates."""
    x_star = np.asarray(x) / physics.LX
    y_star = np.asarray(y) / physics.LY
    u_star = result['basis_u'].evaluate(x_star, y_star) @ result['c_u']
    return u_star * physics.V_SCALE


def evaluate_velocity_y(result: Dict, physics: DarcyPhysics,
                        x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate physical y-velocity at physical coordinates."""
    x_star = np.asarray(x) / physics.LX
    y_star = np.asarray(y) / physics.LY
    v_star = result['basis_v'].evaluate(x_star, y_star) @ result['c_v']
    return v_star * physics.V_SCALE


# ─────────────────────────────────────────────────────────────────────────────
# NiL-N Stub (PINN-style solver -- future implementation)
# ─────────────────────────────────────────────────────────────────────────────

class DarcyPINN:
    """PINN-based (NiL-N) solver for single-phase Darcy flow on SPE10 fields.

    Uses three separate ResNet-style MLPs with SiLU activation for pressure,
    x-velocity, and y-velocity.  Loss is sqrt(K*)-normalised to handle the
    extreme heterogeneity of SPE10 permeability fields.

    Parameters
    ----------
    config : DarcyConfig
    physics : DarcyPhysics
    hidden_dim : int
        Width of hidden layers (default 200).
    num_layers : int
        Number of hidden layers (default 8).
    device : str or torch.device or None
        Compute device; auto-detected if None.
    """

    def __init__(self, config: DarcyConfig, physics: DarcyPhysics,
                 hidden_dim: int = 200, num_layers: int = 8,
                 device=None):
        import torch
        import torch.nn as nn
        from lilq.nn import MLP

        self.config = config
        self.physics = physics

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)

        act = nn.SiLU()
        self.net_P = MLP(2, hidden_dim, 1, num_layers, activation=act).float().to(self.device)
        self.net_U = MLP(2, hidden_dim, 1, num_layers, activation=act).float().to(self.device)
        self.net_V = MLP(2, hidden_dim, 1, num_layers, activation=act).float().to(self.device)

        # Scaling constants (float32 to match network dtype)
        self.X_SCALE = float(physics.LX / 2.0)
        self.Y_SCALE = float(physics.LY / 2.0)
        self.P_SCALE = float(physics.config.P_BOTTOM)
        self.P_MID = float((physics.config.P_TOP + physics.config.P_BOTTOM) / 2.0)
        self.P_HALF = float((physics.config.P_TOP - physics.config.P_BOTTOM) / 2.0)
        self.V_SCALE = float(physics.V_SCALE)
        self.CE_SCALE = float(physics.CE_SCALE)
        self.dP_scale = float(physics.dP_scale)
        self.sqrt_K0 = float(np.sqrt(physics.K0))

        # Collocation tensors
        self._build_collocation()

    def _build_collocation(self):
        import torch
        cfg = self.config
        phy = self.physics

        x_c = ((np.arange(cfg.NX_CELLS) + 0.5) * (phy.LX / cfg.NX_CELLS)).astype(np.float32)
        y_c = ((np.arange(cfg.NY_CELLS) + 0.5) * (phy.LY / cfg.NY_CELLS)).astype(np.float32)
        Xg, Yg = np.meshgrid(x_c, y_c, indexing='ij')

        def _t(arr):
            return torch.tensor(arr.ravel(), dtype=torch.float32,
                                device=self.device).unsqueeze(1)

        self.xpde = _t(Xg).requires_grad_(True)
        self.ypde = _t(Yg).requires_grad_(True)

        sqrt_K = np.sqrt(phy.K_star.ravel()).astype(np.float32)
        self.sqrt_Kx = _t(sqrt_K)
        self.sqrt_Ky = _t(sqrt_K)

        self.xbot = _t(x_c); self.ybot = torch.zeros_like(self.xbot, device=self.device)
        self.xtop = _t(x_c); self.ytop = torch.full_like(self.xtop, phy.LY, device=self.device)
        self.xleft = torch.zeros(cfg.NY_CELLS, 1, device=self.device)
        self.yleft = _t(y_c)
        self.xright = torch.full((cfg.NY_CELLS, 1), phy.LX, device=self.device)
        self.yright = _t(y_c)

    def _norm_input(self, x, y):
        xn = x / self.X_SCALE - 1.0
        yn = y / self.Y_SCALE - 1.0
        return xn.float(), yn.float()

    def _get_P(self, x, y):
        xn, yn = self._norm_input(x, y)
        return self.net_P(torch.cat([xn, yn], dim=1)) * self.P_HALF + self.P_MID

    def _get_U(self, x, y):
        xn, yn = self._norm_input(x, y)
        return self.net_U(torch.cat([xn, yn], dim=1)) * self.V_SCALE

    def _get_V(self, x, y):
        xn, yn = self._norm_input(x, y)
        return self.net_V(torch.cat([xn, yn], dim=1)) * self.V_SCALE

    def _compute_loss(self):
        import torch
        ones = torch.ones_like(self.xpde)

        P = self._get_P(self.xpde, self.ypde)
        U = self._get_U(self.xpde, self.ypde)
        V = self._get_V(self.xpde, self.ypde)

        Px = torch.autograd.grad(P, self.xpde, ones, create_graph=True)[0]
        Py = torch.autograd.grad(P, self.ypde, ones, create_graph=True)[0]
        Ux = torch.autograd.grad(U, self.xpde, ones, create_graph=True)[0]
        Vy = torch.autograd.grad(V, self.ypde, ones, create_graph=True)[0]

        W_PDE, W_BC = 50.0, 20.0

        resDx = (U / self.sqrt_Kx + self.sqrt_Kx * Px) / (self.sqrt_K0 * self.dP_scale)
        resDy = (V / self.sqrt_Ky + self.sqrt_Ky * Py) / (self.sqrt_K0 * self.dP_scale)
        resCE = (Ux + Vy) / self.CE_SCALE

        lDx = W_PDE * torch.mean(resDx ** 2)
        lDy = W_PDE * torch.mean(resDy ** 2)
        lCE = W_PDE * torch.mean(resCE ** 2)

        P_bot = self._get_P(self.xbot, self.ybot)
        lBCD_bot = W_BC * torch.mean(((P_bot - self.config.P_BOTTOM) / self.P_SCALE) ** 2)

        P_top = self._get_P(self.xtop, self.ytop)
        lBCD_top = W_BC * torch.mean(((P_top - self.config.P_TOP) / self.P_SCALE) ** 2)

        U_lft = self._get_U(self.xleft, self.yleft)
        U_rgt = self._get_U(self.xright, self.yright)
        lBCN_lr = W_BC * (torch.mean((U_lft / self.V_SCALE) ** 2) +
                          torch.mean((U_rgt / self.V_SCALE) ** 2))

        total = lDx + lDy + lCE + lBCD_bot + lBCD_top + lBCN_lr
        return total, {'darcy_x': lDx.item(), 'darcy_y': lDy.item(),
                       'continuity': lCE.item(), 'bc_bot': lBCD_bot.item(),
                       'bc_top': lBCD_top.item(), 'bc_lr': lBCN_lr.item()}

    def train(self, max_epochs: int = 150000, lr: float = 1e-3,
              log_every: int = 5000, verbose: bool = True) -> Dict:
        """Train all three networks with Adam + cosine annealing.

        Returns dict with 'history' (list of per-epoch loss dicts) and
        'final_loss'.
        """
        import torch

        all_params = (list(self.net_P.parameters()) +
                      list(self.net_U.parameters()) +
                      list(self.net_V.parameters()))
        optimizer = torch.optim.Adam(all_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5)

        history = []
        t0 = time.time()

        for epoch in range(max_epochs + 1):
            optimizer.zero_grad()
            total, components = self._compute_loss()
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if epoch % log_every == 0:
                rec = {'epoch': epoch, 'total': total.item(), **components}
                history.append(rec)
                if verbose:
                    print(f"  Epoch {epoch:6d}: total={total.item():.4e}  "
                          f"Dx={components['darcy_x']:.3e}  "
                          f"Dy={components['darcy_y']:.3e}  "
                          f"CE={components['continuity']:.3e}")

        elapsed = time.time() - t0
        if verbose:
            print(f"  Training complete: {elapsed:.1f}s, "
                  f"final loss={total.item():.4e}")

        return {'history': history, 'final_loss': total.item(),
                'training_time': elapsed}

    def predict(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate all fields at physical coordinates.

        Returns dict with 'P', 'U', 'V' arrays in physical units.
        """
        import torch
        xt = torch.tensor(x.ravel(), dtype=torch.float32,
                          device=self.device).unsqueeze(1)
        yt = torch.tensor(y.ravel(), dtype=torch.float32,
                          device=self.device).unsqueeze(1)
        with torch.no_grad():
            P = self._get_P(xt, yt).cpu().numpy().ravel()
            U = self._get_U(xt, yt).cpu().numpy().ravel()
            V = self._get_V(xt, yt).cpu().numpy().ravel()
        return {'P': P.reshape(x.shape), 'U': U.reshape(x.shape),
                'V': V.reshape(x.shape)}


def run_nil_n_darcy(config: DarcyConfig, physics: DarcyPhysics,
                    max_epochs: int = 150000, hidden_dim: int = 200,
                    num_layers: int = 8, device=None,
                    verbose: bool = True) -> Dict:
    """Convenience wrapper: create, train, and evaluate a DarcyPINN.

    Returns dict with training metrics and field predictions at cell centres.
    """
    pinn = DarcyPINN(config, physics, hidden_dim=hidden_dim,
                     num_layers=num_layers, device=device)
    if verbose:
        n_params = sum(p.numel() for p in pinn.net_P.parameters()) * 3
        print(f"DarcyPINN: 3 networks x {hidden_dim}w x {num_layers}L "
              f"(SiLU), ~{n_params} params")

    result = pinn.train(max_epochs=max_epochs, verbose=verbose)

    cfg = config
    x_c = (np.arange(cfg.NX_CELLS) + 0.5) * (physics.LX / cfg.NX_CELLS)
    y_c = (np.arange(cfg.NY_CELLS) + 0.5) * (physics.LY / cfg.NY_CELLS)
    Xg, Yg = np.meshgrid(x_c, y_c, indexing='ij')
    fields = pinn.predict(Xg, Yg)

    result['fields'] = fields
    result['pinn'] = pinn
    return result
