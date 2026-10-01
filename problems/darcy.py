"""
SPE10 / Darcy Flow Problem
===========================

Single-phase Darcy flow through heterogeneous permeability (SPE10 benchmark).

Three solver approaches:
1. **FVM** (Finite Volume Method) -- reference solution with harmonic-mean
   transmissibilities and sparse direct solve.
2. **LiL-Q** -- Linear-in-Learnables with lifting function
   h*(x*,y*) = y* + h_tilde*(x*,y*) and sqrt(K*) row-equilibration.
3. **NiL-N** (PINN) -- three MLPs for pressure and velocities, Adam
   (``DarcyPINN``).

Governing equations (dimensionless, sqrt(K*) normalization)::

    Darcy-x:    u*/sqrt(K*) + sqrt(K*) * R * dh_tilde*/dx*  = 0
    Darcy-y:    v*/sqrt(K*) + sqrt(K*) *     dh_tilde*/dy*  = -sqrt(K*)
    Continuity: R * du*/dx* + dv*/dy*                       = 0

Boundary conditions (satisfied exactly by basis construction):
    - h*(y*=0) = 0,  h*(y*=1) = 1   (Dirichlet, via sin(y*) vanishing)
    - dh*/dx* = 0 at x*=0,1          (Neumann, via cos(x*) derivative)
"""

import numpy as np
import os
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
from lilq.instrumentation import EPS_MACH
from lilq.iteration_log import IterationLogger, LilQDiagnosticsTracker
from lilq.provenance import capture_blas_thread_env
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
from lilq.test_errors import max_abs, rel_l2, tensor_grid_values


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

def assemble_fvm(physics: DarcyPhysics):
    """The two-point-flux (TPFA) system ``A p = b`` of :func:`solve_fvm`:
    harmonic-mean transmissibilities, Dirichlet top and bottom, no-flow
    sides. ``p`` is ordered ``j * NX_CELLS + i`` (row-major over
    ``(NY_CELLS, NX_CELLS)``). Returns ``(A, b)`` with ``A`` in CSR form.
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

    return A.tocsr(), b


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
    A, b = assemble_fvm(physics)
    P = spsolve(A, b).reshape(cfg.NY_CELLS, cfg.NX_CELLS)
    return P.T  # Return shape (NX, NY)


def delta_fv(P_h: np.ndarray, P_fvm: np.ndarray, p_bottom: float) -> float:
    """Addendum v2.1 task B9: ``||p_h - p_FV||_2 / ||p_FV - p_bot||_2`` over
    the cell centres. Normalizing by the pressure *increment* rather than
    ``||p_FV||_2`` (as ``fvm_rel_L2`` does) keeps the 3,000 psi offset from
    understating the difference."""
    return float(np.linalg.norm(P_h - P_fvm) / np.linalg.norm(P_fvm - p_bottom))


def tpfa_residual(physics: DarcyPhysics, P_fvm: np.ndarray) -> Dict:
    """Residual of the TPFA system at the FVM solution: ``||A p - b||_2``
    and relative to ``||b||_2``. A direct solve should leave it at round-off."""
    A, b = assemble_fvm(physics)
    r = A @ P_fvm.T.ravel() - b
    return {'tpfa_residual': float(np.linalg.norm(r)),
            'tpfa_residual_rel': float(np.linalg.norm(r) / np.linalg.norm(b))}


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

def make_test_error_fn(basis_h_tilde, physics, P_fvm, n_h):
    """``beta -> eps_p/maxerr_p``: the LiL pressure against the FVM
    reference at the cell centres (Darcy has no exact solution; the same
    comparison the solver's own ``fvm_rel_L2``/``fvm_max_err_psi``
    report). ``beta`` is ``[c_h_tilde; c_u; c_v]``; pressure uses the
    lifting ``P = (y* + h_tilde*) * DELTA_P + P_BOTTOM``."""
    cfg = physics.config
    x_c = (np.arange(cfg.NX_CELLS) + 0.5) / cfg.NX_CELLS
    y_c = (np.arange(cfg.NY_CELLS) + 0.5) / cfg.NY_CELLS

    def test_errors(beta):
        h_tilde = tensor_grid_values(basis_h_tilde, beta[:n_h], [x_c, y_c])
        P = (y_c[None, :] + h_tilde) * physics.DELTA_P + cfg.P_BOTTOM
        return {'eps_p': rel_l2(P, P_fvm), 'maxerr_p': max_abs(P, P_fvm)}

    return test_errors


def solve_lilq_darcy(config: DarcyConfig,
                     physics: DarcyPhysics,
                     verbose: bool = True,
                     iteration_logger=None,
                     diagnostics: bool = True,
                     run_json_path=None) -> Dict:
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
    iteration_logger : ``lilq.iteration_log.IterationLogger``, optional
        When given, Section 3.1 ``iterations.csv`` rows are recorded for
        this solve: ``k=0`` (the solve) and the terminal ``k=1``. Darcy's LiL-Q system is **linear**
        (Darcy-x/y and continuity are already linear in
        h_tilde*/u*/v*) -- there is no Bellman-Kalaba quasilinearization
        loop here, unlike every other problem in this codebase, so this
        is a single direct solve rather than an iterative one.
        Consequently ``order_obs``/``chi``/``stall_flag`` are degenerate
        (no k-1/k-2 history exists) -- the tracker's existing NaN/False
        handling for a lone iterate covers this with no special-casing.
        Because Darcy has no separate BC row block (boundary conditions
        are satisfied exactly by the basis construction -- see the
        module docstring), every row is a "interior" PDE row:
        ``n_interior_rows`` is the full row count and
        ``interior_weight=1.0`` (no lambda-based row weighting exists in
        this system, unlike every other problem). Because the system is
        linear, the nonlinear residual and the linearized residual
        coincide exactly (``compute_residual_vector_fn`` is just
        ``A @ beta - b``, the same operator that defines the system
        itself) -- so ``chi`` correctly evaluates to ``0.0`` rather than
        NaN, confirming there is no linearization error to speak of.
        **Only supported for ``config.solver_method == 'qr'``** (the
        ``gelsy`` driver) -- raises ``NotImplementedError`` for
        ``'lstsq'``, which uses ``numpy.linalg.lstsq`` (a different
        driver entirely; the schema's ``num_rank_gelsy`` column
        specifically means the scipy ``gelsy`` rank). Omitted (``None``,
        the default), behavior is unchanged from before this parameter
        existed.
    run_json_path : str or Path, optional
        When given, writes the Section 3.1 "once per run" ``run.json``
        metadata file for this solve (see ``lilq.run_metadata``).
        Requires ``iteration_logger`` -- raises ``ValueError`` if given
        without it. Reports ``stopping_rule={"type": "single_direct_solve"}``
        and ``K_max=1``: Darcy has no iterative stopping rule to record.

    Returns
    -------
    result : dict
        Keys: 'c_h_tilde', 'c_u', 'c_v', 'metrics',
              'basis_h_tilde', 'basis_u', 'basis_v', 'physics'.
    """
    if iteration_logger is not None and config.solver_method != 'qr':
        raise NotImplementedError(
            "iteration_logger is only supported with config.solver_method="
            "'qr' (scipy's gelsy driver): 'lstsq' uses numpy.linalg.lstsq, "
            "a different driver, and the schema's num_rank_gelsy column "
            "specifically means the gelsy rank."
        )
    if run_json_path is not None and iteration_logger is None:
        raise ValueError("run_json_path requires iteration_logger (for first_stall_iteration).")
    # diagnostics=False (clean timing; the advisor's reply to wave 2): no
    # logger, and total_time stops when the LiL solve ends -- the FVM
    # reference solve and the error metrics after it are not part of the
    # method (0.18-0.27 s of 21-28 s in wave 2) and run off the clock.
    if not diagnostics and iteration_logger is not None:
        raise ValueError("diagnostics=False excludes iteration_logger.")
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
    t_start = time.perf_counter()
    t0 = time.perf_counter()

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

    t_build = time.perf_counter() - t_start
    t_assemble_s = time.perf_counter() - t0

    if verbose:
        print("System: A shape = %s, build time = %.4fs" % (A.shape, t_build))

    # ── Solve ────────────────────────────────────────────────────────────
    t_solve_start = time.perf_counter()
    t0 = time.perf_counter()
    if cfg.solver_method == 'lstsq':
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        rank_gelsy = None
    else:
        coeffs, _, rank_gelsy, _ = linalg.lstsq(A, b, cond=EPS_MACH, lapack_driver='gelsy')
    t_solve = time.perf_counter() - t_solve_start
    t_solve_s = time.perf_counter() - t0

    c_h_tilde = coeffs[:n_h]
    c_u = coeffs[n_h:n_h + n_u]
    c_v = coeffs[n_h + n_u:]

    t_lil_end = time.perf_counter()

    # ── FVM reference ────────────────────────────────────────────────────
    if verbose:
        print("Solving FVM reference...")
    t_fvm_start = time.perf_counter()
    P_fvm = solve_fvm(physics)
    t_fvm = time.perf_counter() - t_fvm_start

    # ── Residuals ────────────────────────────────────────────────────────
    residual = A @ coeffs - b

    t_diag = 0.0  # time in the passive diagnostics, excluded from total_time
    if iteration_logger is not None:
        t_diag0 = time.perf_counter()
        # Single direct linear solve: row k=0 (the solve, "assembled" at
        # the zero vector -- no previous iterate exists for this problem)
        # and the terminal row k=1 (the residual at the solution).
        beta_prev = np.zeros(n_total)
        beta_new = coeffs
        residual_vector_fn = lambda beta: A @ beta - b  # noqa: E731 -- linear system: this literally *is* the nonlinear operator
        total_loss = float(np.sum(residual ** 2))

        tracker = LilQDiagnosticsTracker(
            n_interior_rows=A.shape[0],  # every row is a PDE row -- no separate BC block exists
            interior_weight=1.0,         # no lambda-based row weighting in this system
            test_error_fn=make_test_error_fn(basis_h_tilde, physics, P_fvm, n_h),
        )
        row = tracker.step(
            k=0,
            A_stacked=A, b_stacked=b,
            beta_prev=beta_prev, beta_new=beta_new,
            total_loss=total_loss, rank_gelsy=rank_gelsy,
            t_assemble_s=t_assemble_s, t_solve_s=t_solve_s,
            is_final_iterate=True,
            compute_residual_vector_fn=residual_vector_fn,
        )
        iteration_logger.record(**row)
        iteration_logger.record(**tracker.finish(k=1))
        t_diag = time.perf_counter() - t_diag0

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

    t_total = (time.perf_counter() - t_start - t_diag) if diagnostics else (t_lil_end - t_start)

    metrics = {
        'build_time': t_build,
        'solve_time': t_solve,
        'fvm_time': t_fvm,
        'total_time': t_total,
        'diagnostics_time': t_diag,
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
        'delta_fv': delta_fv(P_lil, P_fvm, cfg.P_BOTTOM),
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

    if run_json_path is not None:
        thread_env = capture_blas_thread_env()
        metadata = build_run_metadata(
            N_total=int(A.shape[0]),
            N_composition={'darcy_x': n_pde, 'darcy_y': n_pde, 'continuity': n_pde},
            P_total=int(n_total),
            P_composition={'h_tilde': int(n_h), 'u': int(n_u), 'v': int(n_v)},
            row_weights={
                # Physics normalization (sqrt(K*) at each collocation
                # point), not a single lambda-based scalar per block like
                # every other problem in this codebase -- Darcy has no
                # lambda_pde/lambda_bc scheme at all (see module docstring).
                'darcy_x': 'per-row sqrt(K*)*R (physics row equilibration, not a block scalar)',
                'darcy_y': 'per-row sqrt(K*) (physics row equilibration, not a block scalar)',
                'continuity': 1.0,
            },
            collocation_construction={
                'method': 'cell centers',
                'NX_CELLS': cfg.NX_CELLS, 'NY_CELLS': cfg.NY_CELLS,
            },
            basis_description={
                'h_tilde': {'family': 'cos(x) x sin(y)', 'modes': cfg.ORDER_H},
                'u': {'family': 'sin(x) x augmented-cos(y)', 'modes': cfg.ORDER_U},
                'v': {'family': 'augmented-cos(x) x augmented-cos(y)', 'modes': cfg.ORDER_V},
            },
            initial_coefficients='n/a (single direct linear solve, no iterative initial guess)',
            solver_driver='gelsy', rcond=EPS_MACH,
            stopping_rule={'type': 'single_direct_solve'},
            K_max=1,
            stopping_reason='direct_solve',
            first_stall_iteration=first_stall_iteration(iteration_logger.rows),
            b2_check=tracker.b2_check,
            kappa_qr_raw_ratio=tracker.kappa_qr_raw_ratio,
            device='cpu',
            thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1),
        )
        write_run_json(run_json_path, metadata)

    return {
        'c_h_tilde': c_h_tilde,
        'c_u': c_u,
        'c_v': c_v,
        'metrics': metrics,
        'P_fvm': P_fvm,
        'P_lil': P_lil,
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
# NiL-N (PINN) solver
# ─────────────────────────────────────────────────────────────────────────────

class DarcyPINN:
    """PINN-based (NiL-N) solver for single-phase Darcy flow on SPE10 fields.

    Uses three separate plain MLPs (``lilq.nn.MLP``) with SiLU activation for pressure,
    x-velocity, and y-velocity.  Loss is sqrt(K*)-normalised to handle the
    extreme heterogeneity of SPE10 permeability fields.

    Parameters
    ----------
    config : DarcyConfig
    physics : DarcyPhysics
    hidden_dim : int
        Width of hidden layers (default 32).
    num_layers : int
        Number of hidden layers (default 2). The defaults are the
        manuscript's network (Table 13: 3 x 1,185 = 3,555 parameters, close
        to LiL's 3,169); the repository had 8 x 200 (~847,000), a later
        experiment in the pre-GitHub notebook that is not in the paper.
    device : str or torch.device or None
        Compute device; auto-detected if None.
    dtype : torch.dtype
        Precision of the networks and every tensor. float64 by default
        (Package 1 v2.0 Section 2: float64 everywhere); the manuscript's
        NiL Darcy values were computed in float32, still available here.
    seed : int or None
        If given, ``torch.manual_seed(seed)`` before the networks are built.
    """

    def __init__(self, config: DarcyConfig, physics: DarcyPhysics,
                 hidden_dim: int = 32, num_layers: int = 2,
                 device=None, dtype=torch.float64, seed=None):
        import torch
        import torch.nn as nn
        from lilq.nn import MLP

        self.config = config
        self.physics = physics

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.dtype = dtype
        self.hidden_dim, self.num_layers, self.seed = hidden_dim, num_layers, seed
        if seed is not None:
            torch.manual_seed(seed)

        act = nn.SiLU()
        self.net_P = MLP(2, hidden_dim, 1, num_layers, activation=act, dtype=dtype).to(self.device)
        self.net_U = MLP(2, hidden_dim, 1, num_layers, activation=act, dtype=dtype).to(self.device)
        self.net_V = MLP(2, hidden_dim, 1, num_layers, activation=act, dtype=dtype).to(self.device)

        # Scaling constants (Python floats: they take the tensors' dtype)
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

        x_c = (np.arange(cfg.NX_CELLS) + 0.5) * (phy.LX / cfg.NX_CELLS)
        y_c = (np.arange(cfg.NY_CELLS) + 0.5) * (phy.LY / cfg.NY_CELLS)
        Xg, Yg = np.meshgrid(x_c, y_c, indexing='ij')

        def _t(arr):
            return torch.tensor(arr.ravel(), dtype=self.dtype,
                                device=self.device).unsqueeze(1)

        self.xpde = _t(Xg).requires_grad_(True)
        self.ypde = _t(Yg).requires_grad_(True)

        sqrt_K = np.sqrt(phy.K_star.ravel())
        self.sqrt_Kx = _t(sqrt_K)
        self.sqrt_Ky = _t(sqrt_K)

        self.xbot = _t(x_c); self.ybot = torch.zeros_like(self.xbot, device=self.device)
        self.xtop = _t(x_c); self.ytop = torch.full_like(self.xtop, phy.LY, device=self.device)
        self.xleft = torch.zeros(cfg.NY_CELLS, 1, dtype=self.dtype, device=self.device)
        self.yleft = _t(y_c)
        self.xright = torch.full((cfg.NY_CELLS, 1), phy.LX, dtype=self.dtype, device=self.device)
        self.yright = _t(y_c)

    def _norm_input(self, x, y):
        xn = x / self.X_SCALE - 1.0
        yn = y / self.Y_SCALE - 1.0
        return xn.to(self.dtype), yn.to(self.dtype)

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

    def network_state(self) -> Dict:
        """The three networks' weights (CPU copies) and what rebuilds them."""
        return {'net_P': {k: v.detach().cpu().clone() for k, v in self.net_P.state_dict().items()},
                'net_U': {k: v.detach().cpu().clone() for k, v in self.net_U.state_dict().items()},
                'net_V': {k: v.detach().cpu().clone() for k, v in self.net_V.state_dict().items()},
                'hidden_dim': self.hidden_dim, 'num_layers': self.num_layers,
                'seed': self.seed, 'dtype': str(self.dtype)}

    def load_network_state(self, state: Dict) -> None:
        for name in ('net_P', 'net_U', 'net_V'):
            getattr(self, name).load_state_dict(state[name])

    def train(self, max_epochs: int = 150000, lr: float = 1e-3,
              log_every: int = 5000, verbose: bool = True,
              checkpoint_path=None, checkpoint_every: int = 5000) -> Dict:
        """Train all three networks with Adam + cosine annealing.

        With ``checkpoint_path``, the networks, the optimizer and scheduler
        states, the epoch and the history are saved there every
        ``checkpoint_every`` epochs (atomically, overwriting), and a
        checkpoint found there at the start is resumed from: the run then
        continues as if it had not stopped (the loop has no randomness),
        and ``training_time`` adds up the time of every segment.

        Returns dict with 'history' (list of per-epoch loss dicts),
        'final_loss', 'training_time', and 'resumed_at' (the epochs a
        checkpoint was resumed from).
        """
        import torch
        from lilq.saved_models import load_checkpoint, save_checkpoint

        all_params = (list(self.net_P.parameters()) +
                      list(self.net_U.parameters()) +
                      list(self.net_V.parameters()))
        optimizer = torch.optim.Adam(all_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5)

        history, resumed_at, start, time_before = [], [], 0, 0.0
        ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
        if ckpt is not None:
            if (ckpt['max_epochs'], ckpt['lr']) != (max_epochs, lr):
                raise ValueError(f"checkpoint {checkpoint_path} is for max_epochs={ckpt['max_epochs']}, "
                                 f"lr={ckpt['lr']}, not {max_epochs}, {lr}")
            self.load_network_state(ckpt['networks'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            history, time_before = ckpt['history'], ckpt['training_time']
            resumed_at = ckpt['resumed_at'] + [ckpt['epoch']]
            start = ckpt['epoch'] + 1
            if verbose:
                print(f"  Resuming from epoch {ckpt['epoch']} ({checkpoint_path})")

        t0 = time.perf_counter()

        for epoch in range(start, max_epochs + 1):
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
            if checkpoint_path and epoch % checkpoint_every == 0 and epoch < max_epochs:
                save_checkpoint(checkpoint_path, {
                    'epoch': epoch, 'max_epochs': max_epochs, 'lr': lr,
                    'networks': self.network_state(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'history': history,
                    'training_time': time_before + time.perf_counter() - t0, 'resumed_at': resumed_at})

        elapsed = time_before + time.perf_counter() - t0
        if verbose:
            print(f"  Training complete: {elapsed:.1f}s, "
                  f"final loss={total.item():.4e}")

        return {'history': history, 'final_loss': total.item(),
                'training_time': elapsed, 'resumed_at': resumed_at}

    def predict(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate all fields at physical coordinates.

        Returns dict with 'P', 'U', 'V' arrays in physical units.
        """
        import torch
        xt = torch.tensor(x.ravel(), dtype=self.dtype,
                          device=self.device).unsqueeze(1)
        yt = torch.tensor(y.ravel(), dtype=self.dtype,
                          device=self.device).unsqueeze(1)
        with torch.no_grad():
            P = self._get_P(xt, yt).cpu().numpy().ravel()
            U = self._get_U(xt, yt).cpu().numpy().ravel()
            V = self._get_V(xt, yt).cpu().numpy().ravel()
        return {'P': P.reshape(x.shape), 'U': U.reshape(x.shape),
                'V': V.reshape(x.shape)}


def load_darcy_pinn(path, config: DarcyConfig = None, physics: DarcyPhysics = None, device='cpu'):
    """``(pinn, saved)``: a trained DarcyPINN from a ``network.pt`` written by
    :func:`run_nil_n_darcy` (or its folder); ``pinn.predict(x, y)``
    evaluates it. ``config`` defaults to the saved one, ``physics`` to
    ``DarcyPhysics(config)``."""
    from pathlib import Path
    from lilq.saved_models import load_checkpoint
    path = Path(path)
    saved = load_checkpoint(path / 'network.pt' if path.is_dir() else path)
    config = config or saved['config']
    physics = physics or DarcyPhysics(config, verbose=False)
    nets = saved['networks']
    dtype = getattr(torch, nets['dtype'].replace('torch.', ''))
    pinn = DarcyPINN(config, physics, hidden_dim=nets['hidden_dim'], num_layers=nets['num_layers'],
                     device=device, dtype=dtype, seed=nets['seed'])
    pinn.load_network_state(nets)
    return pinn, saved


def run_nil_n_darcy(config: DarcyConfig, physics: DarcyPhysics,
                    max_epochs: int = 150000, hidden_dim: int = 32,
                    num_layers: int = 2, device=None,
                    verbose: bool = True, dtype=torch.float64, seed=None,
                    model_dir=None) -> Dict:
    """Convenience wrapper: create, train, and evaluate a DarcyPINN.

    With ``model_dir``: training checkpoints to ``<model_dir>/checkpoint.pt``
    every 5,000 epochs and resumes from it after a crash or a walltime kill;
    the trained networks are saved to ``<model_dir>/network.pt``
    (:func:`load_darcy_pinn`), and the checkpoint is then removed; a
    ``network.pt`` already there is loaded instead of training again.

    Returns dict with training metrics and field predictions at cell centres.
    """
    pinn = DarcyPINN(config, physics, hidden_dim=hidden_dim,
                     num_layers=num_layers, device=device, dtype=dtype, seed=seed)
    if verbose:
        n_params = sum(p.numel() for p in pinn.net_P.parameters()) * 3
        print(f"DarcyPINN: 3 networks x {hidden_dim}w x {num_layers}L "
              f"(SiLU), ~{n_params} params")

    from pathlib import Path
    from lilq.saved_models import load_checkpoint, save_checkpoint
    final_path = Path(model_dir) / 'network.pt' if model_dir else None
    saved = load_checkpoint(final_path) if final_path else None
    if saved is not None:
        if saved['max_epochs'] != max_epochs:
            raise ValueError(f"{final_path} was trained for {saved['max_epochs']} epochs, not {max_epochs}")
        pinn.load_network_state(saved['networks'])
        result = {k: saved[k] for k in ('history', 'final_loss', 'training_time', 'resumed_at')}
    else:
        ckpt_path = Path(model_dir) / 'checkpoint.pt' if model_dir else None
        result = pinn.train(max_epochs=max_epochs, verbose=verbose, checkpoint_path=ckpt_path)
        if final_path:
            save_checkpoint(final_path, {'networks': pinn.network_state(), 'max_epochs': max_epochs,
                                         'config': config, **result})
            ckpt_path.unlink(missing_ok=True)

    cfg = config
    x_c = (np.arange(cfg.NX_CELLS) + 0.5) * (physics.LX / cfg.NX_CELLS)
    y_c = (np.arange(cfg.NY_CELLS) + 0.5) * (physics.LY / cfg.NY_CELLS)
    Xg, Yg = np.meshgrid(x_c, y_c, indexing='ij')
    fields = pinn.predict(Xg, Yg)

    result['fields'] = fields
    result['pinn'] = pinn
    return result
