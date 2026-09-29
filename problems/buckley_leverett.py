"""
Buckley-Leverett Problem Definition (with Optional Gravity)
=============================================================

PDE:  S_t + d_x[f(S)] - D·S_xx = 0   on [0,1] × [0,T]

When N_g = 0 (no gravity):
    f(S) = S² / (S² + M·(1-S)²)
    IC:   S(x,0) = exp(-10·x)
    T_final = 1.0

When N_g ≠ 0 (gravity, Li & Tchelepi 2015):
    f(S) = λ_w/λ_T - (λ_w·λ_n/λ_T)·N_g
    IC:   S(x,0) = 1 - 1/(1 + exp(-100·(x - 0.4)))
    T_final = 0.175

BCs:  S(0,t) = S_left,  S(1,t) = S_right
"""

import functools
import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from lilq.basis import create_basis_2d
from lilq.nn import MLP, calculate_hidden_dim
from lilq.collocation import generate_collocation_points_2d, collocation_to_torch
from lilq.pretraining import pretrain_nn, pretrain_lil
from lilq.solvers import solve_nil_n, solve_nil_q, solve_lil_n, solve_lil_q, line_search_cap
from lilq.utils import set_seed, nn_init_seed, DEVICE
from lilq.instrumentation import EPS_MACH
from lilq.provenance import capture_blas_thread_env
from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
from lilq.test_errors import tensor_grid_values


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BLConfig:
    """Problem configuration for Buckley-Leverett."""
    # Gravity number (0 = no gravity)
    N_g: float = 0.0
    # Viscosity ratio
    M_param: float = 0.5
    # Constant diffusion coefficient
    D_coef: float = -0.1
    # Boundary values
    S_left: float = 1.0
    S_right: float = 0.0
    # Domain
    T_final: float = 0.4
    x_domain: Tuple[float, float] = (0.0, 1.0)
    # Discretization
    N_x: int = 16
    N_t: int = 16
    k_ratio: int = 10
    seed: int = 42  # collocation set (and LiL methods)
    basis_type: str = 'fourier'
    sampling: str = 'random'
    init_seed: Optional[int] = None  # NiL network init; None = seed (lilq.utils.nn_init_seed)
    # Starting point of every method (task B8): 'zero', or 'ic' -- the initial
    # saturation profile extended in time, least-squares fitted to the basis
    # (LiL) or pretrained into the network (NiL). None: the paper's choice,
    # 'zero' with gravity and 'ic' without.
    initial_guess: Optional[str] = None

    @staticmethod
    def with_gravity():
        """Factory for the gravity configuration (Li & Tchelepi 2015)."""
        return BLConfig(N_g=-5.0, M_param=1.0, T_final=0.175)


@dataclass
class BLOptConfig:
    """Optimization settings for BL experiments."""
    max_iterations: int = 10000
    R_tol: float = 1e-5
    lambda_pde: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    max_quasi_iters_nn: int = 50
    max_inner_iters_nn: int = 100
    max_quasi_iters_lil: int = 50
    pretrain_epochs: int = 500
    pretrain_tol: float = 1e-4
    # Pretraining fit grid, points per direction: one 50 x 50 grid for every
    # problem and size, NN and LiL fits alike (the GitHub reference; DECISIONS.md).
    pretrain_grid: int = 50
    # None: 16x the method's own iteration budget (lilq.solvers.line_search_cap),
    # the same rule for every problem; an explicit value overrides it.
    max_line_searches: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

class BLPhysics:
    """Buckley-Leverett physics with optional gravity.

    Relative permeabilities: k_rw = S², k_rn = (1-S)²
    Total mobility: λ_T = k_rw + M·k_rn

    No gravity (N_g=0):
        f(S) = k_rw / λ_T = S²/(S² + M·(1-S)²)

    With gravity:
        f(S) = k_rw/(M·λ_T) - (k_rn·k_rw/(M·λ_T))·N_g
    """

    def __init__(self, config: BLConfig):
        self.N_g = config.N_g
        self.M = config.M_param
        self.D = config.D_coef
        self.S_left = config.S_left
        self.S_right = config.S_right
        self.x_domain = config.x_domain
        self.t_domain = (0.0, config.T_final)
        self._has_gravity = abs(config.N_g) > 1e-12
        self.initial_guess_mode = config.initial_guess or ('zero' if self._has_gravity else 'ic')
        if self.initial_guess_mode not in ('zero', 'ic'):
            raise ValueError(f"initial_guess must be 'zero' or 'ic', not {config.initial_guess!r}")

    # ── Flux ──

    def flux(self, S):
        """Flux function f(S)."""
        if self._has_gravity:
            return self._flux_gravity(S)
        return self._flux_no_gravity(S)

    def _flux_no_gravity(self, S):
        """f(S) = S²/(S² + M·(1-S)²)"""
        S_sq = S ** 2
        denom = S_sq + self.M * (1 - S) ** 2
        return S_sq / (denom + 1e-10)

    def _flux_gravity(self, S):
        """f(S) with gravity term (Li & Tchelepi 2015).

        Total mobility uses physical form: lam_T = k_rw + k_rn (no M).
        M appears only in the denominator: f = k_rw / (M * lam_T).
        """
        k_rw = S ** 2
        k_rn = (1 - S) ** 2
        lam_T = k_rw + k_rn
        f_visc = k_rw / (self.M * lam_T + 1e-10)
        f_grav = k_rn * k_rw / (self.M * lam_T + 1e-10) * self.N_g
        return f_visc - f_grav

    def flux_derivative(self, S):
        """f'(S) via autograd, detached from S's own graph.

        Correct -- and required -- for the quasilinear solver
        (_make_lil_q_system_fn), which deliberately freezes this
        coefficient at the current iterate; every call site outside a
        quasilinear assembly should use flux_derivative_differentiable
        instead. See DECISIONS.md.
        """
        S_t = S.clone().detach().requires_grad_(True) if isinstance(S, torch.Tensor) else \
              torch.tensor(S, dtype=torch.float64, requires_grad=True)
        f = self.flux(S_t)
        df = torch.autograd.grad(f.sum(), S_t, create_graph=False)[0]
        return df.detach()

    def flux_derivative_differentiable(self, S):
        """f'(S) via autograd, preserving the graph through S.

        Required by any gradient-based solver that differentiates the PDE
        residual with respect to trainable parameters upstream of S (e.g.
        LiL-N's solve_lil_n, where S = A_u @ beta) -- backpropagating
        through f'(S) itself, not just through S, is part of the true
        gradient of d_x[f(S)] = f'(S)*S_x with respect to those
        parameters. Using the detached flux_derivative() here silently
        drops that term and was confirmed (DECISIONS.md) to cause LiL-N to
        plateau well short of convergence instead of reaching it.

        S must already require grad (not a detached copy) for the
        backward graph to reach past this function.
        """
        f = self.flux(S)
        df = torch.autograd.grad(f.sum(), S, create_graph=True)[0]
        return df

    def flux_second_derivative(self, S):
        """f''(S) via autograd (detached)."""
        S_t = S.clone().detach().requires_grad_(True) if isinstance(S, torch.Tensor) else \
              torch.tensor(S, dtype=torch.float64, requires_grad=True)
        f = self.flux(S_t)
        df = torch.autograd.grad(f.sum(), S_t, create_graph=True)[0]
        d2f = torch.autograd.grad(df.sum(), S_t)[0]
        return d2f.detach()

    # ── Conditions ──

    def initial_condition(self, x):
        """IC depends on gravity configuration."""
        if self._has_gravity:
            return self._ic_gravity(x)
        return self._ic_no_gravity(x)

    def _ic_no_gravity(self, x):
        if isinstance(x, torch.Tensor):
            return torch.exp(-10 * x.flatten())
        return np.exp(-10 * np.asarray(x).flatten())

    def _ic_gravity(self, x):
        """Smooth step: S(x,0) = 1 - 1/(1 + exp(-100·(x-0.4)))"""
        if isinstance(x, torch.Tensor):
            xf = x.flatten()
            return 1.0 - 1.0 / (1.0 + torch.exp(-100.0 * (xf - 0.4)))
        xf = np.asarray(x).flatten()
        return 1.0 - 1.0 / (1.0 + np.exp(-100.0 * (xf - 0.4)))

    def bc_left(self, t):
        if isinstance(t, torch.Tensor):
            return self.S_left * torch.ones_like(t.flatten())
        return self.S_left * np.ones_like(np.asarray(t).flatten())

    def bc_right(self, t):
        if isinstance(t, torch.Tensor):
            return self.S_right * torch.ones_like(t.flatten())
        return self.S_right * np.ones_like(np.asarray(t).flatten())

    def initial_guess(self, x, t):
        """Starting point every method is fitted to before training: zero, or
        the initial condition extended in time (``initial_guess_mode``)."""
        if self.initial_guess_mode == 'ic':
            return self.initial_condition(x)
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x.flatten())
        return np.zeros_like(np.asarray(x).flatten())


# ─────────────────────────────────────────────────────────────────────────────
# NiL Residuals
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pde_residual_nn(model, x, t, physics: BLPhysics):
    """S_t + d_x[f(S)] + D·S_xx = 0"""
    xt = torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=1)
    S = model(xt)

    dS = torch.autograd.grad(S, xt, torch.ones_like(S), create_graph=True)[0]
    S_x, S_t = dS[:, 0:1], dS[:, 1:2]
    dS_x = torch.autograd.grad(S_x, xt, torch.ones_like(S_x), create_graph=True)[0]
    S_xx = dS_x[:, 0:1]

    f = physics.flux(S)
    f_x = torch.autograd.grad(f.sum(), xt, create_graph=True)[0][:, 0:1]

    return S_t + f_x + physics.D * S_xx


def _compute_bc_residual_nn(model, bc_data):
    """BC + IC residual for BL."""
    loss_fn = nn.MSELoss()
    device = bc_data['device']

    # IC
    xt_ic = torch.cat([bc_data['x_ic'].reshape(-1, 1),
                        bc_data['t_ic'].reshape(-1, 1)], dim=1)
    S_ic = model(xt_ic).flatten()
    ic_loss = loss_fn(S_ic, bc_data['ic_target'].to(device))

    # BCs
    xt_left = torch.cat([bc_data['x_left'].reshape(-1, 1),
                          bc_data['t_left'].reshape(-1, 1)], dim=1)
    xt_right = torch.cat([bc_data['x_right'].reshape(-1, 1),
                           bc_data['t_right'].reshape(-1, 1)], dim=1)
    bc_loss = (loss_fn(model(xt_left).flatten(), bc_data['bc_left_target'].to(device)) +
               loss_fn(model(xt_right).flatten(), bc_data['bc_right_target'].to(device)))

    return bc_loss, ic_loss


# ─────────────────────────────────────────────────────────────────────────────
# LiL System Assembly
# ─────────────────────────────────────────────────────────────────────────────

def _make_lil_q_system_fn(A_u, A_ux, A_ut, A_uxx,
                           A_ic, A_bc_left, A_bc_right,
                           ic_target, bc_l_target, bc_r_target,
                           physics: BLPhysics,
                           lp, li, lb,
                           n_pde, n_ic, n_bc_l, n_bc_r):
    """Linearized system for LiL-Q BL.

    Bellman–Kalaba for S_t + f'(S_prev)·U_x + D·U_xx + f''(S_prev)·S_x_prev·U = rhs
    """
    def assemble_system(beta):
        S_prev = A_u @ beta
        S_x_prev = A_ux @ beta

        S_prev_t = torch.tensor(S_prev, dtype=torch.float64)
        f_p = physics.flux_derivative(S_prev_t).numpy()
        f_pp = physics.flux_second_derivative(S_prev_t).numpy()

        # Linearized PDE operator
        A_linear = (A_ut + f_p.reshape(-1, 1) * A_ux +
                    physics.D * A_uxx +
                    (f_pp * S_x_prev).reshape(-1, 1) * A_u)
        b_linear = f_pp * S_prev * S_x_prev

        w_pde = np.sqrt(lp / n_pde)
        w_ic = np.sqrt(li / n_ic)
        w_bl = np.sqrt(lb / n_bc_l)
        w_br = np.sqrt(lb / n_bc_r)

        A_stacked = np.vstack([
            w_pde * A_linear,
            w_ic * A_ic,
            w_bl * A_bc_left,
            w_br * A_bc_right,
        ])
        b_stacked = np.concatenate([
            w_pde * b_linear,
            w_ic * ic_target,
            w_bl * bc_l_target,
            w_br * bc_r_target,
        ])
        return A_stacked, b_stacked

    return assemble_system


def _make_lil_nonlinear_loss_fn(A_u, A_ux, A_ut, A_uxx,
                                 A_ic, A_bc_left, A_bc_right,
                                 ic_target, bc_l_target, bc_r_target,
                                 physics: BLPhysics,
                                 lp, li, lb):
    """Nonlinear loss for LiL-Q convergence check."""
    def compute_loss(beta):
        S = A_u @ beta
        S_x = A_ux @ beta
        S_t = A_ut @ beta
        S_xx = A_uxx @ beta

        S_t_tensor = torch.tensor(S, dtype=torch.float64)
        f_vals = physics.flux(S_t_tensor).numpy()

        # d_x[f(S)] ≈ f'(S)·S_x
        f_p = physics.flux_derivative(S_t_tensor).numpy()
        f_x = f_p * S_x

        res = S_t + f_x + physics.D * S_xx
        pde = float(np.mean(res ** 2))
        ic = float(np.mean((A_ic @ beta - ic_target) ** 2))
        bc = float(np.mean((A_bc_left @ beta - bc_l_target) ** 2) +
                   np.mean((A_bc_right @ beta - bc_r_target) ** 2))
        total = lp * pde + li * ic + lb * bc
        return total, pde, ic, bc

    return compute_loss


def _make_lil_residual_vector_fn(A_u, A_ux, A_ut, A_uxx,
                                  A_ic, A_bc_left, A_bc_right,
                                  ic_target, bc_l_target, bc_r_target,
                                  physics: BLPhysics,
                                  lp, li, lb,
                                  n_pde, n_ic, n_bc_l, n_bc_r):
    """Weighted nonlinear residual **vector** for Section 3.1's phase
    indicator (Codebase_v3_Proposal.md S2.1) -- the vector form of what
    :func:`_make_lil_nonlinear_loss_fn` reduces to scalars. Stacked
    PDE-then-IC-then-BC-left-then-BC-right and weighted identically to
    :func:`_make_lil_q_system_fn`'s ``A_stacked``/``b_stacked``, by
    construction: ``norm(vector)**2 == total`` from the loss function
    above, exactly. Uses the same (detached, non-differentiable)
    ``physics.flux``/``flux_derivative`` as the loss function -- this is a
    forward-only evaluation for logging, not a gradient path, so the
    graph-preserving ``flux_derivative_differentiable`` (needed only for
    LiL-N's ``.backward()``, see DECISIONS.md) does not apply here.

    Returns ``(beta) -> weighted_residual_vector``.
    """
    w_pde = np.sqrt(lp / n_pde)
    w_ic = np.sqrt(li / n_ic)
    w_bl = np.sqrt(lb / n_bc_l)
    w_br = np.sqrt(lb / n_bc_r)

    def compute_residual_vector(beta):
        S = A_u @ beta
        S_x = A_ux @ beta
        S_t = A_ut @ beta
        S_xx = A_uxx @ beta

        S_tensor = torch.tensor(S, dtype=torch.float64)
        f_p = physics.flux_derivative(S_tensor).numpy()
        f_x = f_p * S_x

        pde_res = S_t + f_x + physics.D * S_xx
        ic_res = A_ic @ beta - ic_target
        bl_res = A_bc_left @ beta - bc_l_target
        br_res = A_bc_right @ beta - bc_r_target

        return np.concatenate([
            w_pde * pde_res, w_ic * ic_res, w_bl * bl_res, w_br * br_res,
        ])

    return compute_residual_vector


def _make_lil_n_loss_fn(A_u, A_ux, A_ut, A_uxx,
                         A_ic, A_bc_left, A_bc_right,
                         ic_target, bc_l_target, bc_r_target,
                         physics: BLPhysics, lp, li, lb, device):
    """Nonlinear loss for LiL-N BL."""
    A_u_t = torch.from_numpy(A_u).to(device)
    A_ux_t = torch.from_numpy(A_ux).to(device)
    A_ut_t = torch.from_numpy(A_ut).to(device)
    A_uxx_t = torch.from_numpy(A_uxx).to(device)
    A_ic_t = torch.from_numpy(A_ic).to(device)
    A_bl_t = torch.from_numpy(A_bc_left).to(device)
    A_br_t = torch.from_numpy(A_bc_right).to(device)
    ic_t = torch.from_numpy(ic_target).to(device)
    bcl_t = torch.from_numpy(bc_l_target).to(device)
    bcr_t = torch.from_numpy(bc_r_target).to(device)

    def compute_loss(beta):
        S = A_u_t @ beta
        S_x = A_ux_t @ beta
        S_t = A_ut_t @ beta
        S_xx = A_uxx_t @ beta

        f_vals = physics.flux(S)
        # Graph-preserving derivative required here -- solve_lil_n
        # differentiates this loss w.r.t. beta directly (unlike LiL-Q's
        # quasilinear assembly, which wants the detached flux_derivative).
        # See DECISIONS.md.
        f_p = physics.flux_derivative_differentiable(S)
        f_x = f_p * S_x

        res = S_t + f_x + physics.D * S_xx
        pde_loss = torch.mean(res ** 2)
        ic_loss = torch.mean((A_ic_t @ beta - ic_t) ** 2)
        bc_loss = (torch.mean((A_bl_t @ beta - bcl_t) ** 2) +
                   torch.mean((A_br_t @ beta - bcr_t) ** 2))

        total = lp * pde_loss + li * ic_loss + lb * bc_loss
        return total, pde_loss, ic_loss, bc_loss

    return compute_loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_lil_solution(basis, coefficients, config: BLConfig, n_eval=200):
    """Evaluate LiL solution on uniform grid."""
    x = np.linspace(config.x_domain[0], config.x_domain[1], n_eval)
    t = np.linspace(0, config.T_final, n_eval)
    X, T = np.meshgrid(x, t, indexing='ij')
    U = basis.reconstruct(coefficients, X.ravel(), T.ravel()).reshape(n_eval, n_eval)
    return X, T, U


def evaluate_nn_solution(model, config: BLConfig, n_eval=200, device=DEVICE):
    """Evaluate NN solution on uniform grid."""
    model.eval()
    x = torch.linspace(config.x_domain[0], config.x_domain[1], n_eval, dtype=torch.float64)
    t = torch.linspace(0, config.T_final, n_eval, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing='ij')
    xt = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)
    with torch.no_grad():
        U = model(xt).reshape(n_eval, n_eval).cpu().numpy()
    return X.numpy(), T.numpy(), U


# ─────────────────────────────────────────────────────────────────────────────
# Top-Level Solver Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_lil_matrices(config: BLConfig, physics: BLPhysics, basis, pts):
    """Precompute basis matrices for LiL methods."""
    x_pde, t_pde = pts['x_pde'], pts['y_pde']
    A_u = basis.evaluate(x_pde, t_pde)
    A_ux = basis.derivative(x_pde, t_pde, dx=1, dy=0)
    A_ut = basis.derivative(x_pde, t_pde, dx=0, dy=1)
    A_uxx = basis.derivative(x_pde, t_pde, dx=2, dy=0)

    A_ic = basis.evaluate(pts['x_ic'], pts['y_ic'])
    ic_target = physics.initial_condition(pts['x_ic']).astype(np.float64)

    A_bc_left = basis.evaluate(pts['x_bc_left'], pts['y_bc_left'])
    A_bc_right = basis.evaluate(pts['x_bc_right'], pts['y_bc_right'])
    bc_l_target = physics.bc_left(pts['y_bc_left']).astype(np.float64)
    bc_r_target = physics.bc_right(pts['y_bc_right']).astype(np.float64)

    return (A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
            ic_target, bc_l_target, bc_r_target)



# ─────────────────────────────────────────────────────────────────────────────
# Test-grid error (Section 3.1 item 10)
# ─────────────────────────────────────────────────────────────────────────────

TEST_GRID = (201, 201)  # not set by the spec for BL; same density as Bratu's
REFERENCE_INTERVALS = 4000  # finite-difference reference: 20 intervals per test-grid spacing


def reference_solution(config: BLConfig, n_intervals: int = REFERENCE_INTERVALS):
    """The saturation on the uniform ``TEST_GRID`` over [x_domain] x [0, T]
    (array ``[i_x, j_t]``), from a finite-difference method-of-lines solve
    of the same PDE, S_t + f(S)_x + D S_xx = 0 (D < 0: diffusion |D|), with
    the same initial and boundary data: second-order central differences on
    ``n_intervals`` equal intervals, the Dirichlet values at both ends, BDF
    time stepping at rtol 1e-9. The problem has no closed-form solution;
    doubling ``n_intervals`` from 2,000 to 4,000 changes the result by about
    1e-6 relative (tests). Cached per configuration."""
    return _reference_cached(config.N_g, config.M_param, config.D_coef, config.S_left, config.S_right,
                             config.T_final, tuple(config.x_domain), int(n_intervals))


@functools.lru_cache(maxsize=8)
def _reference_cached(N_g, M_param, D_coef, S_left, S_right, T_final, x_domain, n_intervals):
    import scipy.sparse as sp
    from scipy.integrate import solve_ivp
    cfg = BLConfig(N_g=N_g, M_param=M_param, D_coef=D_coef, S_left=S_left, S_right=S_right,
                   T_final=T_final, x_domain=x_domain)
    physics = BLPhysics(cfg)
    nx, nt = TEST_GRID
    if n_intervals % (nx - 1):
        raise ValueError(f"n_intervals must be a multiple of {nx - 1}")
    x = np.linspace(*x_domain, n_intervals + 1)
    h = x[1] - x[0]
    t_eval = np.linspace(0.0, T_final, nt)

    def flux(S):
        return physics.flux(torch.from_numpy(S)).numpy()

    def rhs(t, u):
        S = np.concatenate([[S_left], u, [S_right]])
        F = flux(S)
        return -(F[2:] - F[:-2]) / (2.0 * h) - D_coef * (S[2:] - 2.0 * S[1:-1] + S[:-2]) / h ** 2

    S0 = np.asarray(physics.initial_condition(x[1:-1]), dtype=np.float64)
    n = n_intervals - 1
    pattern = sp.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1], format='csr')
    sol = solve_ivp(rhs, (0.0, T_final), S0, method='BDF', t_eval=t_eval,
                    rtol=1e-9, atol=1e-11, jac_sparsity=pattern)
    if not sol.success:
        raise RuntimeError(f"BL reference solve failed: {sol.message}")
    U = np.vstack([np.full(nt, S_left), sol.y, np.full(nt, S_right)])
    # t = 0: the initial condition itself, including at the two end points.
    U[:, 0] = physics.initial_condition(x)
    step = n_intervals // (nx - 1)
    U = U[::step, :]
    U.setflags(write=False)
    return U


def make_test_error_fn(basis, config: BLConfig):
    """``beta -> {'eps_u', 'maxerr_u'}``: the relative L2 error and the
    maximum absolute error of the saturation against
    :func:`reference_solution`, on the uniform ``TEST_GRID`` over
    [x_domain] x [0, T].

    Not the PDE residual's mean square on that grid (the earlier metric):
    for the steep gravity front it was dominated by a few points at t = 0
    and on the domain edges, outside the collocation points' extent, and
    ranked the methods opposite to their actual errors (DECISIONS.md,
    2026-09-28)."""
    xs = np.linspace(*config.x_domain, TEST_GRID[0])
    ts = np.linspace(0.0, config.T_final, TEST_GRID[1])
    S_ref = reference_solution(config)
    norm_ref = float(np.linalg.norm(S_ref))

    def test_errors(beta):
        E = tensor_grid_values(basis, beta, [xs, ts]) - S_ref
        return {'eps_u': float(np.linalg.norm(E)) / norm_ref, 'maxerr_u': float(np.max(np.abs(E)))}

    return test_errors


def run_lil_q(config: BLConfig, opt: BLOptConfig,
              verbose=True, diagnostics_callback=None,
              iteration_logger=None, run_json_path=None):
    """Run LiL-Q for Buckley-Leverett (viscous or gravity, transparently --
    ``physics.flux``/``flux_derivative`` handle both via ``config.N_g``).

    ``iteration_logger`` : ``lilq.iteration_log.IterationLogger``, optional
        When given, a full Section 3.1 ``iterations.csv`` row is recorded
        every outer iteration (see ``lilq.solvers.solve_lil_q``). BL
        supplies the interior-row unweighting (``n_interior_rows=n_pde``,
        ``interior_weight=w_pde``) and the residual-vector function needed
        for ``chi``/``stall_flag`` automatically. Omitted (``None``, the
        default), behavior is unchanged from before this parameter existed.
    run_json_path : str or Path, optional
        When given, writes the Section 3.1 "once per run" ``run.json``
        metadata file for this solve (see ``lilq.run_metadata``).
        Requires ``iteration_logger`` -- raises ``ValueError`` otherwise.
        ``initial_coefficients`` correctly distinguishes the two BL
        variants: gravity starts from zero, viscous (no gravity) from the
        least-squares fit of ``exp(-10*x)`` -- the spec's own callout
        ("the fitted initial profile for viscous Buckley-Leverett").
    """
    if run_json_path is not None and iteration_logger is None:
        raise ValueError("run_json_path requires iteration_logger (for first_stall_iteration).")
    np.random.seed(config.seed)  # NumPy only: this solve is scipy on the CPU (Addendum v2.1 fix 3.3)
    physics = BLPhysics(config)

    basis = create_basis_2d(
        config.basis_type, config.N_x, config.N_t,
        config.x_domain, (0, config.T_final),
    )

    init_coeffs, pretrain_loss = pretrain_lil(
        basis, physics.initial_guess,
        config.x_domain, (0, config.T_final),
        n_grid=opt.pretrain_grid, verbose=verbose,
    )

    pts = generate_collocation_points_2d(
        config.x_domain, (0, config.T_final), config.N_x, config.N_t,
        k_ratio=config.k_ratio, collocation_ratios=(0.9, 0.05, 0.05),
        has_initial_condition=True, seed=config.seed,
        sampling=config.sampling,
    )

    matrices = _prepare_lil_matrices(config, physics, basis, pts)
    (A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
     ic_target, bc_l_target, bc_r_target) = matrices

    n_pde, n_ic = pts['n_pde'], pts['n_ic']
    n_bc_l, n_bc_r = len(pts['x_bc_left']), len(pts['x_bc_right'])

    system_fn = _make_lil_q_system_fn(
        A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
        ic_target, bc_l_target, bc_r_target, physics,
        opt.lambda_pde, opt.lambda_ic, opt.lambda_bc,
        n_pde, n_ic, n_bc_l, n_bc_r,
    )
    loss_fn = _make_lil_nonlinear_loss_fn(
        A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
        ic_target, bc_l_target, bc_r_target, physics,
        opt.lambda_pde, opt.lambda_ic, opt.lambda_bc,
    )

    solve_kwargs = {}
    if iteration_logger is not None:
        residual_vector_fn = _make_lil_residual_vector_fn(
            A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
            ic_target, bc_l_target, bc_r_target, physics,
            opt.lambda_pde, opt.lambda_ic, opt.lambda_bc,
            n_pde, n_ic, n_bc_l, n_bc_r,
        )
        solve_kwargs.update(
            iteration_logger=iteration_logger,
            compute_residual_vector_fn=residual_vector_fn,
            n_interior_rows=n_pde,
            interior_weight=np.sqrt(opt.lambda_pde / n_pde),
            test_error_fn=make_test_error_fn(basis, config),
        )

    coefficients, metrics, summary = solve_lil_q(
        system_fn, loss_fn, init_coeffs,
        max_quasi_iters=opt.max_quasi_iters_lil,
        R_tol=opt.R_tol, verbose=verbose,
        diagnostics_callback=diagnostics_callback,
        **solve_kwargs,
    )
    summary['pretrain_loss'] = float(pretrain_loss)

    if run_json_path is not None:
        thread_env = capture_blas_thread_env()
        metadata = build_run_metadata(
            N_total=n_pde + n_ic + n_bc_l + n_bc_r,
            N_composition={'pde': n_pde, 'ic': n_ic, 'bc_left': n_bc_l, 'bc_right': n_bc_r},
            P_total=int(len(coefficients)),
            P_composition={'u': int(len(coefficients))},
            row_weights={
                'pde': float(np.sqrt(opt.lambda_pde / n_pde)),
                'ic': float(np.sqrt(opt.lambda_ic / n_ic)),
                'bc_left': float(np.sqrt(opt.lambda_bc / n_bc_l)),
                'bc_right': float(np.sqrt(opt.lambda_bc / n_bc_r)),
            },
            collocation_construction={
                'method': 'random-tensor' if config.sampling == 'random' else config.sampling,
                'seed': config.seed,
                'N_x': config.N_x, 'N_t': config.N_t, 'k_ratio': config.k_ratio,
            },
            basis_description={
                'family': config.basis_type,
                'u': {'modes_x': config.N_x, 'modes_t': config.N_t},
            },
            initial_coefficients=(
                'zero' if physics.initial_guess_mode == 'zero'
                else 'initial saturation profile extended in time, least-squares fitted '
                     + ('(smooth step at x = 0.4)' if physics._has_gravity else '(exp(-10 x))')
            ),
            solver_driver='gelsy', rcond=EPS_MACH,
            stopping_rule={'type': 'loss_target', 'value': opt.R_tol},
            K_max=opt.max_quasi_iters_lil,
            stopping_reason='target' if summary['converged'] else 'iteration_cap',
            first_stall_iteration=first_stall_iteration(iteration_logger.rows),
            device='cpu',
            thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1),
            b2_check=summary.get('b2_check'),
            kappa_qr_raw_ratio=summary.get('kappa_qr_raw_ratio'),
        )
        write_run_json(run_json_path, metadata)

    return basis, coefficients, metrics, summary


def run_lil_n(config: BLConfig, opt: BLOptConfig,
              device=DEVICE, verbose=True):
    """Run LiL-N for Buckley-Leverett."""
    set_seed(config.seed)
    physics = BLPhysics(config)

    basis = create_basis_2d(
        config.basis_type, config.N_x, config.N_t,
        config.x_domain, (0, config.T_final),
    )

    init_coeffs, pretrain_loss = pretrain_lil(
        basis, physics.initial_guess,
        config.x_domain, (0, config.T_final),
        n_grid=opt.pretrain_grid, verbose=verbose,
    )

    pts = generate_collocation_points_2d(
        config.x_domain, (0, config.T_final), config.N_x, config.N_t,
        k_ratio=config.k_ratio, collocation_ratios=(0.9, 0.05, 0.05),
        has_initial_condition=True, seed=config.seed,
        sampling=config.sampling,
    )

    matrices = _prepare_lil_matrices(config, physics, basis, pts)
    (A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
     ic_target, bc_l_target, bc_r_target) = matrices

    loss_fn = _make_lil_n_loss_fn(
        A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
        ic_target, bc_l_target, bc_r_target, physics,
        opt.lambda_pde, opt.lambda_ic, opt.lambda_bc, device,
    )

    coefficients, metrics, summary = solve_lil_n(
        loss_fn, init_coeffs, device,
        max_iterations=opt.max_iterations,
        max_line_searches=line_search_cap(opt.max_iterations, opt.max_line_searches),
        R_tol=opt.R_tol, verbose=verbose,
    )
    summary['pretrain_loss'] = float(pretrain_loss)
    return basis, coefficients, metrics, summary


def run_nil_n(config: BLConfig, opt: BLOptConfig,
              device=DEVICE, verbose=True):
    """Run NiL-N for Buckley-Leverett."""
    set_seed(nn_init_seed(config))
    physics = BLPhysics(config)

    n_hidden = 2
    hidden_dim = calculate_hidden_dim(n_hidden, config.N_x, config.N_t)
    model = MLP(hidden_dim=hidden_dim, num_layers=n_hidden).to(device)

    pretrain_loss = pretrain_nn(
        model, physics.initial_guess,
        config.x_domain, (0, config.T_final), device,
        n_grid=opt.pretrain_grid,
        max_epochs=opt.pretrain_epochs,
        tol=opt.pretrain_tol, verbose=verbose,
    )

    pts = generate_collocation_points_2d(
        config.x_domain, (0, config.T_final), config.N_x, config.N_t,
        k_ratio=config.k_ratio, collocation_ratios=(0.9, 0.05, 0.05),
        has_initial_condition=True, seed=config.seed,
        sampling=config.sampling,
    )
    pts_t = collocation_to_torch(pts, device)

    x_pde = pts_t['x_pde'].reshape(-1, 1).requires_grad_(True)
    t_pde = pts_t['y_pde'].reshape(-1, 1).requires_grad_(True)

    bc_data = {
        'x_ic': pts_t['x_ic'], 't_ic': pts_t['y_ic'],
        'ic_target': physics.initial_condition(pts_t['x_ic']),
        'x_left': pts_t['x_bc_left'], 't_left': pts_t['y_bc_left'],
        'x_right': pts_t['x_bc_right'], 't_right': pts_t['y_bc_right'],
        'bc_left_target': physics.bc_left(pts_t['y_bc_left']),
        'bc_right_target': physics.bc_right(pts_t['y_bc_right']),
        'device': device,
    }

    pde_fn = lambda m, x, t: _compute_pde_residual_nn(m, x, t, physics)
    bc_fn = lambda m, bd: _compute_bc_residual_nn(m, bd)

    model, metrics, summary = solve_nil_n(
        pde_fn, bc_fn, model, x_pde, t_pde, bc_data,
        lambda_pde=opt.lambda_pde, lambda_bc=opt.lambda_bc, lambda_ic=opt.lambda_ic,
        max_iterations=opt.max_iterations, max_line_searches=line_search_cap(opt.max_iterations, opt.max_line_searches),
        R_tol=opt.R_tol, verbose=verbose,
    )
    summary['pretrain_loss'] = float(pretrain_loss)
    return model, metrics, summary


def run_nil_q(config: BLConfig, opt: BLOptConfig,
              device=DEVICE, verbose=True):
    """Run NiL-Q for Buckley-Leverett."""
    set_seed(nn_init_seed(config))
    physics = BLPhysics(config)

    n_hidden = 2
    hidden_dim = calculate_hidden_dim(n_hidden, config.N_x, config.N_t)
    model = MLP(hidden_dim=hidden_dim, num_layers=n_hidden).to(device)

    pretrain_loss = pretrain_nn(
        model, physics.initial_guess,
        config.x_domain, (0, config.T_final), device,
        n_grid=opt.pretrain_grid,
        max_epochs=opt.pretrain_epochs,
        tol=opt.pretrain_tol, verbose=verbose,
    )

    pts = generate_collocation_points_2d(
        config.x_domain, (0, config.T_final), config.N_x, config.N_t,
        k_ratio=config.k_ratio, collocation_ratios=(0.9, 0.05, 0.05),
        has_initial_condition=True, seed=config.seed,
        sampling=config.sampling,
    )
    pts_t = collocation_to_torch(pts, device)

    x_pde = pts_t['x_pde'].reshape(-1, 1).requires_grad_(True)
    t_pde = pts_t['y_pde'].reshape(-1, 1).requires_grad_(True)

    bc_data = {
        'x_ic': pts_t['x_ic'], 't_ic': pts_t['y_ic'],
        'ic_target': physics.initial_condition(pts_t['x_ic']),
        'x_left': pts_t['x_bc_left'], 't_left': pts_t['y_bc_left'],
        'x_right': pts_t['x_bc_right'], 't_right': pts_t['y_bc_right'],
        'bc_left_target': physics.bc_left(pts_t['y_bc_left']),
        'bc_right_target': physics.bc_right(pts_t['y_bc_right']),
        'device': device,
    }

    # Note: BL NiL-Q linearization uses the same Bellman-Kalaba structure
    # The flux derivative callbacks handle gravity transparently
    pde_fn = lambda m, x, t: _compute_pde_residual_nn(m, x, t, physics)
    bc_fn = lambda m, bd: _compute_bc_residual_nn(m, bd)

    def lin_fn(model, x, t, frozen_data):
        xt = torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=1)
        if frozen_data is None:
            model.eval()
            xt_g = xt.detach().requires_grad_(True)
            S_g = model(xt_g)
            dS = torch.autograd.grad(S_g, xt_g, torch.ones_like(S_g),
                                     create_graph=True, retain_graph=True)[0]
            S_x_g = dS[:, 0:1]
            dS_x = torch.autograd.grad(S_x_g, xt_g, torch.ones_like(S_x_g),
                                       create_graph=True, retain_graph=True)[0]
            S_xx_g = dS_x[:, 0:1]
            S_prev = S_g.detach()
            S_x_prev = S_x_g.detach()
            S_xx_prev = S_xx_g.detach()
            model.train()

            S_flat = S_prev.flatten()
            S_x_flat = S_x_prev.flatten()
            f_p = physics.flux_derivative(S_flat).reshape(-1, 1)
            f_pp = physics.flux_second_derivative(S_flat).reshape(-1, 1)
            a_coef = f_p
            b_coef = physics.D
            r_coef = f_pp * S_x_prev
            c3_coef = -f_pp * S_prev * S_x_prev
            return {
                'a': a_coef, 'b': b_coef, 'r': r_coef, 'c3': c3_coef,
            }

        a = frozen_data['a']
        b = frozen_data['b']
        r = frozen_data['r']
        c3 = frozen_data['c3']

        U = model(xt)
        dU = torch.autograd.grad(U, xt, torch.ones_like(U), create_graph=True)[0]
        U_x, U_t = dU[:, 0:1], dU[:, 1:2]
        dU_x = torch.autograd.grad(U_x, xt, torch.ones_like(U_x), create_graph=True)[0]
        U_xx = dU_x[:, 0:1]

        return U_t + a * U_x + b * U_xx + r * U + c3

    model, metrics, summary = solve_nil_q(
        pde_fn, lin_fn, bc_fn, model, x_pde, t_pde, bc_data,
        lambda_pde=opt.lambda_pde, lambda_bc=opt.lambda_bc, lambda_ic=opt.lambda_ic,
        max_quasi_iters=opt.max_quasi_iters_nn,
        max_inner_iters=opt.max_inner_iters_nn,
        max_line_searches=line_search_cap(opt.max_quasi_iters_nn * opt.max_inner_iters_nn, opt.max_line_searches),
        R_tol=opt.R_tol, verbose=verbose,
    )
    summary['pretrain_loss'] = float(pretrain_loss)
    return model, metrics, summary
