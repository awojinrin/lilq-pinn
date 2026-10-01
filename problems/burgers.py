"""
Burgers Problem Definition
===========================

Viscous Burgers equation:
    u_t + u·u_x - ν·u_xx = 0   on [-1,1] × [0,T]
    IC:  u(x,0) = -sin(π·x)
    BCs: u(-1,t) = u(1,t) = 0

Bellman–Kalaba quasilinearization of u·u_x:
    Given u_prev, linearized advection term becomes:
    u_prev·U_x + u_x_prev·U    (coefficients frozen from prev iterate)

This module defines ONLY the Burgers-specific physics.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

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
class BurgersConfig:
    """Problem configuration for the Burgers equation."""
    viscosity: float = 0.1
    T_final: float = 1.0
    x_domain: Tuple[float, float] = (-1.0, 1.0)
    N_x: int = 16
    N_t: int = 16
    k_ratio: int = 10
    seed: int = 42  # collocation set (and LiL methods)
    basis_type: str = 'sin_fourier'
    sampling: str = 'random'
    init_seed: Optional[int] = None  # NiL network init; None = seed (lilq.utils.nn_init_seed)


@dataclass
class BurgersOptConfig:
    """Optimization settings for Burgers experiments."""
    max_iterations: int = 10000
    # None: 16x the method's own iteration budget (lilq.solvers.line_search_cap),
    # the same rule for every problem; an explicit value overrides it.
    max_line_searches: Optional[int] = None
    R_tol: float = 1e-5
    # How an L-BFGS run may stop short (lilq.solvers.STALL_RULES): 'pytorch'
    # for the published tables, 'f1' for the four-method stall control.
    stall_rule: str = 'pytorch'
    lambda_pde: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    max_quasi_iters_nn: int = 25
    max_inner_iters_nn: int = 100
    max_quasi_iters_lil: int = 20
    pretrain_epochs: int = 500
    pretrain_tol: float = 1e-4
    # Pretraining fit grid, points per direction: one 50 x 50 grid for every
    # problem and size, NN and LiL fits alike (the GitHub reference; DECISIONS.md).
    pretrain_grid: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

class BurgersPhysics:
    """Burgers equation physics.

    PDE:  u_t + u·u_x - ν·u_xx = 0
    IC:   u(x, 0) = -sin(π·x)
    BCs:  u(-1, t) = 0,  u(1, t) = 0
    """

    def __init__(self, config: BurgersConfig):
        self.nu = config.viscosity
        self.T_final = config.T_final
        self.x_domain = config.x_domain
        self.t_domain = (0.0, config.T_final)

    def initial_condition(self, x):
        """IC: u(x,0) = -sin(π·x)"""
        if isinstance(x, torch.Tensor):
            return -torch.sin(torch.pi * x.flatten())
        return -np.sin(np.pi * np.asarray(x).flatten())

    def bc_left(self, t):
        """u(-1, t) = 0"""
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t.flatten())
        return np.zeros_like(np.asarray(t).flatten())

    def bc_right(self, t):
        """u(1, t) = 0"""
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t.flatten())
        return np.zeros_like(np.asarray(t).flatten())

    def initial_guess(self, x, t):
        """Initial guess: u = 0 everywhere."""
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x.flatten())
        return np.zeros_like(np.asarray(x).flatten())


# ─────────────────────────────────────────────────────────────────────────────
# NiL Residuals
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pde_residual_nn(model, x, t, nu):
    """Full nonlinear PDE: u_t + u·u_x - ν·u_xx."""
    xt = torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=1)
    u = model(xt)

    du = torch.autograd.grad(u, xt, torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_x, u_t = du[:, 0:1], du[:, 1:2]

    du_x = torch.autograd.grad(u_x, xt, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    u_xx = du_x[:, 0:1]

    return u_t + u * u_x - nu * u_xx


def _compute_bc_residual_nn(model, bc_data):
    """BC + IC residual for Burgers."""
    loss_fn = nn.MSELoss()
    device = bc_data['device']

    # IC
    x_ic, t_ic = bc_data['x_ic'], bc_data['t_ic']
    xt_ic = torch.cat([x_ic.reshape(-1, 1), t_ic.reshape(-1, 1)], dim=1)
    u_ic = model(xt_ic).flatten()
    ic_target = bc_data['ic_target'].to(device)
    ic_loss = loss_fn(u_ic, ic_target)

    # Left/right BCs
    xt_left = torch.cat([bc_data['x_left'].reshape(-1, 1),
                          bc_data['t_left'].reshape(-1, 1)], dim=1)
    xt_right = torch.cat([bc_data['x_right'].reshape(-1, 1),
                           bc_data['t_right'].reshape(-1, 1)], dim=1)
    bc_loss = (loss_fn(model(xt_left), torch.zeros_like(model(xt_left))) +
               loss_fn(model(xt_right), torch.zeros_like(model(xt_right))))

    return bc_loss, ic_loss


# ─────────────────────────────────────────────────────────────────────────────
# NiL-Q Linearized Residual
# ─────────────────────────────────────────────────────────────────────────────

def _make_linearized_residual_fn(nu):
    """Create NiL-Q linearized residual function for Burgers.

    Bellman–Kalaba linearization of u·u_x:
        u_prev·U_x + u_x_prev·U - ν·U_xx + U_t = u_prev·u_x_prev
    """
    def compute_linearized_residual(model, x, t, frozen_data):
        xt = torch.cat([x.reshape(-1, 1), t.reshape(-1, 1)], dim=1)

        if frozen_data is None:
            model.eval()
            with torch.no_grad():
                u_prev = model(xt)
            xt_g = xt.detach().requires_grad_(True)
            u_g = model(xt_g)
            du_g = torch.autograd.grad(u_g, xt_g, torch.ones_like(u_g),
                                        create_graph=False)[0]
            u_x_prev = du_g[:, 0:1].detach()
            model.train()
            rhs = (u_prev * u_x_prev).detach()
            return {'u_prev': u_prev, 'u_x_prev': u_x_prev, 'rhs': rhs}

        u_prev = frozen_data['u_prev']
        u_x_prev = frozen_data['u_x_prev']
        rhs = frozen_data['rhs']

        U = model(xt)
        dU = torch.autograd.grad(U, xt, torch.ones_like(U),
                                  create_graph=True, retain_graph=True)[0]
        U_x, U_t = dU[:, 0:1], dU[:, 1:2]
        dU_x = torch.autograd.grad(U_x, xt, torch.ones_like(U_x),
                                    create_graph=True, retain_graph=True)[0]
        U_xx = dU_x[:, 0:1]

        return U_t + u_prev * U_x + u_x_prev * U - nu * U_xx - rhs

    return compute_linearized_residual


# ─────────────────────────────────────────────────────────────────────────────
# LiL System Assembly
# ─────────────────────────────────────────────────────────────────────────────

def _make_lil_n_loss_fn(A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
                         ic_target, nu, lp, li, lb, device):
    """Nonlinear loss for LiL-N Burgers."""
    A_u_t = torch.from_numpy(A_u).to(device)
    A_ux_t = torch.from_numpy(A_ux).to(device)
    A_ut_t = torch.from_numpy(A_ut).to(device)
    A_uxx_t = torch.from_numpy(A_uxx).to(device)
    A_ic_t = torch.from_numpy(A_ic).to(device)
    A_bl_t = torch.from_numpy(A_bc_left).to(device)
    A_br_t = torch.from_numpy(A_bc_right).to(device)
    ic_t = torch.from_numpy(ic_target).to(device)

    def compute_loss(beta):
        u = A_u_t @ beta
        u_x = A_ux_t @ beta
        u_t = A_ut_t @ beta
        u_xx = A_uxx_t @ beta

        res = u_t + u * u_x - nu * u_xx
        pde_loss = torch.mean(res ** 2)

        ic_loss = torch.mean((A_ic_t @ beta - ic_t) ** 2)
        bc_loss = (torch.mean((A_bl_t @ beta) ** 2) +
                   torch.mean((A_br_t @ beta) ** 2))

        total = lp * pde_loss + li * ic_loss + lb * bc_loss
        return total, pde_loss, ic_loss, bc_loss

    return compute_loss


def _make_lil_q_system_fn(A_u, A_ux, A_ut, A_uxx,
                           A_ic, A_bc_left, A_bc_right,
                           ic_target, nu, lp, li, lb,
                           n_pde, n_ic, n_bc_l, n_bc_r):
    """Linearized system assembly for LiL-Q Burgers.

    Bellman–Kalaba: U_t + u_prev·U_x + u_x_prev·U - ν·U_xx = u_prev·u_x_prev
    """
    def assemble_system(beta):
        u_prev = A_u @ beta
        u_x_prev = A_ux @ beta

        # A_linear · β_new = b_linear
        A_linear = A_ut + u_prev.reshape(-1, 1) * A_ux + u_x_prev.reshape(-1, 1) * A_u - nu * A_uxx
        b_linear = u_prev * u_x_prev

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
            w_bl * np.zeros(n_bc_l),
            w_br * np.zeros(n_bc_r),
        ])
        return A_stacked, b_stacked

    return assemble_system


def _make_lil_nonlinear_loss_fn(A_u, A_ux, A_ut, A_uxx,
                                 A_ic, A_bc_left, A_bc_right,
                                 ic_target, nu, lp, li, lb):
    """Nonlinear loss evaluator for LiL-Q convergence check."""
    def compute_loss(beta):
        u = A_u @ beta
        u_x = A_ux @ beta
        u_t = A_ut @ beta
        u_xx = A_uxx @ beta

        res = u_t + u * u_x - nu * u_xx
        pde = float(np.mean(res ** 2))
        ic = float(np.mean((A_ic @ beta - ic_target) ** 2))
        bc = float(np.mean((A_bc_left @ beta) ** 2) +
                   np.mean((A_bc_right @ beta) ** 2))
        total = lp * pde + li * ic + lb * bc
        return total, pde, ic, bc

    return compute_loss


def _make_lil_residual_vector_fn(A_u, A_ux, A_ut, A_uxx,
                                  A_ic, A_bc_left, A_bc_right,
                                  ic_target, nu, lp, li, lb,
                                  n_pde, n_ic, n_bc_l, n_bc_r):
    """Weighted nonlinear residual **vector** for Section 3.1's phase
    indicator (Codebase_v3_Proposal.md S2.1) -- the vector form of what
    :func:`_make_lil_nonlinear_loss_fn` reduces to scalars. Stacked
    PDE-then-IC-then-BC-left-then-BC-right and weighted identically to
    :func:`_make_lil_q_system_fn`'s ``A_stacked``/``b_stacked``, by
    construction: ``norm(vector)**2 == total`` from the loss function
    above, exactly (same weighted sum of squares, one pre-reduction and
    one post).

    Returns ``(beta) -> weighted_residual_vector``.
    """
    w_pde = np.sqrt(lp / n_pde)
    w_ic = np.sqrt(li / n_ic)
    w_bl = np.sqrt(lb / n_bc_l)
    w_br = np.sqrt(lb / n_bc_r)

    def compute_residual_vector(beta):
        u = A_u @ beta
        u_x = A_ux @ beta
        u_t = A_ut @ beta
        u_xx = A_uxx @ beta

        pde_res = u_t + u * u_x - nu * u_xx
        ic_res = A_ic @ beta - ic_target
        bl_res = A_bc_left @ beta
        br_res = A_bc_right @ beta

        return np.concatenate([
            w_pde * pde_res, w_ic * ic_res, w_bl * bl_res, w_br * br_res,
        ])

    return compute_residual_vector


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_nn_solution(model, config: BurgersConfig, n_eval=200, device=DEVICE):
    """Evaluate NN solution on uniform grid."""
    model.eval()
    x = torch.linspace(config.x_domain[0], config.x_domain[1], n_eval, dtype=torch.float64)
    t = torch.linspace(0, config.T_final, n_eval, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing='ij')
    xt = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)
    with torch.no_grad():
        U = model(xt).reshape(n_eval, n_eval).cpu().numpy()
    return X.numpy(), T.numpy(), U


def evaluate_lil_solution(basis, coefficients, config: BurgersConfig, n_eval=200):
    """Evaluate LiL solution on uniform grid."""
    x = np.linspace(config.x_domain[0], config.x_domain[1], n_eval)
    t = np.linspace(0, config.T_final, n_eval)
    X, T = np.meshgrid(x, t, indexing='ij')
    U = basis.reconstruct(coefficients, X.ravel(), T.ravel()).reshape(n_eval, n_eval)
    return X, T, U


# ─────────────────────────────────────────────────────────────────────────────
# Top-Level Solver Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def run_nil_n(config: BurgersConfig, opt: BurgersOptConfig,
              device=DEVICE, verbose=True):
    """Run NiL-N for Burgers."""
    set_seed(nn_init_seed(config))
    physics = BurgersPhysics(config)

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
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.05, 0.10),
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
        'device': device,
    }

    pde_fn = lambda m, x, t: _compute_pde_residual_nn(m, x, t, config.viscosity)
    bc_fn = lambda m, bd: _compute_bc_residual_nn(m, bd)

    model, metrics, summary = solve_nil_n(
        pde_fn, bc_fn, model, x_pde, t_pde, bc_data,
        lambda_pde=opt.lambda_pde, lambda_bc=opt.lambda_bc, lambda_ic=opt.lambda_ic,
        max_iterations=opt.max_iterations, max_line_searches=line_search_cap(opt.max_iterations, opt.max_line_searches),
        R_tol=opt.R_tol, verbose=verbose, stall_rule=opt.stall_rule,
    )
    summary['pretrain_loss'] = float(pretrain_loss)
    return model, metrics, summary


def run_nil_q(config: BurgersConfig, opt: BurgersOptConfig,
              device=DEVICE, verbose=True):
    """Run NiL-Q for Burgers."""
    set_seed(nn_init_seed(config))
    physics = BurgersPhysics(config)

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
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.05, 0.10),
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
        'device': device,
    }

    pde_fn = lambda m, x, t: _compute_pde_residual_nn(m, x, t, config.viscosity)
    bc_fn = lambda m, bd: _compute_bc_residual_nn(m, bd)
    lin_fn = _make_linearized_residual_fn(config.viscosity)

    model, metrics, summary = solve_nil_q(
        pde_fn, lin_fn, bc_fn, model, x_pde, t_pde, bc_data,
        lambda_pde=opt.lambda_pde, lambda_bc=opt.lambda_bc, lambda_ic=opt.lambda_ic,
        max_quasi_iters=opt.max_quasi_iters_nn,
        max_inner_iters=opt.max_inner_iters_nn,
        max_line_searches=line_search_cap(opt.max_quasi_iters_nn * opt.max_inner_iters_nn, opt.max_line_searches),
        R_tol=opt.R_tol, verbose=verbose, stall_rule=opt.stall_rule,
    )
    summary['pretrain_loss'] = float(pretrain_loss)
    return model, metrics, summary


def run_lil_n(config: BurgersConfig, opt: BurgersOptConfig,
              device=DEVICE, verbose=True):
    """Run LiL-N for Burgers."""
    set_seed(config.seed)
    physics = BurgersPhysics(config)

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
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.05, 0.10),
        has_initial_condition=True, seed=config.seed,
        sampling=config.sampling,
    )

    x_pde, t_pde = pts['x_pde'], pts['y_pde']
    A_u = basis.evaluate(x_pde, t_pde)
    A_ux = basis.derivative(x_pde, t_pde, dx=1, dy=0)
    A_ut = basis.derivative(x_pde, t_pde, dx=0, dy=1)
    A_uxx = basis.derivative(x_pde, t_pde, dx=2, dy=0)

    A_ic = basis.evaluate(pts['x_ic'], pts['y_ic'])
    ic_target = physics.initial_condition(pts['x_ic']).astype(np.float64)

    A_bc_left = basis.evaluate(pts['x_bc_left'], pts['y_bc_left'])
    A_bc_right = basis.evaluate(pts['x_bc_right'], pts['y_bc_right'])

    loss_fn = _make_lil_n_loss_fn(
        A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
        ic_target, config.viscosity,
        opt.lambda_pde, opt.lambda_ic, opt.lambda_bc, device,
    )

    coefficients, metrics, summary = solve_lil_n(
        loss_fn, init_coeffs, device,
        max_iterations=opt.max_iterations,
        max_line_searches=line_search_cap(opt.max_iterations, opt.max_line_searches),
        R_tol=opt.R_tol, verbose=verbose, stall_rule=opt.stall_rule,
    )
    summary['pretrain_loss'] = float(pretrain_loss)
    return basis, coefficients, metrics, summary



# ─────────────────────────────────────────────────────────────────────────────
# Test-grid error (Section 3.1 item 10)
# ─────────────────────────────────────────────────────────────────────────────

TEST_GRID = (201, 201)  # not set by the spec for Burgers; same density as Bratu's


def make_test_error_fn(basis, config: BurgersConfig):
    """``beta -> {'eps_u': mean square of u_t + u u_x - nu u_xx}`` on a uniform
    201 x 201 grid over [x_domain] x [0, T] (a residual-MSE benchmark: the
    spec puts the test residual in eps_u)."""
    xs = np.linspace(*config.x_domain, TEST_GRID[0])
    ts = np.linspace(0.0, config.T_final, TEST_GRID[1])

    def test_errors(beta):
        u = tensor_grid_values(basis, beta, [xs, ts])
        u_x = tensor_grid_values(basis, beta, [xs, ts], [1, 0])
        u_t = tensor_grid_values(basis, beta, [xs, ts], [0, 1])
        u_xx = tensor_grid_values(basis, beta, [xs, ts], [2, 0])
        res = u_t + u * u_x - config.viscosity * u_xx
        return {'eps_u': float(np.mean(res ** 2))}

    return test_errors


def run_lil_q(config: BurgersConfig, opt: BurgersOptConfig,
              verbose=True, diagnostics_callback=None,
              iteration_logger=None, run_json_path=None, diagnostics=True):
    """Run LiL-Q for Burgers.

    ``iteration_logger`` : ``lilq.iteration_log.IterationLogger``, optional
        When given, a full Section 3.1 ``iterations.csv`` row is recorded
        every outer iteration (see ``lilq.solvers.solve_lil_q``). Burgers
        supplies the interior-row unweighting (``n_interior_rows=n_pde``,
        ``interior_weight=w_pde``) and the residual-vector function needed
        for ``chi``/``stall_flag`` automatically. Omitted (``None``, the
        default), behavior is unchanged from before this parameter existed.
    run_json_path : str or Path, optional
        When given, writes the Section 3.1 "once per run" ``run.json``
        metadata file for this solve (see ``lilq.run_metadata``).
        Requires ``iteration_logger`` -- raises ``ValueError`` otherwise.
    """
    if run_json_path is not None and iteration_logger is None:
        raise ValueError("run_json_path requires iteration_logger (for first_stall_iteration).")
    # diagnostics=False: clean timing (the advisor's reply to wave 2); see lilq.solvers.solve_lil_q.
    if not diagnostics and (iteration_logger is not None or diagnostics_callback is not None):
        raise ValueError("diagnostics=False excludes iteration_logger and diagnostics_callback.")
    np.random.seed(config.seed)  # NumPy only: this solve is scipy on the CPU (Addendum v2.1 fix 3.3)
    physics = BurgersPhysics(config)

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
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.05, 0.10),
        has_initial_condition=True, seed=config.seed,
        sampling=config.sampling,
    )

    x_pde, t_pde = pts['x_pde'], pts['y_pde']
    A_u = basis.evaluate(x_pde, t_pde)
    A_ux = basis.derivative(x_pde, t_pde, dx=1, dy=0)
    A_ut = basis.derivative(x_pde, t_pde, dx=0, dy=1)
    A_uxx = basis.derivative(x_pde, t_pde, dx=2, dy=0)

    A_ic = basis.evaluate(pts['x_ic'], pts['y_ic'])
    ic_target = physics.initial_condition(pts['x_ic']).astype(np.float64)

    A_bc_left = basis.evaluate(pts['x_bc_left'], pts['y_bc_left'])
    A_bc_right = basis.evaluate(pts['x_bc_right'], pts['y_bc_right'])

    n_pde, n_ic = pts['n_pde'], pts['n_ic']
    n_bc_l, n_bc_r = len(pts['x_bc_left']), len(pts['x_bc_right'])

    system_fn = _make_lil_q_system_fn(
        A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
        ic_target, config.viscosity,
        opt.lambda_pde, opt.lambda_ic, opt.lambda_bc,
        n_pde, n_ic, n_bc_l, n_bc_r,
    )
    loss_fn = _make_lil_nonlinear_loss_fn(
        A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
        ic_target, config.viscosity,
        opt.lambda_pde, opt.lambda_ic, opt.lambda_bc,
    )

    solve_kwargs = {}
    if iteration_logger is not None:
        residual_vector_fn = _make_lil_residual_vector_fn(
            A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
            ic_target, config.viscosity,
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
        diagnostics_callback=diagnostics_callback, diagnostics=diagnostics,
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
            initial_coefficients='zero',
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
