"""
Bratu Problem Definition
=========================

Bratu equation: u_xx + u_yy + λ·exp(u) = 0  on [0,1]²
with homogeneous Dirichlet BCs: u = 0 on ∂Ω.

Quasilinearization (Bellman–Kalaba):
    Given u_prev, the linearized equation is:
    u_xx + u_yy + λ·exp(u_prev)·u = λ·exp(u_prev)·(u_prev - 1)

This module defines ONLY the Bratu-specific physics.
The solver templates from ``lilq.solvers`` handle all optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable

from lilq.basis import create_basis_2d
from lilq.nn import MLP, calculate_hidden_dim
from lilq.metrics import MetricsTracker, QuasilinearMetrics
from lilq.collocation import generate_collocation_points_2d, collocation_to_torch
from lilq.pretraining import pretrain_nn, pretrain_lil
from lilq.solvers import solve_nil_n, solve_nil_q, solve_lil_n, solve_lil_q
from lilq.utils import set_seed, clear_gpu_memory, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BratuConfig:
    """Problem configuration for the Bratu equation."""
    lambda_: float = 1.0
    x_domain: Tuple[float, float] = (0.0, 1.0)
    y_domain: Tuple[float, float] = (0.0, 1.0)
    N_x: int = 16
    N_y: int = 16
    k_ratio: int = 10
    seed: int = 42
    basis_type: str = 'fourier'
    sampling: str = 'random'


@dataclass
class BratuOptConfig:
    """Optimization settings for Bratu experiments."""
    # NiL methods
    max_iterations: int = 10000
    max_line_searches: int = 100000
    R_tol: float = 1e-4
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    # NiL-Q specific
    max_quasi_iters_nn: int = 25
    max_inner_iters_nn: int = 300
    # LiL-Q specific
    max_quasi_iters_lil: int = 25
    # Pretraining
    pretrain_epochs: int = 500
    pretrain_tol: float = 1e-4
    pretrain_grid: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

class BratuPhysics:
    """Bratu equation physics.

    PDE:   u_xx + u_yy + λ·exp(u) = 0
    BCs:   u = 0 on ∂[0,1]²
    """

    def __init__(self, config: BratuConfig):
        self.lambda_ = config.lambda_
        self.x_domain = config.x_domain
        self.y_domain = config.y_domain

    def initial_guess(self, x, y):
        """Initial guess: u(x,y) = 0 everywhere."""
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x.flatten())
        return np.zeros_like(np.asarray(x).flatten())


# ─────────────────────────────────────────────────────────────────────────────
# NiL-N Residuals (autograd-based)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pde_residual_nn(model, x, y, lambda_: float):
    """Full nonlinear PDE residual for Bratu via autograd.

    Returns u_xx + u_yy + λ·exp(u)
    """
    xy = torch.cat([x.unsqueeze(1) if x.dim() == 1 else x,
                     y.unsqueeze(1) if y.dim() == 1 else y], dim=1)
    u = model(xy)

    du = torch.autograd.grad(u, xy, torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_x, u_y = du[:, 0:1], du[:, 1:2]

    du_x = torch.autograd.grad(u_x, xy, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    u_xx = du_x[:, 0:1]

    du_y = torch.autograd.grad(u_y, xy, torch.ones_like(u_y),
                                create_graph=True, retain_graph=True)[0]
    u_yy = du_y[:, 1:2]

    return u_xx + u_yy + lambda_ * torch.exp(u)


def _compute_bc_residual_nn(model, bc_data):
    """Boundary condition residual: u = 0 on all 4 edges.

    Returns (bc_loss, None) — no initial condition for Bratu.
    """
    loss_fn = nn.MSELoss()
    total_bc_loss = torch.tensor(0.0, dtype=torch.float64, device=bc_data['device'])

    for key in ['left', 'right', 'bottom', 'top']:
        x_bc = bc_data[f'x_{key}']
        y_bc = bc_data[f'y_{key}']
        xy = torch.cat([x_bc.unsqueeze(1) if x_bc.dim() == 1 else x_bc,
                         y_bc.unsqueeze(1) if y_bc.dim() == 1 else y_bc], dim=1)
        u_bc = model(xy)
        total_bc_loss = total_bc_loss + loss_fn(u_bc, torch.zeros_like(u_bc))

    return total_bc_loss, None  # No IC for Bratu


# ─────────────────────────────────────────────────────────────────────────────
# NiL-Q Linearized Residual
# ─────────────────────────────────────────────────────────────────────────────

def _make_linearized_residual_fn(lambda_: float):
    """Create the NiL-Q linearized residual function for Bratu.

    Returns a callable that, when frozen_data is None, computes and
    caches the frozen quantities; otherwise uses the cached values.
    """
    def compute_linearized_residual(model, x, y, frozen_data):
        xy = torch.cat([x.unsqueeze(1) if x.dim() == 1 else x,
                         y.unsqueeze(1) if y.dim() == 1 else y], dim=1)

        if frozen_data is None:
            # "Freeze" step: compute quantities from current model
            model.eval()
            with torch.no_grad():
                u_prev = model(xy)
            exp_u_prev = torch.exp(u_prev).detach()
            rhs = (lambda_ * exp_u_prev * (u_prev - 1)).detach()
            model.train()
            return {'exp_u_prev': exp_u_prev, 'rhs': rhs}

        # "Solve" step: compute linearized residual using frozen data
        exp_u_prev = frozen_data['exp_u_prev']
        rhs = frozen_data['rhs']

        u = model(xy)
        du = torch.autograd.grad(u, xy, torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_x, u_y = du[:, 0:1], du[:, 1:2]

        du_x = torch.autograd.grad(u_x, xy, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        u_xx = du_x[:, 0:1]

        du_y = torch.autograd.grad(u_y, xy, torch.ones_like(u_y),
                                    create_graph=True, retain_graph=True)[0]
        u_yy = du_y[:, 1:2]

        return u_xx + u_yy + lambda_ * exp_u_prev * u - rhs

    return compute_linearized_residual


# ─────────────────────────────────────────────────────────────────────────────
# LiL System Assembly Functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_lil_n_loss_fn(A_u, A_uxx, A_uyy, A_bc, lambda_: float,
                         lambda_pde: float, lambda_bc: float, device):
    """Create the nonlinear loss function for LiL-N (Bratu).

    Returns ``(beta) -> (total, pde_loss, ic_loss, bc_loss)``
    """
    A_u_t = torch.from_numpy(A_u).to(device)
    A_uxx_t = torch.from_numpy(A_uxx).to(device)
    A_uyy_t = torch.from_numpy(A_uyy).to(device)
    A_bc_t = torch.from_numpy(A_bc).to(device)

    def compute_loss(beta):
        u = A_u_t @ beta
        u_xx = A_uxx_t @ beta
        u_yy = A_uyy_t @ beta

        residual = u_xx + u_yy + lambda_ * torch.exp(u)
        pde_loss = torch.mean(residual ** 2)

        bc_res = A_bc_t @ beta
        bc_loss = torch.mean(bc_res ** 2)

        total = lambda_pde * pde_loss + lambda_bc * bc_loss
        ic_loss = torch.tensor(0.0)
        return total, pde_loss, ic_loss, bc_loss

    return compute_loss


def _make_lil_q_system_fn(A_u, A_uxx, A_uyy, A_bc, n_pde, n_bc,
                           lambda_: float, lambda_pde: float, lambda_bc: float):
    """Create the linearized system assembly function for LiL-Q (Bratu).

    Returns ``(beta) -> (A_stacked, b_stacked)``
    """
    def assemble_system(beta):
        u_prev = A_u @ beta
        exp_u_prev = np.exp(u_prev)

        # Linearized PDE: (A_uxx + A_uyy + λ·exp(u_prev)·A_u) · β_new = λ·exp(u_prev)·(u_prev - 1)
        A_linear = A_uxx + A_uyy + lambda_ * (exp_u_prev.reshape(-1, 1) * A_u)
        b_linear = lambda_ * exp_u_prev * (u_prev - 1)

        # Weighted stacking
        w_pde = np.sqrt(lambda_pde / n_pde)
        w_bc = np.sqrt(lambda_bc / n_bc)

        A_stacked = np.vstack([w_pde * A_linear, w_bc * A_bc])
        b_stacked = np.concatenate([w_pde * b_linear, w_bc * np.zeros(n_bc)])

        return A_stacked, b_stacked

    return assemble_system


def _make_lil_nonlinear_loss_fn(A_u, A_uxx, A_uyy, A_bc,
                                 lambda_: float, lambda_pde: float, lambda_bc: float):
    """Create the nonlinear loss evaluator for LiL-Q convergence checks.

    Returns ``(beta) -> (total, pde, ic, bc)`` as floats.
    """
    def compute_loss(beta):
        u = A_u @ beta
        pde_res = (A_uxx @ beta) + (A_uyy @ beta) + lambda_ * np.exp(u)
        pde_mse = float(np.mean(pde_res ** 2))
        bc_mse = float(np.mean((A_bc @ beta) ** 2))
        total = lambda_pde * pde_mse + lambda_bc * bc_mse
        return total, pde_mse, 0.0, bc_mse

    return compute_loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_nn_solution(model, config: BratuConfig, n_eval=200, device=DEVICE):
    """Evaluate NN solution on uniform grid."""
    model.eval()
    x = torch.linspace(config.x_domain[0], config.x_domain[1], n_eval)
    y = torch.linspace(config.y_domain[0], config.y_domain[1], n_eval)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    with torch.no_grad():
        U = model(xy).reshape(n_eval, n_eval).cpu().numpy()
    return X.numpy(), Y.numpy(), U


def evaluate_lil_solution(basis, coefficients, config: BratuConfig, n_eval=200):
    """Evaluate LiL solution on uniform grid."""
    x = np.linspace(config.x_domain[0], config.x_domain[1], n_eval)
    y = np.linspace(config.y_domain[0], config.y_domain[1], n_eval)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U = basis.reconstruct(coefficients, X.ravel(), Y.ravel()).reshape(n_eval, n_eval)
    return X, Y, U


def evaluate_lil_residual(basis, coefficients, config: BratuConfig, n_eval=200):
    """Evaluate LiL PDE residual on uniform grid."""
    x = np.linspace(config.x_domain[0], config.x_domain[1], n_eval)
    y = np.linspace(config.y_domain[0], config.y_domain[1], n_eval)
    X, Y = np.meshgrid(x, y, indexing='ij')
    xf, yf = X.ravel(), Y.ravel()
    u = basis.reconstruct(coefficients, xf, yf)
    u_xx = basis.reconstruct_derivative(coefficients, xf, yf, dx=2, dy=0)
    u_yy = basis.reconstruct_derivative(coefficients, xf, yf, dx=0, dy=2)
    res = u_xx + u_yy + config.lambda_ * np.exp(u)
    return X, Y, res.reshape(n_eval, n_eval)


# ─────────────────────────────────────────────────────────────────────────────
# Top-Level Solver Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def run_nil_n(config: BratuConfig, opt: BratuOptConfig,
              device=DEVICE, verbose=True):
    """Run NiL-N (Standard PINN) for Bratu."""
    set_seed(config.seed)
    physics = BratuPhysics(config)

    n_hidden = 2
    hidden_dim = calculate_hidden_dim(n_hidden, config.N_x, config.N_y)
    model = MLP(hidden_dim=hidden_dim, num_layers=n_hidden).to(device)

    # Pretrain
    pretrain_loss = pretrain_nn(
        model, physics.initial_guess,
        config.x_domain, config.y_domain, device,
        n_grid=opt.pretrain_grid, max_epochs=opt.pretrain_epochs,
        tol=opt.pretrain_tol, verbose=verbose,
    )

    # Collocation
    pts = generate_collocation_points_2d(
        config.x_domain, config.y_domain, config.N_x, config.N_y,
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.15),
        has_initial_condition=False, seed=config.seed,
        sampling=config.sampling,
    )
    pts_t = collocation_to_torch(pts, device)

    x_pde = pts_t['x_pde'].unsqueeze(1) if pts_t['x_pde'].dim() == 1 else pts_t['x_pde']
    y_pde = pts_t['y_pde'].unsqueeze(1) if pts_t['y_pde'].dim() == 1 else pts_t['y_pde']
    x_pde.requires_grad_(True)
    y_pde.requires_grad_(True)

    bc_data = {
        'x_left': pts_t['x_bc_left'], 'y_left': pts_t['y_bc_left'],
        'x_right': pts_t['x_bc_right'], 'y_right': pts_t['y_bc_right'],
        'x_bottom': pts_t['x_bc_bottom'], 'y_bottom': pts_t['y_bc_bottom'],
        'x_top': pts_t['x_bc_top'], 'y_top': pts_t['y_bc_top'],
        'device': device,
    }

    pde_fn = lambda m, x, y: _compute_pde_residual_nn(m, x, y, config.lambda_)
    bc_fn = lambda m, bd: _compute_bc_residual_nn(m, bd)

    model, metrics, summary = solve_nil_n(
        pde_fn, bc_fn, model, x_pde, y_pde, bc_data,
        lambda_pde=opt.lambda_pde, lambda_bc=opt.lambda_bc,
        max_iterations=opt.max_iterations,
        max_line_searches=opt.max_line_searches,
        R_tol=opt.R_tol, verbose=verbose,
    )

    summary['pretrain_loss'] = float(pretrain_loss)
    return model, metrics, summary


def run_nil_q(config: BratuConfig, opt: BratuOptConfig,
              device=DEVICE, verbose=True):
    """Run NiL-Q (Quasilinear PINN) for Bratu."""
    set_seed(config.seed)
    physics = BratuPhysics(config)

    n_hidden = 2
    hidden_dim = calculate_hidden_dim(n_hidden, config.N_x, config.N_y)
    model = MLP(hidden_dim=hidden_dim, num_layers=n_hidden).to(device)

    pretrain_loss = pretrain_nn(
        model, physics.initial_guess,
        config.x_domain, config.y_domain, device,
        n_grid=opt.pretrain_grid, max_epochs=opt.pretrain_epochs,
        tol=opt.pretrain_tol, verbose=verbose,
    )

    pts = generate_collocation_points_2d(
        config.x_domain, config.y_domain, config.N_x, config.N_y,
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.15),
        has_initial_condition=False, seed=config.seed,
        sampling=config.sampling,
    )
    pts_t = collocation_to_torch(pts, device)

    x_pde = pts_t['x_pde'].unsqueeze(1).requires_grad_(True)
    y_pde = pts_t['y_pde'].unsqueeze(1).requires_grad_(True)

    bc_data = {
        'x_left': pts_t['x_bc_left'], 'y_left': pts_t['y_bc_left'],
        'x_right': pts_t['x_bc_right'], 'y_right': pts_t['y_bc_right'],
        'x_bottom': pts_t['x_bc_bottom'], 'y_bottom': pts_t['y_bc_bottom'],
        'x_top': pts_t['x_bc_top'], 'y_top': pts_t['y_bc_top'],
        'device': device,
    }

    pde_fn = lambda m, x, y: _compute_pde_residual_nn(m, x, y, config.lambda_)
    bc_fn = lambda m, bd: _compute_bc_residual_nn(m, bd)
    lin_fn = _make_linearized_residual_fn(config.lambda_)

    model, metrics, summary = solve_nil_q(
        pde_fn, lin_fn, bc_fn, model, x_pde, y_pde, bc_data,
        lambda_pde=opt.lambda_pde, lambda_bc=opt.lambda_bc,
        max_quasi_iters=opt.max_quasi_iters_nn,
        max_inner_iters=opt.max_inner_iters_nn,
        max_line_searches=opt.max_line_searches,
        R_tol=opt.R_tol, verbose=verbose,
    )

    summary['pretrain_loss'] = float(pretrain_loss)
    return model, metrics, summary


def run_lil_n(config: BratuConfig, opt: BratuOptConfig,
              device=DEVICE, verbose=True):
    """Run LiL-N (Nonlinear LiL) for Bratu."""
    set_seed(config.seed)
    physics = BratuPhysics(config)

    basis = create_basis_2d(
        config.basis_type, config.N_x, config.N_y,
        config.x_domain, config.y_domain,
    )

    init_coeffs, pretrain_loss = pretrain_lil(
        basis, physics.initial_guess,
        config.x_domain, config.y_domain,
        n_grid=opt.pretrain_grid, verbose=verbose,
    )

    pts = generate_collocation_points_2d(
        config.x_domain, config.y_domain, config.N_x, config.N_y,
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.15),
        has_initial_condition=False, seed=config.seed,
        sampling=config.sampling,
    )

    # Precompute basis matrices at collocation points
    x_pde, y_pde = pts['x_pde'], pts['y_pde']
    A_u = basis.evaluate(x_pde, y_pde)
    A_uxx = basis.derivative(x_pde, y_pde, dx=2, dy=0)
    A_uyy = basis.derivative(x_pde, y_pde, dx=0, dy=2)

    # All 4 boundary edges
    x_bc = np.concatenate([pts['x_bc_left'], pts['x_bc_right'],
                           pts['x_bc_bottom'], pts['x_bc_top']])
    y_bc = np.concatenate([pts['y_bc_left'], pts['y_bc_right'],
                           pts['y_bc_bottom'], pts['y_bc_top']])
    A_bc = basis.evaluate(x_bc, y_bc)

    loss_fn = _make_lil_n_loss_fn(
        A_u, A_uxx, A_uyy, A_bc,
        config.lambda_, opt.lambda_pde, opt.lambda_bc, device,
    )

    coefficients, metrics, summary = solve_lil_n(
        loss_fn, init_coeffs, device,
        lambda_pde=opt.lambda_pde, lambda_bc=opt.lambda_bc,
        max_iterations=opt.max_iterations,
        max_line_searches=opt.max_line_searches,
        R_tol=opt.R_tol, verbose=verbose,
    )

    summary['pretrain_loss'] = float(pretrain_loss)
    return basis, coefficients, metrics, summary


def run_lil_q(config: BratuConfig, opt: BratuOptConfig,
              verbose=True, diagnostics_callback=None):
    """Run LiL-Q (Quasilinear LiL) for Bratu."""
    set_seed(config.seed)
    physics = BratuPhysics(config)

    basis = create_basis_2d(
        config.basis_type, config.N_x, config.N_y,
        config.x_domain, config.y_domain,
    )

    init_coeffs, pretrain_loss = pretrain_lil(
        basis, physics.initial_guess,
        config.x_domain, config.y_domain,
        n_grid=opt.pretrain_grid, verbose=verbose,
    )

    pts = generate_collocation_points_2d(
        config.x_domain, config.y_domain, config.N_x, config.N_y,
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.15),
        has_initial_condition=False, seed=config.seed,
        sampling=config.sampling,
    )

    x_pde, y_pde = pts['x_pde'], pts['y_pde']
    A_u = basis.evaluate(x_pde, y_pde)
    A_uxx = basis.derivative(x_pde, y_pde, dx=2, dy=0)
    A_uyy = basis.derivative(x_pde, y_pde, dx=0, dy=2)

    x_bc = np.concatenate([pts['x_bc_left'], pts['x_bc_right'],
                           pts['x_bc_bottom'], pts['x_bc_top']])
    y_bc = np.concatenate([pts['y_bc_left'], pts['y_bc_right'],
                           pts['y_bc_bottom'], pts['y_bc_top']])
    A_bc = basis.evaluate(x_bc, y_bc)

    n_bc = len(x_bc)
    n_pde = pts['n_pde']

    system_fn = _make_lil_q_system_fn(
        A_u, A_uxx, A_uyy, A_bc, n_pde, n_bc,
        config.lambda_, opt.lambda_pde, opt.lambda_bc,
    )
    loss_fn = _make_lil_nonlinear_loss_fn(
        A_u, A_uxx, A_uyy, A_bc,
        config.lambda_, opt.lambda_pde, opt.lambda_bc,
    )

    coefficients, metrics, summary = solve_lil_q(
        system_fn, loss_fn, init_coeffs,
        max_quasi_iters=opt.max_quasi_iters_lil,
        R_tol=opt.R_tol, verbose=verbose,
        diagnostics_callback=diagnostics_callback,
    )

    summary['pretrain_loss'] = float(pretrain_loss)
    return basis, coefficients, metrics, summary
