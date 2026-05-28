"""
Bratu Equation Core Module - Modular LiL Implementation
========================================================

Contains all four solution methods:
1. NiL-N: Standard PINN (Nonlinear-in-Learnables, Nonlinear PDE)
2. NiL-Q: Quasilinear PINN (Nonlinear-in-Learnables, Quasilinearized PDE)
3. LiL-N: Nonlinear LiL (Linear-in-Learnables, Nonlinear PDE)
4. LiL-Q: Quasilinear LiL (Linear-in-Learnables, Quasilinearized PDE)

LiL methods now accept any basis from the lil_basis module:
    - Chebyshev x Chebyshev
    - Fourier (sin, cos, both) in any combination
    - ELM (Extreme Learning Machine)
    - Mixed tensor products (e.g., Chebyshev x Sin)

All methods support pre-training to an initial guess for fair comparison.

TRACKING: For L-BFGS methods, we track BOTH:
  - Iterations (optimizer steps)
  - Line searches (function evaluations)
This allows plotting convergence against either metric.

Problem:
    PDE: Laplacian(u) + lambda*exp(u) = 0  on  [0,1] x [0,1]
    BC:  u = 0  on all boundaries (homogeneous Dirichlet)

Author: Gbenga / Claude collaboration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Union
import scipy

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lilq_shared.basis import (
    Chebyshev1D, Fourier1D, TensorProductBasis2D,
    ELMBasis2D, AugmentedBasis1D,
    create_chebyshev_basis_2d, create_fourier_basis_2d, create_mixed_basis_2d
)
from lilq_shared.utils import (
    set_seed, clear_gpu_memory, calculate_hidden_dim, MLP, MetricsTracker
)

# Force float64 precision
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=16)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class BratuPhysics:
    """Physics configuration for Bratu equation.

    PDE: Laplacian(u) + lambda*exp(u) = 0
    Domain: x in [0, 1], y in [0, 1]
    BC: u = 0 on all boundaries
    """
    lambda_: float = 6.2
    x_domain: Tuple[float, float] = (0.0, 1.0)
    y_domain: Tuple[float, float] = (0.0, 1.0)

    def boundary_condition(self, x, y):
        """Boundary condition: u = 0 on all boundaries."""
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        else:
            return np.zeros_like(np.asarray(x))

    def initial_guess(self, x, y):
        """Initial guess function for pre-training.

        Uses a bump function that satisfies BCs (zero on boundaries)
        and has a reasonable shape for the Bratu solution.
        """
        if isinstance(x, torch.Tensor):
            x_flat = x.flatten()
            y_flat = y.flatten()
            return torch.zeros_like(x_flat)
        else:
            x_flat = np.asarray(x).flatten()
            y_flat = np.asarray(y).flatten()
            return np.zeros_like(x_flat)


@dataclass
class DiscretizationConfig:
    """Discretization parameters."""
    N_x: int = 15
    N_y: int = 15
    k_ratio: int = 10
    collocation_ratios: Tuple[float, float] = (0.85, 0.15)
    domain_sampling: str = "uniform"
    boundary_sampling: str = "uniform"
    seed: int = 42


@dataclass
class OptimizationConfig:
    """Optimization parameters."""
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    R_tol: float = 1e-4
    max_iterations: int = 15000
    max_line_searches: int = 150000
    max_quasi_iters: int = 100
    n_epochs_lbfgs_per_iter: int = 200
    regularization_weight: float = 1e-10
    n_hidden_layers: int = 2
    n_epochs_adam: int = 0
    n_epochs_lbfgs: int = 15000
    lr_adam: float = 1e-3
    pretrain_epochs: int = 500
    pretrain_tol: float = 1e-4


# =============================================================================



# =============================================================================
# BASIS CREATION CONVENIENCE FUNCTIONS
# =============================================================================

def create_bratu_basis(basis_type: str, N_x: int, N_y: int,
                       x_domain: Tuple[float, float] = (0.0, 1.0),
                       y_domain: Tuple[float, float] = (0.0, 1.0),
                       elm_seed: int = 42,
                       elm_activation: str = 'tanh') -> Union[TensorProductBasis2D, ELMBasis2D]:
    """
    Create a 2D basis for the Bratu problem.

    Parameters
    ----------
    basis_type : str
        One of:
          'chebyshev'    : Chebyshev(x) x Chebyshev(y), order N_x-1 and N_y-1
          'sin_sin'      : Sin(x) x Sin(y), N_x and N_y modes
          'cos_cos'      : Cos(x) x Cos(y), N_x and N_y modes
          'fourier'      : Full Fourier (cos+sin) x (cos+sin)
          'cheb_sin'     : Chebyshev(x) x Sin(y)
          'sin_cheb'     : Sin(x) x Chebyshev(y)
          'elm'          : Extreme Learning Machine, n_hidden = N_x * N_y
        Or any 'typeX_typeY' string where typeX and typeY are each one of
        'chebyshev', 'sin', 'cos', 'fourier'.
    N_x, N_y : int
        Number of basis functions per direction (interpretation depends on type).
        For Chebyshev, order = N-1 giving N terms.
        For Fourier sin/cos, n_modes = N.
        For ELM, total hidden units = N_x * N_y.
    x_domain, y_domain : tuple
        Physical domains.
    elm_seed : int
        Random seed for ELM weight initialization.
    elm_activation : str
        Activation for ELM ('tanh' or 'sigmoid').

    Returns
    -------
    basis : TensorProductBasis2D or ELMBasis2D
    """
    basis_type = basis_type.lower().strip()

    if basis_type == 'chebyshev':
        return create_chebyshev_basis_2d(N_x - 1, N_y - 1, x_domain, y_domain)

    elif basis_type == 'sin_sin':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain,
                                       mode_x='sin', mode_y='sin')

    elif basis_type == 'cos_cos':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain,
                                       mode_x='cos', mode_y='cos')

    elif basis_type == 'fourier':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain,
                                       mode_x='both', mode_y='both')

    elif basis_type == 'cheb_sin':
        return TensorProductBasis2D(
            Chebyshev1D(N_x - 1, x_domain),
            Fourier1D(N_y, y_domain, mode='sin')
        )

    elif basis_type == 'sin_cheb':
        return TensorProductBasis2D(
            Fourier1D(N_x, x_domain, mode='sin'),
            Chebyshev1D(N_y - 1, y_domain)
        )

    elif basis_type == 'cheb_cos':
        return TensorProductBasis2D(
            Chebyshev1D(N_x - 1, x_domain),
            Fourier1D(N_y, y_domain, mode='cos')
        )

    elif basis_type == 'cos_cheb':
        return TensorProductBasis2D(
            Fourier1D(N_x, x_domain, mode='cos'),
            Chebyshev1D(N_y - 1, y_domain)
        )

    elif basis_type == 'cos_sin':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain,
                                       mode_x='cos', mode_y='sin')

    elif basis_type == 'sin_cos':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain,
                                       mode_x='sin', mode_y='cos')

    elif basis_type == 'elm':
        n_hidden = N_x * N_y
        return ELMBasis2D(
            n_hidden=n_hidden,
            domain_x=x_domain,
            domain_y=y_domain,
            activation=elm_activation,
            seed=elm_seed
        )

    else:
        raise ValueError(
            f"Unknown basis_type '{basis_type}'. "
            f"Supported: chebyshev, sin_sin, cos_cos, fourier, "
            f"cheb_sin, sin_cheb, cheb_cos, cos_cheb, cos_sin, sin_cos, elm"
        )


# =============================================================================
# COLLOCATION POINT GENERATION
# =============================================================================

def generate_collocation_points(
    physics: BratuPhysics,
    discretization: DiscretizationConfig,
    device: torch.device
) -> Dict:
    """Generate collocation points for all methods."""

    np.random.seed(discretization.seed)

    n_coefs = discretization.N_x * discretization.N_y
    ratios = discretization.collocation_ratios
    normalized_ratios = [r / sum(ratios) for r in ratios]

    x_min, x_max = physics.x_domain
    y_min, y_max = physics.y_domain

    n_pde = discretization.k_ratio * normalized_ratios[0] * n_coefs
    n_pde_dim = math.ceil(np.sqrt(n_pde))

    if discretization.domain_sampling == "random":
        x_pde = np.random.uniform(x_min + 1e-5, x_max - 1e-5, n_pde_dim).astype(np.float64)
        y_pde = np.random.uniform(y_min + 1e-5, y_max - 1e-5, n_pde_dim).astype(np.float64)
    else:
        x_pde = np.linspace(x_min + 1e-5, x_max - 1e-5, n_pde_dim, dtype=np.float64)
        y_pde = np.linspace(y_min + 1e-5, y_max - 1e-5, n_pde_dim, dtype=np.float64)

    pts_pde = np.stack(np.meshgrid(x_pde, y_pde), axis=-1).reshape([-1, 2]).astype(np.float64)

    n_bc = math.ceil(discretization.k_ratio * normalized_ratios[1] * n_coefs / 4)
    if discretization.boundary_sampling == "random":
        x_bc = np.random.uniform(x_min, x_max, n_bc).astype(np.float64)
        y_bc = np.random.uniform(y_min, y_max, n_bc).astype(np.float64)
    else:
        x_bc = np.linspace(x_min, x_max, n_bc, dtype=np.float64)
        y_bc = np.linspace(y_min, y_max, n_bc, dtype=np.float64)

    pts_bc_left = np.column_stack([np.zeros(n_bc, dtype=np.float64), y_bc])
    pts_bc_right = np.column_stack([np.ones(n_bc, dtype=np.float64), y_bc])
    pts_bc_bottom = np.column_stack([x_bc, np.zeros(n_bc, dtype=np.float64)])
    pts_bc_top = np.column_stack([x_bc, np.ones(n_bc, dtype=np.float64)])
    pts_bc = np.vstack([pts_bc_left, pts_bc_right, pts_bc_bottom, pts_bc_top])

    return {
        'pts_pde': pts_pde,
        'pts_bc': pts_bc,
        'x_bc': x_bc,
        'y_bc': y_bc,
        'n_bc': n_bc,
        'n_pde': pts_pde.shape[0]
    }


# =============================================================================
# PRE-TRAINING UTILITIES
# =============================================================================

def pretrain_nn_on_initial_guess(
    model: MLP,
    physics: BratuPhysics,
    device: torch.device,
    discretization: DiscretizationConfig,
    optimization: OptimizationConfig
) -> float:
    """Pre-train neural network on initial guess function."""
    if optimization.pretrain_epochs <= 0:
        return 0.0

    model.train()

    n_fit = max(50, discretization.N_x * discretization.N_y)
    n_x = int(np.sqrt(n_fit))
    n_y = n_x

    x_fit = np.linspace(physics.x_domain[0], physics.x_domain[1], n_x, dtype=np.float64)
    y_fit = np.linspace(physics.y_domain[0], physics.y_domain[1], n_y, dtype=np.float64)
    X, Y = np.meshgrid(x_fit, y_fit)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    target_np = physics.initial_guess(x_flat, y_flat)

    xy_tensor = torch.tensor(
        np.column_stack([x_flat, y_flat]),
        dtype=torch.float64, device=device
    )
    target_tensor = torch.tensor(
        target_np.reshape(-1, 1),
        dtype=torch.float64, device=device
    )

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn='strong_wolfe'
    )

    loss_fn = nn.MSELoss()
    final_loss = float('inf')

    for epoch in range(optimization.pretrain_epochs):
        def closure():
            optimizer.zero_grad()
            pred = model(xy_tensor)
            loss = loss_fn(pred, target_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            pred = model(xy_tensor)
            final_loss = loss_fn(pred, target_tensor).item()

        if final_loss < optimization.pretrain_tol:
            break

    model.train()
    return final_loss


def pretrain_lil_on_initial_guess(
    basis,
    physics: BratuPhysics,
    discretization: DiscretizationConfig,
    optimization: OptimizationConfig
) -> Tuple[np.ndarray, float]:
    """Pre-train LiL coefficients on initial guess function.

    Works with any basis from lil_basis (TensorProductBasis2D or ELMBasis2D).
    """
    n_coefs = basis.n_basis

    if optimization.pretrain_epochs <= 0:
        return np.random.uniform(-1.0, 1.0, n_coefs).astype(np.float64) * 1e-4, 0.0

    n_fit = max(100, 2 * n_coefs)
    n_x = int(np.sqrt(n_fit))
    n_y = n_x

    x_fit = np.linspace(physics.x_domain[0], physics.x_domain[1], n_x, dtype=np.float64)
    y_fit = np.linspace(physics.y_domain[0], physics.y_domain[1], n_y, dtype=np.float64)
    X, Y = np.meshgrid(x_fit, y_fit)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    target = physics.initial_guess(x_flat, y_flat)

    A = basis.evaluate(x_flat, y_flat)

    coefficients = scipy.linalg.lstsq(A, target, lapack_driver='gelsy')[0]

    pred = A @ coefficients
    mse = float(np.mean((pred - target) ** 2))

    return coefficients, mse


# =============================================================================
# PINN RESIDUAL FUNCTIONS
# =============================================================================

def compute_pde_residual_pinn(model: MLP, x: torch.Tensor, y: torch.Tensor,
                              lambda_: float) -> torch.Tensor:
    """Compute Bratu PDE residual: Laplacian(u) + lambda*exp(u)"""
    xy = torch.cat((x, y), dim=1)
    u = model(xy)

    du = torch.autograd.grad(
        outputs=u, inputs=xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_x = du[:, 0:1]
    u_y = du[:, 1:2]

    du_x = torch.autograd.grad(
        outputs=u_x, inputs=xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]
    u_xx = du_x[:, 0:1]

    du_y = torch.autograd.grad(
        outputs=u_y, inputs=xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0]
    u_yy = du_y[:, 1:2]

    residual = u_xx + u_yy + lambda_ * torch.exp(u)
    return residual


def compute_bc_residual(model: MLP, x_bc: torch.Tensor, y_bc: torch.Tensor) -> torch.Tensor:
    """Compute boundary condition residuals (all 4 sides, u = 0)."""
    device = x_bc.device

    left = torch.cat((torch.zeros_like(y_bc, device=device), y_bc), dim=1)
    right = torch.cat((torch.ones_like(y_bc, device=device), y_bc), dim=1)
    bottom = torch.cat((x_bc, torch.zeros_like(x_bc, device=device)), dim=1)
    top = torch.cat((x_bc, torch.ones_like(x_bc, device=device)), dim=1)

    pts_bc = torch.cat((left, right, bottom, top), dim=0)
    u_bc = model(pts_bc)

    return u_bc


# =============================================================================
# METHOD 1: STANDARD PINN (NiL-N)
# =============================================================================

def solve_nonlinear_pinn(
    physics: BratuPhysics,
    discretization: DiscretizationConfig,
    optimization: OptimizationConfig,
    device: torch.device,
    verbose: bool = True
    ) -> Tuple[MLP, MetricsTracker, Dict]:
    """Solve Bratu equation using standard PINN (NiL-N method)."""
    set_seed(discretization.seed)

    if verbose:
        print("=" * 80)
        print("METHOD: Standard PINN (NiL-N)")
        print("=" * 80)

    hidden_dim = calculate_hidden_dim(
        optimization.n_hidden_layers,
        discretization.N_x,
        discretization.N_y
    )
    model = MLP(
        input_dim=2,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=optimization.n_hidden_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Parameters: {n_params}")

    pretrain_loss = pretrain_nn_on_initial_guess(
        model, physics, device, discretization, optimization
    )
    if verbose:
        print(f"  Pre-training loss: {pretrain_loss:.6e}")

    points = generate_collocation_points(physics, discretization, device)

    x_pde = torch.tensor(points['pts_pde'][:, 0:1], dtype=torch.float64,
                         device=device, requires_grad=True)
    y_pde = torch.tensor(points['pts_pde'][:, 1:2], dtype=torch.float64,
                         device=device, requires_grad=True)
    x_bc = torch.tensor(points['x_bc'].reshape(-1, 1), dtype=torch.float64, device=device)
    y_bc = torch.tensor(points['y_bc'].reshape(-1, 1), dtype=torch.float64, device=device)

    loss_fn = nn.MSELoss()
    lambda_pde = optimization.lambda_pde
    lambda_bc = optimization.lambda_bc

    metrics = MetricsTracker()
    metrics.start()

    iteration_counter = [0]
    func_eval_counter = [0]

    def compute_loss():
        residual = compute_pde_residual_pinn(model, x_pde, y_pde, physics.lambda_)
        pde_loss = loss_fn(residual, torch.zeros_like(residual))

        bc_residual = compute_bc_residual(model, x_bc, y_bc)
        bc_loss = loss_fn(bc_residual, torch.zeros_like(bc_residual))

        total = lambda_pde * pde_loss + lambda_bc * bc_loss
        return total, pde_loss, bc_loss

    model.train()
    total_loss, pde_loss, bc_loss = compute_loss()
    metrics.record(0, 0, total_loss.item(), pde_loss.item(), bc_loss.item())

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=15,
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    def closure():
        func_eval_counter[0] += 1
        optimizer.zero_grad()
        total, _, _ = compute_loss()
        total.backward()
        return total

    converged = False
    max_iters = optimization.max_iterations
    max_line_searches = optimization.max_line_searches

    while iteration_counter[0] < max_iters and func_eval_counter[0] < max_line_searches:
        optimizer.step(closure)
        iteration_counter[0] += 1

        total_loss, pde_loss, bc_loss = compute_loss()
        metrics.record(iteration_counter[0], func_eval_counter[0],
                      total_loss.item(), pde_loss.item(), bc_loss.item())

        if verbose and iteration_counter[0] % 500 == 0:
            print(f"  Iter {iteration_counter[0]} (evals: {func_eval_counter[0]}): loss={total_loss.item():.6e}")

        if total_loss.item() < optimization.R_tol:
            if verbose:
                print(f"  Converged at iter {iteration_counter[0]} ({func_eval_counter[0]} evals)")
            converged = True
            break

    total_time = metrics.data['wall_time'][-1]

    summary = {
        'method': 'NiL-N',
        'total_iterations': int(iteration_counter[0]),
        'total_line_searches': int(func_eval_counter[0]),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_params),
        'pretrain_loss': float(pretrain_loss),
        'converged': bool(converged)
    }

    return model, metrics, summary


# =============================================================================
# METHOD 2: QUASILINEAR PINN (NiL-Q)
# =============================================================================

def solve_quasilinear_pinn(
    physics: BratuPhysics,
    discretization: DiscretizationConfig,
    optimization: OptimizationConfig,
    device: torch.device,
    verbose: bool = True
    ) -> Tuple[MLP, MetricsTracker, Dict]:
    """Solve Bratu equation using quasilinear PINN (NiL-Q method)."""
    set_seed(discretization.seed)

    if verbose:
        print("=" * 80)
        print("METHOD: Quasilinear PINN (NiL-Q)")
        print("=" * 80)

    hidden_dim = calculate_hidden_dim(
        optimization.n_hidden_layers,
        discretization.N_x,
        discretization.N_y
    )
    model = MLP(
        input_dim=2,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=optimization.n_hidden_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Parameters: {n_params}")

    pretrain_loss = pretrain_nn_on_initial_guess(
        model, physics, device, discretization, optimization
    )
    if verbose:
        print(f"  Pre-training loss: {pretrain_loss:.6e}")

    points = generate_collocation_points(physics, discretization, device)

    x_pde = torch.tensor(points['pts_pde'][:, 0:1], dtype=torch.float64,
                         device=device, requires_grad=True)
    y_pde = torch.tensor(points['pts_pde'][:, 1:2], dtype=torch.float64,
                         device=device, requires_grad=True)
    x_bc = torch.tensor(points['x_bc'].reshape(-1, 1), dtype=torch.float64, device=device)
    y_bc = torch.tensor(points['y_bc'].reshape(-1, 1), dtype=torch.float64, device=device)

    loss_fn = nn.MSELoss()
    lambda_pde = optimization.lambda_pde
    lambda_bc = optimization.lambda_bc

    metrics = MetricsTracker()
    metrics.start()

    total_iterations = [0]
    total_func_evals = [0]

    model.train()
    residual = compute_pde_residual_pinn(model, x_pde, y_pde, physics.lambda_)
    pde_loss = loss_fn(residual, torch.zeros_like(residual))
    bc_residual = compute_bc_residual(model, x_bc, y_bc)
    bc_loss = loss_fn(bc_residual, torch.zeros_like(bc_residual))
    total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss

    metrics.record(0, 0, total_loss.item(), pde_loss.item(), bc_loss.item())

    converged = False
    n_quasi_iters = 0
    max_line_searches = optimization.max_line_searches

    optimizer = optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=1,
            max_eval=15,
            tolerance_grad=1e-8,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn='strong_wolfe'
        )

    for quasi_iter in range(optimization.max_quasi_iters):
        n_quasi_iters = quasi_iter + 1
        if verbose:
            print(f"\n  Quasilinear iteration {quasi_iter + 1}")

        model.eval()
        xy_pde = torch.cat([x_pde, y_pde], dim=1)
        with torch.no_grad():
            u_prev = model(xy_pde)

        exp_u_prev = torch.exp(u_prev).detach()
        rhs_target = (physics.lambda_ * exp_u_prev * (u_prev - 1)).detach()

        model.train()

        def compute_linearized_residual():
            xy = torch.cat([x_pde, y_pde], dim=1)
            u_curr = model(xy)

            du = torch.autograd.grad(
                outputs=u_curr, inputs=xy,
                grad_outputs=torch.ones_like(u_curr),
                create_graph=True, retain_graph=True
            )[0]
            u_x = du[:, 0:1]
            u_y = du[:, 1:2]

            du_x = torch.autograd.grad(
                outputs=u_x, inputs=xy,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True, retain_graph=True
            )[0]
            u_xx = du_x[:, 0:1]

            du_y = torch.autograd.grad(
                outputs=u_y, inputs=xy,
                grad_outputs=torch.ones_like(u_y),
                create_graph=True, retain_graph=True
            )[0]
            u_yy = du_y[:, 1:2]

            residual = u_xx + u_yy + physics.lambda_ * exp_u_prev * u_curr - rhs_target
            return residual

        inner_iterations = [0]
        inner_func_evals = [0]
        max_inner_iters = optimization.n_epochs_lbfgs_per_iter

        while inner_iterations[0] < max_inner_iters:
            def closure():
                inner_func_evals[0] += 1
                total_func_evals[0] += 1
                optimizer.zero_grad()
                linear_res = compute_linearized_residual()
                linear_loss = loss_fn(linear_res, torch.zeros_like(linear_res))

                bc_res = compute_bc_residual(model, x_bc, y_bc)
                bc_loss = loss_fn(bc_res, torch.zeros_like(bc_res))

                total = lambda_pde * linear_loss + lambda_bc * bc_loss
                total.backward()
                return total

            optimizer.step(closure)
            inner_iterations[0] += 1
            total_iterations[0] += 1

            residual = compute_pde_residual_pinn(model, x_pde, y_pde, physics.lambda_)
            pde_loss = loss_fn(residual, torch.zeros_like(residual))
            bc_res = compute_bc_residual(model, x_bc, y_bc)
            bc_loss = loss_fn(bc_res, torch.zeros_like(bc_res))
            total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss

            metrics.record(total_iterations[0], total_func_evals[0],
                          total_loss.item(), pde_loss.item(), bc_loss.item())

            if total_loss.item() < optimization.R_tol:
                break

            if total_func_evals[0] >= max_line_searches:
                break

        if verbose:
            print(f"    Loss: {metrics.data['loss'][-1]:.6e} (iter: {total_iterations[0]}, evals: {total_func_evals[0]})")

        if metrics.data['loss'][-1] < optimization.R_tol:
            if verbose:
                print(f"  Converged at quasi-iteration {quasi_iter + 1}")
            converged = True
            break

        if total_func_evals[0] >= max_line_searches:
            if verbose:
                print(f"  Max line searches reached")
            break

    total_time = metrics.data['wall_time'][-1]

    summary = {
        'method': 'NiL-Q',
        'total_iterations': int(total_iterations[0]),
        'total_line_searches': int(total_func_evals[0]),
        'n_quasi_iters': int(n_quasi_iters),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_params),
        'pretrain_loss': float(pretrain_loss),
        'converged': bool(converged)
    }

    return model, metrics, summary


# =============================================================================
# METHOD 3: NONLINEAR LiL (LiL-N)
# =============================================================================

def solve_nonlinear_lil(
    physics: BratuPhysics,
    discretization: DiscretizationConfig,
    optimization: OptimizationConfig,
    basis,
    device: torch.device,
    verbose: bool = True
    ) -> Tuple[object, np.ndarray, MetricsTracker, Dict]:
    """Solve Bratu equation using nonlinear LiL (LiL-N method).

    Parameters
    ----------
    basis : TensorProductBasis2D or ELMBasis2D
        The LiL basis to use. Created via create_bratu_basis() or directly.

    Returns
    -------
    basis : the basis object (for evaluation)
    coefficients : np.ndarray of solved coefficients
    metrics : MetricsTracker
    summary : dict
    """
    set_seed(discretization.seed)

    if verbose:
        print("=" * 80)
        print("METHOD: Nonlinear LiL (LiL-N)")
        print(f"  Basis: {basis}")
        print("=" * 80)

    n_coefs = basis.n_basis

    if verbose:
        print(f"  Coefficients: {n_coefs}")

    init_coeffs, pretrain_loss = pretrain_lil_on_initial_guess(
        basis, physics, discretization, optimization
    )
    if verbose:
        print(f"  Pre-training loss: {pretrain_loss:.6e}")

    points = generate_collocation_points(physics, discretization, device)

    x_pde = points['pts_pde'][:, 0]
    y_pde = points['pts_pde'][:, 1]
    x_bc_pts = points['pts_bc'][:, 0]
    y_bc_pts = points['pts_bc'][:, 1]

    A_u = basis.evaluate(x_pde, y_pde)
    A_uxx = basis.derivative(x_pde, y_pde, dx=2, dy=0)
    A_uyy = basis.derivative(x_pde, y_pde, dx=0, dy=2)
    A_bc = basis.evaluate(x_bc_pts, y_bc_pts)

    A_u_t = torch.from_numpy(A_u).to(device)
    A_uxx_t = torch.from_numpy(A_uxx).to(device)
    A_uyy_t = torch.from_numpy(A_uyy).to(device)
    A_bc_t = torch.from_numpy(A_bc).to(device)

    beta = torch.from_numpy(init_coeffs).to(device).requires_grad_(True)

    lambda_pde = torch.tensor(optimization.lambda_pde, dtype=torch.float64, device=device)
    lambda_bc = torch.tensor(optimization.lambda_bc, dtype=torch.float64, device=device)

    metrics = MetricsTracker()
    metrics.start()

    iteration_counter = [0]
    func_eval_counter = [0]

    def compute_loss():
        u = A_u_t @ beta
        u_xx = A_uxx_t @ beta
        u_yy = A_uyy_t @ beta

        residual_pde = u_xx + u_yy + physics.lambda_ * torch.exp(u)
        pde_loss = torch.mean(residual_pde ** 2)

        bc_res = A_bc_t @ beta
        bc_loss = torch.mean(bc_res ** 2)

        total = lambda_pde * pde_loss + lambda_bc * bc_loss
        return total, pde_loss, bc_loss

    with torch.no_grad():
        total_loss, pde_loss, bc_loss = compute_loss()
        metrics.record(0, 0, total_loss.item(), pde_loss.item(), bc_loss.item())

    optimizer = optim.LBFGS(
        [beta],
        lr=1.0,
        max_iter=1,
        max_eval=15,
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    def closure():
        func_eval_counter[0] += 1
        optimizer.zero_grad()
        total, _, _ = compute_loss()
        total.backward()
        return total

    converged = False
    max_iters = optimization.max_iterations
    max_line_searches = optimization.max_line_searches

    while iteration_counter[0] < max_iters and func_eval_counter[0] < max_line_searches:
        optimizer.step(closure)
        iteration_counter[0] += 1

        with torch.no_grad():
            total_loss, pde_loss, bc_loss = compute_loss()
            metrics.record(iteration_counter[0], func_eval_counter[0],
                          total_loss.item(), pde_loss.item(), bc_loss.item())

        if verbose and iteration_counter[0] % 500 == 0:
            print(f"  Iter {iteration_counter[0]} (evals: {func_eval_counter[0]}): loss={total_loss.item():.6e}")

        if total_loss.item() < optimization.R_tol:
            if verbose:
                print(f"  Converged at iter {iteration_counter[0]} ({func_eval_counter[0]} evals)")
            converged = True
            break

    coefficients = beta.detach().cpu().numpy()
    total_time = metrics.data['wall_time'][-1]

    summary = {
        'method': 'LiL-N',
        'total_iterations': int(iteration_counter[0]),
        'total_line_searches': int(func_eval_counter[0]),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_coefs),
        'pretrain_loss': float(pretrain_loss),
        'converged': bool(converged)
    }

    return basis, coefficients, metrics, summary


# =============================================================================
# METHOD 4: QUASILINEAR LiL (LiL-Q)
# =============================================================================

def solve_quasilinear_lil(
    physics: BratuPhysics,
    discretization: DiscretizationConfig,
    optimization: OptimizationConfig,
    basis,
    verbose: bool = True,
    analyze_svd: bool = False
    ) -> Tuple[object, np.ndarray, MetricsTracker, Dict]:
    """Solve Bratu equation using quasilinear LiL (LiL-Q method).

    Parameters
    ----------
    basis : TensorProductBasis2D or ELMBasis2D
        The LiL basis to use. Created via create_bratu_basis() or directly.

    Returns
    -------
    basis : the basis object (for evaluation)
    coefficients : np.ndarray of solved coefficients
    metrics : MetricsTracker
    summary : dict

    NOTE: Uses direct linear solves, so iteration = func_eval for this method.
    """
    set_seed(discretization.seed)

    if verbose:
        print("=" * 80)
        print("METHOD: Quasilinear LiL (LiL-Q)")
        print(f"  Basis: {basis}")
        print("=" * 80)

    n_coefs = basis.n_basis

    if verbose:
        print(f"  Coefficients: {n_coefs}")

    beta, pretrain_loss = pretrain_lil_on_initial_guess(
        basis, physics, discretization, optimization
    )
    if verbose:
        print(f"  Pre-training loss: {pretrain_loss:.6e}")

    points = generate_collocation_points(physics, discretization, DEVICE)

    x_pde = points['pts_pde'][:, 0]
    y_pde = points['pts_pde'][:, 1]
    x_bc_pts = points['pts_bc'][:, 0]
    y_bc_pts = points['pts_bc'][:, 1]

    A_u = basis.evaluate(x_pde, y_pde)
    A_uxx = basis.derivative(x_pde, y_pde, dx=2, dy=0)
    A_uyy = basis.derivative(x_pde, y_pde, dx=0, dy=2)
    A_bc = basis.evaluate(x_bc_pts, y_bc_pts)

    lambda_pde = optimization.lambda_pde
    lambda_bc = optimization.lambda_bc

    metrics = MetricsTracker()
    metrics.start()

    def compute_nonlinear_residual(b):
        return (A_uxx @ b) + (A_uyy @ b) + physics.lambda_ * np.exp(A_u @ b)

    pde_res = compute_nonlinear_residual(beta)
    pde_mse = float(np.mean(pde_res ** 2))
    bc_mse = float(np.mean((A_bc @ beta) ** 2))
    total_loss = lambda_pde * pde_mse + lambda_bc * bc_mse

    metrics.record(0, 0, total_loss, pde_mse, bc_mse)

    if verbose:
        print(f"  Initial loss: {total_loss:.6e}")

    converged = False
    n_quasi_iters = 0

    for quasi_iter in range(optimization.max_quasi_iters):
        n_quasi_iters = quasi_iter + 1

        u_prev = A_u @ beta
        exp_u_prev = np.exp(u_prev)

        A_linear = A_uxx + A_uyy + physics.lambda_ * (exp_u_prev.reshape(-1, 1) * A_u)
        b_linear = physics.lambda_ * exp_u_prev * (u_prev - 1)

        n_pde = points['n_pde']
        n_bc_total = points['pts_bc'].shape[0]

        weight_pde = np.sqrt(lambda_pde / n_pde)
        weight_bc = np.sqrt(lambda_bc / n_bc_total)

        matrices = [weight_pde * A_linear, weight_bc * A_bc]
        targets = [weight_pde * b_linear, weight_bc * np.zeros(n_bc_total, dtype=np.float64)]

        A_stacked = np.vstack(matrices)
        b_stacked = np.concatenate(targets)

        beta = scipy.linalg.lstsq(A_stacked, b_stacked, lapack_driver='gelsy')[0]

        linear_mse = float(np.mean((A_linear @ beta - b_linear) ** 2))

        pde_res = compute_nonlinear_residual(beta)
        pde_mse = float(np.mean(pde_res ** 2))
        bc_mse = float(np.mean((A_bc @ beta) ** 2))
        total_loss = lambda_pde * pde_mse + lambda_bc * bc_mse
        A_cond = 0.0
        if analyze_svd:
            A_cond = np.linalg.cond(A_stacked)

            # 1. Compute the SVD to get the singular values (sigma)
            U, sigma, Vh = np.linalg.svd(A_stacked, full_matrices=False)
            
            # 2. Compute the condition number kappa
            # \kappa := \sigma_max / \sigma_min
            kappa = sigma[0] / sigma[-1]
            
            # 3. Compute the quartiles (Percentiles: 100%, 75%, 50%, 25%, 0%)
            # q4 corresponds to maximum (sigma[0]) and q0 corresponds to minimum (sigma[-1])
            q4, q3, q2, q1, q0 = np.percentile(sigma, [100, 75, 50, 25, 0])
            
            # 4. Define the tolerance and compute the numerical rank
            # Machine epsilon for float64
            epsilon = np.finfo(A_stacked.dtype).eps 
            tolerance = max(A_stacked.shape) * sigma[0] * epsilon
            
            # Count singular values strictly greater than the tolerance
            numerical_rank = np.sum(sigma > tolerance)
            
            # Print clean summary containing the quartiles
            print(f"  SVD analysis:\n"
                f"   sigma_max = {q4:.4e}, Q3 = {q3:.4e}, median = {q2:.4e}, Q1 = {q1:.4e}, sigma_min = {q0:.4e}\n"
                f"   kappa = {kappa:.4e}, epsilon = {epsilon:.4e}, \n"
                f"   tolerance = {tolerance:.4e}, numerical rank = {numerical_rank}/{A_stacked.shape[1]}")

        metrics.record(quasi_iter + 1, quasi_iter + 1, total_loss, pde_mse, bc_mse, linear_mse, A_cond)

        if verbose and (quasi_iter + 1) % 10 == 0:
            print(f"  Iter {quasi_iter + 1}: loss={total_loss:.6e}")

        if total_loss < optimization.R_tol:
            if verbose:
                print(f"  Converged at iteration {quasi_iter + 1}")
            converged = True
            break

    coefficients = beta.astype(np.float64)
    total_time = metrics.data['wall_time'][-1]

    summary = {
        'method': 'LiL-Q',
        'total_iterations': int(n_quasi_iters),
        'total_line_searches': int(n_quasi_iters),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_coefs),
        'pretrain_loss': float(pretrain_loss),
        'converged': bool(converged)
    }

    return basis, coefficients, metrics, summary


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_nn_solution(model, physics, discretization, n_eval=200, device=DEVICE):
    """Evaluate NN solution on uniform grid."""
    model.eval()

    x = torch.linspace(physics.x_domain[0], physics.x_domain[1], n_eval)
    y = torch.linspace(physics.y_domain[0], physics.y_domain[1], n_eval)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    xy = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    with torch.no_grad():
        U = model(xy).reshape(n_eval, n_eval).cpu().numpy()

    return X.numpy(), Y.numpy(), U


def evaluate_nn_residual(model, physics, discretization, n_eval=200, device=DEVICE):
    """Evaluate NN PDE residual on uniform grid."""
    model.eval()

    x_r = torch.linspace(physics.x_domain[0], physics.x_domain[1], n_eval).unsqueeze(1)
    y_r = torch.linspace(physics.y_domain[0], physics.y_domain[1], n_eval).unsqueeze(1)
    X_r, Y_r = torch.meshgrid(x_r.squeeze(), y_r.squeeze(), indexing='ij')

    x_flat = X_r.flatten().unsqueeze(1).requires_grad_(True).to(device)
    y_flat = Y_r.flatten().unsqueeze(1).requires_grad_(True).to(device)

    res = compute_pde_residual_pinn(model, x_flat, y_flat, physics.lambda_)
    residual = res.detach().cpu().numpy().reshape(n_eval, n_eval)

    return X_r.detach().numpy(), Y_r.detach().numpy(), residual


def evaluate_lil_solution(basis, coefficients, physics, discretization, n_eval=200):
    """Evaluate LiL solution on uniform grid.

    Works with any basis from lil_basis.
    """
    x = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval)
    y = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x_flat = X.ravel()
    y_flat = Y.ravel()

    U = basis.reconstruct(coefficients, x_flat, y_flat).reshape(n_eval, n_eval)

    return X, Y, U


def evaluate_lil_residual(basis, coefficients, physics, discretization, n_eval=200):
    """Evaluate LiL PDE residual on uniform grid.

    Works with any basis from lil_basis.
    """
    x = np.linspace(physics.x_domain[0], physics.x_domain[1], n_eval)
    y = np.linspace(physics.y_domain[0], physics.y_domain[1], n_eval)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x_flat = X.ravel()
    y_flat = Y.ravel()

    u = basis.reconstruct(coefficients, x_flat, y_flat)
    u_xx = basis.reconstruct_derivative(coefficients, x_flat, y_flat, dx=2, dy=0)
    u_yy = basis.reconstruct_derivative(coefficients, x_flat, y_flat, dx=0, dy=2)

    res = u_xx + u_yy + physics.lambda_ * np.exp(u)
    residual = res.reshape(n_eval, n_eval)

    return X, Y, residual


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Bratu Equation Core Module (Modular LiL)")
    print(f"Device: {DEVICE}")
    print("\nThis module tracks BOTH iterations AND line searches.")
    print("Use run_bratu_experiments.py to run experiments.")
    print("\nSupported LiL basis types:")
    print("  chebyshev, sin_sin, cos_cos, fourier,")
    print("  cheb_sin, sin_cheb, cheb_cos, cos_cheb,")
    print("  cos_sin, sin_cos, elm")
