"""
Generic Solver Templates for LiL-Q
====================================

Reusable solver implementations for the four methods:
    - NiL-N: Standard PINN (Nonlinear-in-Learnables, Nonlinear PDE)
    - NiL-Q: Quasilinear PINN (Nonlinear-in-Learnables, Quasilinearized PDE)
    - LiL-N: Nonlinear LiL (Linear-in-Learnables, Nonlinear PDE)
    - LiL-Q: Quasilinear LiL (Linear-in-Learnables, Quasilinearized PDE)

Each solver accepts callback functions for the problem-specific parts
(PDE residual, boundary conditions, quasilinearization formula).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
from typing import Tuple, Callable, Dict, Optional

from .nn import MLP, calculate_hidden_dim
from .metrics import MetricsTracker, QuasilinearMetrics
from .utils import set_seed, clear_gpu_memory


# Force float64 precision for all solvers
torch.set_default_dtype(torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: NiL-N — Standard PINN (Nonlinear-in-Learnables, Nonlinear PDE)
# ─────────────────────────────────────────────────────────────────────────────

def solve_nil_n(
    compute_pde_residual: Callable,
    compute_bc_residual: Callable,
    model: MLP,
    x_pde: torch.Tensor,
    y_pde: torch.Tensor,
    bc_data: dict,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_ic: float = 10.0,
    max_iterations: int = 10000,
    max_line_searches: int = 100000,
    R_tol: float = 1e-4,
    verbose: bool = True,
) -> Tuple[MLP, MetricsTracker, Dict]:
    """Standard PINN solver (NiL-N method).

    Parameters
    ----------
    compute_pde_residual : callable
        ``(model, x_pde, y_pde) -> residual_tensor``
    compute_bc_residual : callable
        ``(model, bc_data) -> (bc_loss_tensor, ic_loss_tensor_or_None)``
    model : MLP
        Pre-initialized neural network.
    x_pde, y_pde : torch.Tensor
        Interior collocation point coordinates (requires_grad=True).
    bc_data : dict
        Boundary/IC point data (problem-specific structure).
    lambda_pde, lambda_bc, lambda_ic : float
        Loss weights.
    max_iterations, max_line_searches : int
        Stopping criteria.
    R_tol : float
        Convergence tolerance on total loss.
    verbose : bool
        Print progress.

    Returns
    -------
    model, metrics, summary
    """
    loss_fn = nn.MSELoss()
    metrics = MetricsTracker()
    metrics.start()

    iteration_counter = [0]
    func_eval_counter = [0]

    def compute_loss():
        residual = compute_pde_residual(model, x_pde, y_pde)
        pde_loss = loss_fn(residual, torch.zeros_like(residual))

        bc_loss, ic_loss = compute_bc_residual(model, bc_data)

        total = lambda_pde * pde_loss + lambda_bc * bc_loss
        ic_val = 0.0
        if ic_loss is not None:
            total = total + lambda_ic * ic_loss
            ic_val = ic_loss.item()

        return total, pde_loss.item(), ic_val, bc_loss.item()

    # Record initial state
    model.train()
    total_loss, pde_val, ic_val, bc_val = compute_loss()
    metrics.record(0, 0, total_loss.item(), pde_val, ic_val, bc_val)

    optimizer = optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=1, max_eval=15,
        tolerance_grad=1e-8, tolerance_change=1e-9,
        history_size=100, line_search_fn='strong_wolfe',
    )

    def closure():
        func_eval_counter[0] += 1
        optimizer.zero_grad()
        total, _, _, _ = compute_loss()
        total.backward()
        return total

    converged = False
    while iteration_counter[0] < max_iterations and func_eval_counter[0] < max_line_searches:
        optimizer.step(closure)
        iteration_counter[0] += 1

        total_loss, pde_val, ic_val, bc_val = compute_loss()
        metrics.record(
            iteration_counter[0], func_eval_counter[0],
            total_loss.item(), pde_val, ic_val, bc_val,
        )

        if verbose and iteration_counter[0] % 500 == 0:
            print(f"  Iter {iteration_counter[0]} (evals: {func_eval_counter[0]}): "
                  f"loss={total_loss.item():.6e}")

        if total_loss.item() < R_tol:
            if verbose:
                print(f"  Converged at iter {iteration_counter[0]} "
                      f"({func_eval_counter[0]} evals)")
            converged = True
            break

    total_time = metrics.data['wall_time'][-1]
    n_params = sum(p.numel() for p in model.parameters())

    summary = {
        'method': 'NiL-N',
        'total_iterations': int(iteration_counter[0]),
        'total_line_searches': int(func_eval_counter[0]),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_params),
        'converged': bool(converged),
    }

    return model, metrics, summary


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: NiL-Q — Quasilinear PINN (Nonlinear, Quasilinearized PDE)
# ─────────────────────────────────────────────────────────────────────────────

def solve_nil_q(
    compute_pde_residual: Callable,
    compute_linearized_residual_fn: Callable,
    compute_bc_residual: Callable,
    model: MLP,
    x_pde: torch.Tensor,
    y_pde: torch.Tensor,
    bc_data: dict,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_ic: float = 10.0,
    max_quasi_iters: int = 25,
    max_inner_iters: int = 300,
    max_line_searches: int = 100000,
    R_tol: float = 1e-4,
    verbose: bool = True,
) -> Tuple[MLP, MetricsTracker, Dict]:
    """Quasilinear PINN solver (NiL-Q method).

    Parameters
    ----------
    compute_pde_residual : callable
        Full nonlinear: ``(model, x, y) -> residual``
    compute_linearized_residual_fn : callable
        ``(model, x, y, frozen_data) -> linearized_residual``
        where ``frozen_data`` is computed from the previous iterate.
    compute_bc_residual : callable
        ``(model, bc_data) -> (bc_loss, ic_loss_or_None)``
    model : MLP
        Pre-initialized neural network.
    """
    loss_fn = nn.MSELoss()
    metrics = MetricsTracker()
    metrics.start()

    total_iterations = [0]
    total_func_evals = [0]

    def compute_full_loss():
        residual = compute_pde_residual(model, x_pde, y_pde)
        pde_loss = loss_fn(residual, torch.zeros_like(residual))
        bc_loss, ic_loss = compute_bc_residual(model, bc_data)
        total = lambda_pde * pde_loss + lambda_bc * bc_loss
        ic_val = 0.0
        if ic_loss is not None:
            total = total + lambda_ic * ic_loss
            ic_val = ic_loss.item()
        return total, pde_loss.item(), ic_val, bc_loss.item()

    # Record initial state
    model.train()
    total_loss, pde_val, ic_val, bc_val = compute_full_loss()
    metrics.record(0, 0, total_loss.item(), pde_val, ic_val, bc_val)

    optimizer = optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=1, max_eval=15,
        tolerance_grad=1e-8, tolerance_change=1e-9,
        history_size=100, line_search_fn='strong_wolfe',
    )

    converged = False
    n_quasi_iters = 0

    for quasi_iter in range(max_quasi_iters):
        n_quasi_iters = quasi_iter + 1
        if verbose:
            print(f"\n  Quasilinear iteration {quasi_iter + 1}")

        # Freeze current iterate
        model.eval()
        frozen_data = compute_linearized_residual_fn(model, x_pde, y_pde, None)
        model.train()

        # Inner L-BFGS loop on linearized problem
        for inner_iter in range(max_inner_iters):
            def closure():
                total_func_evals[0] += 1
                optimizer.zero_grad()
                lin_res = compute_linearized_residual_fn(model, x_pde, y_pde, frozen_data)
                lin_loss = loss_fn(lin_res, torch.zeros_like(lin_res))
                bc_loss, ic_loss = compute_bc_residual(model, bc_data)
                total = lambda_pde * lin_loss + lambda_bc * bc_loss
                if ic_loss is not None:
                    total = total + lambda_ic * ic_loss
                total.backward()
                return total

            optimizer.step(closure)
            total_iterations[0] += 1

            # Evaluate full nonlinear loss
            total_loss, pde_val, ic_val, bc_val = compute_full_loss()
            metrics.record(
                total_iterations[0], total_func_evals[0],
                total_loss.item(), pde_val, ic_val, bc_val,
            )

            if total_loss.item() < R_tol or total_func_evals[0] >= max_line_searches:
                break

        if verbose:
            print(f"    Loss: {metrics.data['loss'][-1]:.6e} "
                  f"(iter: {total_iterations[0]}, evals: {total_func_evals[0]})")

        if metrics.data['loss'][-1] < R_tol:
            if verbose:
                print(f"  Converged at quasi-iteration {quasi_iter + 1}")
            converged = True
            break

        if total_func_evals[0] >= max_line_searches:
            if verbose:
                print("  Max line searches reached")
            break

    total_time = metrics.data['wall_time'][-1]
    n_params = sum(p.numel() for p in model.parameters())

    summary = {
        'method': 'NiL-Q',
        'total_iterations': int(total_iterations[0]),
        'total_line_searches': int(total_func_evals[0]),
        'n_quasi_iters': int(n_quasi_iters),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_params),
        'converged': bool(converged),
    }

    return model, metrics, summary


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: LiL-N — Nonlinear LiL (Linear-in-Learnables, Nonlinear PDE)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lil_n(
    compute_loss_fn: Callable,
    init_coeffs: np.ndarray,
    device: torch.device,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_ic: float = 10.0,
    max_iterations: int = 10000,
    max_line_searches: int = 100000,
    R_tol: float = 1e-4,
    verbose: bool = True,
) -> Tuple[np.ndarray, MetricsTracker, Dict]:
    """Nonlinear LiL solver (LiL-N method).

    Optimizes basis coefficients using L-BFGS on the nonlinear PDE residual.

    Parameters
    ----------
    compute_loss_fn : callable
        ``(beta) -> (total_loss, pde_loss, ic_loss, bc_loss)``
        where ``beta`` is a torch.Tensor of coefficients.
        All return values should be torch.Tensor scalars.
    init_coeffs : np.ndarray
        Initial coefficient vector from pre-training.
    device : torch.device
        Computation device.

    Returns
    -------
    coefficients, metrics, summary
    """
    beta = torch.from_numpy(init_coeffs).to(device).requires_grad_(True)
    n_coefs = len(init_coeffs)

    metrics = MetricsTracker()
    metrics.start()

    iteration_counter = [0]
    func_eval_counter = [0]

    # Record initial state (no torch.no_grad — some loss fns use autograd internally)
    total_loss, pde_loss, ic_loss, bc_loss = compute_loss_fn(beta)
    metrics.record(0, 0, total_loss.item(), pde_loss.item(),
                  ic_loss.item() if isinstance(ic_loss, torch.Tensor) else ic_loss,
                  bc_loss.item())

    optimizer = optim.LBFGS(
        [beta], lr=1.0, max_iter=1, max_eval=15,
        tolerance_grad=1e-8, tolerance_change=1e-9,
        history_size=100, line_search_fn='strong_wolfe',
    )

    def closure():
        func_eval_counter[0] += 1
        optimizer.zero_grad()
        total, _, _, _ = compute_loss_fn(beta)
        total.backward()
        return total

    converged = False
    while iteration_counter[0] < max_iterations and func_eval_counter[0] < max_line_searches:
        optimizer.step(closure)
        iteration_counter[0] += 1

        # No torch.no_grad — some loss fns use autograd internally (e.g. BL flux_derivative)
        total_loss, pde_loss, ic_loss, bc_loss = compute_loss_fn(beta)
        metrics.record(
            iteration_counter[0], func_eval_counter[0],
            total_loss.item(), pde_loss.item(),
            ic_loss.item() if isinstance(ic_loss, torch.Tensor) else ic_loss,
            bc_loss.item(),
        )

        if verbose and iteration_counter[0] % 500 == 0:
            print(f"  Iter {iteration_counter[0]} (evals: {func_eval_counter[0]}): "
                  f"loss={total_loss.item():.6e}")

        if total_loss.item() < R_tol:
            if verbose:
                print(f"  Converged at iter {iteration_counter[0]} "
                      f"({func_eval_counter[0]} evals)")
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
        'converged': bool(converged),
    }

    return coefficients, metrics, summary


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4: LiL-Q — Quasilinear LiL (Linear-in-Learnables, Quasilinearized)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lil_q(
    assemble_system_fn: Callable,
    compute_nonlinear_loss_fn: Callable,
    init_coeffs: np.ndarray,
    max_quasi_iters: int = 100,
    R_tol: float = 1e-4,
    verbose: bool = True,
    diagnostics_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, QuasilinearMetrics, Dict]:
    """Quasilinear LiL solver (LiL-Q method).

    At each outer iteration:
        1. Evaluate the current solution and derivatives.
        2. Assemble the linearized weighted least-squares system.
        3. Solve via QR (``scipy.linalg.lstsq``).
        4. Check convergence on the full nonlinear loss.

    SVD/condition number analysis is **not** embedded in this solver.
    Use ``diagnostics_callback`` or ``lilq.analysis`` for opt-in analysis.

    Parameters
    ----------
    assemble_system_fn : callable
        ``(beta) -> (A_stacked, b_stacked)``
        where ``A_stacked`` is the weighted system matrix and ``b_stacked``
        is the weighted RHS vector. The system is solved via lstsq.
    compute_nonlinear_loss_fn : callable
        ``(beta) -> (total_loss, pde_loss, ic_loss, bc_loss)``
        Evaluates the full nonlinear loss for convergence checking.
        All return values are floats.
    init_coeffs : np.ndarray
        Initial coefficient vector from pre-training.
    max_quasi_iters : int
        Maximum outer iterations.
    R_tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    diagnostics_callback : callable, optional
        ``(A_stacked, beta, quasi_iter)`` called at each iteration for
        opt-in SVD/conditioning analysis. Does NOT affect runtime of
        the core solver when ``None``.

    Returns
    -------
    coefficients, metrics, summary
    """
    beta = init_coeffs.copy().astype(np.float64)
    n_coefs = len(beta)

    metrics = QuasilinearMetrics()

    # Record initial state
    total_loss, pde_loss, ic_loss, bc_loss = compute_nonlinear_loss_fn(beta)
    metrics.record(0, 0, nl_res=total_loss, update_norm=0.0,
                   pde_loss=pde_loss, ic_loss=ic_loss, bc_loss=bc_loss,
                   total_loss=total_loss)

    if verbose:
        print(f"  Initial loss: {total_loss:.6e}")

    converged = False
    n_quasi_iters = 0

    for quasi_iter in range(max_quasi_iters):
        n_quasi_iters = quasi_iter + 1

        # Assemble and solve linearized system
        A_stacked, b_stacked = assemble_system_fn(beta)
        beta_new = scipy.linalg.lstsq(A_stacked, b_stacked, lapack_driver='gelsy')[0]

        # Opt-in diagnostics (SVD, condition number)
        if diagnostics_callback is not None:
            diagnostics_callback(A_stacked, beta_new, quasi_iter)

        # Convergence metrics
        update_norm = float(np.linalg.norm(beta_new - beta) /
                           (np.linalg.norm(beta_new) + 1e-30))
        beta = beta_new

        total_loss, pde_loss, ic_loss, bc_loss = compute_nonlinear_loss_fn(beta)

        metrics.record(
            quasi_iter + 1, quasi_iter + 1,
            nl_res=total_loss, update_norm=update_norm,
            pde_loss=pde_loss, ic_loss=ic_loss, bc_loss=bc_loss,
            total_loss=total_loss,
        )

        if verbose and (quasi_iter + 1) % 5 == 0:
            print(f"  Iter {quasi_iter + 1}: loss={total_loss:.6e}, "
                  f"d_beta={update_norm:.3e}")

        if total_loss < R_tol:
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
        'final_loss': float(metrics.data['total_loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_coefs),
        'converged': bool(converged),
    }

    return coefficients, metrics, summary
