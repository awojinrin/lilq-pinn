"""
Pretraining Utilities for LiL-Q
================================

Pre-train neural networks and basis coefficients to an initial guess
function, ensuring all methods start from a comparable baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
from typing import Tuple, Callable


def nn_pretrain_grid_side(target_dof: int, floor: int) -> int:
    """Per-dimension fit-grid side length for NN pretraining.

    Restores the pre-GitHub-consolidation behavior of scaling the
    pretraining fit grid with the problem's own basis DOF count, floored at
    a per-problem minimum so small configurations still get a usable fit.
    Each problem's ``(floor, target_dof)`` pair is its own historical value
    (they differ across Bratu/Burgers/Buckley-Leverett) -- see
    ``DECISIONS.md`` for the exact numbers and provenance.

    Parameters
    ----------
    target_dof : int
        Nominal total fit-point target, e.g. ``N_x * N_y``.
    floor : int
        Minimum total fit-point count.

    Returns
    -------
    int
        Grid points per dimension (a square grid of this side length is
        used for the fit).
    """
    return int(np.sqrt(max(floor, target_dof)))


def pretrain_nn(
    model: nn.Module,
    initial_guess_fn: Callable,
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
    device: torch.device,
    n_grid: int = 50,
    max_epochs: int = 500,
    tol: float = 1e-4,
    verbose: bool = True,
) -> float:
    """Pre-train a neural network to match an initial guess function.

    Uses L-BFGS optimization to fit the NN output to target values
    produced by ``initial_guess_fn(x, y)`` on a grid.

    Parameters
    ----------
    model : nn.Module
        Neural network to pre-train (modified in-place).
    initial_guess_fn : callable
        Function ``(x, y) -> values`` providing the target. Should accept
        both numpy arrays and torch tensors.
    x_domain, y_domain : tuple
        Physical domain (min, max) for each direction.
    device : torch.device
        Computation device.
    n_grid : int
        Grid resolution per dimension for fitting points.
    max_epochs : int
        Maximum L-BFGS epochs.
    tol : float
        Convergence tolerance on MSE loss.
    verbose : bool
        Print progress.

    Returns
    -------
    float
        Final pre-training MSE loss.
    """
    if max_epochs <= 0:
        if verbose:
            print("  Pre-training skipped (max_epochs <= 0)")
        return 0.0

    model.train()

    x_fit = np.linspace(x_domain[0], x_domain[1], n_grid, dtype=np.float64)
    y_fit = np.linspace(y_domain[0], y_domain[1], n_grid, dtype=np.float64)
    X, Y = np.meshgrid(x_fit, y_fit)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # Get target values from initial guess
    target_np = initial_guess_fn(x_flat, y_flat)
    if isinstance(target_np, torch.Tensor):
        target_np = target_np.detach().cpu().numpy()
    target_np = np.asarray(target_np, dtype=np.float64).flatten()

    xy_tensor = torch.tensor(
        np.column_stack([x_flat, y_flat]),
        dtype=torch.float64, device=device,
    )
    target_tensor = torch.tensor(
        target_np.reshape(-1, 1),
        dtype=torch.float64, device=device,
    )

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn='strong_wolfe',
    )

    loss_fn = nn.MSELoss()
    final_loss = float('inf')

    if verbose:
        print(f"  Pre-training NN ({max_epochs} epochs)...")

    for epoch in range(max_epochs):
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

        if verbose and epoch % 200 == 0:
            print(f"    Epoch {epoch:4d}: loss = {final_loss:.6e}")

        if final_loss < tol:
            if verbose:
                print(f"    Converged at epoch {epoch}: loss {final_loss:.6e} < {tol}")
            break

    if verbose:
        print(f"  Pre-training complete. Final loss: {final_loss:.6e}")

    model.train()
    return final_loss


def pretrain_lil(
    basis,
    initial_guess_fn: Callable,
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
    n_grid: int = 100,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Pre-train LiL basis coefficients via least-squares fitting.

    Fits coefficients so that ``basis.evaluate(x, y) @ coefficients``
    approximates ``initial_guess_fn(x, y)`` on a grid.

    Parameters
    ----------
    basis : TensorProductBasis2D or ELMBasis2D
        Basis object.
    initial_guess_fn : callable
        Function ``(x, y) -> values`` providing the target.
    x_domain, y_domain : tuple
        Physical domain (min, max) for each direction.
    n_grid : int
        Minimum total fit-point count (not per-dimension -- the per-
        dimension side length is ``sqrt(max(n_grid, 2 * n_basis))``).
        Default 100 matches the historical value used identically across
        Bratu, Burgers, and Buckley-Leverett pre-consolidation; see
        ``DECISIONS.md``.
    verbose : bool
        Print progress.

    Returns
    -------
    coefficients : np.ndarray of shape (n_basis,)
        Fitted coefficients.
    mse : float
        Final mean squared error.
    """
    n_coefs = basis.n_basis

    n_fit = nn_pretrain_grid_side(2 * n_coefs, floor=n_grid)

    x_fit = np.linspace(x_domain[0], x_domain[1], n_fit, dtype=np.float64)
    y_fit = np.linspace(y_domain[0], y_domain[1], n_fit, dtype=np.float64)
    X, Y = np.meshgrid(x_fit, y_fit)
    x_flat = X.ravel()
    y_flat = Y.ravel()

    target = initial_guess_fn(x_flat, y_flat)
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    target = np.asarray(target, dtype=np.float64).flatten()

    if verbose:
        print("  Pre-training LiL (least squares)...")

    A = basis.evaluate(x_flat, y_flat)
    coefficients = scipy.linalg.lstsq(A, target, lapack_driver='gelsy')[0]

    pred = A @ coefficients
    mse = float(np.mean((pred - target) ** 2))

    if verbose:
        print(f"  Pre-training complete. MSE: {mse:.6e}")

    return coefficients, mse
