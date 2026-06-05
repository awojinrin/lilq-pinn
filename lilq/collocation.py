"""
Collocation Point Generation for LiL-Q
========================================

Generic collocation point generation for 2D domains, handling the common
patterns across all PDE problems: interior grid, boundary edges, and
optional initial condition line.
"""

import math
import numpy as np
import torch
from typing import Tuple, Dict


def generate_collocation_points_2d(
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
    N_x: int,
    N_y: int,
    k_ratio: int = 10,
    collocation_ratios: Tuple[float, ...] = (0.85, 0.15),
    sampling: str = "random",
    seed: int = 42,
    has_initial_condition: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate collocation points for 2D PDE problems.

    For problems **without** an initial condition (e.g., Bratu):
        ``collocation_ratios = (ratio_pde, ratio_bc)``

    For problems **with** an initial condition (e.g., Burgers, BL):
        ``collocation_ratios = (ratio_pde, ratio_ic, ratio_bc)``

    Parameters
    ----------
    x_domain, y_domain : tuple
        Physical domain bounds (x_min, x_max) and (y_min, y_max).
        For time-dependent problems, y_domain is the time domain (0, T).
    N_x, N_y : int
        Basis function counts per direction (determines collocation density).
    k_ratio : int
        Oversampling ratio: total collocation ≈ k_ratio × N_x × N_y.
    collocation_ratios : tuple
        Relative allocation to (PDE, BC) or (PDE, IC, BC).
    sampling : str
        'uniform' for linspace grids, 'random' for random points.
    seed : int
        Random seed for reproducible point placement.
    has_initial_condition : bool
        If True, expects 3-element collocation_ratios and generates
        IC points along y=y_min.

    Returns
    -------
    dict with keys:
        'x_pde', 'y_pde' : 1D arrays of interior PDE points
        'n_pde' : int, number of interior points
        'x_bc_*', 'y_bc_*' : boundary point arrays for each edge
        'n_bc' : int, number of BC points per edge
        'x_ic', 'y_ic', 'n_ic' : IC point arrays (if has_initial_condition)
    """
    np.random.seed(seed)

    n_coefs = N_x * N_y
    x_min, x_max = x_domain
    y_min, y_max = y_domain

    # Normalize ratios
    ratios = list(collocation_ratios)
    total_ratio = sum(ratios)
    norm_ratios = [r / total_ratio for r in ratios]

    # ── Interior PDE points ──
    n_pde = k_ratio * norm_ratios[0] * n_coefs
    n_pde_dim = max(math.ceil(np.sqrt(n_pde)), 5)

    eps = 1e-6
    if sampling == "random":
        xp = np.sort(np.random.uniform(x_min + eps, x_max - eps, n_pde_dim)).astype(np.float64)
        yp = np.sort(np.random.uniform(y_min + eps, y_max - eps, n_pde_dim)).astype(np.float64)
    else:
        xp = np.linspace(x_min + eps, x_max - eps, n_pde_dim, dtype=np.float64)
        yp = np.linspace(y_min + eps, y_max - eps, n_pde_dim, dtype=np.float64)

    xx, yy = np.meshgrid(xp, yp)
    x_pde = xx.ravel()
    y_pde = yy.ravel()

    result = {
        'x_pde': x_pde,
        'y_pde': y_pde,
        'n_pde': len(x_pde),
    }

    # ── Initial condition points (along y = y_min) ──
    if has_initial_condition:
        ic_ratio_idx = 1
        bc_ratio_idx = 2
        n_ic = max(10, math.ceil(k_ratio * norm_ratios[ic_ratio_idx] * n_coefs))

        if sampling == "random":
            x_ic = np.sort(np.random.uniform(x_min, x_max, n_ic)).astype(np.float64)
        else:
            x_ic = np.linspace(x_min, x_max, n_ic, dtype=np.float64)

        result['x_ic'] = x_ic
        result['y_ic'] = np.full_like(x_ic, y_min)
        result['n_ic'] = n_ic
    else:
        bc_ratio_idx = 1

    # ── Boundary condition points ──
    n_bc = max(10, math.ceil(k_ratio * norm_ratios[bc_ratio_idx] * n_coefs / 4))

    if sampling == "random":
        tx = np.sort(np.random.uniform(x_min, x_max, n_bc)).astype(np.float64)
        ty = np.sort(np.random.uniform(y_min, y_max, n_bc)).astype(np.float64)
    else:
        tx = np.linspace(x_min, x_max, n_bc, dtype=np.float64)
        ty = np.linspace(y_min, y_max, n_bc, dtype=np.float64)

    result['x_bc_left'] = np.full(n_bc, x_min, dtype=np.float64)
    result['y_bc_left'] = ty
    result['x_bc_right'] = np.full(n_bc, x_max, dtype=np.float64)
    result['y_bc_right'] = ty
    result['x_bc_bottom'] = tx
    result['y_bc_bottom'] = np.full(n_bc, y_min, dtype=np.float64)
    result['x_bc_top'] = tx
    result['y_bc_top'] = np.full(n_bc, y_max, dtype=np.float64)
    result['n_bc'] = n_bc

    return result


def collocation_to_torch(points: Dict[str, np.ndarray],
                         device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert numpy collocation points to torch tensors on the given device.

    Returns a new dictionary with the same keys but torch.Tensor values.
    Numeric scalars (n_pde, n_bc, etc.) are preserved as integers.
    PDE interior points get ``requires_grad=True`` for autograd.
    """
    result = {}
    for key, val in points.items():
        if isinstance(val, np.ndarray):
            t = torch.tensor(val, dtype=torch.float64, device=device)
            if 'pde' in key:
                t = t.requires_grad_(True)
            result[key] = t
        else:
            result[key] = val  # int counts
    return result
