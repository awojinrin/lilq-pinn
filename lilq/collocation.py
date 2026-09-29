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
from typing import Tuple, Dict, Optional


def points_1d(a: float, b: float, n: int, sampling: str) -> np.ndarray:
    """n points on [a, b] of one family: sorted uniform random ('random',
    'scattered'), equispaced ('uniform'), or Chebyshev-Gauss-Lobatto ('cgl',
    endpoints included, clustered towards them). Random draws use NumPy's
    global generator, as the paper's construction does."""
    if sampling in ("random", "scattered"):
        return np.sort(np.random.uniform(a, b, n)).astype(np.float64)
    if sampling == "cgl":
        t = np.cos(np.pi * np.arange(n) / (n - 1))[::-1]           # -1 .. 1
        return (a + (b - a) * (t + 1.0) / 2.0).astype(np.float64)
    return np.linspace(a, b, n, dtype=np.float64)


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
    floor: Optional[int] = None,
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
        'random' (the paper's): tensor grid of sorted uniform random
        abscissae; 'uniform': equispaced tensor grid; 'cgl':
        Chebyshev-Gauss-Lobatto tensor grid; 'scattered': uniformly random
        points, not a tensor grid (the same number of them). Boundary and
        initial points follow the same family along each edge.
    seed : int
        Random seed for reproducible point placement.
    has_initial_condition : bool
        If True, expects 3-element collocation_ratios and generates
        IC points along y=y_min.
    floor : int, optional
        Minimum points per direction and per boundary/initial edge. None
        keeps the paper's minimums (5 per direction, 10 per edge);
        Component C lowers them so that small N/P are reachable.

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
    n_pde_dim = max(math.ceil(np.sqrt(n_pde)), 5 if floor is None else floor)

    eps = 1e-6
    if sampling not in ("random", "uniform", "cgl", "scattered"):
        raise ValueError(f"unknown sampling {sampling!r}")
    if sampling == "scattered":
        n = n_pde_dim ** 2
        x_pde = np.random.uniform(x_min + eps, x_max - eps, n).astype(np.float64)
        y_pde = np.random.uniform(y_min + eps, y_max - eps, n).astype(np.float64)
    else:
        xp = points_1d(x_min + eps, x_max - eps, n_pde_dim, sampling)
        yp = points_1d(y_min + eps, y_max - eps, n_pde_dim, sampling)
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
        n_ic = max(10 if floor is None else floor, math.ceil(k_ratio * norm_ratios[ic_ratio_idx] * n_coefs))

        x_ic = points_1d(x_min, x_max, n_ic, sampling)

        result['x_ic'] = x_ic
        result['y_ic'] = np.full_like(x_ic, y_min)
        result['n_ic'] = n_ic
    else:
        bc_ratio_idx = 1

    # ── Boundary condition points ──
    n_bc = max(10 if floor is None else floor, math.ceil(k_ratio * norm_ratios[bc_ratio_idx] * n_coefs / 4))

    tx = points_1d(x_min, x_max, n_bc, sampling)
    ty = points_1d(y_min, y_max, n_bc, sampling)

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


def count_distinct_rows(equation, x, y) -> int:
    """Distinct collocation rows: rows of the same equation at the same
    point are identical (a tensor grid's corner lies on two edges, so its
    boundary rows appear twice). Addendum v2.2 Section 2.9."""
    keys = set(zip(np.asarray(equation).tolist(), np.asarray(x, dtype=np.float64).tolist(),
                   np.asarray(y, dtype=np.float64).tolist()))
    return len(keys)


def write_collocation_rows(path, blocks) -> int:
    """Save every collocation row, in the order the system is assembled, to
    ``path`` (``.npz``): ``x``, ``y``, ``block`` (e.g. ``bc_u_left``),
    ``equation`` (the row's equation: rows with the same equation and point
    are identical), ``weight`` (the row weight), and ``n_distinct``. What an
    a-posteriori computation of the sampling constants c1, c2 on each grid
    needs (Addendum v2.2 Section 2.8.3). ``blocks``: a list of ``(block,
    equation, x, y, weight)``. Returns ``n_distinct``."""
    import os
    xs, ys, blk, eq, w = [], [], [], [], []
    for block, equation, x, y, weight in blocks:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        xs.append(x); ys.append(y)
        blk += [block] * len(x); eq += [equation] * len(x)
        w.append(np.full(len(x), float(weight)))
    x, y = np.concatenate(xs), np.concatenate(ys)
    n_distinct = count_distinct_rows(eq, x, y)
    path = str(path)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, x=x, y=y, block=np.array(blk), equation=np.array(eq),
                        weight=np.concatenate(w), n_distinct=np.array(n_distinct))
    os.replace(tmp, path)
    return n_distinct


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
