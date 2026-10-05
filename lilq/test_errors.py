"""
Test-grid evaluation for the per-iterate error columns of ``iterations.csv``
============================================================================

Computational_Package_1_v2.md Section 3.1 item 10 logs the test errors of
every iterate (``eps_u``, ``eps_v``, ``eps_p``, ``eps_p_meanfree``,
``maxerr_*``) on fixed test grids never used for collocation (Section 2).
Evaluating a full basis matrix on such a grid every iteration would cost
far more than the solve (a 301 x 401 grid against 625 basis functions is
~75M entries per field), so this module exploits the tensor-product
structure instead: on a tensor grid, $u = \\Phi_x \\Theta \\Phi_y^\\top$
(and its N-D analogue), which is exact and nearly free.

Both ``TensorProductBasis2D`` and ``TensorProductBasisND`` order
coefficients row-major (first dimension outermost), which is what
:func:`tensor_grid_values` assumes; the tests check it against
``basis.evaluate``.
"""

from typing import Optional, Sequence

import numpy as np


def _factor_bases(basis):
    if hasattr(basis, "bases"):
        return list(basis.bases)
    if hasattr(basis, "basis_x") and hasattr(basis, "basis_y"):
        return [basis.basis_x, basis.basis_y]
    raise TypeError(f"{type(basis).__name__} is not a tensor-product basis")


def tensor_grid_values(basis, coeffs: np.ndarray, axes: Sequence[np.ndarray],
                       orders: Optional[Sequence[int]] = None) -> np.ndarray:
    """Values (or a mixed derivative) of ``basis @ coeffs`` on the tensor
    grid ``axes[0] x axes[1] x ...``, shape ``(len(axes[0]), len(axes[1]), ...)``.

    ``orders[d]`` is the derivative order along dimension ``d`` (default 0).
    """
    factors = _factor_bases(basis)
    if orders is None:
        orders = [0] * len(factors)
    result = np.asarray(coeffs, dtype=np.float64).reshape([b.n_basis for b in factors])
    for b, pts, order in zip(factors, axes, orders):
        pts = np.asarray(pts, dtype=np.float64)
        M = b.evaluate(pts) if order == 0 else b.derivative(pts, order=order)
        # Contract the leading coefficient axis; the point axis goes last,
        # so after every dimension the axes are (pts_0, pts_1, ...).
        result = np.tensordot(result, M, axes=([0], [1]))
    return result


def grid_evaluator(basis, axes: Sequence[np.ndarray]):
    """``coeffs -> values on the tensor grid axes[0] x axes[1]``. A
    tensor-product basis uses :func:`tensor_grid_values`; any other basis (a
    random-feature ELM, say) is evaluated on the grid once and the matrix
    kept, so each later call is one matrix-vector product."""
    try:
        _factor_bases(basis)
        return lambda coeffs: tensor_grid_values(basis, coeffs, axes)
    except TypeError:
        mesh = np.meshgrid(*[np.asarray(a, dtype=np.float64) for a in axes], indexing='ij')
        Phi = basis.evaluate(*[m.ravel() for m in mesh])
        shape = mesh[0].shape
        return lambda coeffs: (Phi @ np.asarray(coeffs, dtype=np.float64)).reshape(shape)


def rel_l2(pred: np.ndarray, exact: np.ndarray) -> float:
    """``||pred - exact||_2 / ||exact||_2`` over the test grid (equal to the
    relative RMS error every problem in this codebase reports)."""
    return float(np.linalg.norm(pred - exact) / max(np.linalg.norm(exact), 1e-300))


def max_abs(pred: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(pred - exact)))
