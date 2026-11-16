"""Errors and best-approximation errors in a Sobolev-type norm of X (Package 5, item 1).

The norm of X is a weighted sum of squared component norms, every component weighted equally:
- H2 in two space dimensions: u, u_x, u_y, u_xx, u_xy, u_yy;
- H^{2,1} in space-time: u, u_x, u_xx, u_t.

The integrals are a quadrature (points and weights) supplied by the caller. Nothing here knows the
problem: the caller gives the reference's components at the quadrature points and, for a trial space,
the matrices of the basis's components there (and the components of a lifting, if the space is affine).
"""

import numpy as np
import scipy.linalg as sla

EPS = np.finfo(float).eps


class XNorm:
    """Relative errors in the norm of X against a reference.

    ``weights``: quadrature weights (any positive scale); ``reference``: ``{component: values}`` at the
    quadrature points, in the order of ``components``."""

    def __init__(self, weights, reference, components):
        self.components = tuple(components)
        self.sw = np.sqrt(np.asarray(weights, float))
        self.ref = {c: np.asarray(reference[c], float) for c in self.components}
        self.den = float(np.sqrt(sum(np.sum((self.ref[c] * self.sw) ** 2) for c in self.components)))

    def norm(self, fields):
        """The norm of X of ``{component: values}``, relative to the reference's."""
        return float(np.sqrt(sum(np.sum((np.asarray(fields[c]) * self.sw) ** 2) for c in self.components))) / self.den

    def error(self, fields):
        """The relative error of ``{component: values}``."""
        return self.norm({c: np.asarray(fields[c]) - self.ref[c] for c in self.components})

    def error_of(self, mats, beta, lift=None):
        """The relative error of the trial function with coefficients ``beta``: ``mats`` gives each
        component's basis matrix, ``lift`` each component's lifting values (affine spaces)."""
        return self.error({c: mats[c] @ beta + (0 if lift is None else lift[c]) for c in self.components})

    def best(self, mats, lift=None, driver='gelsy'):
        """The relative distance of the reference to the trial space in the norm of X: least squares on
        all components stacked (``driver`` 'gelsy' or 'gelsd', rcond = eps)."""
        A = np.vstack([mats[c] * self.sw[:, None] for c in self.components])
        b = np.concatenate([(self.ref[c] - (0 if lift is None else lift[c])) * self.sw for c in self.components])
        coef = sla.lstsq(A, b, lapack_driver=driver, cond=EPS, check_finite=False)[0]
        return float(np.linalg.norm(A @ coef - b)) / self.den


def relative_updates(weights, mats, betas, components, lift=None):
    """``||u^(k+1) - u^(k)|| / ||u^(k+1)||`` for k = 0..K-1 in the norm of ``components`` (Package 5,
    Addendum 2): the coefficients ``betas`` (K+1 rows, iterate 0 first), the basis matrices ``mats`` and
    the quadrature ``weights``. For an affine space, ``lift`` (each component's lifting values) enters
    the norm of u^(k+1) and cancels in the difference. Needs no reference."""
    sw = np.sqrt(np.asarray(weights, float))[:, None]
    B = np.asarray(betas, float).T                                           # P x (K+1)
    num, den = 0.0, 0.0
    for c in components:
        F = mats[c] @ B                                                      # points x (K+1)
        if lift is not None:
            F = F + np.asarray(lift[c], float)[:, None]
        num = num + np.sum(((F[:, 1:] - F[:, :-1]) * sw) ** 2, axis=0)
        den = den + np.sum((F[:, 1:] * sw) ** 2, axis=0)
    return np.sqrt(num) / np.sqrt(den)


def gauss_legendre_2d(n, box):
    """A tensor Gauss-Legendre rule of n x n nodes on ``box = ((a, b), (c, d))``: ``(X, Y, W)``,
    the weights summing to 1 (first coordinate outer)."""
    g, w = np.polynomial.legendre.leggauss(n)
    (a, b), (c, d) = box
    x, y = a + (b - a) * (g + 1) / 2, c + (d - c) * (g + 1) / 2
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X.ravel(), Y.ravel(), np.outer(w / 2, w / 2).ravel()
