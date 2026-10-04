"""
Square Chebyshev collocation and least-squares Chebyshev collocation for Bratu (Package 2, item 1, P2-3)
=======================================================================================================

The classical baseline of Package 2 (the advisor's instructions of 4 October
2026, Section 4.1), ported from the advisor's pilot
(``analysis_scripts/bratu_pilot/bratu_pilot.py``, attached to the package).
The problem is 2D Bratu, Laplacian(u) + lambda e^u = 0 on (0,1)^2, u = 0 on
the boundary, lambda = 6.2, from zero (the lower branch).

**One code path.** Every method is the same Newton (Gauss-Newton) iteration:
``assemble`` builds the linearized system from the trial space's values and
Laplacians at the rows, weighted per row; the methods differ only in

* the trial space (``TrialSpace``): tensor Chebyshev polynomials
  T_i(2x-1) T_j(2y-1), i, j < p (``'tensor'``), or the same times
  b = x(1-x)y(1-y), which satisfies the boundary condition identically
  (``'hard'``);
* the rows and their weights (``Layout``): square collocation on the p x p
  Chebyshev-Gauss-Lobatto (CGL) grid with the boundary rows replaced by the
  Dirichlet condition (unweighted; LU); or least squares (``gelsy``) on a CGL
  grid with N/P = r, the interior rows weighted by tensor Clenshaw-Curtis
  weights (``hard``), or interior and boundary rows by equal mean-square
  weights with lambda_bc (``weakMS``, the formulation of the tested members).

Methods (``METHODS``):

========================  ===========================================================  ===============
label                     formulation                                                  free coeff.
========================  ===========================================================  ===============
``SQ-CGL``                square, p x p CGL grid, Dirichlet rows, LU                   (p - 2)^2
``SQ-equi``               the same on equispaced nodes                                 (p - 2)^2
``LS-hard-CGL-<r>``       'hard' space, interior CGL rows, Clenshaw-Curtis weights     p^2
``LS-weakMS-CGL-<r>``     weak boundary rows, equal mean-square weights, lambda_bc     p^2
``LS-weakCC<l>-CGL-<r>``  weak boundary rows, Clenshaw-Curtis weights, lambda_bc = l   p^2
========================  ===========================================================  ===============

The package's ``LS-weakMS-CGL-3`` uses lambda_bc = 10 (the paper's value;
Section 4.1). The pilot ran it with lambda_bc = 1; where they differ, the
instructions win, and the pilot's value is available as ``lam_bc=1``.

**Iteration.** Newton from zero, at most ``K_max`` = 20 solves, stopping when
||beta^(k+1) - beta^(k)|| / ||beta^(k+1)|| < 1e-9 (the Kovasznay rule of the
paper). The history records per iteration the residual norms, the coefficient
change, chi, the error against a reference (if given) and the assembly and
solve times (timed separately; diagnostics are outside both). ``kappa`` is the
2-norm condition number of the system matrix at the last solve, by SVD.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.linalg as sla
from numpy.polynomial import chebyshev as C

LAM = 6.2
K_MAX = 20
TOL = 1e-9
PLATEAU = 0.05            # the plateau iterate: the first within 5% of the final error


def cgl(m):
    """The m Chebyshev-Gauss-Lobatto points on [0, 1], ascending."""
    return 0.5 * (1 - np.cos(np.pi * np.arange(m) / (m - 1)))


def cc_weights(m):
    """Clenshaw-Curtis weights on [0, 1] for ``cgl(m)``; they sum to 1."""
    n = m - 1
    theta = np.pi * np.arange(m) / n
    w = np.zeros(m)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n ** 2 - 1)
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
        v -= np.cos(n * theta[1:-1]) / (n ** 2 - 1)
    else:
        w[0] = w[n] = 1.0 / n ** 2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
    w[1:-1] = 2 * v / n
    return w / 2.0


def cheb_1d(x, p):
    """T_j(2x - 1), j < p, and its first and second x-derivatives at ``x``."""
    t = 2 * np.asarray(x, dtype=float) - 1
    V = C.chebvander(t, p - 1)
    eye = np.eye(p)
    V1 = np.stack([2 * C.chebval(t, C.chebder(eye[j], 1)) for j in range(p)], 1)
    V2 = np.stack([4 * C.chebval(t, C.chebder(eye[j], 2)) for j in range(p)], 1)
    return V, V1, V2


@dataclass(frozen=True)
class TrialSpace:
    """``kind`` 'tensor' (T_i T_j) or 'hard' (b T_i T_j, b = x(1-x)y(1-y)),
    i, j < ``p``; coefficients ordered x-index outer."""
    kind: str
    p: int

    @property
    def n(self):
        return self.p * self.p

    def _factors(self, z):
        V, V1, V2 = cheb_1d(z, self.p)
        if self.kind == 'tensor':
            return V, V2
        b, b1 = z * (1 - z), 1 - 2 * z
        F = b[:, None] * V
        F2 = -2.0 * V + 2 * b1[:, None] * V1 + b[:, None] * V2
        return F, F2

    def rows(self, x, y):
        """``(Phi, Lap)``: the basis and its Laplacian at the points (x, y)."""
        Fx, Fx2 = self._factors(np.asarray(x, float))
        Fy, Fy2 = self._factors(np.asarray(y, float))
        Phi = (Fx[:, :, None] * Fy[:, None, :]).reshape(len(x), -1)
        Lap = (Fx2[:, :, None] * Fy[:, None, :] + Fx[:, :, None] * Fy2[:, None, :]).reshape(len(x), -1)
        return Phi, Lap

    def values(self, beta, x, y):
        """The field at the points (x, y)."""
        return self.rows(np.ravel(x), np.ravel(y))[0] @ beta

    def grid_values(self, beta, axis):
        """The field on the tensor grid ``axis`` x ``axis`` (array [i, j])."""
        Fx, _ = self._factors(np.asarray(axis, float))
        return Fx @ np.asarray(beta).reshape(self.p, self.p) @ Fx.T


@dataclass
class Layout:
    """The collocation rows: interior points with row weights, boundary points
    with row weights (none for 'hard'), and whether the system is square."""
    xi: np.ndarray
    yi: np.ndarray
    wi: np.ndarray
    xb: np.ndarray
    yb: np.ndarray
    wb: np.ndarray
    square: bool

    @property
    def N(self):
        return len(self.xi) + len(self.xb)


def _tensor(g, w1=None):
    X, Y = np.meshgrid(g, g, indexing='ij')
    X, Y = X.ravel(), Y.ravel()
    on_bd = np.isclose(X, 0) | np.isclose(X, 1) | np.isclose(Y, 0) | np.isclose(Y, 1)
    W = None if w1 is None else np.outer(w1, w1).ravel()
    return X, Y, W, on_bd


def square_layout(p, nodes='cgl'):
    """The p x p grid (CGL or equispaced) with the boundary rows replaced by
    the Dirichlet condition: (p - 2)^2 interior rows and 4p - 4 boundary rows,
    unweighted."""
    g = cgl(p) if nodes == 'cgl' else np.linspace(0, 1, p)
    X, Y, _, bd = _tensor(g)
    one = np.ones
    return Layout(X[~bd], Y[~bd], one((~bd).sum()), X[bd], Y[bd], one(bd.sum()), True)


def hard_layout(p, ratio, m=None):
    """Interior rows of an m-point CGL tensor grid, (m - 2)^2 >= ratio p^2,
    weighted by the square roots of the tensor Clenshaw-Curtis weights."""
    if m is None:
        m = int(np.ceil(np.sqrt(ratio) * p - 1e-12)) + 2
    X, Y, W, bd = _tensor(cgl(m), cc_weights(m))
    return Layout(X[~bd], Y[~bd], np.sqrt(W[~bd]), np.zeros(0), np.zeros(0), np.zeros(0), False)


def weak_ms_layout(p, ratio, lam_bc):
    """An m-point CGL tensor grid with m^2 >= ratio p^2 (interior and boundary
    points), equal mean-square weights: interior 1/sqrt(n_int), boundary
    sqrt(lam_bc / n_bd)."""
    m = int(np.ceil(np.sqrt(ratio) * p - 1e-12))
    X, Y, _, bd = _tensor(cgl(m))
    nI, nB = int((~bd).sum()), int(bd.sum())
    return Layout(X[~bd], Y[~bd], np.full(nI, 1 / np.sqrt(nI)), X[bd], Y[bd], np.full(nB, np.sqrt(lam_bc / nB)), False)


def weak_cc_layout(p, ratio, lam_bc):
    """An m-point CGL tensor grid, m^2 >= ratio p^2: interior rows with the
    tensor Clenshaw-Curtis weights, each edge's rows with the 1D weights
    times lam_bc (the corners on two edges)."""
    m = int(np.ceil(np.sqrt(ratio) * p - 1e-12))
    g, w1 = cgl(m), cc_weights(m)
    X, Y, W, bd = _tensor(g, w1)
    xb = np.concatenate([g, g, 0 * g, 0 * g + 1])
    yb = np.concatenate([0 * g, 0 * g + 1, g, g])
    wb = np.sqrt(lam_bc * np.concatenate([w1] * 4))
    return Layout(X[~bd], Y[~bd], np.sqrt(W[~bd]), xb, yb, wb, False)


def method(label, p, lam_bc=10.0):
    """``(TrialSpace, Layout)`` of a method label (see ``METHODS``)."""
    if label == 'SQ-CGL':
        return TrialSpace('tensor', p), square_layout(p, 'cgl')
    if label == 'SQ-equi':
        return TrialSpace('tensor', p), square_layout(p, 'equi')
    kind, _, r = label.rpartition('-CGL-')
    ratio = float(r)
    if kind == 'LS-hard':
        return TrialSpace('hard', p), hard_layout(p, ratio)
    if kind == 'LS-weakMS':
        return TrialSpace('tensor', p), weak_ms_layout(p, ratio, lam_bc)
    if kind.startswith('LS-weakCC'):
        return TrialSpace('tensor', p), weak_cc_layout(p, ratio, float(kind[len('LS-weakCC'):] or lam_bc))
    raise ValueError(f"unknown method {label!r}")


METHODS = ('SQ-CGL', 'SQ-equi', 'LS-hard-CGL-1.5', 'LS-hard-CGL-3', 'LS-weakMS-CGL-3', 'LS-weakCC100-CGL-1.5')


def free_coefficients(label, p):
    return (p - 2) ** 2 if label.startswith('SQ') else p * p


class Rows:
    """The trial space evaluated at a layout's rows, once per solve."""

    def __init__(self, space: TrialSpace, layout: Layout):
        self.space, self.layout = space, layout
        self.PhiI, self.LapI = space.rows(layout.xi, layout.yi)
        self.PhiB = space.rows(layout.xb, layout.yb)[0] if len(layout.xb) else np.zeros((0, space.n))


def assemble(rows: Rows, beta, lam=LAM):
    """The linearized system at beta, weighted per row:
    A beta_new = f with A = [w_i (Lap + lam e^u Phi); w_b Phi_b],
    f = [w_i lam e^u (u - 1); 0], and the nonlinear residual R at beta."""
    L = rows.layout
    uI = rows.PhiI @ beta
    e = lam * np.exp(uI)
    A = np.vstack([L.wi[:, None] * (rows.LapI + e[:, None] * rows.PhiI), L.wb[:, None] * rows.PhiB])
    f = np.concatenate([L.wi * e * (uI - 1), np.zeros(len(L.wb))])
    R = np.concatenate([L.wi * (rows.LapI @ beta + e), L.wb * (rows.PhiB @ beta)])
    return A, f, R


def solve_linear(A, f, square):
    if square:
        return sla.lu_solve(sla.lu_factor(A, check_finite=False), f, check_finite=False)
    return sla.lstsq(A, f, lapack_driver='gelsy', cond=np.finfo(float).eps, check_finite=False)[0]


def newton(label, p, k_max=K_MAX, tol=TOL, reference=None, lam_bc=10.0, diagnostics=True):
    """Newton from zero for one method and size. ``reference``: an
    ``(axis, field)`` pair, the reference solution on the tensor grid
    axis x axis, for the per-iteration error. Returns a dict with the
    history, the stop, the plateau iterate, the final coefficients, kappa
    and the rank of the last system (SVD)."""
    space, layout = method(label, p, lam_bc)
    t0 = time.perf_counter()
    rows = Rows(space, layout)
    t_setup = time.perf_counter() - t0
    beta = np.zeros(space.n)
    hist, prev_Rlin, k_stop = [], None, None
    for k in range(k_max):
        t1 = time.perf_counter()
        A, f, R = assemble(rows, beta)
        t2 = time.perf_counter()
        new = solve_linear(A, f, layout.square)
        t3 = time.perf_counter()
        rec = {'k': k, 't_assemble_s': t2 - t1, 't_solve_s': t3 - t2}
        if diagnostics:
            Rlin = A @ new - f
            rec.update(norm_R=float(np.linalg.norm(R)), norm_Rlin=float(np.linalg.norm(Rlin)),
                       norm_dbeta=float(np.linalg.norm(new - beta)),
                       chi=float(np.linalg.norm(R - prev_Rlin) / max(np.linalg.norm(prev_Rlin), 1e-300))
                       if prev_Rlin is not None else None)
            if reference is not None:
                rec['err'] = error(space, beta, *reference)
            prev_Rlin = Rlin
        rel = np.linalg.norm(new - beta) / max(np.linalg.norm(new), 1e-300)
        rec['rel_dbeta'] = float(rel)
        hist.append(rec)
        beta = new
        if rel < tol:
            k_stop = k + 1
            break
    final = {'k': len(hist), 'rel_dbeta': None, 't_assemble_s': None, 't_solve_s': None}
    if diagnostics and reference is not None:
        final['err'] = error(space, beta, *reference)
    hist.append(final)
    out = {'method': label, 'p': p, 'free_coeff': free_coefficients(label, p), 'P_trial': space.n, 'N': layout.N,
           'k_stop': k_stop, 'stop': 'tolerance' if k_stop else 'k_max', 'beta': beta, 'space': space,
           'history': hist, 't_setup_s': t_setup}
    if diagnostics:
        A, _, _ = assemble(rows, beta)
        s = np.linalg.svd(A, compute_uv=False)
        out['kappa'] = float(s[0] / s[-1])
        out['rank'] = int((s > max(A.shape) * np.finfo(float).eps * s[0]).sum())
        if reference is not None:
            errs = [h['err'] for h in hist]
            out['err_final'] = errs[-1]
            out['k_plateau'] = next(i for i, e in enumerate(errs) if e <= (1 + PLATEAU) * errs[-1])
    return out


def error(space, beta, axis, ref):
    """Relative discrete L2 error on the tensor grid axis x axis."""
    u = space.grid_values(beta, axis)
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def reference(p=48, axis=None):
    """The SQ-CGL solution at degree p (Section 6.2's Bratu reference):
    ``(beta, space, field on axis x axis)``."""
    res = newton('SQ-CGL', p, diagnostics=False)
    axis = np.linspace(0, 1, 201) if axis is None else axis
    return res['beta'], res['space'], res['space'].grid_values(res['beta'], axis), res
