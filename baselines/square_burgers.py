"""
Square space-time Chebyshev collocation for Burgers (Package 2, item 1, Section 4.3)
===================================================================================

The optional third classical baseline of the advisor's instructions of 4
October 2026 (Section 4.3; run because item 1 stays below 250 SU). The
viscous Burgers problem of Section 6.3:
- the equation u_t + u u_x - nu u_xx = 0 on (-1, 1) x (0, 1], with
  nu = 0.1;
- the initial condition u(x, 0) = -sin(pi x);
- the boundary condition u(+-1, t) = 0.

**Trial space:** tensor Chebyshev T_i(x) T_j(2t - 1), i, j < p
(p_x = p_t = p), coefficients x-index outer, the same space as the
certified-grid runs of item 3.

**Rows**, on the p x p Chebyshev-Gauss-Lobatto (CGL) grid, x in [-1, 1] and
t in [0, 1]:
- the boundary condition at every point with x = +-1, the corners
  included, where the initial datum -sin(+-pi) = 0 agrees;
- the initial condition at the p - 2 interior-x points of t = 0;
- the equation at the remaining (p - 2)(p - 1) points (interior x,
  t in (0, 1]).

So the system is square, with p^2 equations and p^2 unknowns. The constraint
rows fix 3p - 2 coefficients, which leaves the (p - 2)(p - 1) free
coefficients of Section 4.3. That is the counting of SQ-CGL on Bratu,
(p - 2)^2.

**Iteration.** As in Section 4.1 and ``baselines/square_chebyshev.py``:
- Newton (the quasilinearization: u_t + u_k u_x + u_{k,x} u - nu u_xx =
  u_k u_{k,x}) from zero, solved by LU;
- at most K_max = 20 solves, stopping when the relative coefficient change
  is below 1e-9;
- the history and plateau conventions of ``square_chebyshev``.

The error is the relative discrete L2 error on the 201 x 201 test grid
against the Cole-Hopf reference of Section 6.2. ``kappa`` and ``rank`` are
those of the last Jacobian, by SVD.
"""

import time

import numpy as np
from numpy.polynomial import chebyshev as C

from baselines.square_chebyshev import PLATEAU, TOL, solve_linear

NU = 0.1
K_MAX = 20
X_DOMAIN, T_DOMAIN = (-1.0, 1.0), (0.0, 1.0)


def cgl(m, a, b):
    """The m CGL points on [a, b], ascending."""
    return a + (b - a) * 0.5 * (1 - np.cos(np.pi * np.arange(m) / (m - 1)))


def cheb(z, n, a, b):
    """T_j(xi(z)), j < n, and its first and second z-derivatives (xi maps [a, b] to [-1, 1])."""
    t = 2 * (np.asarray(z, float) - a) / (b - a) - 1
    s = 2 / (b - a)
    eye = np.eye(n)
    V = C.chebvander(t, n - 1)
    V1 = np.stack([s * C.chebval(t, C.chebder(eye[j], 1)) for j in range(n)], 1)
    V2 = np.stack([s * s * C.chebval(t, C.chebder(eye[j], 2)) for j in range(n)], 1)
    return V, V1, V2


def free_coefficients(p):
    return (p - 2) * (p - 1)


class System:
    """The square system's rows at size p: the basis and its derivatives at
    the equation points, and the constraint rows with their data."""

    def __init__(self, p):
        self.p = p
        xg, tg = cgl(p, *X_DOMAIN), cgl(p, *T_DOMAIN)
        X, T = np.meshgrid(xg, tg, indexing='ij')
        X, T = X.ravel(), T.ravel()
        boundary = np.isclose(np.abs(X), 1.0)
        initial = np.isclose(T, 0.0) & ~boundary
        pde = ~boundary & ~initial
        self.n_pde, self.n_initial, self.n_boundary = int(pde.sum()), int(initial.sum()), int(boundary.sum())
        Vx, Vx1, Vx2 = cheb(X, p, *X_DOMAIN)
        Vt, Vt1, _ = cheb(T, p, *T_DOMAIN)
        k = lambda A, B: (A[:, :, None] * B[:, None, :]).reshape(len(X), -1)  # noqa: E731
        Phi, Phi_x, Phi_t, Phi_xx = k(Vx, Vt), k(Vx1, Vt), k(Vx, Vt1), k(Vx2, Vt)
        self.Phi, self.Phi_x, self.Phi_t, self.Phi_xx = Phi[pde], Phi_x[pde], Phi_t[pde], Phi_xx[pde]
        self.Phi_c = np.vstack([Phi[initial], Phi[boundary]])
        self.data_c = np.concatenate([-np.sin(np.pi * X[initial]), np.zeros(self.n_boundary)])
        self.points = {'pde': (X[pde], T[pde]), 'initial': (X[initial], T[initial]),
                       'boundary': (X[boundary], T[boundary])}

    @property
    def n(self):
        return self.p * self.p

    def assemble(self, beta):
        """``(A, f, R)``: the Newton system at beta (A beta_new = f) and the
        nonlinear residual at beta."""
        u, ux = self.Phi @ beta, self.Phi_x @ beta
        J = self.Phi_t + u[:, None] * self.Phi_x + ux[:, None] * self.Phi - NU * self.Phi_xx
        A = np.vstack([J, self.Phi_c])
        f = np.concatenate([u * ux, self.data_c])
        R = np.concatenate([self.Phi_t @ beta + u * ux - NU * (self.Phi_xx @ beta), self.Phi_c @ beta - self.data_c])
        return A, f, R

    def grid_values(self, beta, x, t):
        """The field on the tensor grid x by t (array [i, j])."""
        return cheb(x, self.p, *X_DOMAIN)[0] @ np.asarray(beta).reshape(self.p, self.p) @ cheb(t, self.p, *T_DOMAIN)[0].T


def error(system, beta, reference):
    (x, t), u_ref = reference
    return float(np.linalg.norm(system.grid_values(beta, x, t) - u_ref) / np.linalg.norm(u_ref))


def newton(p, k_max=K_MAX, tol=TOL, reference=None, diagnostics=True):
    """Newton from zero at size p, in the shape of ``square_chebyshev.newton``.
    ``reference``: ``((x, t), u)`` on the test grid, for the per-iteration error."""
    t0 = time.perf_counter()
    system = System(p)
    t_setup = time.perf_counter() - t0
    beta = np.zeros(system.n)
    hist, prev_Rlin, k_stop = [], None, None
    for k in range(k_max):
        t1 = time.perf_counter()
        A, f, R = system.assemble(beta)
        t2 = time.perf_counter()
        new = solve_linear(A, f, True)
        t3 = time.perf_counter()
        rec = {'k': k, 't_assemble_s': t2 - t1, 't_solve_s': t3 - t2}
        if diagnostics:
            Rlin = A @ new - f
            rec.update(norm_R=float(np.linalg.norm(R)), norm_Rlin=float(np.linalg.norm(Rlin)),
                       norm_dbeta=float(np.linalg.norm(new - beta)),
                       chi=float(np.linalg.norm(R - prev_Rlin) / max(np.linalg.norm(prev_Rlin), 1e-300))
                       if prev_Rlin is not None else None)
            if reference is not None:
                rec['err'] = error(system, beta, reference)
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
        final['err'] = error(system, beta, reference)
    hist.append(final)
    out = {'method': 'SQ-CGL', 'p': p, 'free_coeff': free_coefficients(p), 'P_trial': system.n, 'N': system.n,
           'k_stop': k_stop, 'stop': 'tolerance' if k_stop else 'k_max', 'beta': beta, 'system': system,
           'history': hist, 't_setup_s': t_setup}
    if diagnostics:
        A, _, _ = system.assemble(beta)
        s = np.linalg.svd(A, compute_uv=False)
        out['kappa'] = float(s[0] / s[-1])
        out['rank'] = int((s > max(A.shape) * np.finfo(float).eps * s[0]).sum())
        if reference is not None:
            errs = [h['err'] for h in hist]
            out['err_final'] = errs[-1]
            out['k_plateau'] = next(i for i, e in enumerate(errs) if e <= (1 + PLATEAU) * errs[-1])
    return out
