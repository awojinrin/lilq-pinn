"""
Square spectral collocation for Kovasznay flow, P_N - P_{N-2} (Package 2, item 1, Section 4.2)
=============================================================================================

The classical baseline on Kovasznay (the advisor's instructions of 4 October
2026, Section 4.2). Square collocation of the steady Navier-Stokes equations
with equal-order velocity and pressure spaces has spurious pressure modes, so
the pressure space is two orders lower:

* velocity components u, v: tensor Chebyshev T_i(xi(x)) T_j(eta(y)), i, j < p_d
  (xi, eta map the domain's sides to [-1, 1]);
* pressure: the same with i, j < p_d - 2.

Rows, on the p_d x p_d Chebyshev-Gauss-Lobatto (CGL) grid of the domain:
the x- and y-momentum and the continuity equation at the (p_d - 2)^2
interior points, and the Dirichlet velocity data (the exact solution) at
the 4 p_d - 4 boundary points. Unknowns and equations both number
3 p_d^2 - 4 p_d + 4. One continuity row -- the one at the interior point
nearest the corner (-0.5, -0.5) -- is replaced by a pressure pin: the exact
pressure either at the corner itself (``pin='corner'``) or at that interior
point (``pin='interior'``, the fallback of Section 4.2).

The equations are the code's convective form (``problems.kovasznay``):
u u_x + v u_y + p_x - nu (u_xx + u_yy) = 0, the same for v with p_y, and
u_x + v_y = 0, Re = 40. Newton (the quasilinearization) from zero, stopping
when ||beta^(k+1) - beta^(k)|| / ||beta^(k+1)|| < 1e-9, K_max = 60. Errors on
the paper's 301 x 401 test grid: relative L2 of u, v, p (pin gauge) and of the
mean-free p, as ``problems.kovasznay.make_test_error_fn``.
"""

import time

import numpy as np
import scipy.linalg as sla
from numpy.polynomial import chebyshev as C

RE = 40.0
NU = 1.0 / RE
X_DOMAIN, Y_DOMAIN = (-0.5, 1.0), (-0.5, 1.5)
K_MAX, TOL = 60, 1e-9
TEST_GRID = (301, 401)
CORNER = (X_DOMAIN[0], Y_DOMAIN[0])
LAM = RE / 2.0 - np.sqrt(RE ** 2 / 4.0 + 4.0 * np.pi ** 2)


def exact(x, y):
    e = np.exp(LAM * x)
    return (1 - e * np.cos(2 * np.pi * y), LAM / (2 * np.pi) * e * np.sin(2 * np.pi * y),
            0.5 * (1 - np.exp(2 * LAM * x)))


def cgl(m, a, b):
    return a + (b - a) * 0.5 * (1 - np.cos(np.pi * np.arange(m) / (m - 1)))


def cheb(z, n, a, b):
    """T_j(xi(z)), j < n, and its first and second z-derivatives."""
    t = 2 * (np.asarray(z, float) - a) / (b - a) - 1
    s = 2 / (b - a)
    eye = np.eye(n)
    V = C.chebvander(t, n - 1)
    V1 = np.stack([s * C.chebval(t, C.chebder(eye[j], 1)) for j in range(n)], 1) if n > 1 else 0 * V
    V2 = np.stack([s * s * C.chebval(t, C.chebder(eye[j], 2)) for j in range(n)], 1) if n > 2 else 0 * V
    return V, V1, V2


def tensor(x, y, n):
    """Values and derivatives (val, dx, dy, dxx, dyy) of the n x n tensor basis."""
    Vx, Vx1, Vx2 = cheb(x, n, *X_DOMAIN)
    Vy, Vy1, Vy2 = cheb(y, n, *Y_DOMAIN)
    k = lambda A, B: (A[:, :, None] * B[:, None, :]).reshape(len(x), -1)  # noqa: E731
    return {'val': k(Vx, Vy), 'dx': k(Vx1, Vy), 'dy': k(Vx, Vy1), 'dxx': k(Vx2, Vy), 'dyy': k(Vx, Vy2)}


class System:
    """The rows of the square system at degree p_d, evaluated once."""

    def __init__(self, p_d, pin='corner'):
        self.p_d, self.pin = p_d, pin
        self.nv, self.npr = p_d * p_d, (p_d - 2) ** 2
        self.n = 2 * self.nv + self.npr
        xg, yg = cgl(p_d, *X_DOMAIN), cgl(p_d, *Y_DOMAIN)
        X, Y = np.meshgrid(xg, yg, indexing='ij')
        X, Y = X.ravel(), Y.ravel()
        bd = np.isclose(X, X_DOMAIN[0]) | np.isclose(X, X_DOMAIN[1]) | np.isclose(Y, Y_DOMAIN[0]) | np.isclose(Y, Y_DOMAIN[1])
        self.xi, self.yi, self.xb, self.yb = X[~bd], Y[~bd], X[bd], Y[bd]
        self.Vi, self.Pi = tensor(self.xi, self.yi, p_d), tensor(self.xi, self.yi, p_d - 2)
        self.Vb = tensor(self.xb, self.yb, p_d)['val']
        ub, vb, _ = exact(self.xb, self.yb)
        self.ub, self.vb = ub, vb
        # the continuity row replaced by the pin: the interior point nearest the corner
        self.pin_row = int(np.argmin((self.xi - CORNER[0]) ** 2 + (self.yi - CORNER[1]) ** 2))
        px, py = (CORNER if pin == 'corner' else (self.xi[self.pin_row], self.yi[self.pin_row]))
        self.pin_point = (float(px), float(py))
        self.pin_phi = tensor(np.array([px]), np.array([py]), p_d - 2)['val'][0]
        self.pin_value = float(exact(np.array([px]), np.array([py]))[2][0])

    def split(self, beta):
        return beta[:self.nv], beta[self.nv:2 * self.nv], beta[2 * self.nv:]

    def assemble(self, beta):
        """Newton's system at beta: J beta_new = f."""
        a, b, c = self.split(beta)
        V, Pp = self.Vi, self.Pi
        u, v = V['val'] @ a, V['val'] @ b
        ux, uy, vx, vy = V['dx'] @ a, V['dy'] @ a, V['dx'] @ b, V['dy'] @ b
        conv = u[:, None] * V['dx'] + v[:, None] * V['dy']
        lap = V['dxx'] + V['dyy']
        Z = np.zeros((len(u), self.npr))
        # x-momentum: (u.grad) u_new + (u_new.grad) u + p_x - nu lap u_new = (u.grad) u
        xm = np.hstack([conv + ux[:, None] * V['val'] - NU * lap, uy[:, None] * V['val'], Pp['dx']])
        ym = np.hstack([vx[:, None] * V['val'], conv + vy[:, None] * V['val'] - NU * lap, Pp['dy']])
        ct = np.hstack([V['dx'], V['dy'], Z])
        fx, fy, fc = u * ux + v * uy, u * vx + v * vy, np.zeros(len(u))
        # pin: replace the continuity row nearest the corner
        ct[self.pin_row] = np.concatenate([np.zeros(2 * self.nv), self.pin_phi])
        fc[self.pin_row] = self.pin_value
        Zb = np.zeros((len(self.xb), self.nv))
        Zp = np.zeros((len(self.xb), self.npr))
        bu = np.hstack([self.Vb, Zb, Zp])
        bv = np.hstack([Zb, self.Vb, Zp])
        J = np.vstack([xm, ym, ct, bu, bv])
        f = np.concatenate([fx, fy, fc, self.ub, self.vb])
        return J, f


def test_errors(system, beta):
    """eps_u, eps_v, eps_p (pin gauge), eps_p_meanfree on the 301 x 401 grid."""
    xs, ys = np.linspace(*X_DOMAIN, TEST_GRID[0]), np.linspace(*Y_DOMAIN, TEST_GRID[1])
    a, b, c = system.split(beta)
    rel = lambda p, e: float(np.linalg.norm(p - e) / np.linalg.norm(e))  # noqa: E731
    Vx, Vy = cheb(xs, system.p_d, *X_DOMAIN)[0], cheb(ys, system.p_d, *Y_DOMAIN)[0]
    Px, Py = cheb(xs, system.p_d - 2, *X_DOMAIN)[0], cheb(ys, system.p_d - 2, *Y_DOMAIN)[0]
    u = Vx @ a.reshape(system.p_d, system.p_d) @ Vy.T
    v = Vx @ b.reshape(system.p_d, system.p_d) @ Vy.T
    p = Px @ c.reshape(system.p_d - 2, system.p_d - 2) @ Py.T
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    ue, ve, pe = exact(X, Y)
    return {'eps_u': rel(u, ue), 'eps_v': rel(v, ve), 'eps_p': rel(p, pe),
            'eps_p_meanfree': rel(p - p.mean(), pe - pe.mean())}


def newton(p_d, pin='corner', k_max=K_MAX, tol=TOL, diagnostics=True):
    """Newton from zero. Returns the history (per iteration: times, the
    coefficient change and, with diagnostics, the errors), the stop, the
    errors, and the rank and kappa of the Jacobian at the final iterate
    (SVD)."""
    system = System(p_d, pin)
    beta = np.zeros(system.n)
    hist, k_stop = [], None
    for k in range(k_max):
        t1 = time.perf_counter()
        J, f = system.assemble(beta)
        t2 = time.perf_counter()
        new = sla.lu_solve(sla.lu_factor(J, check_finite=False), f, check_finite=False)
        t3 = time.perf_counter()
        rel = float(np.linalg.norm(new - beta) / max(np.linalg.norm(new), 1e-300))
        rec = {'k': k, 't_assemble_s': t2 - t1, 't_solve_s': t3 - t2, 'rel_dbeta': rel}
        if diagnostics:
            rec.update(test_errors(system, beta))
        hist.append(rec)
        beta = new
        if rel < tol:
            k_stop = k + 1
            break
    out = {'p_d': p_d, 'pin': pin, 'pin_point': system.pin_point, 'n_unknowns': system.n,
           'n_equations': 3 * (p_d - 2) ** 2 + 2 * (4 * p_d - 4), 'P_velocity': 2 * system.nv,
           'P_pressure': system.npr, 'k_stop': k_stop, 'stop': 'tolerance' if k_stop else 'k_max',
           'beta': beta, 'history': hist, 'system': system}
    out.update(test_errors(system, beta))
    if diagnostics:
        J, _ = system.assemble(beta)
        s = np.linalg.svd(J, compute_uv=False)
        out['kappa'] = float(s[0] / s[-1])
        out['rank'] = int((s > J.shape[0] * np.finfo(float).eps * s[0]).sum())
        out['sigma_min'], out['sigma_max'] = float(s[-1]), float(s[0])
        # rank at zero, the first Jacobian (Stokes-like): a structural check independent of Newton
        J0, _ = system.assemble(np.zeros(system.n))
        s0 = np.linalg.svd(J0, compute_uv=False)
        out['rank_at_zero'] = int((s0 > J0.shape[0] * np.finfo(float).eps * s0[0]).sum())
        out['kappa_at_zero'] = float(s0[0] / s0[-1])
    return out
