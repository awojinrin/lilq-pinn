"""
Nonlinear Poisson-Boltzmann equation around three inclusions
============================================================

    -Lap u + kappa^2 sinh(u) = rho   in Omega,      u = g   on dOmega,

with kappa^2 = 4.

**The domain.** Omega is the set r < 1 + 0.3 cos(5 theta) in polar
coordinates about the origin, minus three closed disks (``HOLES``: centre,
radius). Its boundary has four curves, and its area is
pi (1 + 0.3^2/2) - sum pi r_i^2 = 3.010274.

**The manufactured solution** is three Gaussians centred at the disks:

    u* = sum_i a_i exp(-w d_i^2),   (a_i) = (2, -1.5, 0.8),   w = 6,

with d_i the distance to the centre of disk i. rho and g follow from u*. The
derivatives are in closed form: with e_i = exp(-w d_i^2), X_i = x - cx_i and
Y_i = y - cy_i,
- u_x = sum -2 w a_i X_i e_i;
- u_xx = sum a_i (4 w^2 X_i^2 - 2 w) e_i;
- u_xy = sum 4 w^2 a_i X_i Y_i e_i;
- Lap u* = sum a_i (4 w^2 d_i^2 - 4 w) e_i.
(The reference script derives them symbolically; ``tests/test_poisson_boltzmann.py``
compares the two.)

**The trial space** is T_i(x/BOX) T_j(y/BOX), 0 <= i, j < p (P = p^2,
x-major), tensor Chebyshev on the bounding box [-BOX, BOX]^2 with
BOX = 1.32.

The point sets (``halton`` and ``boundary``) are those of the reference
implementation ``ref_pb_inclusions.py`` of the advisor's Package 4
(12 November 2026), unchanged, so that the point sets coincide. The
systems and the fine quadrature follow the same script.
"""

import numpy as np
from numpy.polynomial import legendre as Lg
from scipy.stats import qmc

from lilq.basis import Chebyshev1D

KAP2, BOX, LAM, RATIO = 4.0, 1.32, 1000.0, 8
HOLES = [(0.35, 0.25, 0.18), (-0.45, -0.20, 0.20), (0.0, -0.60, 0.12)]
AMPL = [2.0, -1.5, 0.8]
WIDTH = 6.0
KEYS = ['v', 'x', 'y', 'xx', 'xy', 'yy']
AREA = np.pi * (1 + 0.3 ** 2 / 2) - sum(np.pi * r * r for _, _, r in HOLES)


def R(th):
    return 1 + 0.3 * np.cos(5 * th)


def dR(th):
    return -1.5 * np.sin(5 * th)


# ---------------------------------------------------------------- the manufactured solution

def exact(x, y):
    """u* and its derivatives: a dict with keys v, x, y, xx, xy, yy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {k: np.zeros(np.broadcast(x, y).shape) for k in KEYS}
    w = WIDTH
    for a, (cx, cy, _) in zip(AMPL, HOLES):
        X, Y = x - cx, y - cy
        e = a * np.exp(-w * (X * X + Y * Y))
        out['v'] += e
        out['x'] += -2 * w * X * e
        out['y'] += -2 * w * Y * e
        out['xx'] += (4 * w * w * X * X - 2 * w) * e
        out['yy'] += (4 * w * w * Y * Y - 2 * w) * e
        out['xy'] += 4 * w * w * X * Y * e
    return out


def u_exact(x, y):
    return exact(x, y)['v']


def rho(x, y):
    d = exact(x, y)
    return -(d['xx'] + d['yy']) + KAP2 * np.sinh(d['v'])


# ---------------------------------------------------------------- geometry and point sets

def inside(X, Y):
    m = np.hypot(X, Y) < R(np.arctan2(Y, X))
    for cx, cy, r in HOLES:
        m &= np.hypot(X - cx, Y - cy) > r
    return m


def halton(n, seed):
    """First n points of the scrambled Halton sequence on the bounding box that fall in Omega."""
    h = qmc.Halton(d=2, scramble=True, seed=seed)
    out = np.empty((0, 2))
    while len(out) < n:
        q = (h.random(4 * n) - 0.5) * 2 * BOX
        out = np.vstack([out, q[inside(q[:, 0], q[:, 1])]])
    return out[:n, 0], out[:n, 1]


def boundary(nb):
    """About nb points on the four boundary curves, in proportion to their lengths; equal arc length on the
    flower, equal angle on the circles (midpoint offsets). Weights: arc length per point / total length."""
    th0 = np.linspace(0, 2 * np.pi, 4001)
    ds = np.sqrt(R(th0) ** 2 + dR(th0) ** 2)
    s = np.concatenate([[0], np.cumsum((ds[1:] + ds[:-1]) / 2 * np.diff(th0))])
    L0 = s[-1]
    lens = [L0] + [2 * np.pi * r for _, _, r in HOLES]
    tot = sum(lens)
    n0 = max(8, int(round(nb * L0 / tot)))
    th = np.interp((np.arange(n0) + 0.5) * L0 / n0, s, th0)
    X = list(R(th) * np.cos(th))
    Y = list(R(th) * np.sin(th))
    W = [L0 / n0] * n0
    for (cx, cy, r), L in zip(HOLES, lens[1:]):
        n = max(6, int(round(nb * L / tot)))
        a = (np.arange(n) + 0.5) * 2 * np.pi / n
        X += list(cx + r * np.cos(a))
        Y += list(cy + r * np.sin(a))
        W += [L / n] * n
    W = np.array(W)
    return np.array(X), np.array(Y), W / W.sum()


def fine_quadrature(nr=70, nth=420):
    """Polar Gauss rule of the flower with the nodes inside the disks removed; weights renormalized to sum 1."""
    g, w = Lg.leggauss(nr)
    rr = (g + 1) / 2
    th = (np.arange(nth) + 0.5) * 2 * np.pi / nth
    Rr = R(th)
    X = np.outer(rr, Rr * np.cos(th)).ravel()
    Y = np.outer(rr, Rr * np.sin(th)).ravel()
    W = np.outer(w / 2 * rr, Rr ** 2).ravel()
    m = inside(X, Y)
    return X[m], Y[m], W[m] / W[m].sum()


def area_by_rule(nr=70, nth=420):
    """The area of Omega by the fine rule before normalization (check M6)."""
    g, w = Lg.leggauss(nr)
    rr = (g + 1) / 2
    th = (np.arange(nth) + 0.5) * 2 * np.pi / nth
    Rr = R(th)
    X = np.outer(rr, Rr * np.cos(th)).ravel()
    Y = np.outer(rr, Rr * np.sin(th)).ravel()
    W = np.outer(w / 2 * rr, Rr ** 2 * 2 * np.pi / nth).ravel()
    return W[inside(X, Y)].sum()


# ---------------------------------------------------------------- the trial space

def cheb(z, p, d):
    """T_j(z/BOX), j < p, or its d-th derivative, at the points z."""
    c = Chebyshev1D(order=p - 1, domain=(-BOX, BOX))
    return c.evaluate(z) if d == 0 else c.derivative(z, d)


def basis(X, Y, p, keys=KEYS):
    vx = [cheb(X, p, d) for d in range(3)]
    vy = [cheb(Y, p, d) for d in range(3)]
    full = dict(v=(0, 0), x=(1, 0), y=(0, 1), xx=(2, 0), xy=(1, 1), yy=(0, 2))
    return {key: (vx[full[key][0]][:, :, None] * vy[full[key][1]][:, None, :]).reshape(len(X), -1) for key in keys}


# ---------------------------------------------------------------- errors and best approximation

class Norms:
    """Relative L2 and H2 errors on the fine quadrature, and the best-approximation errors of the trial space.
    H2: the function, both first derivatives and the three second derivatives, weighted equally."""

    def __init__(self, nr=70, nth=420):
        self.nr, self.nth = nr, nth
        self.X, self.Y, W = fine_quadrature(nr, nth)
        self.sw = np.sqrt(W)
        d = exact(self.X, self.Y)
        self.ref = [d[k] for k in KEYS]
        self.den0 = np.linalg.norm(self.ref[0] * self.sw)
        self.den2 = np.sqrt(sum(np.sum((r * self.sw) ** 2) for r in self.ref))
        self._p = None

    def _b(self, p):
        if self._p != p:
            self._p, self._bb = p, basis(self.X, self.Y, p)
        return self._bb

    def errors(self, beta, p):
        b = self._b(p)
        e0 = np.linalg.norm((b['v'] @ beta - self.ref[0]) * self.sw) / self.den0
        e2 = np.sqrt(sum(np.sum(((b[k] @ beta - r) * self.sw) ** 2) for k, r in zip(KEYS, self.ref))) / self.den2
        return float(e0), float(e2)

    def best(self, p):
        from scipy.linalg import lstsq
        b = self._b(p)
        eps = np.finfo(float).eps
        q = lstsq(b['v'] * self.sw[:, None], self.ref[0] * self.sw, lapack_driver='gelsy', cond=eps)[0]
        d0 = np.linalg.norm((b['v'] @ q - self.ref[0]) * self.sw) / self.den0
        M = np.vstack([b[k] * self.sw[:, None] for k in KEYS])
        rhs = np.concatenate([r * self.sw for r in self.ref])
        q2 = lstsq(M, rhs, lapack_driver='gelsy', cond=eps)[0]
        return float(d0), float(np.linalg.norm(M @ q2 - rhs) / self.den2)


# ---------------------------------------------------------------- collocation systems

class System:
    """Rows: interior points (PDE) with squared weights Wi, boundary points (Dirichlet) with Wb.
    ``linearized(b)`` is the quasilinearized system at b, ``residual(b)`` the weighted nonlinear residual.
    ``assemble``/``residual`` with ``exclude`` follow the Section 6 recipe's interface (blocks
    'interior' and 'boundary')."""

    def __init__(self, p, Xi, Yi, Wi, Xb, Yb, Wb):
        bi = basis(Xi, Yi, p, ['v', 'xx', 'yy'])
        self.V, self.Lap = bi['v'], bi['xx'] + bi['yy']
        self.Vb = basis(Xb, Yb, p, ['v'])['v']
        self.wi, self.wb = np.sqrt(Wi), np.sqrt(Wb)
        self.rho, self.g = rho(Xi, Yi), u_exact(Xb, Yb)
        self.p, self.P = p, p * p
        self.n_interior, self.n_boundary = len(Xi), len(Xb)
        self.N = len(Xi) + len(Xb)
        self.points = {'interior': (np.asarray(Xi), np.asarray(Yi), np.asarray(Wi)),
                       'boundary': (np.asarray(Xb), np.asarray(Yb), np.asarray(Wb))}

    def residual(self, b, exclude=()):
        assert not exclude
        return np.concatenate([(-self.Lap @ b + KAP2 * np.sinh(self.V @ b) - self.rho) * self.wi,
                               (self.Vb @ b - self.g) * self.wb])

    def linearized(self, b):
        u = self.V @ b
        c = KAP2 * np.cosh(u)
        A = np.vstack([(-self.Lap + c[:, None] * self.V) * self.wi[:, None], self.Vb * self.wb[:, None]])
        return A, np.concatenate([(self.rho - KAP2 * np.sinh(u) + c * u) * self.wi, self.g * self.wb])

    def assemble(self, b, exclude=()):
        assert not exclude
        return self.linearized(b)


def lilq_system(p, seed, ratio=RATIO, lam=LAM, nb=None, interior=None):
    """The LiL-Q set: ``ratio * P`` Halton points in Omega with equal squared weights summing to 1, and the
    boundary set of about 12p points with squared weights lam times arc length per point / total length.
    ``interior`` gives the Halton points instead of drawing them (shipped point sets)."""
    P = p * p
    n = int(round(ratio * P))
    Xi, Yi = interior if interior is not None else halton(n, seed)
    Xi, Yi = Xi[:n], Yi[:n]
    Xb, Yb, Wb = boundary(12 * p if nb is None else nb)
    return System(p, Xi, Yi, np.full(len(Xi), 1 / len(Xi)), Xb, Yb, lam * Wb)


def square_system(p, seed, kind, halton_points=None):
    """N = P rows, no weights. ``kind`` 'halton' (scattered) or 'fekete' (approximate Fekete points by a
    column-pivoted QR of the transposed Vandermonde matrix)."""
    from scipy.linalg import qr
    P = p * p
    hp = halton_points or (lambda n: halton(n, seed))
    Xb, Yb, _ = boundary(4 * p)
    nb = len(Xb)
    if kind == 'halton':
        Xi, Yi = hp(P - nb)
    else:
        Xbc, Ybc, _ = boundary(12 * p)
        _, _, pb = qr(basis(Xbc, Ybc, p, ['v'])['v'].T, pivoting=True, mode='economic')
        Xb, Yb = Xbc[pb[:nb]], Ybc[pb[:nb]]
        Xc, Yc = hp(8 * P)
        _, _, pi = qr(basis(Xc, Yc, p, ['v'])['v'].T, pivoting=True, mode='economic')
        Xi, Yi = Xc[pi[:P - nb]], Yc[pi[:P - nb]]
    return System(p, Xi, Yi, np.ones(len(Xi)), Xb, Yb, np.ones(nb))


def under_system(p, seed, halton_points=None):
    """N = P/2 rows: the points of boundary(2p) and Halton points in Omega, no weights."""
    P = p * p
    hp = halton_points or (lambda n: halton(n, seed))
    Xb, Yb, _ = boundary(2 * p)
    nb = len(Xb)
    Xi, Yi = hp(P // 2 - nb)
    return System(p, Xi, Yi, np.ones(len(Xi)), Xb, Yb, np.ones(nb))
