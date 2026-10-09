"""
Package 3 (the advisor's instructions of 8 October 2026): tensor CGL grids with
Clenshaw-Curtis weights, and the sampling constants of Section 5.6 on them
==========================================================================

Shared by the package's three items. The nodes and weights are B10's
(``lilq.collocation.points_1d(..., 'cgl')`` and ``clenshaw_curtis_weights``),
endpoints included, as P2-16 (``experiments/p2_16_certified.py``).

**Collocation sets and weights (Section 2.2).** A grid is a dict of blocks:
``{name: {'points': (n, d) array, 'w': (n,) CC weights, 'share': s,
'free': free coordinates, 'Ms': CGL counts per free direction}}``. The
squared row weight of an equation with weight lambda on a block is
``lambda * share * w``:
- interior: ``share = 1``, ``w`` the tensor CC weights (sum 1);
- each auxiliary block (boundary line or face, initial line or slab):
  ``share = |block| / |dOmega|``, with ``|dOmega|`` the measure of the whole
  boundary of the space or space-time domain, parts carrying no rows
  included.

These differ from the paper's runs, which put lambda on each edge or face.

**Constants (Section 2.3).**
- :func:`interior_constants_kronecker`: on a tensor CC-CGL grid the weighted
  Gram matrix of a tensor orthonormal Legendre basis of Q_q is a Kronecker
  product of 1D Gram matrices. So c1^2 and c2^2 are products of the 1D
  extreme eigenvalues (the advisor's ``gram_1d_extremes``). Q_q has q modes
  per direction; for a trial space with p modes per direction,
  q = m * p + 1 (P2-16's convention).
- :func:`interior_constants_direct`: the same from the SVD of the weighted
  evaluation matrix (check K1).
- :func:`block_constants`: one auxiliary block and one component. The space
  is the traces of the trial space (tensor Chebyshev in the block's free
  coordinates) plus the datum. It is orthonormalized under the block's mean
  square (tensor Gauss-Legendre) by a column-pivoted QR of the
  unit-normalized columns. Columns with |R_ii| < 1e-10 |R_11| are dropped.
  c1 and c2 are the extreme singular values of
  (sqrt(w) x evaluation at the nodes) R^-1, with w the block's weights
  normalized to sum 1 (the block-matched convention of Section 6). It is
  the advisor's ``block_constants`` extended to faces and slabs.

**Check K2** (:func:`cc_exactness`, C4 of P2-16 at any dimension):
- the integral of a tensor polynomial of degree <= M_d - 1 in each direction
  is exact to 1e-12, relative to the polynomial's L2 norm;
- the squared L2 norm of one of degree <= floor((M_d - 1)/2) is exact to
  1e-12.
"""

import math

import numpy as np
import scipy.linalg as sla
from numpy.polynomial import chebyshev as C
from numpy.polynomial import legendre as L

from lilq.collocation import clenshaw_curtis_weights, points_1d

COLUMN_DROP_TOL = 1e-10
CC_EXACTNESS_TOL = 1e-12


# ---------------------------------------------------------------- 1D and tensor rules

def cgl_rule(a, b, M):
    """M CGL nodes on [a, b] (ascending, endpoints included) and their CC weights (sum 1)."""
    return points_1d(a, b, M, 'cgl'), clenshaw_curtis_weights(M)


def tensor_points(axes):
    """The tensor grid of the 1D node sets ``axes``, first index outermost: (n, d)."""
    grids = np.meshgrid(*axes, indexing='ij')
    return np.stack([g.ravel() for g in grids], axis=1)


def tensor_weights(ws):
    """The tensor product of 1D weights, in the order of :func:`tensor_points`."""
    w = np.asarray(ws[0], float)
    for v in ws[1:]:
        w = np.multiply.outer(w, np.asarray(v, float))
    return w.ravel()


def tensor_rule(intervals, Ms):
    """The tensor CC-CGL rule: points (n, d) and weights (n,), summing to 1."""
    rules = [cgl_rule(a, b, M) for (a, b), M in zip(intervals, Ms)]
    return tensor_points([r[0] for r in rules]), tensor_weights([r[1] for r in rules])


def gauss_rule(intervals, n):
    """Tensor Gauss-Legendre with n points per direction, weights normalized to sum 1."""
    t, w = L.leggauss(n)
    axes = [a + (b - a) * (t + 1) / 2 for a, b in intervals]
    return tensor_points(axes), tensor_weights([w / 2] * len(intervals))


# ---------------------------------------------------------------- the grids of items 1 and 2

def _block(points, w, share, free, Ms):
    return {'points': points, 'w': np.asarray(w, float), 'share': float(share), 'free': tuple(free), 'Ms': tuple(Ms)}


def beltrami_grid(M, intervals=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)), rule='cc'):
    """Item 1's collocation set (Section 3.2): the M^4 tensor CGL grid on the
    space-time box; each of the six faces with the M^3 grid in its free
    coordinates (two space, one time); the initial slab t = 0 with the M^3
    grid in (x, y, z). |dOmega| counts the six faces and both slabs (40 on
    [-1, 1]^3 x [0, 1]); the final slab carries no rows.

    ``rule='gauss'``: the same blocks with M Gauss-Legendre points per
    direction (weights summing to 1), for the Y-norm of rho_r (Section 3.5)."""
    make = {'cc': tensor_rule, 'gauss': lambda iv, Ms: gauss_rule(iv, Ms[0])}[rule]
    length = [b - a for a, b in intervals]
    face = lambda d: math.prod(length[i] for i in range(4) if i != d)  # noqa: E731  (measure of a face normal to d)
    boundary = 2 * sum(face(d) for d in range(4))         # six faces and the two slabs
    pts, w = make(intervals, (M,) * 4)
    blocks = {'interior': _block(pts, w, 1.0, (0, 1, 2, 3), (M,) * 4)}
    for d, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        free = tuple(i for i in range(4) if i != d)
        fp, fw = make([intervals[i] for i in free], (M,) * 3)
        for side, value in (('-', intervals[d][0]), ('+', intervals[d][1])):
            p = np.empty((len(fp), 4))
            p[:, list(free)] = fp
            p[:, d] = value
            blocks[f'{name}{side}'] = _block(p, fw, face(d) / boundary, free, (M,) * 3)
    sp, sw = make(intervals[:3], (M,) * 3)
    p = np.column_stack([sp, np.full(len(sp), intervals[3][0])])
    blocks['initial'] = _block(p, sw, face(3) / boundary, (0, 1, 2), (M,) * 3)
    return {'blocks': blocks, 'M': M, 'rule': rule, 'boundary_measure': boundary,
            'N_points': sum(len(b['points']) for b in blocks.values())}


def bl_grid(P, r, T):
    """Item 2's collocation set (Section 4.1): the P2-16 Burgers rule on
    x in [0, 1], t in [0, T]:
    - M = ceil(sqrt(0.9 r P)) interior CGL points per direction;
    - ceil(0.05 r P) CGL points on the initial line;
    - ceil(0.025 r P) on each lateral line, the (n + 1)-point rule on [0, T]
      without t = 0, whose weights are kept as they are (sum 1 - w_0), as
      P2-16.

    |dOmega| = 2 (1 + T); the line t = T carries no rows."""
    M = math.ceil(math.sqrt(0.9 * r * P))
    n_i, n_b = math.ceil(0.05 * r * P), math.ceil(0.025 * r * P)
    boundary = 2 * (1.0 + T)
    pts, w = tensor_rule(((0.0, 1.0), (0.0, T)), (M, M))
    blocks = {'interior': _block(pts, w, 1.0, (0, 1), (M, M))}
    xi, wi = cgl_rule(0.0, 1.0, n_i)
    blocks['initial'] = _block(np.column_stack([xi, 0 * xi]), wi, 1.0 / boundary, (0,), (n_i,))
    tb, wb = cgl_rule(0.0, T, n_b + 1)
    for name, xe in (('left', 0.0), ('right', 1.0)):
        blocks[name] = _block(np.column_stack([0 * tb[1:] + xe, tb[1:]]), wb[1:], T / boundary, (1,), (n_b,))
    return {'blocks': blocks, 'M': M, 'n_initial': n_i, 'n_lateral': n_b, 'boundary_measure': boundary,
            'N_points': sum(len(b['points']) for b in blocks.values())}


# ---------------------------------------------------------------- constants

def legendre_orth(z, a, b, q):
    """The first q Legendre polynomials on [a, b], orthonormal for the normalized measure dz / (b - a)."""
    t = 2 * (np.asarray(z, float) - a) / (b - a) - 1
    return L.legvander(t, q - 1) * np.sqrt(2 * np.arange(q) + 1)


def gram_1d_extremes(M, q):
    """(lambda_min, lambda_max) of the CC-weighted Gram matrix of the first q
    orthonormal Legendre polynomials at the M CGL points."""
    x, w = cgl_rule(-1.0, 1.0, M)
    V = legendre_orth(x, -1.0, 1.0, q)
    e = np.linalg.eigvalsh((V * w[:, None]).T @ V)
    return float(e.min()), float(e.max())


def interior_constants_kronecker(M, q, dim):
    """c1, c2 on Q_q (q modes per direction) at the M^dim tensor CC-CGL grid.
    Not computable (rule C6) when dim Q_q >= the number of points, q >= M."""
    row = {'M': M, 'q': q, 'dim': dim, 'dim_ambient': q ** dim, 'N_interior': M ** dim}
    if q >= M:
        return {**row, 'computable': False, 'c1': None, 'c2': None, 'c2_over_c1': None}
    lo, hi = gram_1d_extremes(M, q)
    return {**row, 'computable': True, 'c1': math.sqrt(lo ** dim), 'c2': math.sqrt(hi ** dim),
            'c2_over_c1': (hi / lo) ** (dim / 2), 'c1_squared': lo ** dim, 'c2_squared': hi ** dim}


def interior_constants_direct(points, w, intervals, q):
    """c1, c2 by the SVD of diag(sqrt(w)) Phi, Phi the tensor orthonormal
    Legendre basis of Q_q at the points (w normalized to sum 1)."""
    w = np.asarray(w, float) / np.sum(w)
    n = len(points)
    Phi = np.ones((n, 1))
    for d, (a, b) in enumerate(intervals):
        B = legendre_orth(points[:, d], a, b, q)
        Phi = (Phi[:, :, None] * B[:, None, :]).reshape(n, -1)
    Phi *= np.sqrt(w)[:, None]
    s = sla.svd(Phi, compute_uv=False, overwrite_a=True, check_finite=False)
    return {'c1': float(s[-1]), 'c2': float(s[0]), 'c2_over_c1': float(s[0] / s[-1])}


def _chebyshev_tensor(points, intervals, p):
    """The tensor Chebyshev basis with p[d] modes in direction d, at points (n, d)."""
    V = np.ones((len(points), 1))
    for d, (a, b) in enumerate(intervals):
        t = 2 * (points[:, d] - a) / (b - a) - 1
        B = C.chebvander(t, p[d] - 1)
        V = (V[:, :, None] * B[:, None, :]).reshape(len(points), -1)
    return V


def block_constants(nodes, w, intervals, p, data=None, n_gl=None, tol=COLUMN_DROP_TOL):
    """(c1, c2) on one auxiliary block (Section 2.3(b)).

    ``nodes`` (n, d) are in the block's free coordinates, ranging over
    ``intervals``. ``p`` is the number of trial modes per free direction (an
    int or a tuple): the traces of a tensor Chebyshev trial space on the
    block. ``data``, a callable of the (n, d) points, adds the datum on the
    block. Gauss-Legendre uses ``n_gl`` points per direction; the default is
    max(4 n, 400) on lines, and max(4 M, 32) on faces and slabs, with M the
    block's CGL count per direction."""
    nodes = np.asarray(nodes, float)
    if nodes.ndim == 1:                                   # a line: (n,) -> (n, 1)
        nodes = nodes[:, None]
    dim = len(intervals)
    p = (p,) * dim if np.isscalar(p) else tuple(p)
    w = np.asarray(w, float) / np.sum(w)
    if n_gl is None:
        n_gl = max(4 * len(nodes), 400) if dim == 1 else max(4 * len(np.unique(nodes[:, 0])), 32)
    zq, wq = gauss_rule(intervals, n_gl)

    def basis(z):
        V = _chebyshev_tensor(z, intervals, p)
        return np.hstack([V, np.asarray(data(z), float)[:, None]]) if data is not None else V
    Vq = basis(zq) * np.sqrt(wq)[:, None]
    scale = np.linalg.norm(Vq, axis=0)
    Vq /= scale
    _, R, piv = sla.qr(Vq, mode='economic', pivoting=True)
    diag = np.abs(np.diag(R))
    k = int((diag > tol * diag[0]).sum())
    Vh = (basis(nodes) * np.sqrt(w)[:, None] / scale)[:, piv[:k]]
    X = sla.solve_triangular(R[:k, :k], Vh.T, trans='T').T     # Vh R^-1
    s = np.linalg.svd(X, compute_uv=False)
    return {'c1': float(s[-1]), 'c2': float(s[0]), 'c2_over_c1': float(s[0] / s[-1]),
            'columns': Vq.shape[1], 'dropped': Vq.shape[1] - k, 'n_gl': n_gl}


# ---------------------------------------------------------------- check K2

def cc_exactness(points, w, intervals, Ms, rank=4, rng=None, tol=CC_EXACTNESS_TOL):
    """Check K2 on one tensor CC rule (points (n, d), weights summing to 1, Ms
    the 1D counts). The test polynomials are sums of ``rank`` products of
    random orthonormal-Legendre series in each direction. So the check runs on
    the rule's own flattened points and weights at any dimension, and the exact
    values follow from the coefficients: the integral of a product is the
    product of the constant coefficients, and the inner product of two
    products is the product of the coefficient dot products."""
    rng = np.random.default_rng(0) if rng is None else rng
    w = np.asarray(w, float)
    dim = len(intervals)

    def poly(degs):
        coef = [rng.standard_normal((rank, deg + 1)) for deg in degs]
        vals = np.ones((rank, len(points)))
        for d, (a, b) in enumerate(intervals):
            vals *= coef[d] @ legendre_orth(points[:, d], a, b, degs[d] + 1).T
        gram = np.ones((rank, rank))
        for c in coef:
            gram *= c @ c.T
        return vals.sum(0), np.prod([c[:, 0] for c in coef], axis=0).sum(), gram.sum()
    f, integral, f_norm2 = poly([M - 1 for M in Ms])
    h, _, h_norm2 = poly([(M - 1) // 2 for M in Ms])
    err_int = abs(w @ f - integral) / math.sqrt(f_norm2)
    err_norm = abs(w @ h ** 2 - h_norm2) / h_norm2
    return {'Ms': list(Ms), 'dim': dim, 'weight_sum_minus_1': float(w.sum() - 1),
            'integral_degree_M_minus_1_err_over_L2_norm': float(err_int),
            'norm_degree_half_rel_err': float(err_norm),
            'passed': bool(err_int < tol and err_norm < tol)}
