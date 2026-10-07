"""
Reference solutions for the scalar benchmarks (Package 2, Section 6.2)
=====================================================================

**Burgers.** The viscous Burgers problem of the manuscript's Section 6.3,
u_t + u u_x = nu u_xx on (-1, 1) x (0, T], u(x, 0) = -sin(pi x),
u(+-1, t) = 0, has the Cole-Hopf solution

    u(x, t) = - int sin(pi (x - eta)) F(x - eta) G(eta) d eta / int F(x - eta) G(eta) d eta,
    F(y) = exp(-cos(pi y) / (2 pi nu)),   G(eta) = exp(-eta^2 / (4 nu t)),

(the integrals over the real line; the 2-periodic odd extension of the
initial datum keeps u = 0 at x = +-1). :func:`cole_hopf` evaluates the two
integrals by adaptive quadrature (``scipy.integrate.quad_vec``, relative
tolerance 1e-13), with the exponent shifted by its bound 1/(2 pi nu) so
that nothing overflows at small nu, over |eta| <= s sqrt(4c + 60), s =
sqrt(4 nu t), c = 1/(2 pi nu), beyond which the integrands are below 1e-26
of their peak. :func:`cole_hopf_ux` gives u_x the same way, for the check
against Basdevant et al. (1986): u_x(0, 1.6037/pi) = -152.00516 at
nu = 0.01/pi.

:func:`cole_hopf_series` is an independent evaluation of the same solution:
on (-1, 1) with u_x-free (Neumann) data for phi, Cole-Hopf gives
phi(x, t) = I_0(c) + 2 sum_n (-1)^n I_n(c) exp(-nu n^2 pi^2 t) cos(n pi x),
u = -2 nu phi_x / phi, with modified Bessel functions I_n (computed scaled,
``scipy.special.ive``). At nu = 0.1 the two agree to 3e-15 on the test grid.
The series is a check at moderate nu only: phi is of order exp(-2c) of its
terms near x = 0, so at nu = 0.01/pi (c = 50) the sum cancels catastrophically,
and the Basdevant check uses the quadrature.

**Bratu.** The reference is the square Chebyshev collocation solution at
p = 48 (``baselines.square_chebyshev.reference``), stored as an ``.npz``.

:func:`load_reference` reads either ``.npz`` (axes and field), and
:func:`rel_l2` is the error measure of Package 2: the relative discrete L2
error on the reference's test grid.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ive

BASDEVANT = {'nu': 0.01 / np.pi, 'x': 0.0, 't': 1.6037 / np.pi, 'u_x': -152.00516}


def _halfwidth(nu, t):
    s = np.sqrt(4 * nu * t)
    return s * np.sqrt(4 / (2 * np.pi * nu) + 60.0)


def _kernels(x, eta, nu, t):
    """exp(E - E_max) for the numerator and denominator integrands, and
    the y = x - eta values; E = -cos(pi y)/(2 pi nu) - eta^2/(4 nu t),
    E_max = 1/(2 pi nu)."""
    y = x - eta
    c = 1.0 / (2 * np.pi * nu)
    w = np.exp(-np.cos(np.pi * y) * c - c - eta ** 2 / (4 * nu * t))
    return y, w


def cole_hopf(x, t, nu, epsrel=1e-13):
    """u at the points ``x`` (array) and the time ``t`` (scalar)."""
    x = np.asarray(x, dtype=float)
    if t == 0:
        return -np.sin(np.pi * x)
    L = _halfwidth(nu, t)

    def integrand(eta):
        y, w = _kernels(x, eta, nu, t)
        return np.concatenate([np.sin(np.pi * y) * w, w])

    val, _ = quad_vec(integrand, -L, L, epsabs=0.0, epsrel=epsrel, norm='max', limit=20000)
    n = len(x)
    return -val[:n] / val[n:]


def cole_hopf_ux(x, t, nu, epsrel=1e-13):
    """u_x at the points ``x`` and the time ``t`` (t > 0)."""
    x = np.asarray(x, dtype=float)
    L = _halfwidth(nu, t)
    c = 1.0 / (2 * np.pi * nu)

    def integrand(eta):
        y, w = _kernels(x, eta, nu, t)
        g1 = np.sin(np.pi * y) * c * np.pi            # d/dy of -cos(pi y) c
        s, co = np.sin(np.pi * y), np.cos(np.pi * y)
        return np.concatenate([s * w, w, (np.pi * co + s * g1) * w, g1 * w])

    val, _ = quad_vec(integrand, -L, L, epsabs=0.0, epsrel=epsrel, norm='max', limit=20000)
    n = len(x)
    H, G, H1, G1 = val[:n], val[n:2 * n], val[2 * n:3 * n], val[3 * n:]
    return -(H1 * G - H * G1) / G ** 2


def cole_hopf_series(x, t, nu, n_terms=None):
    """u from the cosine series of the Cole-Hopf transform (independent of
    :func:`cole_hopf`)."""
    x = np.asarray(x, dtype=float)
    c = 1.0 / (2 * np.pi * nu)
    if n_terms is None:
        n_terms = int(c + 40 * np.sqrt(c) + 60)
    n = np.arange(1, n_terms + 1)
    a = 2 * (-1.0) ** n * ive(n, c) / ive(0, c) * np.exp(-nu * n ** 2 * np.pi ** 2 * t)
    phi = 1.0 + np.cos(np.pi * np.outer(x, n)) @ a
    phi_x = -np.pi * np.sin(np.pi * np.outer(x, n)) @ (n * a)
    return -2 * nu * phi_x / phi


def burgers_reference(x, t, nu, epsrel=1e-13):
    """u on the tensor grid ``x`` x ``t`` (array [i, j]) by quadrature."""
    return np.stack([cole_hopf(x, tj, nu, epsrel) for tj in t], axis=1)


def load_reference(path):
    """``(axes, field, meta)`` from a reference ``.npz``: Bratu (x, y, u) or
    Burgers (x, t, u)."""
    with np.load(path) as z:
        axes = [z['x'], z['y'] if 'y' in z.files else z['t']]
        meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
        return axes, z['u'].copy(), meta


def rel_l2(u, ref):
    """The relative discrete L2 error on the test grid."""
    return float(np.linalg.norm(np.asarray(u) - ref) / np.linalg.norm(ref))


def make_eps_ref_fn(basis, reference_npz):
    """``beta -> {'eps_ref': ...}``: the error of ``basis @ beta`` against the
    reference in ``reference_npz`` on its test grid (tensor-product
    evaluation), for an iteration log's ``test_error_fn``."""
    from lilq.test_errors import tensor_grid_values
    axes, ref, _ = load_reference(reference_npz)

    def eps_ref(beta):
        return {'eps_ref': rel_l2(tensor_grid_values(basis, beta, axes), ref)}

    return eps_ref


def save_npz_atomic(path, **arrays):
    """``np.savez_compressed`` to a temporary file beside ``path``, then a
    rename: a job reading ``path`` while another writes it (concurrent Stage 2
    jobs that each make a missing reference) sees the old file or the new one,
    never a partial one. The writers are deterministic, so both are the same."""
    import os
    path = Path(path)
    tmp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_text_atomic(path, text):
    import os
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def reference_path(reference_dir, benchmark):
    return Path(reference_dir) / {'bratu': 'bratu_ref_p48.npz', 'burgers': 'burgers_cole_hopf.npz'}[benchmark]
