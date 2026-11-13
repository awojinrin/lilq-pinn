"""Package 4, item 2: the Poisson-Boltzmann problem around three inclusions."""

import numpy as np
import pytest

from experiments import p4_2_pb_inclusions as drv
from experiments.p4_common import identity_scaled, iterate_to_rule
from problems import poisson_boltzmann as pb

# Check M6: the advisor's values of u* and rho (instructions, Section 5)
M6_POINTS = [
    (0.273923, -0.460427, 5.042075450052e-01, 4.823286228989e+00),
    (-0.918053, -0.966945, -9.547121981571e-03, 7.868748818285e-01),
    (0.626540, 0.825511, 1.732547915903e-01, -5.316103703370e+00),
    (0.213272, 0.458993, 1.368447251037e+00, 2.871002443147e+01),
    (0.087250, 0.870145, 1.312517771519e-01, -4.858502796205e+00),
]


def test_manufactured_solution_at_the_advisors_points():
    # the table gives the points to 6 decimals, so compare at 1e-5 relative
    for x, y, u, r in M6_POINTS:
        assert pb.u_exact(x, y) == pytest.approx(u, rel=1e-5)
        assert pb.rho(x, y) == pytest.approx(r, rel=1e-5)


def test_closed_form_derivatives_by_finite_differences():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-1, 1, 20), rng.uniform(-1, 1, 20)
    h = 1e-4
    d = pb.exact(x, y)
    u = pb.u_exact
    np.testing.assert_allclose(d['x'], (u(x + h, y) - u(x - h, y)) / (2 * h), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(d['y'], (u(x, y + h) - u(x, y - h)) / (2 * h), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(d['xx'], (u(x + h, y) - 2 * u(x, y) + u(x - h, y)) / h ** 2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(d['yy'], (u(x, y + h) - 2 * u(x, y) + u(x, y - h)) / h ** 2, rtol=1e-4, atol=1e-4)
    fxy = (u(x + h, y + h) - u(x + h, y - h) - u(x - h, y + h) + u(x - h, y - h)) / (4 * h * h)
    np.testing.assert_allclose(d['xy'], fxy, rtol=1e-4, atol=1e-4)


def test_closed_form_equals_the_symbolic_derivation():
    sp = pytest.importorskip('sympy')
    X, Y = sp.symbols('x y')
    U = sum(a * sp.exp(-pb.WIDTH * ((X - cx) ** 2 + (Y - cy) ** 2)) for a, (cx, cy, _) in zip(pb.AMPL, pb.HOLES))
    rho_sym = sp.lambdify((X, Y), -(sp.diff(U, X, 2) + sp.diff(U, Y, 2)) + pb.KAP2 * sp.sinh(U), 'numpy')
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-1.3, 1.3, 50), rng.uniform(-1.3, 1.3, 50)
    np.testing.assert_allclose(pb.rho(x, y), rho_sym(x, y), rtol=1e-12, atol=1e-12)


def test_area_and_geometry():
    assert pb.AREA == pytest.approx(3.010274, abs=1e-6)
    assert pb.area_by_rule() == pytest.approx(3.009239, abs=1e-6)
    Xb, Yb, Wb = pb.boundary(200)
    assert Wb.sum() == pytest.approx(1.0) and len(Xb) == len(Wb)
    on_flower = np.isclose(np.hypot(Xb, Yb), pb.R(np.arctan2(Yb, Xb)))
    on_disk = np.zeros_like(on_flower)
    for cx, cy, r in pb.HOLES:
        on_disk |= np.isclose(np.hypot(Xb - cx, Yb - cy), r)
    assert np.all(on_flower | on_disk)


def test_halton_draws_are_prefixes():
    """Every set of the package is a prefix of one shipped draw per seed."""
    a, b = pb.halton(500, 2), pb.halton(1700, 2)
    np.testing.assert_array_equal(a[0], b[0][:500])
    np.testing.assert_array_equal(a[1], b[1][:500])
    assert np.all(pb.inside(*b))


def test_identity_of_remark_1():
    S = pb.lilq_system(6, 0)
    beta = np.random.default_rng(3).standard_normal(S.P) * 0.1
    A, f = S.linearized(beta)
    assert identity_scaled(A, f, beta, S.residual(beta)) < S.P * np.finfo(float).eps


def test_lilq_matches_the_reference_at_p10():
    """expected_pb.json, P = 100, seed 0: the rule fires at step 2, stall 2.688, errors 4.854e-2 and 3.705e-1."""
    S = pb.lilq_system(10, 0)
    beta, k_s, stall, _ = iterate_to_rule(S, np.zeros(S.P))
    e0, e2 = pb.Norms().errors(beta, 10)
    assert k_s == 2
    assert stall == pytest.approx(2.688, rel=1e-3)
    assert (e0, e2) == pytest.approx((4.854e-2, 3.705e-1), rel=1e-3)


def test_shipped_points_rebuild_the_same_set(tmp_path):
    drv.draw_points(tmp_path)
    h = drv.Halton(tmp_path / drv.ITEM / 'points')
    S1 = drv.lilq_set(h, 10, 1)
    S2 = pb.lilq_system(10, 1)
    np.testing.assert_array_equal(S1.points['interior'][0], S2.points['interior'][0])
    assert S1.N == S2.N


def test_newton_square_statuses():
    S = pb.square_system(10, 0, 'fekete')          # the smallest size of 2b
    b, it, status, _ = drv.newton(S, 'lu')
    assert status in ('converged', 'diverged', 'budget', 'singular') and it <= drv.NEWTON_K_MAX
