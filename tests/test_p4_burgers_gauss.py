"""Package 4, item 1: Burgers on Gauss points with enrichment at stall."""

import numpy as np
import pytest

from experiments import p4_1_burgers_gauss as m
from experiments.p4_common import identity_scaled
from numpy.polynomial import chebyshev as C


def test_space_is_the_reference_scripts():
    """sin(n pi (x + 1)/2) times T_j(2t - 1), as ``xb``/``tb`` of the reference script."""
    p = 5
    space = m.Space(p)
    x = np.linspace(-1, 1, 7)
    t = np.linspace(0, 1, 6)
    n = np.arange(1, p + 1)
    a = n * np.pi / 2
    S = space.x_mats(x)
    np.testing.assert_allclose(S[0], np.sin(a * (x[:, None] + 1)), atol=1e-14)
    np.testing.assert_allclose(S[1], a * np.cos(a * (x[:, None] + 1)), atol=1e-13)
    np.testing.assert_allclose(S[2], -a * a * np.sin(a * (x[:, None] + 1)), atol=1e-12)
    T = space.t_mats(t)
    for j in range(p):
        c = np.zeros(j + 1)
        c[j] = 1
        np.testing.assert_allclose(T[0][:, j], C.chebval(2 * t - 1, c), atol=1e-14)
        d = C.chebder(c) if j else np.zeros(1)
        np.testing.assert_allclose(T[1][:, j], 2 * C.chebval(2 * t - 1, d), atol=1e-12)


def test_point_set_and_weights():
    p = 4
    rows = m.Rows(m.Space(p))
    assert rows.N == 9 * p * p + 3 * p
    np.testing.assert_allclose((rows.W ** 2).sum(), 1.0)          # interior: normalized, lambda = 1
    np.testing.assert_allclose((rows.w0 ** 2).sum(), m.LAM0)      # initial: lambda_0 times normalized
    assert np.all((rows.x > -1) & (rows.x < 1)) and np.all((rows.t > 0) & (rows.t < 1))


def test_identity_of_remark_1():
    """A beta - f is the weighted nonlinear residual at beta (check M3)."""
    space = m.Space(6)
    rows = m.Rows(space)
    beta = np.random.default_rng(0).standard_normal(space.P) * 0.1
    A, f = rows.assemble(beta)
    N = rows.residual(beta)
    assert identity_scaled(A, f, beta, N) < space.P * m.EPS_MACH
    np.testing.assert_allclose(A @ beta - f, N, atol=1e-13)


def test_embedding_keeps_the_function():
    space, big = m.Space(6), m.Space(10)
    beta = np.random.default_rng(1).standard_normal(space.P)
    x, t = np.linspace(-1, 1, 11), np.linspace(0, 1, 9)
    np.testing.assert_allclose(big.values(m.embed(beta, 6, 10), x, t), space.values(beta, x, t), atol=1e-12)


def test_single_solve_matches_the_reference_at_p6():
    """expected_burgers.json at p = 6: 5 solves, stall level 1.31e-1."""
    space = m.Space(6)
    _, k_s, stall, _ = m.iterate_to_rule(m.Rows(space), np.zeros(space.P), m.N_S)
    assert k_s + 1 == 5
    assert stall == pytest.approx(0.1306264121480671, rel=1e-10)


def test_enrichment_levels_and_variants():
    """Tolerance 0.05 stops after the p = 10 level; E2 takes fewer solves there."""
    e1 = m.enrichment(0.05, m.VARIANTS['E1'])
    e2 = m.enrichment(0.05, m.VARIANTS['E2'])
    assert [lev['p'] for lev in e1] == [6, 10] and [lev['p'] for lev in e2] == [6, 10]
    assert (e1[1]['solves'], e2[1]['solves']) == (4, 3)
    assert e1[1]['stall_level'] == pytest.approx(e2[1]['stall_level'], rel=1e-6)


def test_colexact_satisfies_the_initial_and_boundary_data():
    x, t = np.linspace(-1, 1, 21), np.array([0.0, 0.3, 1.0])
    u = m.colexact(x, t)
    np.testing.assert_allclose(u[:, 0], -np.sin(np.pi * x), atol=1e-15)
    assert np.abs(u[[0, -1], 1:]).max() < 1e-12
