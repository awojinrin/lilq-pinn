"""Package 2, item 8 (P2-10): the manufactured elasticity solution; the paper's
case is unchanged; the lateral-traction finding is recorded."""

import csv

import numpy as np
import pytest

from problems.elasticity import (CompatibleManufacturedElasticityPhysics, ElasticityConfig, ElasticityPhysics,
                                 ManufacturedElasticityPhysics, pi)


def test_the_paper_cases_traction_data_are_the_former_constants():
    ph = ElasticityPhysics(ElasticityConfig())
    x = np.linspace(0, 1, 17)
    assert np.array_equal(ph.traction_top_syy(x), ph.C11 * ph.Q * np.sin(pi * x))
    assert np.array_equal(ph.traction_lateral_sxx(np.zeros(17), x), np.zeros(17))


def _solutions(sp, X, Y):
    return {ManufacturedElasticityPhysics: (Y * (1 - Y) * sp.exp(X * Y) / 10, X * (1 - X) * Y * sp.exp(X + Y) / 20),
            CompatibleManufacturedElasticityPhysics: (
                sp.exp(sp.cos(sp.pi * X)) * sp.exp(sp.cos(2 * sp.pi * Y)) * sp.sin(sp.pi * Y) / 10,
                sp.sin(sp.pi * X) * sp.exp(sp.cos(2 * sp.pi * X)) * Y * sp.exp(Y) / 20)}


@pytest.mark.parametrize('cls', [ManufacturedElasticityPhysics, CompatibleManufacturedElasticityPhysics])
def test_manufactured_derivatives_against_sympy(cls):
    sp = pytest.importorskip('sympy')
    X, Y = sp.symbols('x y')
    u, v = _solutions(sp, X, Y)[cls]
    ph = cls(ElasticityConfig())
    lam, mu, C11 = ph.lam, ph.mu, ph.C11
    sym = {'exact_ux': u, 'exact_uy': v, 'exact_exx': sp.diff(u, X), 'exact_eyy': sp.diff(v, Y),
           'exact_exy': (sp.diff(u, Y) + sp.diff(v, X)) / 2,
           'body_force_x': C11 * sp.diff(u, X, 2) + mu * sp.diff(u, Y, 2) + (lam + mu) * sp.diff(v, X, Y),
           'body_force_y': mu * sp.diff(v, X, 2) + C11 * sp.diff(v, Y, 2) + (lam + mu) * sp.diff(u, X, Y)}
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)
    for name, expr in sym.items():
        want = sp.lambdify((X, Y), expr, 'numpy')(x, y)
        np.testing.assert_allclose(getattr(ph, name)(x, y), want, rtol=1e-13, atol=1e-15)


def test_manufactured_boundary_data():
    ph = ManufacturedElasticityPhysics(ElasticityConfig())
    t = np.linspace(0, 1, 21)
    zero, one = np.zeros_like(t), np.ones_like(t)
    for f, x, y in ((ph.exact_ux, t, zero), (ph.exact_uy, t, zero), (ph.exact_ux, t, one),
                    (ph.exact_uy, zero, t), (ph.exact_uy, one, t)):
        assert np.abs(f(x, y)).max() < 1e-15
    assert np.array_equal(ph.traction_top_syy(t), ph.exact_syy(t, one))
    assert np.abs(ph.traction_lateral_sxx(one, t)).max() > 0.05          # not zero: the change of data


def test_compatible_solution_meets_the_papers_boundary_data():
    """The reply's solution: every displacement condition, and sigma_xx = 0 on
    the lateral faces, so the lateral datum is the paper's zero."""
    ph = CompatibleManufacturedElasticityPhysics(ElasticityConfig())
    t = np.linspace(0, 1, 41)
    zero, one = np.zeros_like(t), np.ones_like(t)
    for f, x, y in ((ph.exact_ux, t, zero), (ph.exact_uy, t, zero), (ph.exact_ux, t, one),
                    (ph.exact_uy, zero, t), (ph.exact_uy, one, t)):
        assert np.abs(f(x, y)).max() < 1e-15
    for xe in (zero, one):
        assert np.abs(ph.exact_sxx(xe, t)).max() < 1e-13
        assert np.array_equal(ph.traction_lateral_sxx(xe, t), np.zeros_like(t))
    assert np.array_equal(ph.traction_top_syy(t), ph.exact_syy(t, one))
    assert np.abs(ph.exact_syy(t, one)).max() > 0.05                       # the top traction is not trivial


def test_item_8_rows(tmp_path):
    import experiments.p2_10_elasticity_manufactured as m
    rows = m.run(tmp_path, sizes=(5, 10, 15))
    with open(tmp_path / 'P2_10_elasticity_manufactured' / 'rows.csv') as f:
        assert [(r['solution'], int(r['P'])) for r in csv.DictReader(f)] == [
            ('compatible', 50), ('compatible', 200), ('compatible', 450),
            ('specified', 50), ('specified', 200), ('specified', 450)]
    compatible, rows = rows[:3], rows[3:]
    for r in compatible:                     # converges with delta_P, which falls geometrically
        assert r['rank_svd'] == r['P'] and r['lateral_sxx_required_max'] == 0
        assert 1 <= r['ratio_ux'] < 5 and 1 <= r['ratio_uy'] < 20
    assert compatible[2]['delta_ux'] < 1e-3 * compatible[1]['delta_ux'] < 1e-3 * compatible[0]['delta_ux']
    assert compatible[2]['eps_ux'] < 1e-6 and compatible[2]['eps_uy'] < 1e-5
    for r in rows:                           # the specified solution: the errors stall
        assert r['rank_svd'] == r['P'] and 0 < r['delta_ux'] < 0.02 and 0 < r['delta_uy'] < 0.02
        assert r['ratio_ux'] == pytest.approx(r['eps_ux'] / r['delta_ux'])
        assert r['lateral_sxx_trial_max'] < 1e-10 < 0.05 < r['lateral_sxx_required_max']
        assert r['eps_ux'] > 0.2
    assert rows[1]['delta_ux'] < rows[0]['delta_ux']
    for solution in ('compatible', 'specified'):
        assert (tmp_path / 'P2_10_elasticity_manufactured' / solution / 'P50' / 'run.json').exists()
