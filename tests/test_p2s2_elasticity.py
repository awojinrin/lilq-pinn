"""Package 2, item 8 (P2-10): the manufactured elasticity solution; the paper's
case is unchanged; the lateral-traction finding is recorded."""

import csv

import numpy as np
import pytest

from problems.elasticity import ElasticityConfig, ElasticityPhysics, ManufacturedElasticityPhysics, pi


def test_the_paper_cases_traction_data_are_the_former_constants():
    ph = ElasticityPhysics(ElasticityConfig())
    x = np.linspace(0, 1, 17)
    assert np.array_equal(ph.traction_top_syy(x), ph.C11 * ph.Q * np.sin(pi * x))
    assert np.array_equal(ph.traction_lateral_sxx(np.zeros(17), x), np.zeros(17))


def test_manufactured_derivatives_against_sympy():
    sp = pytest.importorskip('sympy')
    X, Y = sp.symbols('x y')
    u = Y * (1 - Y) * sp.exp(X * Y) / 10
    v = X * (1 - X) * Y * sp.exp(X + Y) / 20
    ph = ManufacturedElasticityPhysics(ElasticityConfig())
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


def test_item_8_rows(tmp_path):
    import experiments.p2_10_elasticity_manufactured as m
    rows = m.run(tmp_path, sizes=(5, 10))
    with open(tmp_path / 'P2_10_elasticity_manufactured' / 'rows.csv') as f:
        assert [int(r['P']) for r in csv.DictReader(f)] == [50, 200]
    for r in rows:
        assert r['rank_svd'] == r['P'] and 0 < r['delta_ux'] < 0.02 and 0 < r['delta_uy'] < 0.02
        assert r['ratio_ux'] == pytest.approx(r['eps_ux'] / r['delta_ux'])
        assert r['lateral_sxx_trial_max'] < 1e-10 < 0.05 < r['lateral_sxx_required_max']
    assert rows[1]['delta_ux'] < rows[0]['delta_ux']
    assert (tmp_path / 'P2_10_elasticity_manufactured' / 'P50' / 'run.json').exists()
