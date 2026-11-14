"""Package 5, item 4: the irregular domain at larger p, random features, delta_P two ways."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import p4_2_pb_inclusions as p42
from experiments import p5_4_pb as m
from problems import poisson_boltzmann as pb

REPO = Path(__file__).resolve().parents[1]


def test_the_package5_points_extend_package4s():
    """data/package5: 16,000 points per seed whose first 8,000 are the shipped Package 4 draws."""
    for seed in m.SEEDS:
        with np.load(REPO / 'data' / 'package5' / f'halton_seed{seed}.npz') as new, \
                np.load(REPO / 'data' / 'package4' / f'halton_seed{seed}.npz') as old:
            assert len(new['x']) == m.HALTON_N
            assert np.array_equal(new['x'][:len(old['x'])], old['x']) and np.array_equal(new['y'][:len(old['y'])], old['y'])
            assert pb.inside(new['x'], new['y']).all()


def test_random_features_are_the_advisors():
    """``rf_basis`` of ref_pb_rf.py: the draw order a, b, c and the closed-form derivatives."""
    f = pb.RandomFeatures(50, 4.0, 1)
    rng = np.random.default_rng(101)
    a, b, c = rng.uniform(-4, 4, 50), rng.uniform(-4, 4, 50), rng.uniform(-4, 4, 50)
    assert np.array_equal(f.a, a) and np.array_equal(f.b, b) and np.array_equal(f.c, c)
    X, Y = np.array([0.1, -0.3]), np.array([0.2, 0.5])
    B = f.basis(X, Y)
    h = 1e-4
    v = lambda x, y: f.basis(x, y, ['v'])['v']  # noqa: E731
    np.testing.assert_allclose(B['v'], np.tanh(np.outer(X, a) + np.outer(Y, b) + c), rtol=1e-15)
    np.testing.assert_allclose(B['x'], (v(X + h, Y) - v(X - h, Y)) / (2 * h), atol=1e-6)
    np.testing.assert_allclose(B['yy'], (v(X, Y + h) - 2 * v(X, Y) + v(X, Y - h)) / h ** 2, atol=1e-4)
    np.testing.assert_allclose(B['xy'], (v(X + h, Y + h) - v(X + h, Y - h) - v(X - h, Y + h) + v(X - h, Y - h))
                               / (4 * h * h), atol=1e-4)


def test_the_random_feature_set_is_the_advisors():
    """8P interior points (Halton seed 0) and 12 round(sqrt(P)) boundary points, weight 1,000."""
    halton = p42.Halton(REPO / 'data' / 'package5')
    S = pb.lilq_system(pb.RandomFeatures(200, 2.0, 0), 0, interior=halton.take(0, 1600))
    Xb, _, Wb = pb.boundary(12 * 14)
    assert S.P == 200 and S.n_interior == 1600 and S.n_boundary == len(Xb)
    np.testing.assert_allclose(S.wb ** 2, pb.LAM * Wb, rtol=1e-15)
    with pytest.raises(ValueError):
        pb.lilq_system(pb.RandomFeatures(200, 2.0, 0), 0, interior=halton.take(0, 100))


def test_the_chebyshev_system_is_unchanged():
    halton = p42.Halton(REPO / 'data' / 'package4')
    S = pb.lilq_system(10, 0, interior=halton.take(0, 800))
    assert S.P == 100 and S.n_boundary == len(pb.boundary(120)[0])
    np.testing.assert_array_equal(S.V, pb.basis(*halton.take(0, 800), 10, ['v'])['v'])


def test_delta_two_ways_agree_on_a_well_conditioned_space():
    n = pb.Norms(20, 120)
    a, b = n.best(6, 'gelsy'), n.best(6, 'gelsd')
    assert a == pytest.approx(b, rel=1e-10)
    assert n.best(6) == a                                   # the default is gelsy, as delivered


def test_rf_run_delta_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'RF_K_MAX', 6)
    monkeypatch.setattr(m, 'RF_P', (16,))
    monkeypatch.setattr(m, 'RF_SIGMA', (2.0,))
    monkeypatch.setattr(m, 'RF_DRAWS', (0,))
    monkeypatch.setattr(m, 'SIZES_P4', (4,))
    monkeypatch.setattr(m, 'SIZES_A', ())
    real = pb.Norms
    monkeypatch.setattr(pb, 'Norms', lambda: real(20, 120))
    halton = p42.Halton(REPO / 'data' / 'package5')
    m.run_rf(tmp_path, halton)
    d = m.rf_dir(tmp_path, 16, 2.0, 0)
    run = json.loads((d / 'run.json').read_text())
    assert run['package5']['item'] == '4b' and run['basis_description']['draw'] == 0
    assert np.load(d / 'betas.npy').shape == (7, 16) and (d / 'beta_returned.npy').exists()
    m.run_delta(tmp_path)
    assert {f.stem for f in (tmp_path / m.ITEM / 'delta_P').glob('*.json')} == {'cheb_p4', 'rf_P16_sigma2_draw0'}
    exp = tmp_path / 'exp.json'
    exp.write_text(json.dumps([{'P': 16, 'sigma': 2.0, 'seed': 0, 'rank': 16, 'rule': 3, 'solves': 4, 'stall': 1.0,
                                'eL2': 1.0, 'eH2': 1.0, 'dL2': 1.0, 'dH2': 1.0, 'kappa': 1.0}]))
    out = m.summarize(tmp_path, exp)
    assert len(out['4b']) == 1 and len(out['4b_expected']) == 1 and len(out['delta']) == 2
    row = out['delta'][0]
    assert {'delta_H2_gelsy', 'delta_H2_gelsd', 'rel_diff_H2'} <= set(row)



def test_rf_runs_a_chosen_part_of_the_set(tmp_path, monkeypatch):
    """--P and --sigma split 4b among processes; by default the whole set runs."""
    calls = []
    monkeypatch.setattr(pb, 'Norms', lambda: None)
    monkeypatch.setattr(m, 'rf_run', lambda out, h, P, sigma, draw, norms: calls.append((P, sigma, draw)))
    m.run_rf(tmp_path, None, sizes=(1600,), sigmas=(4.0,))
    assert calls == [(1600, 4.0, d) for d in m.RF_DRAWS]
    calls.clear()
    m.run_rf(tmp_path, None)
    assert len(calls) == len(m.RF_P) * len(m.RF_SIGMA) * len(m.RF_DRAWS)


def test_the_seeds_of_2c_side_by_side_keep_their_own_rows(tmp_path):
    """Item 4's job runs ``under`` for each seed in its own process: one file per seed, merged by the summary."""
    pts = str(REPO / 'data' / 'package5')
    for seed in (0, 1):
        m.main(['under', '--p', '6', '--seeds', str(seed), '--out', str(tmp_path), '--points', pts])
    under = tmp_path / m.ITEM / 'underdetermined'
    assert sorted(f.name for f in under.glob('*.csv')) == ['underdetermined_seed0.csv', 'underdetermined_seed1.csv']
    out = m.summarize(tmp_path)
    assert [(int(r['p']), int(r['seed'])) for r in out['2c']] == [(6, 0), (6, 1)]
    assert (under / 'underdetermined.csv').exists()
