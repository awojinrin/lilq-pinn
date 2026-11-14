"""Package 5, item 2: LiL-N by nonlinear least squares (``experiments/p5_2_nlls.py``)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import p4_1_burgers_gauss as p41
from experiments import p5_2_nlls as m

REPO = Path(__file__).resolve().parents[1]
POINTS = REPO / 'data' / 'package4'


def small_burgers(p=5):
    space = p41.Space(p)
    return m.Problem('burgers', p, p41.Rows(space), lambda b: {'eL2': float(np.linalg.norm(b))}, {'eL2': 1.0})


def small_pb(p=5):
    return m.poisson_boltzmann(p, POINTS)


@pytest.mark.parametrize('make', [small_burgers, small_pb])
def test_the_jacobian_is_the_derivative_of_the_residual(make):
    """The analytic Jacobian handed to least_squares: A(beta) = dN/dbeta (centred differences)."""
    prob = make()
    rng = np.random.default_rng(0)
    beta, d = 0.1 * rng.standard_normal(prob.P), rng.standard_normal(prob.P)
    h = 1e-6
    fd = (prob.residual(beta + h * d) - prob.residual(beta - h * d)) / (2 * h)
    np.testing.assert_allclose(prob.jacobian(beta) @ d, fd, rtol=1e-6, atol=1e-7 * np.abs(fd).max())


def test_the_lbfgs_loss_and_gradient():
    prob = small_burgers()
    beta = 0.1 * np.random.default_rng(1).standard_normal(prob.P)
    t = torch.from_numpy(beta.copy()).requires_grad_(True)
    loss, *_ = m._SquaredResidual(prob)(t)
    loss.backward()
    r = prob.residual(beta)
    assert float(loss.detach()) == pytest.approx(float(r @ r), rel=1e-14)
    np.testing.assert_allclose(t.grad.numpy(), 2 * prob.jacobian(beta).T @ r, rtol=1e-13)


def test_every_solver_reaches_lilq_on_a_small_burgers(tmp_path):
    """The point of item 2: on a small case all solvers end at LiL-Q's residual."""
    prob = small_burgers(6)
    rows = {name: m.run(prob, name, tmp_path, timing=False) for name in m.SOLVERS}
    ref = rows['lilq']['norm_N']
    for name in m.LS_METHODS + ('lbfgs_scipy',):
        assert rows[name]['norm_N'] == pytest.approx(ref, rel=1e-6), name
    assert rows['lbfgs']['norm_N'] == pytest.approx(ref, rel=1e-4)
    assert rows['lilq']['solves'] >= 3 and rows['trf']['nfev'] <= m.LS_MAX_NFEV
    d = m.run_dir_of(tmp_path, 'burgers', 6, 'trf')
    assert (d / 'beta.npy').exists() and json.loads((d / 'result.json').read_text())['solver'] == 'trf'
    betas = np.load(m.run_dir_of(tmp_path, 'burgers', 6, 'lilq') / 'betas.npy')
    assert len(betas) == rows['lilq']['solves'] + 1 and not betas[0].any()
    np.testing.assert_array_equal(betas[-1], np.load(m.run_dir_of(tmp_path, 'burgers', 6, 'lilq') / 'beta.npy'))


def test_timing_repeats_the_count(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'timed', lambda once, warmup: {'median_s': 1.0, 'spread': 0.0, 'solves': [once()[1]]})
    r = m.run(small_burgers(), 'lilq', tmp_path)
    assert r['timing_matches'] and r['time_s'] == 1.0


def test_the_lbfgs_solvers_are_timed_by_their_one_logged_pass(tmp_path, monkeypatch):
    """The advisor's reply to the cost check (14 November 2026), answer 2."""
    def no_protocol(once, warmup):
        raise AssertionError('the L-BFGS solvers take no timing protocol')
    monkeypatch.setattr(m, 'timed', no_protocol)
    monkeypatch.setattr(m, 'LBFGS_MAX_IT', 20)
    for name in m.ONE_PASS:
        r = m.run(small_burgers(), name, tmp_path)
        assert r['time_s'] > 0 and r['timing'].startswith('one logged pass') and r['timing_matches']


def test_summary_and_comparison(tmp_path):
    prob = small_burgers(6)
    for name in ('lilq', 'trf'):
        m.run(prob, name, tmp_path, timing=False)
    exp = tmp_path / 'expected.json'
    exp.write_text(json.dumps([{'case': 'burgers p6', 'P': 36, 'dP': 1.0,
                                'lilq': {'solves': 5, 'err': 1.0, 'N': 1.0},
                                'trf': {'nfev': 7, 'njev': 7, 'err': 1.0, 'N': 1.0}}]))
    rows, cmp_rows = m.summarize(tmp_path, exp)
    assert [r['solver'] for r in rows] == ['lilq', 'trf']
    assert {r['solver'] for r in cmp_rows} == {'lilq', 'trf'} and 'nfev_expected' in cmp_rows[1]
    assert (tmp_path / m.ITEM / 'terminal.csv').exists()


def test_timed_uses_a_given_warmup():
    from experiments.p4_common import timed
    calls = []

    def once():
        calls.append(1)
        return 2.0, 5
    out = timed(once, warmup=(3.0, 5))
    assert len(calls) == 3 and out['warmup_s'] == 3.0 and out['solves'] == [5]
    timed(once)
    assert len(calls) == 7
