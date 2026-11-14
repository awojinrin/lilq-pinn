"""Package 5, items 1, 2 and 4: the checks against the delivered runs (``experiments/p5_checks.py``)."""

import json

import pytest

from experiments import p5_1_weights as w
from experiments import p5_2_nlls as n
from experiments import p5_4_pb as m
from experiments import p5_checks as c
from experiments.p4_common import write_csv


def _log(path, rows, stall=None):
    path.mkdir(parents=True, exist_ok=True)
    write_csv(path / 'iterations.csv', rows)
    (path / 'run.json').write_text(json.dumps({'first_stall_iteration': stall}))


def _rows(R, Rlin, eps, chi, f=1.0):
    return [{'k': k, 'norm_R_h': a, 'norm_Rlin_h': b, 'eps_ref': e, 'chi': x, 'norm_f_h': f}
            for k, (a, b, e, x) in enumerate(zip(R, Rlin, eps, chi))]


BASE = dict(R=[1.0, 0.5, 0.1, 0.09, 0.09], Rlin=[1e-15, 0.05, 0.09, 0.09, 0.09],
            eps=[0.3, 0.1, 0.02, 0.019, 0.019], chi=[1e9, 0.7, 0.01, 1e-5, 1e-7])


def test_c1_identical_logs_pass_and_differences_after_the_stall_are_recorded_only(tmp_path):
    _log(tmp_path / 'there', _rows(BASE['R'], BASE['Rlin'], BASE['eps'], BASE['chi']), stall=3)
    late = dict(BASE, R=BASE['R'][:4] + [0.0901], Rlin=[3e-16] + BASE['Rlin'][1:])   # after the stall; a round-off step
    _log(tmp_path / 'here', _rows(late['R'], late['Rlin'], late['eps'], late['chi']))
    r = c.compare_logs(tmp_path / 'here', tmp_path / 'there')
    assert r['pass'] and r['iterates_compared'] == 4 and r['rule_step'] == r['rule_step_delivered']


def test_c1_a_difference_before_the_stall_fails(tmp_path):
    _log(tmp_path / 'there', _rows(BASE['R'], BASE['Rlin'], BASE['eps'], BASE['chi']), stall=3)
    eps = list(BASE['eps'])
    eps[2] *= 1 + 1e-6
    _log(tmp_path / 'here', _rows(BASE['R'], BASE['Rlin'], eps, BASE['chi']))
    r = c.compare_logs(tmp_path / 'here', tmp_path / 'there')
    assert not r['pass'] and r['max_rel_eps_ref'] == pytest.approx(1e-6, rel=1e-6)


def test_c1_finds_the_delivered_runs(tmp_path, monkeypatch):
    assert c.delivered_dir_c1('bratu', 100, 'P2', 'P3').as_posix() == 'P2/P2_16_certified/bratu_P100_NP10'
    assert c.delivered_dir_c1('bl_sine_gravity', 64, 'P2', 'P3').as_posix() == \
        'P3/P3_2b_bl_lifted_sine_cheb/gravity_P64_NP10_zero'
    assert c.delivered_dir_c1('bl_cheb_viscous', 64, 'P2', 'P3').as_posix() == \
        'P3/P3_2a_bl_chebyshev_certified/viscous_P64_NP10_ic'
    monkeypatch.setattr(w, 'families', lambda: ('bratu',))
    monkeypatch.setattr(w, 'sizes', lambda family: (25,))
    rows = _rows(BASE['R'], BASE['Rlin'], BASE['eps'], BASE['chi'])
    _log(w.run_dir_of(tmp_path / 'p5', 'bratu', 25, 10.0), rows)
    out = c.c1(tmp_path / 'p5', tmp_path / 'p2', tmp_path / 'p3')
    assert out['pass'] is None and out['rows'][0]['note'] == 'no delivered run'
    _log(c.delivered_dir_c1('bratu', 25, tmp_path / 'p2', tmp_path / 'p3'), rows, stall=3)
    assert c.c1(tmp_path / 'p5', tmp_path / 'p2', tmp_path / 'p3')['pass'] is True


def _package4(root):
    write_csv((root / 'P4_1a_burgers_gauss').mkdir(parents=True) or root / 'P4_1a_burgers_gauss' / 'terminal.csv',
              [{'p': 10, 'returned_iterate': 5, 'error': 4e-3, 'norm_N_returned': 0.0323}])
    write_csv((root / 'P4_2_pb_inclusions' / 'lilq').mkdir(parents=True) or
              root / 'P4_2_pb_inclusions' / 'lilq' / 'terminal.csv',
              [{'p': 10, 'seed': s, 'returned_iterate': 3, 'eL2': 0.0485, 'eH2': 0.370, 'norm_N_returned': 2.69,
                'delta_L2': 0.0216, 'delta_H2': 0.2555} for s in (0, 1)])


def _result(root, problem, p, **kw):
    d = n.run_dir_of(root, problem, p, 'lilq')
    d.mkdir(parents=True, exist_ok=True)
    (d / 'result.json').write_text(json.dumps(kw))


def test_c2_item2_lilq_against_package4(tmp_path):
    _package4(tmp_path / 'p4')
    _result(tmp_path / 'p5', 'burgers', 10, solves=5, eL2=4e-3 * (1 + 1e-4), norm_N=0.0323)
    _result(tmp_path / 'p5', 'pb', 10, solves=3, eL2=0.0485, eH2=0.370, norm_N=2.70)
    out = c.c2(tmp_path / 'p5', tmp_path / 'p4')
    assert out['pass'] is True and len(out['rows']) == 2
    _result(tmp_path / 'p5', 'pb', 10, solves=4, eL2=0.0485, eH2=0.370, norm_N=2.69)     # one more solve
    assert c.c2(tmp_path / 'p5', tmp_path / 'p4')['pass'] is False
    _result(tmp_path / 'p5', 'pb', 10, solves=3, eL2=0.0485, eH2=0.380, norm_N=2.69)     # 3% in H2
    assert c.c2(tmp_path / 'p5', tmp_path / 'p4')['pass'] is False


def test_c3_delta_of_package4_and_collect(tmp_path):
    _package4(tmp_path / 'p4')
    d = tmp_path / 'p5' / m.ITEM / 'delta_P'
    d.mkdir(parents=True)
    (d / 'cheb_p10.json').write_text(json.dumps({'delta_L2_gelsy': 0.0216 * (1 + 1e-3), 'delta_H2_gelsy': 0.2555}))
    assert c.c3(tmp_path / 'p5', tmp_path / 'p4')['pass'] is True
    (d / 'cheb_p10.json').write_text(json.dumps({'delta_L2_gelsy': 0.0216 * 1.01, 'delta_H2_gelsy': 0.2555}))
    assert c.c3(tmp_path / 'p5', tmp_path / 'p4')['pass'] is False
    summary = c.collect(tmp_path / 'p5', tmp_path / 'p2', tmp_path / 'p3', tmp_path / 'p4')
    assert summary == {'C1': None, 'C2': None, 'C3': False}
    assert json.loads((tmp_path / 'p5' / 'checks.json').read_text())['summary'] == summary
