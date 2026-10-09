"""Package 3, Addendum 1 (item 4): Burgers LiL-Q at larger sizes (``experiments/p3_4_burgers_large_P.py``)."""

import csv
import json

import numpy as np
import pytest

import experiments.p3_4_burgers_large_P as m


def test_setups():
    from experiments.run_burgers import paper_setup
    _, logged = m.setup(30)
    assert (logged.R_tol, logged.max_quasi_iters_lil) == (0.0, 60)          # no target at the new sizes
    _, timed = m.setup(32, logged=False, max_iter=7)
    assert (timed.R_tol, timed.max_quasi_iters_lil) == (0.0, 7)
    _, control = m.setup(25, logged=False, max_iter=4)                       # the paper pass: its own target
    assert control.R_tol == paper_setup(25)[1].R_tol and control.max_quasi_iters_lil == 4
    config, _ = m.setup(35)
    assert (config.N_x, config.N_t, config.basis_type, config.k_ratio) == (35, 35, 'sin_fourier', 10)


def _fake_reference(tmp_path):
    x, t = np.linspace(-1, 1, 41), np.linspace(0, 1, 31)
    X, T = np.meshgrid(x, t, indexing='ij')
    np.savez(tmp_path / 'burgers_cole_hopf.npz', x=x, t=t, u=-np.sin(np.pi * X) * np.exp(-T))
    return tmp_path


def test_a_logged_pass_and_its_clean_runs(tmp_path):
    """A small logged pass (P = 25): the log, the loss at every iterate (0..60),
    the B2 identity; and clean runs to an iterate reach its logged loss exactly (L4)."""
    ref = _fake_reference(tmp_path)
    d = m.logged_pass(5, tmp_path / 'out', ref)
    log, losses = m.read_log(d)
    assert sorted(losses) == list(range(61)) and len(log) == 61
    assert json.loads((d / 'run.json').read_text())['b2_check']['max_rel_err_over_run'] < 1e-10
    t = m.clean_timing(5, 6, repeats=2)
    assert t['iterations'] == [6, 6] and all(float(v) == losses[6] for v in t['final_losses'])
    assert len(t['times_s']) == 2 and t['median_s'] > 0


def _log(tmp_path, eps, chi, rlin):
    d = tmp_path / 'run'
    d.mkdir()
    with open(d / 'iterations.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['k', 'eps_ref', 'chi', 'norm_Rlin_h', 'rel_dbeta'])
        w.writeheader()
        for k, e in enumerate(eps):
            last = k == len(eps) - 1
            w.writerow({'k': k, 'eps_ref': e, 'chi': '' if last else chi[k], 'norm_Rlin_h': '' if last else rlin[k],
                        'rel_dbeta': '' if last else 0.1})
    (d / 'losses.csv').write_text('iterate,loss\n0,1.0\n')
    return d


def test_stopping_iterates(tmp_path):
    eps = [1.0, 0.1, 2e-5, 1.5e-5, 1.4e-5, 1.4e-5]
    chi = [1.0, 0.5, 0.01, 0.01, 0.01]
    rlin = [1.0, 0.5, 0.2, 0.2, 0.2]
    log, _ = m.read_log(_log(tmp_path, eps, chi, rlin))
    k_e, k_r, i = m.stopping_iterates(log)
    assert k_e == 3                                                          # the first iterate at or under 1.6e-5
    assert k_r == i + 1 and k_r is not None


def test_l1_compares_k_up_to_4(tmp_path):
    def write(d, rows):
        d.mkdir(parents=True)
        with open(d / 'iterations.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['k', 'norm_R_h', 'norm_Rlin_h', 'eps_ref'])
            w.writeheader()
            w.writerows(rows)
    rows = [{'k': k, 'norm_R_h': 1.0 / (k + 1), 'norm_Rlin_h': 0.5 / (k + 1), 'eps_ref': 0.1 / (k + 1)} for k in range(6)]
    p2 = [dict(r) for r in rows[:5]]
    p2[4]['norm_Rlin_h'] = ''                                                # P2-12's terminal row
    write(tmp_path / 'here', rows)
    write(tmp_path / 'p2_12', p2)
    assert m.check_l1(tmp_path / 'here', tmp_path / 'p2_12')['passed']
    p2[2]['eps_ref'] = 0.1 / 3 * (1 + 1e-15)
    write(tmp_path / 'p2_12b', p2)
    assert not m.check_l1(tmp_path / 'here', tmp_path / 'p2_12b')['passed']


def test_40_runs_only_if_no_size_reaches_the_target(tmp_path, monkeypatch):
    done = []
    monkeypatch.setattr(m, 'logged_pass', lambda N, out, ref: done.append(N) or tmp_path / str(N))
    monkeypatch.setattr(m, 'read_log', lambda d: ([], {}))
    monkeypatch.setattr(m, 'stopping_iterates', lambda log: (None, 5, 4))
    m.run_all(tmp_path, tmp_path, timing=False)
    assert done == [25, 30, 32, 35, 40]
    done.clear()
    monkeypatch.setattr(m, 'stopping_iterates', lambda log: (7, 5, 4))
    m.run_all(tmp_path, tmp_path, timing=False)
    assert done == [25, 30, 32, 35]
