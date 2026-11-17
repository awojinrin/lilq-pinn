"""Package 5, item 3: the network runs (``experiments/p5_3_networks.py``).

The wrapper keeps the four-method run and changes only the stopping; the
history file; the budget; the warm-up; checks C1, C2 and C4."""

import csv
import json
import math

import numpy as np
import pytest
import torch

import experiments.p5_3_networks as n3


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def ref(tmp_path):
    """Stand-in references on small grids (the values only enter the error)."""
    x = np.linspace(0, 1, 11)
    np.savez(tmp_path / 'bratu_ref_p48.npz', x=x, y=x, u=np.outer(np.sin(np.pi * x), np.sin(np.pi * x)))
    xb, t = np.linspace(-1, 1, 11), np.linspace(0, 1, 11)
    np.savez(tmp_path / 'burgers_cole_hopf.npz', x=xb, t=t, u=-np.outer(np.sin(np.pi * xb), np.exp(-t)))
    return tmp_path


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize('method', ['nil_n', 'nil_q'])
def test_lbfgs_runs_log_the_error_every_10_iterations(ref, method):
    _, rec, rows, metrics = n3.solve('burgers', 25, method, 0, ref, budget_s=None, max_iterations=25)
    assert [r['iteration'] for r in rows] == [0, 10, 20, 25] and rec['iterations'] == 25
    assert rec['stopping_reason'] == 'iteration_cap' and all(math.isfinite(r['error']) for r in rows)
    on_record = dict(zip(metrics.data['iteration'], metrics.data['wall_time']))
    assert all(r['time_s'] == on_record[r['iteration']] for r in rows)
    assert rec['error_time_s'] > 0 and rec['training_time_s'] == metrics.data['wall_time'][-1]


def test_the_wrapper_changes_only_the_stopping(ref, monkeypatch):
    """The solver gets the four-method run's arguments, except: no target, tolerances 0,
    no caps (NiL-Q keeps its inner cap per linearization)."""
    import problems.burgers as mod
    from experiments.run_burgers import paper_setup
    seen = {}
    real_q, real_n = mod.solve_nil_q, mod.solve_nil_n

    def spy(real, key):
        def f(*a, **k):
            seen[key] = k
            return real(*a, **k)
        return f
    monkeypatch.setattr(mod, 'solve_nil_q', spy(real_q, 'q'))
    monkeypatch.setattr(mod, 'solve_nil_n', spy(real_n, 'n'))
    n3.solve('burgers', 25, 'nil_q', 0, ref, budget_s=None, max_iterations=3)
    n3.solve('burgers', 25, 'nil_n', 0, ref, budget_s=None, max_iterations=3)
    config, opt = paper_setup(5)
    q, n = seen['q'], seen['n']
    assert q['max_inner_iters'] == opt.max_inner_iters_nn and 'max_iterations' not in q
    assert q['R_tol'] == n['R_tol'] == 0.0 and q['stall_rule'] == n['stall_rule'] == 'f1'
    assert (q['lambda_pde'], q['lambda_bc']) == (n['lambda_pde'], n['lambda_bc']) == (opt.lambda_pde, opt.lambda_bc)
    assert q['max_quasi_iters'] == n['max_iterations'] == n3.NO_CAP           # the short runs' cap is the probe's
    assert n['max_line_searches'] == q['max_line_searches'] == n3.NO_CAP
    assert q['probe'].max_iterations == n['probe'].max_iterations == 3
    assert n3.NO_CAP >= 10 ** 9                       # a budget run: no cap can bind before 600 s


def test_a_budget_ends_the_run(ref, tmp_path):
    d = n3.run_dir_of(tmp_path, 'burgers', 25, 'nil_q', 0, 'cpu')
    rec = n3.run('burgers', 25, 'nil_q', 0, ref, tmp_path, budget_s=0.5)
    assert rec['stopping_reason'] == 'budget' and 0.5 <= rec['training_time_s'] < 1.5
    rows = _rows(d / 'history.csv')
    assert list(rows[0]) == list(n3.HISTORY_COLUMNS) and rows[0]['iteration'] == '0'
    assert int(rows[-1]['iteration']) == rec['iterations'] and float(rows[-1]['time_s']) == rec['training_time_s']
    assert (d / 'metrics.csv').exists() and (d / 'network.pt').exists()
    assert json.loads((d / 'run.json').read_text())['method'] == 'NiL-Q (L-BFGS)'


def test_lm_logs_the_error_every_iteration_and_stops_at_the_budget(ref, tmp_path):
    rec = n3.run('bratu', 25, 'lm', 0, ref, tmp_path, budget_s=0.3)
    d = n3.run_dir_of(tmp_path, 'bratu', 25, 'lm', 0, 'cpu')
    rows = _rows(d / 'history.csv')
    assert [int(r['iteration']) for r in rows] == list(range(rec['iterations'] + 1))
    assert rec['stopping_reason'] == 'budget' and rec['training_time_s'] >= 0.3
    assert rec['error_time_s'] > 0 and (d / 'log.csv').exists()


def test_c2_and_c4_runs(ref, tmp_path):
    res = n3.c2_c4('burgers', 25, 'nil_n', ref, tmp_path, iterations=20)
    assert res['c2_passed'] and res['iterations'] == 20 and res['error_time_s'] > 0 and res['pilot'] is None
    assert set(res) >= {'time_with_s', 'time_without_s', 'time_rel_diff', 'c4_passed'}


@pytest.mark.parametrize('method', ['nil_n', 'lm'])
def test_c2_and_c4_size_their_runs_from_a_pilot(ref, tmp_path, method):
    res = n3.c2_c4('bratu', 25, method, ref, tmp_path, seconds=0.5)
    pilot = res['pilot']
    assert pilot['iterations'] == n3.C2C4_PILOT[method] and pilot['target_s'] == 0.5
    rate = pilot['training_time_s'] / pilot['iterations']
    assert res['iterations'] == max(n3.C2C4_PILOT[method], math.ceil(0.5 / rate))
    assert res['c2_passed']


def test_c1_against_a_package1_history(ref, tmp_path):
    d =n3.run_dir_of(tmp_path / 'out', 'burgers', 25, 'nil_n', 0, 'cpu')
    _, rec, rows, metrics = n3.solve('burgers', 25, 'nil_n', 0, ref, budget_s=None, max_iterations=120,
                                     log_error=False)
    d.mkdir(parents=True)
    (d / 'run.json').write_text(json.dumps(rec))
    metrics.to_dataframe().to_csv(d / 'metrics.csv', index=False, float_format='%.17g')
    old = n3._package1_history(tmp_path / 'p1', 'burgers', 25, 'nil_n', 0, 'cpu')
    old.parent.mkdir(parents=True)

    def write(scale_at_50):
        with open(old, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['iteration', 'loss'])
            for i, v in zip(metrics.data['iteration'], metrics.data['loss']):
                w.writerow([i, repr(v * (scale_at_50 if i == 50 else 1.0))])
    write(1.0)
    res = n3.c1(d, tmp_path / 'p1')
    assert res['passed'] and res['max_rel_diff'] == 0.0 and res['iterations_compared'] == 101
    write(1.0 + 1e-5)
    assert not n3.c1(d, tmp_path / 'p1')['passed']
    assert n3.c1(d, tmp_path / 'nowhere') is None


def test_bl_uses_the_stored_reference_of_the_lil_runs(tmp_path):
    from experiments.run_bl import paper_setup
    x, t = np.linspace(0, 1, 5), np.linspace(0, 0.4, 4)
    u = np.outer(1 - x, 1 + t)
    np.savez(tmp_path / 'bl_fd_ref_viscous_nu0.1.npz', x=x, t=t, u=u)
    config, _ = paper_setup(8, False)
    (xa, ta), ua = n3.reference('bl', config, tmp_path)
    assert np.array_equal(xa, x) and np.array_equal(ta, t) and np.array_equal(ua, u)


def test_the_warm_up_runs_every_method(ref, tmp_path):
    w = n3.warm_up('burgers', 'cpu', ref, scratch=tmp_path)
    assert set(w) == set(n3.METHODS) and all(s > 0 for s in w.values())
    assert not list(tmp_path.glob('.warmup_*'))


# ---------------------------------------------------------------- Batch 5: the jobs' processes

def test_parts_split_the_runs_round_robin():
    runs = [(P, m, s) for P in (25, 100, 225) for m in n3.METHODS for s in n3.SEEDS]
    a, b = n3.part_of(runs, '0/2'), n3.part_of(runs, '1/2')
    assert sorted(a + b) == sorted(runs) and not set(a) & set(b) and abs(len(a) - len(b)) <= 1
    assert {P for P, _, _ in a} == {P for P, _, _ in b} == {25, 100, 225}      # every size in each part
    assert n3.part_of(runs, '0/1') == runs
    with pytest.raises(ValueError):
        n3.part_of(runs, '2/2')


def test_two_processes_of_a_job_write_their_own_records(ref, tmp_path, monkeypatch):
    """Two tags on one problem and device: separate warm-up and provenance records, each run's tag kept."""
    monkeypatch.setattr(n3, 'BUDGET_S', 0.2)
    monkeypatch.setattr(n3, 'save_provenance', lambda d: (d.mkdir(parents=True, exist_ok=True),
                                                          (d / 'hardware.json').write_text('{}')))
    for part, tag in (('0/2', 'gpu-burgers-0'), ('1/2', 'gpu-burgers-1')):
        n3.main(['run', '--bench', 'burgers', '--out', str(tmp_path), '--reference-dir', str(ref), '--P', '25',
                 '--methods', 'nil_n', '--seeds', '0', '1', '--budget', '0.2', '--part', part, '--tag', tag])
    root = tmp_path / n3.ITEM / 'networks'
    assert sorted(p.name for p in (root / 'warmup').iterdir()) == ['burgers_cpu_gpu-burgers-0.json',
                                                                   'burgers_cpu_gpu-burgers-1.json']
    assert sorted(p.name for p in (root / 'provenance').iterdir()) == ['burgers_cpu_gpu-burgers-0',
                                                                       'burgers_cpu_gpu-burgers-1']
    tags = {s: json.loads((n3.run_dir_of(tmp_path, 'burgers', 25, 'nil_n', s, 'cpu') / 'run.json').read_text())['tag']
            for s in (0, 1)}
    assert tags == {0: 'gpu-burgers-0', 1: 'gpu-burgers-1'}
    assert not list(root.glob('.warmup_*'))


def _history(d, times, tag):
    d.mkdir(parents=True, exist_ok=True)
    n3.write_history(d / 'history.csv', [{'iteration': 10 * i, 'evaluations': i, 'time_s': t, 'loss': 1.0,
                                          'error': 1.0} for i, t in enumerate(times)])
    (d / 'run.json').write_text(json.dumps({'benchmark': 'burgers', 'P': 625, 'method': 'NiL-N (L-BFGS)',
                                            'method_key': 'nil_n', 'seed': 0, 'device': 'cuda', 'tag': tag}))


def test_pairing_compares_the_clock_at_the_last_common_iteration(tmp_path):
    _history(n3.solo_dir_of(tmp_path, 'burgers', 625, 'nil_n', 0, 'cuda'), [0.0, 10.0, 20.0, 30.0], 'solo')
    _history(n3.run_dir_of(tmp_path, 'burgers', 625, 'nil_n', 0, 'cuda'), [0.0, 10.2, 20.5, 31.0, 41.0, 52.0], 'pair')
    res = n3.pairing(tmp_path)
    (row,) = res['rows']
    assert row['iteration'] == 30 and row['paired_over_solo_minus_1'] == pytest.approx(1 / 30)
    assert row['passed'] and res['passed'] and (tmp_path / n3.ITEM / 'networks' / 'checks' / 'pairing.json').exists()
    assert row['evaluations_solo'] == row['evaluations_paired'] == 3                  # the same path
    _history(n3.run_dir_of(tmp_path, 'burgers', 625, 'nil_n', 0, 'cuda'), [0.0, 11.0, 22.0, 33.0], 'pair')
    assert n3.pairing(tmp_path)['passed'] is False                                     # 10% slower when paired


def test_pairing_without_its_paired_run_has_no_verdict(tmp_path):
    _history(n3.solo_dir_of(tmp_path, 'bratu', 225, 'nil_n', 0, 'cuda'), [0.0, 1.0], 'solo')
    res = n3.pairing(tmp_path)
    assert res['rows'][0]['passed'] is None and res['passed'] is False
    (n3.run_dir_of(tmp_path, 'burgers', 625, 'nil_n', 0, 'cuda')).mkdir(parents=True)
    n3.write_history(n3.run_dir_of(tmp_path, 'burgers', 625, 'nil_n', 0, 'cuda') / 'history.csv', [])
    assert n3.pairing(tmp_path)['rows'][0]['passed'] is None                           # cut before its run.json


def test_pairing_survives_a_solo_run_cut_before_its_records(tmp_path):
    d = n3.solo_dir_of(tmp_path, 'bl', 1024, 'nil_n', 0, 'cuda')
    d.mkdir(parents=True)
    (d / 'log.csv').write_text('iteration\n')                                          # until_end stopped it
    res = n3.pairing(tmp_path)
    assert res['rows'] == [{'solo': d.name, 'passed': None,
                            'reason': 'the solo run did not finish (no run.json or history.csv)'}]
    assert res['passed'] is False




def test_the_solo_run_goes_to_its_own_folder(ref, tmp_path, monkeypatch):
    monkeypatch.setattr(n3, 'save_provenance', lambda d: d.mkdir(parents=True, exist_ok=True))
    n3.main(['solo', '--bench', 'burgers', '--P', '25', '--method', 'nil_n', '--out', str(tmp_path),
             '--reference-dir', str(ref), '--device', 'cpu', '--budget', '0.2', '--tag', 'solo'])
    d = n3.solo_dir_of(tmp_path, 'burgers', 25, 'nil_n', 0, 'cpu')
    assert json.loads((d / 'run.json').read_text())['tag'] == 'solo' and (d / 'history.csv').exists()
    assert not n3.run_dir_of(tmp_path, 'burgers', 25, 'nil_n', 0, 'cpu').exists()
