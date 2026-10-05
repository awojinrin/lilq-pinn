"""Package 2, item 7 (P2-1), Kovasznay: the series against Section 10, one
small run end to end, the exponent fit, check C8', the resume, and the
solver's per-iteration timing kept with the diagnostics off."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import experiments.p2_1_scaling as m

REPO = Path(__file__).resolve().parents[1]


def test_series_are_section_10():
    assert m.SERIES['kovasznay_P']['sizes'] == (25, 30, 35, 40, 50)          # P = 1,875 .. 7,500
    assert [3 * p * p for p in m.SERIES['kovasznay_P']['sizes']] == [1875, 2700, 3675, 4800, 7500]
    assert m.SERIES['kovasznay_N']['sizes'] == (3, 5, 10, 20) and m.N_SERIES_PD == 20     # P = 1,200
    import experiments.component_c as cc
    for ratio in (3, 5, 10, 20):
        k, rows = cc.k_for_ratio('kovasznay', 20, ratio)
        assert abs(rows / 1200 - ratio) / ratio < 0.05
        config = m.kovasznay_config('kovasznay_N', ratio, 'cuda')
        assert config.k_ratio == k and config.sampling == 'uniform' and config.use_gpu
    from experiments.run_kovasznay import K_RATIO, MAX_ITER, TOL
    c = m.kovasznay_config('kovasznay_P', 25, 'cpu')
    assert (c.N_x, c.k_ratio, c.max_iter, c.tol, c.use_gpu) == (25, K_RATIO, MAX_ITER, TOL, False)


def test_solver_keeps_the_timing_parts_without_diagnostics():
    from problems.kovasznay import solve_kovasznay
    r = solve_kovasznay(m.kovasznay_config('kovasznay_P', 5, 'cpu'), verbose=False, diagnostics=False)
    h = r['history']
    n = len(h['iteration'])
    assert len(h['t_assemble']) == len(h['t_solve']) == n and all(t > 0 for t in h['t_solve'])
    assert h['gpu_mem_peak_bytes'] == [None] * n


def test_one_small_run(tmp_path):
    rec = m.run_one('kovasznay_P', 5, 'cpu', tmp_path)                        # P = 75
    d = tmp_path / 'P2_1_scaling' / 'kovasznay' / 'kovasznay_P_5_cpu'
    assert json.loads((d / 'run.json').read_text()) == rec
    assert rec['P'] == 75 and rec['rank'] <= 75 and rec['kappa'] >= 1 and rec['peak_host_bytes'] > 0
    assert rec['peak_gpu_bytes'] is None and 0 < rec['eps_u'] < 1 and rec['total_time_s'] > 0
    with open(d / 'iterations.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == rec['iterations'] and {'t_assemble_s', 't_solve_s'} <= set(rows[0])
    assert 'single timing' in rec['timing_protocol']


def test_exponent():
    x = np.array([1875, 2700, 3675, 4800, 7500])
    assert m.exponent(x, 2e-10 * x ** 3) == pytest.approx(3.0)
    assert m.exponent([1], [2]) is None


def _fake_package1(tmp_path, eps, time_s):
    B = tmp_path / 'p1' / 'B_instrumentation'
    for dev in ('cpu', 'cuda'):
        (B / f'kovasznay_P1875_{dev}_paper').mkdir(parents=True)
        (B / f'kovasznay_P1875_{dev}_paper' / 'summary.json').write_text(json.dumps(
            {'n_outer_iters': 6, **{f'test_{k}': v for k, v in eps.items()}}))
    (B / 'clean_timing').mkdir()
    (B / 'clean_timing' / 'clean_timing.csv').write_text(
        f'run,quantity,clean_time_s\nkovasznay_P1875_cpu_paper,solve_time_total,{time_s}\n')
    return tmp_path / 'p1'


def test_check_c8prime(tmp_path):
    eps = {'eps_u': 7.0e-13, 'eps_v': 5.3e-12, 'eps_p': 1.8e-12, 'eps_p_meanfree': 1.6e-12}
    p1 = _fake_package1(tmp_path, eps, 4.0)
    run = {'series': 'kovasznay_P', 'size': 25, 'P': 1875, 'iterations': 6, **eps}
    runs = [{**run, 'device': 'cpu', 'total_time_s': 4.4}, {**run, 'device': 'cuda', 'total_time_s': 1.0}]
    assert m.check_c8prime(runs, p1)['passed']
    runs[0]['total_time_s'] = 4.7                                             # 17.5% slower
    assert not m.check_c8prime(runs, p1)['passed']
    runs[0]['total_time_s'] = 4.0
    runs[1]['eps_u'] = 7.0e-13 * (1 + 1e-9)                                   # nine digits only
    c8 = m.check_c8prime(runs, p1)
    assert not c8['passed'] and c8['points'][0]['passed'] and not c8['points'][1]['passed']


def test_series_skips_finished_runs(tmp_path, monkeypatch):
    d = tmp_path / 'P2_1_scaling' / 'kovasznay' / 'kovasznay_P_25_cpu'
    d.mkdir(parents=True)
    (d / 'run.json').write_text('{}')
    calls = []
    import lilq.provenance
    monkeypatch.setattr(lilq.provenance, 'save_provenance', lambda *a, **k: None)    # it calls git itself
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError))
    m.run_series(['kovasznay_P'], 'cpu', tmp_path, sizes=[25])
    assert calls == []


def test_item_7a_jobs():
    for name, cls in (('p2s2_scaling_cpu_a', 'timed-cpu'), ('p2s2_scaling_gpu', 'timed')):
        text = (REPO / 'scripts' / 'cluster' / 'package2' / f'{name}.slurm').read_text()
        assert f'# lilq-resources: {cls} ' in text
        assert 'p2_1_scaling.py series --series kovasznay_P kovasznay_N' in text
