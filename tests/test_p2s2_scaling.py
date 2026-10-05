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
    assert 'single timing' in rec['timing_protocol'] and rec['status'] == 'ok'


def test_the_timing_is_saved_before_the_svd(tmp_path, monkeypatch):
    """A kill during the off-clock SVD keeps the timing (status 'timed'), and the
    summary still uses it for the times."""
    def killed(*a, **k):
        raise KeyboardInterrupt('walltime')
    monkeypatch.setattr(m.np.linalg, 'svd', killed)
    with pytest.raises(KeyboardInterrupt):
        m.run_one('kovasznay_P', 5, 'cpu', tmp_path)
    rec = json.loads((tmp_path / 'P2_1_scaling' / 'kovasznay' / 'kovasznay_P_5_cpu' / 'run.json').read_text())
    assert rec['status'] == 'timed' and rec['total_time_s'] > 0 and rec['P'] == 75 and 'kappa' not in rec
    assert [r['status'] for r in m._runs(tmp_path)] == ['timed']


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
    run = {'series': 'kovasznay_P', 'size': 25, 'P': 1875, 'iterations': 6, 'status': 'ok', **eps}
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


def test_item_7_jobs():
    import re
    text = {n: (REPO / 'scripts' / 'cluster' / 'package2' / f'{n}.slurm').read_text()
            for n in ('p2s2_scaling_cpu_a', 'p2s2_scaling_cpu_b', 'p2s2_scaling_gpu')}
    assert '# lilq-resources: timed-cpu ' in text['p2s2_scaling_cpu_a'] and '# lilq-resources: timed ' in text['p2s2_scaling_gpu']
    assert 'series --series kovasznay_P kovasznay_N --device cpu' in text['p2s2_scaling_cpu_a']
    assert 'series --series beltrami --sizes 6 7 --device cpu' in text['p2s2_scaling_cpu_a']
    assert 'series --series beltrami --sizes 8 --device cpu' in text['p2s2_scaling_cpu_b']
    assert 'series --series kovasznay_P kovasznay_N beltrami --device cuda' in text['p2s2_scaling_gpu']
    h, mnt = map(int, re.search(r'--time=(\d+):(\d+)', text['p2s2_scaling_cpu_b']).groups())
    assert (h + mnt / 60) * 48 <= 300          # Section 1's skip rule for the largest Beltrami size


# ---------------------------------------------------------------- Beltrami (batch 4)

def _tiny_beltrami(use_gpu=False):
    from problems.beltrami import BeltramiConfig
    return BeltramiConfig(N_vel=3, N_p=3, basis_type='chebyshev', n_pressure_pin_levels=3, max_iter=2,
                          N_x=4, N_y=4, N_z=4, N_t=4, N_bc=3, N_t_bc=3, N_ic=4, use_gpu=use_gpu)


def test_beltrami_series_is_section_10():
    from experiments.run_beltrami_pinned import pinned_config
    assert m.beltrami_config(6, 'cpu') == pinned_config()                   # the paper's run, exactly
    P = {n: 3 * n ** 4 + m.BELTRAMI[n][0] ** 4 for n in (6, 7, 8)}
    assert P == {6: 7984, 7: 13764, 8: 22288}
    for n, grid in ((6, 8), (7, 10), (8, 11)):
        c = m.beltrami_config(n, 'cuda')
        assert (c.N_x, c.N_t, c.N_ic, c.N_bc, c.N_t_bc) == (grid, grid, grid, grid - 2, grid - 2)
        assert c.n_pressure_pin_levels == c.N_p and c.use_gpu


def test_beltrami_rows_formula_and_final_system():
    from problems.beltrami import solve_beltrami
    config = _tiny_beltrami()
    r = solve_beltrami(config, verbose=False, diagnostics=False, return_final_system=True)
    assert r['A_final'].shape == (m.beltrami_rows(config), 3 * 3 ** 4 + 3 ** 4)
    h = r['history']
    assert len(h['t_assemble']) == len(h['t_solve']) == len(h['iteration']) and h['gpu_mem_peak_bytes'][0] is None


def test_beltrami_gpu_path_matches_the_cpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    from problems.beltrami import solve_beltrami
    c = solve_beltrami(_tiny_beltrami(), verbose=False, diagnostics=False)
    g = solve_beltrami(_tiny_beltrami(use_gpu=True), verbose=False, diagnostics=False)
    tc = np.concatenate([c[f'theta_{f}'] for f in 'uvwp'])
    tg = np.concatenate([g[f'theta_{f}'] for f in 'uvwp'])
    assert np.linalg.norm(tg - tc) / np.linalg.norm(tc) < 1e-8
    assert g['history']['gpu_mem_peak_bytes'][0] > 0 and set(g['history']['gpu_parts'][0]) == {'h2d_s', 'qr_solve_s', 'd2h_s'}


def test_beltrami_gpu_without_cuda_raises(monkeypatch):
    import problems.beltrami as pb
    monkeypatch.setattr(pb, 'HAS_TORCH_CUDA', False)
    with pytest.raises(RuntimeError, match='no CUDA device'):
        pb._lstsq(np.eye(3), np.ones(3), use_gpu=True)                     # no silent CPU fallback


def test_did_not_fit_is_recorded(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    monkeypatch.setattr(m, '_solve', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('CUDA out of memory.')))
    rec = m.run_one('beltrami', 8, 'cuda', tmp_path)
    assert rec['status'] == 'did_not_fit' and rec['P'] == 22288 and rec['N'] == m.beltrami_rows(m.beltrami_config(8, 'cuda'))
    assert rec['A_bytes'] == 8 * rec['N'] * rec['P']
    monkeypatch.setattr(m, '_solve', lambda *a, **k: (_ for _ in ()).throw(ValueError('something else')))
    with pytest.raises(ValueError):                                           # any other failure is not hidden
        m.run_one('beltrami', 8, 'cuda', tmp_path / 'b')


def test_check_c8prime_beltrami(tmp_path):
    eps = {'eps_u': 7.0e-13, 'eps_v': 5.3e-12, 'eps_p': 1.8e-12, 'eps_p_meanfree': 1.6e-12}
    p1 = _fake_package1(tmp_path, eps, 4.0)
    B = p1 / 'B_instrumentation'
    (B / 'beltrami_pinned').mkdir()
    snaps = [{'t': t, 'u': 3e-4, 'v': 3e-4, 'w': 3e-4, 'p': 7.5e-3, 'p_pin_gauge': 1.8e-2} for t in (0.0, 0.5, 1.0)]
    (B / 'beltrami_pinned' / 'report.json').write_text(json.dumps(
        {'n_outer_iters': 4, 'rel_l2_p_pin_gauge': 5.4e-3, 'snapshots': snaps}))
    with open(B / 'clean_timing' / 'clean_timing.csv', 'a') as f:
        f.write('beltrami_pinned,solve_time_total,267.0\n')
    run = {'series': 'beltrami', 'size': 6, 'P': 7984, 'device': 'cpu', 'iterations': 4, 'total_time_s': 280.0,
           'rel_l2_p_pin_gauge': 5.4e-3, **{f't{s["t"]:g}_{f}': s[f] for s in snaps for f in m.SNAPSHOT_FIELDS}}
    c8 = m.check_c8prime([run], p1)
    assert c8['passed'] and c8['points'][0]['point'] == 'beltrami_pinned' and len(c8['points'][0]['errors_rel_diff']) == 16
    run['t1_p'] *= 1 + 1e-8
    assert not m.check_c8prime([run], p1)['passed']
