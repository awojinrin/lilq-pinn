"""The advisor's reply to wave 1, item 2.2 (timing) and Section 1 (the
Beltrami report's flag)."""

import json

import numpy as np

import experiments.component_b as cb


def test_a_paper_pass_run_follows_an_untimed_warm_up_run(tmp_path):
    """Every timed (paper-pass) run is preceded by one complete untimed run
    of the same configuration on the same device, its cold time recorded;
    the kmax pass, which is not timed, has none."""
    runs = cb.build_runs(benchmarks=('bratu', 'kovasznay'), passes=('paper', 'kmax'), devices=('cpu',),
                         smoke=True)
    assert [cb.execute_run(r, tmp_path, verbose=False) for r in runs] == ['ok'] * len(runs)
    for r in runs:
        summary = json.loads((tmp_path / r.name / 'summary.json').read_text())
        if r.pass_ == 'paper':
            assert isinstance(summary['warmup_run_time_s'], float) and summary['warmup_run_time_s'] > 0
        else:
            assert summary['warmup_run_time_s'] is None


def test_warm_up_run_is_called_once_and_only_for_the_paper_pass():
    calls = []
    assert cb._warm_up_run('paper', lambda: calls.append(1) or 0.25) == 0.25
    assert cb._warm_up_run('kmax', lambda: calls.append(1) or 0.25) is None
    assert calls == [1]


def test_elasticity_repeats_time_the_phase_of_t_cum(tmp_path):
    """Wave 1's median of five (0.069 s at the smallest size) was 2-35x the
    logged solve, because it repeated solve_time_total. The repeats now time
    assembly + solve, the phase of t_cum_s, and agree with it within a
    factor of 2; the QR time is their median too."""
    (run,) = cb._elasticity_runs(smoke=False, sizes=(5,))
    assert cb.execute_run(run, tmp_path, verbose=False) == 'ok'
    s = json.loads((tmp_path / run.name / 'summary.json').read_text())
    assert len(s['timing_repeats_s']) == cb.SHORT_RUN_REPEATS
    assert 0.5 <= s['time_lil_s'] / s['t_cum_s'] <= 2.0
    assert s['time_lil_s'] == np.median([r['time_lil_s'] for r in s['timing_repeats_s']])
    assert s['solve_time_qr'] == np.median([r['solve_time_qr'] for r in s['timing_repeats_s']])
    assert s['solve_time_qr'] <= s['time_lil_s'] <= s['solve_time_total']
    assert s['warmup_run_time_s'] > 0


def test_beltrami_report_has_no_improvement_verdict():
    """0.7515% against the paper's rounded 0.752% is no difference; the
    report gives both numbers and no flag."""
    from pathlib import Path
    source = (Path(cb.__file__).parent / 'run_beltrami_pinned.py').read_text(encoding='utf-8')
    assert 'improved' not in source
    assert "'paper_t1_pressure_error_pct'" in source
