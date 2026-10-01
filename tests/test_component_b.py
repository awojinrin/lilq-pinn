"""Tests for experiments/component_b.py -- the Section 3.3 driver."""

import csv
import json

import pytest
import torch

import experiments.component_b as cb

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_full_plan_matches_section_3_3():
    runs = cb.build_runs(devices=('cpu', 'cuda'))
    names = [r.name for r in runs]
    assert len(names) == len(set(names)) == 63
    # Scalar benchmarks: paper sizes x {paper, kmax}, CPU.
    assert sum(r.benchmark == 'bratu' for r in runs) == 3 * 2
    assert sum(r.benchmark == 'burgers' for r in runs) == 5 * 2
    assert sum(r.benchmark == 'bl' for r in runs) == 4 * 2
    assert sum(r.benchmark == 'bl_gravity' for r in runs) == 4 * 2
    # Kovasznay: five sizes x {cpu, cuda} x {paper, kmax}.
    assert sum(r.benchmark == 'kovasznay' for r in runs) == 5 * 2 * 2
    # Beltrami: paper and a K_max = 8 pass (Addendum v2.2 2.7).
    assert [r.name for r in runs if r.benchmark == 'beltrami'] == ['beltrami_P7984_cpu_paper',
                                                                   'beltrami_P7984_cpu_kmax']
    # Single-pass benchmarks: linear, one solve (Addendum v2.2 2.7.4).
    assert sum(r.benchmark == 'elasticity' for r in runs) == 5
    assert [r.config_label for r in runs if r.benchmark == 'darcy'] == ['S1', 'S2', 'S3', 'SPE10']
    assert {r.pass_ for r in runs if r.benchmark in ('elasticity', 'darcy')} == {'paper'}


def test_kmax_only_pass_skips_single_pass_benchmarks():
    runs = cb.build_runs(passes=('kmax',), devices=('cpu',))
    assert {r.benchmark for r in runs} == {'bratu', 'burgers', 'bl', 'bl_gravity', 'kovasznay', 'beltrami'}


def test_smoke_runs_end_to_end_resume_and_index(tmp_path):
    root = tmp_path / 'B_instrumentation'
    runs = cb.build_runs(benchmarks=('bratu', 'elasticity'), devices=('cpu',), smoke=True)
    assert [cb.execute_run(r, root, verbose=False) for r in runs] == ['ok'] * 3
    for r in runs:
        run_dir = root / r.name
        for f in ('run.json', 'iterations.csv', 'summary.json', 'hardware.json', 'environment.txt'):
            assert (run_dir / f).exists(), (r.name, f)
        summary = json.loads((run_dir / 'summary.json').read_text())
        assert summary['run'] == r.name and summary['iterations'] >= 1

    # kmax runs the whole (smoke-capped) budget.
    kmax = json.loads((root / 'bratu_P25_cpu_kmax' / 'summary.json').read_text())
    assert kmax['iterations'] == kmax['K_max'] and kmax['R_tol'] == 0.0

    # Resume: completed runs are skipped.
    assert [cb.execute_run(r, root, verbose=False) for r in runs] == ['done'] * 3

    with open(cb.write_index(root), newline='') as f:
        rows = list(csv.DictReader(f))
    assert {r['run'] for r in rows} == {r.name for r in runs}
    assert all(r['status'] == 'ok' for r in rows)


def test_failure_writes_traceback_and_is_retried(tmp_path):
    root = tmp_path / 'B_instrumentation'
    calls = []

    def boom(run_dir):
        calls.append(1)
        raise RuntimeError("solver exploded")

    run = cb.Run('bratu', 'P25', 'cpu', 'paper', boom)
    assert cb.execute_run(run, root, verbose=False) == 'failed'
    assert 'RuntimeError: solver exploded' in (root / run.name / 'error.txt').read_text()
    assert not (root / run.name / 'summary.json').exists()
    assert cb.execute_run(run, root, verbose=False) == 'failed'  # not marked done: retried
    assert len(calls) == 2
    with open(cb.write_index(root), newline='') as f:
        assert list(csv.DictReader(f))[0]['status'] == 'failed'



@requires_cuda
def test_gpu_equivalence_smoke(tmp_path):
    path = cb.run_gpu_equivalence(tmp_path, smoke=True, repeats=1, verbose=False)
    with open(path, newline='') as f:
        (row,) = list(csv.DictReader(f))
    assert set(row) == set(cb.EQUIVALENCE_COLUMNS)
    assert row['P'] == '75' and row['equivalent'] == row['equivalent_amended'] == 'True'
    assert float(row['beta_rel_diff']) <= 1e-8
    assert all(float(row[c]) > 0 for c in ('t_gelsy_cpu_s', 't_gels_cpu_s', 't_qr_gpu_s'))
    assert int(row['gpu_mem_peak_bytes']) > 0 and int(row['gpu_mem_estimate_bytes']) > 0


@requires_cuda
def test_kovasznay_gpu_run_json_records_section_3_2_diagnostics(tmp_path):
    from lilq.iteration_log import IterationLogger
    from problems.kovasznay import KovasznayConfig, solve_kovasznay, gpu_memory_estimate_bytes
    logger = IterationLogger()
    solve_kovasznay(KovasznayConfig(N_x=5, N_y=5, max_iter=5, use_gpu=True), verbose=False,
                    iteration_logger=logger, run_json_path=tmp_path / 'run.json')
    gpu = json.loads((tmp_path / 'run.json').read_text())['gpu_qr']
    assert gpu['mem_estimate_bytes'] == gpu_memory_estimate_bytes(json.loads(
        (tmp_path / 'run.json').read_text())['N_total'], 75)
    assert 0 < gpu['min_diag_ratio'] <= 1 and gpu['flagged_iterations'] == []


def test_reproduction_check_applies_b1_tolerances_and_orders_discrepancies_first(tmp_path):
    from experiments.paper_values import PAPER_VALUES
    root = tmp_path / 'B_instrumentation'

    def write(name, **summary):
        (root / name).mkdir(parents=True)
        (root / name / 'summary.json').write_text(json.dumps(summary))

    # Bratu P25 (paper: 2 iterations, kappa 1.7e2): equal iterations, kappa within 10x.
    write('bratu_P25_cpu_paper', iterations=2, t_cum_s=0.001, kappa_final=1.0e3)
    # Kovasznay P675 (paper E_u 2.0e-5, E_p 8.1e-4): E_u within 2x, E_p 7x off.
    write('kovasznay_P675_cpu_paper', n_outer_iters=7, t_cum_s=0.6, rel_l2_u=1.5e-5,
          rel_l2_v=1.3e-4, rel_l2_p=1.1e-4)

    with open(cb.reproduction_check(root), newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(PAPER_VALUES)
    by = {(r['benchmark'], r['config'], r['quantity']): r for r in rows}
    assert by[('bratu', 'P25', 'LiL-Q iterations')]['status'] == 'ok'
    assert by[('bratu', 'P25', 'kappa (final iterate)')]['status'] == 'ok'          # 5.9x < 10x
    assert by[('bratu', 'P25', 'LiL-Q runtime (s)')]['status'] == 'reported'
    assert by[('kovasznay', 'P675', 'iterations')]['status'] == 'violation'         # 7 != 6
    assert by[('kovasznay', 'P675', 'E_u')]['status'] == 'ok'
    assert by[('kovasznay', 'P675', 'E_p')]['status'] == 'violation'
    assert by[('burgers', 'P25', 'LiL-Q iterations')]['status'] == 'missing'
    order = {'violation': 0, 'missing': 1, 'reported': 2, 'ok': 3}
    assert [order[r['status']] for r in rows] == sorted(order[r['status']] for r in rows)


def test_saved_reference_is_the_grid_the_errors_use(tmp_path):
    """Recompute a real Kovasznay run's logged test error from the saved
    reference field: they must agree, so reference/ holds the actual grid."""
    import numpy as np
    from lilq.iteration_log import IterationLogger
    from lilq.test_errors import rel_l2, tensor_grid_values
    from problems.kovasznay import KovasznayConfig, solve_kovasznay

    ref = cb.save_reference(tmp_path)
    expected = {'kovasznay', 'bratu', 'burgers', 'bl', 'bl_gravity', 'elasticity', 'beltrami',
                'darcy_S1', 'darcy_S2', 'darcy_S3', 'darcy_SPE10'}
    assert {p.stem for p in ref.glob('*.npz')} == expected
    assert (ref / 'README.txt').exists()

    k = np.load(ref / 'kovasznay.npz')
    assert k['u'].shape == (301, 401)
    logger = IterationLogger()
    r = solve_kovasznay(KovasznayConfig(N_x=6, N_y=6, max_iter=5), verbose=False, iteration_logger=logger)
    u = tensor_grid_values(r['basis_u'], r['theta_u'], [k['x'], k['y']])
    assert rel_l2(u, k['u']) == pytest.approx(logger.rows[-1]['eps_u'], rel=1e-12)

    d = np.load(ref / 'darcy_S1.npz')
    assert d['P_fvm'].shape == (60, 220)
    assert np.load(ref / 'beltrami.npz')['p'].shape == (21, 21, 21, 11)


def test_saved_code_records_the_commit_and_the_source(tmp_path):
    code = cb.save_code(tmp_path)
    prov = json.loads((code / 'PROVENANCE.json').read_text())
    assert prov['available'] and len(prov['commit']) == 40
    for d in cb.CODE_DIRS:
        assert (code / d).is_dir()
    assert (code / 'experiments' / 'component_b.py').exists()
    assert (code / 'DECISIONS.md').exists()
    assert not list(code.rglob('__pycache__'))


def test_reproduction_check_separates_round_off_from_violations(tmp_path, monkeypatch):
    import experiments.paper_values as pvmod
    from experiments.paper_values import PaperValue
    (tmp_path / 'x_P1_cpu_paper').mkdir()
    (tmp_path / 'x_P1_cpu_paper' / 'summary.json').write_text(json.dumps({'a': 3e-16, 'b': 3e-6}))
    monkeypatch.setattr(pvmod, 'PAPER_VALUES', [
        PaperValue('x', 'P1', 'tiny error', 1e-15, 'error', 't', lambda s: s['a']),   # both round-off
        PaperValue('x', 'P1', 'real error', 1e-5, 'error', 't', lambda s: s['b']),    # 3.3x off: violation
    ])
    with open(cb.reproduction_check(tmp_path), newline='') as f:
        status = {r['quantity']: r['status'] for r in csv.DictReader(f)}
    assert status == {'tiny error': 'round-off', 'real error': 'violation'}


def test_kmax_pass_lengths():
    """Addendum v2.2 Section 2.7: K_max = 60 in the kmax pass for the scalar
    benchmarks and Kovasznay, 8 for Beltrami, zero tolerance throughout. The
    paper pass has K_max = 60 too since the advisor's follow-up of 1 October
    2026; it differs from the kmax pass in its stopping rule."""
    runs = {(r.benchmark, r.config_label, r.device, r.pass_): r for r in cb.build_runs(devices=('cpu',))}
    for b in ('bratu', 'burgers', 'bl', 'bl_gravity'):
        for (bench, label, _, pass_), run in runs.items():
            if bench != b:
                continue
            opt = run.execute.__defaults__[1]
            if pass_ == 'kmax':
                assert opt.max_quasi_iters_lil == cb.KMAX_PASS_ITERS == 60 and opt.R_tol == 0.0
            else:
                assert opt.max_quasi_iters_lil == 60 and opt.R_tol > 0
    kov = [r for k, r in runs.items() if k[0] == 'kovasznay']
    assert {(r.pass_, r.execute.__defaults__[0].max_iter, r.execute.__defaults__[0].tol > 0) for r in kov} == \
        {('paper', 60, True), ('kmax', 60, False)}
    bel = {r.pass_: r.execute.__defaults__[0] for k, r in runs.items() if k[0] == 'beltrami'}
    assert (bel['kmax'].max_iter, bel['kmax'].tol) == (cb.BELTRAMI_KMAX_PASS_ITERS, 0.0) == (8, 0.0)
    assert bel['paper'].tol > 0 and bel['paper'].conditioning_every_iteration and bel['paper'].max_iter == 60


def test_bl_logs_retained_kappa_and_beta_norm():
    from lilq.iteration_log import solve_rows
    from problems.buckley_leverett import BLConfig, BLOptConfig, run_lil_q
    logger = cb.IterationLogger()
    run_lil_q(BLConfig(N_x=6, N_t=6), BLOptConfig(max_quasi_iters_lil=3), verbose=False, iteration_logger=logger)
    rows = logger.rows
    assert all(r["kappa_retained"] <= r["kappa_raw"] == r["kappa"] for r in solve_rows(rows))
    assert all(r["norm_beta"] is not None for r in rows)          # terminal row too


def test_short_paper_runs_are_timed_five_times():
    """Addendum v2.2 Section 2.10: a paper-pass run under 1 s is reported as
    the median of five repeats, all five kept; longer runs and the kmax pass
    report their own time."""
    times = iter([0.05, 0.03, 0.04, 0.06, 0.02])
    reported, repeats = cb._timing_repeats(0.2, lambda: next(times), 'paper', smoke=False)
    assert repeats == [0.05, 0.03, 0.04, 0.06, 0.02] and reported == 0.04
    assert cb._timing_repeats(0.2, lambda: 1 / 0, 'kmax', smoke=False) == (0.2, None)
    assert cb._timing_repeats(3.0, lambda: 1 / 0, 'paper', smoke=False) == (3.0, None)


def test_scalar_summary_has_training_time_beside_t_cum(tmp_path):
    runs = cb.build_runs(benchmarks=('bratu',), passes=('paper',), devices=('cpu',), smoke=True)
    assert cb.execute_run(runs[0], tmp_path, verbose=False) == 'ok'
    s = json.loads((tmp_path / runs[0].name / 'summary.json').read_text())
    assert s['training_time'] > 0 and 't_cum_s' in s and 'training_time_single_run' in s
