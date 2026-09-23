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
    assert len(names) == len(set(names)) == 62
    # Scalar benchmarks: paper sizes x {paper, kmax}, CPU.
    assert sum(r.benchmark == 'bratu' for r in runs) == 3 * 2
    assert sum(r.benchmark == 'burgers' for r in runs) == 5 * 2
    assert sum(r.benchmark == 'bl' for r in runs) == 4 * 2
    assert sum(r.benchmark == 'bl_gravity' for r in runs) == 4 * 2
    # Kovasznay: five sizes x {cpu, cuda} x {paper, kmax}.
    assert sum(r.benchmark == 'kovasznay' for r in runs) == 5 * 2 * 2
    # Single-pass benchmarks (no K_max pass per Section 3.3).
    assert [r.name for r in runs if r.benchmark == 'beltrami'] == ['beltrami_P7984_cpu_paper']
    assert sum(r.benchmark == 'elasticity' for r in runs) == 5
    assert [r.config_label for r in runs if r.benchmark == 'darcy'] == ['S1', 'S2', 'S3', 'SPE10']
    assert {r.pass_ for r in runs if r.benchmark in ('elasticity', 'beltrami', 'darcy')} == {'paper'}


def test_kmax_only_pass_skips_single_pass_benchmarks():
    runs = cb.build_runs(passes=('kmax',), devices=('cpu',))
    assert {r.benchmark for r in runs} == {'bratu', 'burgers', 'bl', 'bl_gravity', 'kovasznay'}


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
    assert row['P'] == '75' and row['equivalent'] == 'True'
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
