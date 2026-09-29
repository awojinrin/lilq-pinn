"""Addendum v2.2 Section 4.2: the three waves' results folders, the wave
report, and package1 assembled from the waves."""

import importlib.util
import json
import tarfile

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / 'cluster' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_package_is_assembled_from_the_waves_with_their_commits(tmp_path):
    assemble = _load('assemble_package')
    res = tmp_path / 'results'
    _write(res / 'wave1' / 'COMMIT', json.dumps({'commit': 'aaa', 'tree_hash': 't1'}))
    _write(res / 'wave1' / 'B_instrumentation' / 'bratu_P25_cpu_paper' / 'summary.json', '{"x": 1}')
    _write(res / 'wave1' / 'A_calibration' / 'search' / 'F1_configs.json', 'same')
    _write(res / 'wave1' / 'A_calibration' / 'tuning_log.md', 'wave 1 lines\n')
    _write(res / 'wave2' / 'COMMIT', json.dumps({'commit': 'bbb', 'tree_hash': 't2'}))
    _write(res / 'wave2' / 'A_calibration' / 'search' / 'F1_configs.json', 'same')          # copied from wave 1
    _write(res / 'wave2' / 'A_calibration' / 'tuning_log.md', 'wave 1 lines\nwave 2 lines\n')
    _write(res / 'wave2' / 'A_calibration' / 'full' / 'F1_representative.json', '{}')
    record = assemble.assemble(res, res / 'package1')
    out = res / 'package1'
    assert (out / 'B_instrumentation' / 'bratu_P25_cpu_paper' / 'summary.json').read_text() == '{"x": 1}'
    assert (out / 'A_calibration' / 'tuning_log.md').read_text().endswith('wave 2 lines\n')   # later wave wins
    assert not (out / 'COMMIT').exists()
    waves = json.loads((out / 'WAVES.json').read_text())
    assert waves['waves']['1']['commit'] == 'aaa' and waves['waves']['2']['commit'] == 'bbb'
    assert waves['waves']['3'] is None
    assert [o['path'] for o in record['overrides']] == ['A_calibration/tuning_log.md']


def test_wave_report_packs_the_small_files_not_the_models(tmp_path):
    report = _load('wave_report')
    res = tmp_path / 'results'
    root = res / 'wave1'
    _write(root / 'COMMIT', '{"commit": "aaa"}')
    _write(root / 'B_instrumentation' / 'gpu_cpu_equivalence.csv', 'P\n75\n')
    _write(root / 'B_instrumentation' / 'bratu_P25_cpu_kmax' / 'iterations.csv', 'k\n0\n')
    _write(root / 'B_instrumentation' / 'bratu_P25_cpu_kmax' / 'solution.pt', 'big')
    _write(root / 'B_instrumentation' / 'four_method_jobs' / 'bratu_gpu' / 'models' / 'm' / 'history.csv', 'i\n')
    _write(root / 'B_instrumentation' / 'four_method_jobs' / 'bratu_gpu' / 'models' / 'm' / 'network.pt', 'big')
    _write(root / 'A_calibration' / 'checks' / 'a1.json', '{"passed": true}')
    _write(root / 'C_oversampling' / 'runs' / 'r' / 'collocation.npz', 'big')
    _write(root / 'C_oversampling' / 'results' / 'oversampling.csv', 'benchmark\n')
    report.main(['--wave', '1', '--results', str(res)])
    with tarfile.open(res / 'wave1_report.tar.gz') as tar:
        names = set(tar.getnames())
    assert {'wave1/COMMIT', 'wave1/B_instrumentation/gpu_cpu_equivalence.csv',
            'wave1/B_instrumentation/bratu_P25_cpu_kmax/iterations.csv',
            'wave1/B_instrumentation/four_method_jobs/bratu_gpu/models/m/history.csv',
            'wave1/A_calibration/checks/a1.json', 'wave1/C_oversampling/results/oversampling.csv',
            'wave1/sacct.txt', 'wave1/report_log.txt'} <= names
    assert not any(n.endswith(('.pt', '.npz')) for n in names)


def test_wave_report_builds_the_index_and_check_b1_from_real_runs(tmp_path):
    """The report's own steps on real (smoke) Component B output: the run
    index, check B1's table, and both in the tarball, with no failed step."""
    import experiments.component_b as cb
    report = _load('wave_report')
    res = tmp_path / 'results'
    root = res / 'wave1'
    runs = cb.build_runs(benchmarks=('bratu',), passes=('paper', 'kmax'), devices=('cpu',), smoke=True)
    assert [cb.execute_run(r, root / 'B_instrumentation', verbose=False) for r in runs] == ['ok'] * len(runs)
    report.main(['--wave', '1', '--results', str(res)])
    log = (root / 'report_log.txt').read_text()
    assert 'exit 1' not in log and log.count('exit 0') == 2
    with tarfile.open(res / 'wave1_report.tar.gz') as tar:
        names = set(tar.getnames())
    assert {'wave1/B_instrumentation/runs_index.csv', 'wave1/B_instrumentation/reproduction_check.csv',
            'wave1/B_instrumentation/bratu_P25_cpu_kmax/iterations.csv'} <= names


def test_every_job_script_names_a_resource_class_and_waves_use_them():
    classes = {}
    for script in (REPO / 'scripts' / 'cluster').glob('*.slurm'):
        line = next(l for l in script.read_text().splitlines() if l.startswith('# lilq-resources:'))
        classes[script.stem] = line.split()[2]
    assert classes['10a_timed_lilq_cpu'] == classes['21_four_method_cpu'] == classes['32_A_cpu'] == 'timed-cpu'
    assert classes['10b_timed_lilq_gpu'] == classes['20_four_method_gpu'] == 'timed'
    waves = ''.join((REPO / 'scripts' / 'cluster' / f'submit_wave{n}.sh').read_text() for n in (1, 2, 3))
    for job in classes:
        assert f'$S/{job}.slurm' in waves, job              # every job belongs to a wave


@pytest.mark.skipif(__import__('os').name == 'nt', reason='runs the bash submission scripts')
def test_dry_run_checks_every_job_and_submits_nothing(tmp_path):
    """DRY_RUN=1 puts each wave-1 job through `sbatch --test-only` with the
    profile's resources, and fails for a job the scheduler would refuse (Grace
    refuses --mem=0, so whole-node jobs must request the memory)."""
    import os
    import subprocess
    fake = tmp_path / 'sbatch'
    fake.write_text('#!/bin/bash\n'
                    'for a in "$@"; do [[ "$a" == --mem=0 ]] && { echo "refused: --mem can NOT be 0" >&2; exit 1; }; done\n'
                    'for a in "$@"; do [[ "$a" == --test-only ]] && { echo "$*" >> "$(dirname "$0")/log"; '
                    'echo "sbatch: Job 1 to start at soon" >&2; exit 0; }; done\nexit 1\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", DRY_RUN='1')
    out = subprocess.run(['bash', str(REPO / 'scripts/cluster/submit_wave1.sh')], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 9 and not any('--dependency' in l for l in log)
    assert all('--mem=360G' in l for l in log if '10a_timed_lilq_cpu' in l or '10b_timed_lilq_gpu' in l)
    out = subprocess.run(['bash', str(REPO / 'scripts/cluster/submit_wave1.sh')],
                         env=dict(env, CLUSTER='faster'), capture_output=True, text=True)
    assert out.returncode != 0 and 'DRY RUN FAILED' in out.stdout
