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
    waves = ''.join((REPO / 'scripts' / 'cluster' / f'submit_wave{n}.sh').read_text() for n in (1, 2, 3, 4))
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


def test_timing_reruns_and_classes_of_wave_2():
    """The reply to wave 1: timing reruns in timed jobs (item 2.2) and job
    21 at 4 h (item 2.4)."""
    classes = {s.stem: next(l for l in s.read_text().splitlines() if l.startswith('# lilq-resources:')).split()[2]
               for s in (REPO / 'scripts' / 'cluster').glob('*.slurm')}
    assert classes['11a_timing_reruns_cpu'] == 'timed-cpu' and classes['11b_timing_reruns_gpu'] == 'timed'
    assert classes['12_timing_reruns_beltrami_darcy'] == 'timed-cpu'
    assert '#SBATCH --time=04:00:00' in (REPO / 'scripts/cluster/21_four_method_cpu.slurm').read_text()
    wave2 = (REPO / 'scripts/cluster/submit_wave2.sh').read_text()
    assert '11a_timing_reruns_cpu.slurm' in wave2 and '11b_timing_reruns_gpu.slurm' in wave2
    assert sum(l.startswith('submit') and '--array=0-3' in l for l in wave2.splitlines()) == 2


def test_timed_class_holds_the_node_with_one_gpu_and_no_exclusive():
    """Item 2.3: --exclusive allocated (and charged) both A100s."""
    sbatch = (REPO / 'scripts/cluster/sbatch.sh').read_text()
    timed = next(l for l in sbatch.splitlines() if l.strip().startswith('timed)'))
    assert '--exclusive' not in timed and '--cpus-per-task="$NODE_CORES"' in sbatch
    assert 'GPU_GRES=gpu:a100:1' in (REPO / 'scripts/cluster/profiles/grace.sh').read_text()


def test_component_a_endings_flag_runs_that_stopped_before_their_budget(tmp_path):
    report = _load('wave_report')
    root = tmp_path / 'wave2'
    _write(root / 'A_calibration' / 'screening' / 'F1_00_s0' / 'run.json',
           json.dumps({'end_reason': 'budget', 'budget_s': 600.0, 'wall_s': 600.0}))
    _write(root / 'A_calibration' / 'screening' / 'F2_03_s0' / 'run.json',
           json.dumps({'end_reason': 'converged', 'budget_s': 600.0, 'wall_s': 212.5}))
    _write(root / 'A_calibration' / 'full' / 'F1_07_s2' / 'run.json',
           json.dumps({'end_reason': 'failure: non-finite loss', 'budget_s': 3600.0, 'wall_s': 50.0}))
    _write(root / 'A_calibration' / 'screening' / 'F1_00_s0' / 'log.csv', 'it,loss\n')
    _write(root / 'A_calibration' / 'screening' / 'F1_selection.json', '{}')
    early = report.component_a_endings(root)
    assert sorted(r['run'] for r in early) == ['full/F1_07_s2', 'screening/F2_03_s0']
    import csv
    rows = list(csv.DictReader(open(root / 'A_calibration' / 'run_endings.csv')))
    assert len(rows) == 3 and {r['family'] for r in rows} == {'F1', 'F2'}
    packed = {p.as_posix() for p in report.report_files(root)}
    assert {'A_calibration/run_endings.csv', 'A_calibration/screening/F1_00_s0/log.csv',
            'A_calibration/screening/F1_00_s0/run.json', 'A_calibration/screening/F1_selection.json'} <= packed


@pytest.mark.skipif(__import__('os').name == 'nt', reason='runs the bash submission scripts')
def test_wave_2_dry_run_requests_one_gpu_per_timed_job(tmp_path):
    """Wave 2 on new code: preflight, A1 gate, Component A, 20/21 with all
    four tasks, the timing reruns and the report -- every timed GPU job one
    A100, all 48 cores and 360G, without --exclusive."""
    import os
    import subprocess
    fake = tmp_path / 'sbatch'
    fake.write_text('#!/bin/bash\n'
                    'for a in "$@"; do [[ "$a" == --test-only ]] && { echo "$*" >> "$(dirname "$0")/log"; '
                    'echo "sbatch: Job 1 to start at soon" >&2; exit 0; }; done\nexit 1\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", DRY_RUN='1')
    out = subprocess.run(['bash', str(REPO / 'scripts/cluster/submit_wave2.sh')], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 12
    gpu_timed = [l for l in log if '--partition=gpu' in l and '--cpus-per-task=48' in l]
    assert len(gpu_timed) == 5                      # 30, 31, 33, 20, 11b
    assert all('--gres=gpu:a100:1' in l and '--mem=360G' in l and '--exclusive' not in l for l in gpu_timed)
    assert all('--array=0-3' in l for l in log if 'four_method' in l)


@pytest.mark.skipif(__import__('os').name == 'nt', reason='runs bash')
@pytest.mark.parametrize('script, device', [('20_four_method_gpu', 'gpu'), ('21_four_method_cpu', 'cpu')])
def test_b4_tasks_run_controls_and_wave_2_bratu_controls_wave_1(script, device):
    """Item 2.5: every task runs its own stall controls; in wave 2 the Bratu
    task runs only wave 1's Bratu controls, into a folder of its own."""
    import subprocess
    text = (REPO / 'scripts' / 'cluster' / f'{script}.slurm').read_text()
    block = text[text.index('PROBLEMS='):text.index('\nfi\n') + 4]

    def resolve(wave, task):
        cmd = (f'B=/R/wave{wave}/B RESULTS=/R LILQ_WAVE={wave} SLURM_ARRAY_TASK_ID={task}\n{block}'
               'echo "$OUT|${MODE[*]}"')
        return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True).stdout.strip()
    assert resolve(2, 1) == f'/R/wave2/B/four_method_jobs/burgers_{device}|--controls'
    assert resolve(2, 0) == (f'/R/wave2/B/four_method_jobs/bratu_{device}_controls|--controls-only '
                             f'--controls-from /R/wave1/B_instrumentation/four_method_jobs/bratu_{device}')
    assert resolve(1, 0) == f'/R/wave1/B/four_method_jobs/bratu_{device}|--controls'


def test_wave_4_scripts_and_classes():
    """Wave 4 (the advisor's reply to wave 2, his follow-up, Addendum v2.3)."""
    classes = {s.stem: next(l for l in s.read_text().splitlines() if l.startswith('# lilq-resources:')).split()[2]
               for s in (REPO / 'scripts' / 'cluster').glob('*.slurm')}
    assert classes['13a_clean_timing_cpu'] == 'timed-cpu' and classes['13b_clean_timing_gpu'] == 'timed'
    assert classes['35_A_f2_full'] == 'timed' and classes['34_A_f1_repick'] == 'shared-gpu'
    assert classes['44_b10_cgl_cc'] == classes['45_b8_kmax60'] == 'cpu'
    f1 = (REPO / 'scripts/cluster/34_A_f1_repick.slurm').read_text()
    assert '--keep-top F1_04 F1_15 F1_18' in f1
    f2 = (REPO / 'scripts/cluster/35_A_f2_full.slurm').read_text()
    assert '--keep-top F2_10 F2_05 F2_14' in f2                       # pinned (reply on wave 3, item 2.4)
    assert '13b_clean_timing_gpu.slurm --dependency=afterok:$pre,afterany:$tcpu' in \
        (REPO / 'scripts/cluster/submit_wave4.sh').read_text()
    wave3 = (REPO / 'scripts/cluster/submit_wave3.sh').read_text()
    assert '90_finalize' not in [l.split()[2].split('/')[-1].replace('.slurm', '') for l in wave3.splitlines()
                                 if l.startswith('submit ')]
    assert '$S/90_finalize.slurm' in (REPO / 'scripts/cluster/submit_wave4.sh').read_text()


def test_representative_thresholds(tmp_path):
    """For each representative run: the first logged time eps_u <= 1e-4,
    1e-6, 1e-8, 1e-9, and the final errors (the reply to wave 2, Section 2)."""
    report = _load('wave_report')
    A = tmp_path / 'wave4' / 'A_calibration'
    _write(A / 'full' / 'F2_representative.json', json.dumps({'representative': 'F2_05'}))
    for stage, seed, trace in (('full', 0, [(1.0, 1e-3), (2.0, 5e-5), (3.0, 1e-7), (4.0, 5e-9)]),
                               ('full_cpu', 0, [(2.0, 1e-3), (9.0, 2e-6)])):
        d = A / stage / f'F2_05_s{seed}'
        _write(d / 'run.json', json.dumps({'seed': seed, 'device': 'cuda' if stage == 'full' else 'cpu',
                                           'eps_u': trace[-1][1], 'eps_p_meanfree': 1e-6, 'wall_s': 3600.0,
                                           'end_reason': 'budget'}))
        _write(d / 'log.csv', 'iter,t_cum_s,eps_u\n' + ''.join(f'{i},{t},{e}\n' for i, (t, e) in enumerate(trace))
               + '9,9.5,\n')
    _write(A / 'full' / 'F2_03_s0' / 'log.csv', 'iter,t_cum_s,eps_u\n0,1.0,1e-12\n')       # not the representative
    rows = {r['stage']: r for r in report.representative_thresholds(tmp_path / 'wave4')}
    assert set(rows) == {'full', 'full_cpu'}
    full = rows['full']
    assert (full['t_eps_u_le_0.0001'], full['t_eps_u_le_1e-06'], full['t_eps_u_le_1e-08'],
            full['t_eps_u_le_1e-09']) == (2.0, 3.0, 4.0, None)
    assert rows['full_cpu']['t_eps_u_le_1e-06'] is None and rows['full_cpu']['t_eps_u_le_0.0001'] == 9.0
    assert (tmp_path / 'wave4' / 'A_calibration' / 'representative_thresholds.csv').exists()


def test_package_assembles_four_waves(tmp_path):
    assemble = _load('assemble_package')
    res = tmp_path / 'results'
    for n, commit in ((1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')):
        _write(res / f'wave{n}' / 'COMMIT', json.dumps({'commit': commit}))
    _write(res / 'wave2' / 'A_calibration' / 'full' / 'F2_representative.json', '{"representative": "F2_16"}')
    _write(res / 'wave4' / 'A_calibration' / 'full' / 'F2_representative.json', '{"representative": "F2_05"}')
    record = assemble.assemble(res, res / 'package1')
    assert record['waves']['4']['commit'] == 'd'
    assert json.loads((res / 'package1' / 'A_calibration/full/F2_representative.json').read_text())['representative'] == 'F2_05'
    assert record['overrides'] == [{'path': 'A_calibration/full/F2_representative.json', 'from_wave': 4}]


def test_package_files_are_copies_so_finalize_cannot_change_a_wave(tmp_path):
    """90_finalize rewrites package files in place (the run index, the
    reproduction check); a hard link would rewrite the wave's own file too
    (the advisor's reply on wave 3, Section 1, item 2)."""
    import os
    assemble = _load('assemble_package')
    res = tmp_path / 'results'
    _write(res / 'wave2' / 'COMMIT', json.dumps({'commit': 'b'}))
    _write(res / 'wave2' / 'B_instrumentation' / 'runs_index.csv', 'run,status\nx,ok\n')
    assemble.assemble(res, res / 'package1')
    pkg = res / 'package1' / 'B_instrumentation' / 'runs_index.csv'
    assert os.stat(pkg).st_nlink == 1
    with open(pkg, 'w') as f:                                        # as write_index does
        f.write('rebuilt\n')
    assert (res / 'wave2' / 'B_instrumentation' / 'runs_index.csv').read_text() == 'run,status\nx,ok\n'
    # A package1 left from an assembly by hard links: rebuilt from scratch, the
    # wave's file untouched, stale files gone.
    pkg.unlink()
    os.link(res / 'wave2' / 'B_instrumentation' / 'runs_index.csv', pkg)
    (res / 'package1' / 'stale.txt').write_text('x')
    record = assemble.assemble(res, res / 'package1')
    assert os.stat(pkg).st_nlink == 1 and record['overrides'] == [] and not (res / 'package1' / 'stale.txt').exists()
    assert (res / 'wave2' / 'B_instrumentation' / 'runs_index.csv').read_text() == 'run,status\nx,ok\n'
    other = tmp_path / 'not_a_package'
    _write(other / 'keep.txt', 'x')
    with pytest.raises(FileExistsError):
        assemble.assemble(res, other)
    assert (other / 'keep.txt').exists()


@pytest.mark.skipif(__import__('os').name == 'nt', reason='runs the bash submission scripts')
def test_wave_4_submission_copies_component_a_and_chains_the_jobs(tmp_path):
    """A real (fake-sbatch) submission: wave 2's search, screening runs and
    F1 finalist runs are copied into wave 4, and 15 jobs are submitted with
    the right dependencies."""
    import os
    import subprocess
    res = tmp_path / 'results'
    w2 = res / 'wave2' / 'A_calibration'
    for p in ('search/F2_configs.json', 'screening/F2_05_s0/run.json', 'full/F1_18_s0/run.json',
              'full/F1_04_s4/run.json', 'full/F1_15_s2/run.json', 'full/F2_16_s0/run.json', 'tuning_log.md'):
        _write(w2 / p, '{}')
    _write(res / 'wave3' / 'COMMIT', json.dumps({'commit': 'c'}))
    fake = tmp_path / 'sbatch'
    fake.write_text('#!/bin/bash\nn=$(cat "$(dirname "$0")/n" 2>/dev/null || echo 100); echo $((n+1)) > "$(dirname "$0")/n"\n'
                    'echo "$*" >> "$(dirname "$0")/log"; echo $((n+1))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", LILQ_RESULTS=str(res), YES='1')
    out = subprocess.run(['bash', str(REPO / 'scripts/cluster/submit_wave4.sh')], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    w4 = res / 'wave4' / 'A_calibration'
    assert (w4 / 'search' / 'F2_configs.json').exists() and (w4 / 'screening' / 'F2_05_s0' / 'run.json').exists()
    assert (w4 / 'full' / 'F1_18_s0').exists() and (w4 / 'full' / 'F1_04_s4').exists() and (w4 / 'full' / 'F1_15_s2').exists()
    assert not (w4 / 'full' / 'F2_16_s0').exists() and (w4 / 'tuning_log.md').exists()
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 14
    assert sum('32_A_cpu' in l for l in log) == 2 and any('--array=1' in l and '32_A_cpu' in l for l in log)
    assert 'afterany' in log[-1] and '90_finalize' in log[-1]
    # 13b waits for 13a (both write clean_timing.csv); the fake ids count from 101.
    tcpu = 101 + next(i for i, l in enumerate(log) if '13a_clean_timing_cpu' in l)
    assert f'afterany:{tcpu}' in next(l for l in log if '13b_clean_timing_gpu' in l)
