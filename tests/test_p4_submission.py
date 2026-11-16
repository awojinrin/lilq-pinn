"""Package 4's submission: the job scripts (as generated, LF), their classes and the SU plan,
the inputs the preflight checks, the p4 stage in the cluster scripts and the report, and the
submission script (bash): every job once, in order, with its dependencies, and the dry run."""

import importlib.util
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
P4 = REPO / 'scripts' / 'cluster' / 'package4'
SUBMIT = P4 / 'submit_p4.sh'
CLASS = {'p4_preflight': ('cpu', '00:30:00'), 'p4_timed': ('timed-cpu', '00:20:00'),
         'p4_item1': ('cpu', '00:40:00'), 'p4_item2': ('cpu', '00:20:00'),
         'p4_item2_constants': ('cpu', '01:15:00'), 'p4_report': ('cpu', '00:15:00')}
AFTER_PREFLIGHT = {'p4_timed', 'p4_item1', 'p4_item2'}
RATE = {'cpu': 24, 'timed-cpu': 48}                       # SU per hour on Grace
CAP = 100


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hours(wall):
    h, m, s = map(int, wall.split(':'))
    return h + m / 60 + s / 3600


def test_job_scripts_their_classes_and_the_su_plan():
    assert {p.stem for p in P4.glob('*.slurm')} == set(CLASS)
    for name, (cls, wall) in CLASS.items():
        t = (P4 / f'{name}.slurm').read_text()
        assert re.search(rf'^# lilq-resources: {cls} ', t, re.M), name
        assert f'#SBATCH --time={wall}\n' in t, name
        assert re.search(r'#SBATCH --job-name=lilq-p4-[a-z0-9-]+\n', t), name   # the report finds the logs by it
        assert 'source scripts/cluster/env.sh\nset -euo pipefail\n' in t, name
    requested = sum(RATE[c] * _hours(w) for c, w in CLASS.values())
    assert requested == pytest.approx(88) and requested <= CAP                  # within the cap even at the walltimes
    assert '88 SU requested' in SUBMIT.read_text()


def test_job_scripts_are_as_generated_and_lf(tmp_path):
    gen = _module('p4_generate_jobs', P4 / '_generate_jobs.py')
    gen.HERE = tmp_path
    gen.main()
    for p in tmp_path.glob('*.slurm'):
        assert p.read_bytes() == (P4 / p.name).read_bytes(), p.name
    for p in P4.iterdir():
        assert not p.is_file() or b'\r' not in p.read_bytes(), p.name                              # bash on the cluster


def test_every_stage_runs_once_and_reads_checked_inputs():
    text = {name: (P4 / f'{name}.slurm').read_text() for name in CLASS}
    stages = {'p4_1_burgers_gauss.py': ('logged', 'enrich', 'constants', 'time', 'summarize'),
              'p4_2_pb_inclusions.py': ('points', 'compare-points', 'lilq', 'square', 'under', 'rows', 'weight',
                                        'time', 'm7', 'solution', 'summarize')}
    everything = ''.join(text.values())
    for driver, names in stages.items():
        for stage in names:
            assert len(re.findall(rf'{driver} {stage} ', everything)) == 1, (driver, stage)
    assert len(re.findall(r'p4_2_pb_inclusions.py constants ', text['p4_item2_constants'])) == 2   # p <= 26, then p = 30
    assert 'p4_checks.py m0' in text['p4_timed'] and 'p4_checks.py provenance' in text['p4_timed']
    assert 'p4_checks.py collect' in text['p4_report'] and 'stage_report.py --stage p4' in text['p4_report']
    pre = text['p4_preflight']
    assert 'MISSING INPUT' in pre and 'pytest' in pre
    for job, t in text.items():                                                 # every input a job reads is checked first
        for path in re.findall(r'"\$RESULTS/([^"]+)"', t):
            assert path in pre, (job, path)
    for seed in (0, 1, 2):                                                      # the shipped points, in the bundle
        assert f'data/package4/halton_seed{seed}.npz' in pre
        assert (REPO / 'data' / 'package4' / f'halton_seed{seed}.npz').exists()


def test_item2f_runs_the_seeds_side_by_side_with_p30_last():
    t = (P4 / 'p4_item2_constants.slurm').read_text()
    assert 'LILQ_WALL_S=4500 ' in t                                              # 1:15:00
    assert 'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8' in t
    assert t.count('seed_runs 0 &') == 1 and t.count('--no-extremes & pids') == 2
    assert 'until_end python experiments/p4_2_pb_inclusions.py constants --out "$PKG" --points data/package4 --seeds "$s" --p 30 --no-extremes' in t


def test_p4_is_a_stage_of_the_cluster_scripts():
    cluster = REPO / 'scripts' / 'cluster'
    assert 'p4)   export PKG="$RESULTS/package4"' in (cluster / 'env.sh').read_text()
    assert '|p3|p4|p5|p5t|p5a)$' in (cluster / 'sbatch.sh').read_text()
    assert 'LILQ_WAVE" == p4' in (cluster / 'submit_lib.sh').read_text()
    assert _module('stage_report', cluster / 'package2' / 'stage_report.py').STAGES['p4'] == 'package4'


def test_stage_report_packs_package4(tmp_path, monkeypatch):
    sr = _module('stage_report', REPO / 'scripts' / 'cluster' / 'package2' / 'stage_report.py')
    pkg = tmp_path / 'results' / 'package4' / 'P4_checks'
    pkg.mkdir(parents=True)
    (pkg / 'm0.json').write_text('{}')
    logs = tmp_path / 'logs'
    logs.mkdir()
    for name in ('lilq-p4-timed.401.out', 'lilq-p4-report.499.out', 'lilq-p3-k0.301.out'):
        (logs / name).write_text('log\n')
    monkeypatch.setenv('SLURM_JOB_ID', '499')
    monkeypatch.delenv('CLUSTER', raising=False)
    tar_path = sr.report('p4', tmp_path / 'results', logs_dir=logs, sacct=False)
    assert tar_path.name == 'package4.tar.gz'
    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert {'package4/P4_checks/m0.json', 'package4/slurm_logs/lilq-p4-timed.401.out'} <= names
    assert not any('report.499' in n or 'p3' in n for n in names)


# ---------------------------------------------------------------- the submission script (bash)

def _fake_cluster(tmp_path, dry):
    """sbatch (as Package 3's tests) and a `python` that is this one."""
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    if dry:
        (bin_ / 'sbatch').write_text('#!/bin/bash\nfor a in "$@"; do [[ "$a" == --test-only ]] && { echo "$*" >> '
                                     '"$(dirname "$0")/log"; echo "sbatch: Job 1 to start at soon" >&2; exit 0; }; done\nexit 1\n')
    else:
        (bin_ / 'sbatch').write_text('#!/bin/bash\nn=$(cat "$(dirname "$0")/n" 2>/dev/null || echo 100); '
                                     'echo $((n+1)) > "$(dirname "$0")/n"\necho "$*" >> "$(dirname "$0")/log"; echo $((n+1))\n')
    (bin_ / 'python').write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    for p in bin_.iterdir():
        p.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", LILQ_ACCOUNT='000000000000', YES='1',
               LILQ_RESULTS=str(tmp_path / 'results'), **({'DRY_RUN': '1'} if dry else {}))
    env.pop('CLUSTER', None)
    return env, bin_ / 'log'


def _job(line):
    return re.search(r'(p4_[A-Za-z0-9_]+)\.slurm', line).group(1)


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_dry_run_checks_all_6_jobs_and_submits_nothing(tmp_path):
    env, log = _fake_cluster(tmp_path, True)
    out = subprocess.run(['bash', str(SUBMIT)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'Package 4 (results/package4)' in out.stdout and '88 SU requested' in out.stdout
    lines = log.read_text().splitlines()
    assert len(lines) == 6 and all('--account=000000000000' in l and '--test-only' in l for l in lines)
    assert not any('--dependency' in l for l in lines)
    for l in lines:
        assert ('--exclusive' in l) == (CLASS[_job(l)][0] == 'timed-cpu'), l


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_submission_chains_the_jobs(tmp_path):
    env, log = _fake_cluster(tmp_path, False)
    out = subprocess.run(['bash', str(SUBMIT)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 6
    ids = {_job(l): 101 + i for i, l in enumerate(lines)}
    assert set(ids) == set(CLASS)
    by_job = {_job(l): l for l in lines}
    assert '--dependency' not in lines[0] and _job(lines[0]) == 'p4_preflight'
    for job in AFTER_PREFLIGHT:
        assert f"--dependency=afterok:{ids['p4_preflight']} " in by_job[job], job
    assert f"--dependency=afterok:{ids['p4_item2']} " in by_job['p4_item2_constants']
    rep = by_job['p4_report']
    assert _job(lines[-1]) == 'p4_report' and 'afterany' in rep
    assert all(f':{ids[j]}' in rep for j in AFTER_PREFLIGHT | {'p4_item2_constants'})
    assert f":{ids['p4_preflight']}" not in rep


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash job script')
def test_until_end_stops_a_run_before_the_walltime(tmp_path):
    """The item 2f cap: a command still running 5 min before the walltime is stopped, and the job goes on."""
    t = (P4 / 'p4_item2_constants.slurm').read_text()
    until_end = next(l for l in t.splitlines() if l.startswith('until_end()'))
    script = f'set -euo pipefail\nLILQ_WALL_S=302\n{until_end}\n( until_end sleep 30 )\necho went-on\n'
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0 and 'STOPPED OR FAILED (exit 124): sleep 30' in out.stdout and 'went-on' in out.stdout


# ---------------------------------------------------------------- the report's checks.json

def test_checks_json_marks_checks_not_evaluated_on_the_cluster(tmp_path):
    """M6 needs the advisor's expected values, which the cluster does not have: no verdict there,
    never a false failure; a failed part fails the check."""
    import json
    sys.path.insert(0, str(REPO))
    from experiments import p4_checks
    assert p4_checks.combined_pass([True, True]) is True
    assert p4_checks.combined_pass([True, None]) is None and p4_checks.combined_pass([]) is None
    assert p4_checks.combined_pass([None, False]) is False
    (tmp_path / 'P4_2_pb_inclusions').mkdir()
    (tmp_path / 'checks_item1.json').write_text(json.dumps({'M3': {'pass': True}, 'M9_item1': {'pass': True}}))
    (tmp_path / 'P4_2_pb_inclusions' / 'checks_item2.json').write_text(json.dumps(
        {'M3': {'pass': True}, 'M6': {'pass': None, 'area_pass': True}, 'M9_item2': {'pass': False}}))
    summary = p4_checks.collect(tmp_path)
    assert summary == {'M3': True, 'M9': False, 'M6': None}


def test_kappa_at_the_returned_iterate_is_the_next_solve_row():
    sys.path.insert(0, str(REPO))
    from experiments.p4_common import kappa_returned
    rows = [{'kappa_retained': str(10.0 * k), 'kappa_raw': str(20.0 * k)} for k in range(5)]
    assert kappa_returned(rows, 2) == {'kappa_retained_returned': 30.0, 'kappa_raw_returned': 60.0}
    assert all(v != v for v in kappa_returned(rows, 4).values())                 # the log ends at the rule: NaN
    assert all(v != v for v in kappa_returned(rows, None).values())
