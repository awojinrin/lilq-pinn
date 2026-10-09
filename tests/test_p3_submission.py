"""Package 3's submission: the job scripts (as generated, LF), wave 1 submitted once in order
with its dependencies, the dry run, B3 level 2's gate (wave 2) and its submission, and the
p3 stage in the cluster scripts and the report."""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
P3 = REPO / 'scripts' / 'cluster' / 'package3'
SUBMIT, SUBMIT_L2 = P3 / 'submit_p3.sh', P3 / 'submit_p3_b3l2.sh'
COMPUTE = {'p3_k0', 'p3_beltrami_small', 'p3_beltrami_b2', 'p3_beltrami_b3_l1', 'p3_bl', 'p3_affine',
           'p3_burgers_large_P'}
CLASS = {'p3_preflight': ('cpu', '01:00:00'), 'p3_k0': ('timed-cpu', '02:00:00'),
         'p3_beltrami_small': ('timed-cpu', '03:00:00'), 'p3_beltrami_b2': ('timed-cpu', '04:00:00'),
         'p3_beltrami_b3_l1': ('timed-cpu', '04:00:00'), 'p3_beltrami_b3_l2': ('timed-cpu', '07:00:00'),
         'p3_bl': ('cpu', '02:00:00'), 'p3_affine': ('timed-cpu', '01:30:00'),
         'p3_burgers_large_P': ('timed-cpu', '01:00:00'), 'p3_report': ('cpu', '00:30:00')}
RATE = {'cpu': 24, 'timed-cpu': 48}                       # SU per hour on Grace (su_plan.csv)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hours(wall):
    h, m, s = map(int, wall.split(':'))
    return h + m / 60 + s / 3600


def test_job_scripts_their_classes_and_the_su_plan():
    assert {p.stem for p in P3.glob('*.slurm')} == set(CLASS)
    for name, (cls, wall) in CLASS.items():
        t = (P3 / f'{name}.slurm').read_text()
        assert re.search(rf'^# lilq-resources: {cls} ', t, re.M), name
        assert f'#SBATCH --time={wall}\n' in t, name
        assert re.search(r'#SBATCH --job-name=lilq-p3-[a-z0-9-]+\n', t), name   # the report and the gate find the logs by it
        assert 'source scripts/cluster/env.sh\nset -euo pipefail\n' in t, name
    wave1 = sum(RATE[c] * _hours(w) for n, (c, w) in CLASS.items() if n != 'p3_beltrami_b3_l2')
    assert wave1 == 828 and wave1 + 336 == 1164                                # under the cap of 850 even at the caps
    assert RATE['timed-cpu'] * _hours(CLASS['p3_beltrami_b3_l2'][1]) == 336


def test_job_scripts_are_as_generated_and_lf(tmp_path):
    gen = _module('p3_generate_jobs', P3 / '_generate_jobs.py')
    gen.HERE = tmp_path
    gen.main()
    for p in tmp_path.glob('*.slurm'):
        assert p.read_bytes() == (P3 / p.name).read_bytes(), p.name
    for p in P3.iterdir():
        assert not p.is_file() or b'\r' not in p.read_bytes(), p.name                              # bash on the cluster


def test_runs_have_their_caps_and_inputs():
    small = (P3 / 'p3_beltrami_small.slurm').read_text()
    assert small.count('capped 1h python $B run --size B1') == 5
    assert '--tag _rerun' in small and '--pin-scale 1e-3' in small and '--pin-scale 1e3' in small
    assert 'p3_k0_k8.py k8' in small and 'package1_v2.0.0/package1' in small
    assert (P3 / 'p3_beltrami_b2.slurm').read_text().count('capped 2h') == 2
    assert 'capped 235m' in (P3 / 'p3_beltrami_b3_l1.slurm').read_text()        # inside the 4 h walltime
    assert 'capped 415m' in (P3 / 'p3_beltrami_b3_l2.slurm').read_text()        # inside the 7 h walltime
    assert '--with-checks' in (P3 / 'p3_bl.slurm').read_text()
    pre = (P3 / 'p3_preflight.slurm').read_text()
    assert 'p3_checks.py k1' in pre and 'p3_checks.py k2' in pre and 'MISSING INPUT' in pre
    for job in COMPUTE:                                                           # every input a job reads is checked first
        for path in re.findall(r'"\$RESULTS/([^"]+)"', (P3 / f'{job}.slurm').read_text()):
            assert path in pre, (job, path)


def test_p3_is_a_stage_of_the_cluster_scripts():
    cluster = REPO / 'scripts' / 'cluster'
    assert 'p3)   export PKG="$RESULTS/package3"' in (cluster / 'env.sh').read_text()
    assert '|p3)$' in (cluster / 'sbatch.sh').read_text()
    assert 'LILQ_WAVE" == p3' in (cluster / 'submit_lib.sh').read_text()
    assert _module('stage_report', cluster / 'package2' / 'stage_report.py').STAGES['p3'] == 'package3'


def test_wave1_submits_every_job_once():
    text = SUBMIT.read_text()
    for job in COMPUTE | {'p3_preflight', 'p3_report'}:
        assert text.count(f'/{job}.slurm') == 1, job
    assert 'p3_beltrami_b3_l2.slurm' not in text                                  # wave 2
    l2 = SUBMIT_L2.read_text()
    assert l2.count('/p3_beltrami_b3_l2.slurm') == 1 and l2.count('/p3_report.slurm') == 1


def test_stage_report_packs_package3(tmp_path, monkeypatch):
    sr = _module('stage_report', REPO / 'scripts' / 'cluster' / 'package2' / 'stage_report.py')
    pkg = tmp_path / 'results' / 'package3' / 'P3_checks'
    pkg.mkdir(parents=True)
    (pkg / 'k1.json').write_text('{}')
    logs = tmp_path / 'logs'
    logs.mkdir()
    for name in ('lilq-p3-k0.301.out', 'lilq-p3-report.399.out', 'lilq-p2s2-bl.5.out'):
        (logs / name).write_text('log\n')
    monkeypatch.setenv('SLURM_JOB_ID', '399')
    monkeypatch.delenv('CLUSTER', raising=False)
    tar_path = sr.report('p3', tmp_path / 'results', logs_dir=logs, sacct=False)
    assert tar_path.name == 'package3.tar.gz'
    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert {'package3/P3_checks/k1.json', 'package3/slurm_logs/lilq-p3-k0.301.out'} <= names
    assert not any('report.399' in n or 'p2s2' in n for n in names)


# ---------------------------------------------------------------- B3 level 2's gate

WAVE1 = {'301': 'lilq-p3-preflight', '302': 'lilq-p3-k0', '303': 'lilq-p3-beltrami-small',
         '304': 'lilq-p3-beltrami-b2', '305': 'lilq-p3-beltrami-b3-l1', '306': 'lilq-p3-bl',
         '307': 'lilq-p3-affine', '308': 'lilq-p3-burgers-large-p', '309': 'lilq-p3-report'}
HOURS = {'301': 0.2, '302': 1.2, '303': 0.6, '304': 1.1, '305': 2.6, '306': 1.3, '307': 1.0, '308': 0.25, '309': 0.02}


def _wave1_done(root, k0=True, l1_status='complete', s_per_it=1000.0, kappa_s=1500.0, hours=None):
    """A results tree and logs as wave 1 leaves them; the sacct text it would give."""
    pkg = root / 'results' / 'package3'
    (pkg / 'P3_checks').mkdir(parents=True)
    if k0 is not None:
        (pkg / 'P3_checks' / 'k0.json').write_text(json.dumps({'check': 'K0', 'passed': k0}))
    if l1_status is not None:
        (pkg / 'P3_1_beltrami_certified' / 'B3_L1').mkdir(parents=True)
        p3 = {'status': l1_status, 'size': 'B3', 'level': 1}
        if l1_status == 'complete':
            p3.update(seconds_per_iteration=s_per_it, kappa_time_s=kappa_s)
        (pkg / 'P3_1_beltrami_certified' / 'B3_L1' / 'run.json').write_text(json.dumps({'package3': p3}))
    logs = root / 'logs'
    logs.mkdir()
    for job, name in WAVE1.items():
        (logs / f'{name}.{job}.out').write_text('log\n')
    hours = hours or HOURS
    rate = lambda name: 24 if name in ('lilq-p3-preflight', 'lilq-p3-bl', 'lilq-p3-report') else 48  # noqa: E731
    sacct = ''.join(f'{job}|{name}|billing={rate(name)},cpu={rate(name)},mem=1G,node=1|{round(hours[job] * 3600)}|COMPLETED\n'
                    for job, name in WAVE1.items())
    return root / 'results', logs, sacct


def _gate():
    return _module('b3l2_gate', P3 / 'b3l2_gate.py')


def test_gate_launches_when_all_three_hold(tmp_path):
    results, logs, sacct = _wave1_done(tmp_path)
    code, lines = _gate().decide(results, logs, queue=lambda: [], accounting=lambda ids: sacct)
    su = sum((24 if WAVE1[j] in ('lilq-p3-preflight', 'lilq-p3-bl', 'lilq-p3-report') else 48) * h for j, h in HOURS.items())
    assert code == 0, lines
    assert any('K0: passed' in l for l in lines)
    assert any('= 5.04 h, < 6.5 h' in l for l in lines)                        # 1.91 x (8,000 + 1,500) s
    assert any(f'budget: {su:.1f} SU charged by 9 Package 3 jobs + 336' in l for l in lines)


@pytest.mark.parametrize('case, code, words', [
    ('rule', 1, 'not < 6.5 h'),
    ('budget', 1, 'over 850'),
    ('k0_failed', 1, 'K0: FAILED'),
    ('k0_missing', 1, 'K0 did not finish'),
    ('l1_capped', 1, "did not complete (status 'running'"),
    ('l1_missing', 1, 'B3 level 1 did not run'),
    ('queued', 2, 'wave 1 is not over'),
    ('l2_queued', 1, 'queued already'),
    ('l2_ran', 1, 'has run already'),
    ('no_squeue', 2, 'squeue could not be read'),
    ('no_sacct', 2, 'sacct could not be read'),
    ('sacct_short', 2, 'no record of job(s) 309'),
    ('sacct_running', 2, 'still shows job(s) 305 running'),
    ('nothing_ran', 2, 'no Package 3 job has run yet'),
])
def test_gate_refuses_or_waits(tmp_path, case, code, words):
    kw = {'rule': dict(s_per_it=1500.0, kappa_s=1000.0),                          # 1.91 x 13,000 s = 6.90 h
          'budget': dict(hours={**HOURS, '305': 4.0, '303': 3.0, '304': 2.0}),  # 586 SU charged, over 514
          'k0_failed': dict(k0=False), 'k0_missing': dict(k0=None),
          'l1_capped': dict(l1_status='running'), 'l1_missing': dict(l1_status=None)}.get(case, {})
    results, logs, sacct = _wave1_done(tmp_path, **kw)
    queue, accounting = (lambda: [('309', 'lilq-p3-report', 'RUNNING')]), (lambda ids: sacct)   # wave 1's report may still run
    if case == 'queued':
        queue = lambda: [('305', 'lilq-p3-beltrami-b3-l1', 'RUNNING'), ('400', 'other-job', 'PENDING')]  # noqa: E731
    elif case == 'l2_queued':
        queue = lambda: [('410', 'lilq-p3-beltrami-b3-l2', 'PENDING')]           # noqa: E731
    elif case == 'l2_ran':
        (logs / 'lilq-p3-beltrami-b3-l2.410.out').write_text('log\n')
    elif case == 'no_squeue':
        queue = lambda: None                                                      # noqa: E731
    elif case == 'no_sacct':
        accounting = lambda ids: None                                             # noqa: E731
    elif case == 'sacct_short':
        accounting = lambda ids: ''.join(l + '\n' for l in sacct.splitlines() if not l.startswith('309|'))  # noqa: E731
    elif case == 'sacct_running':
        accounting = lambda ids: sacct.replace('|9360|COMPLETED', '|9360|RUNNING')   # noqa: E731  (job 305, 2.6 h)
    elif case == 'nothing_ran':
        for p in logs.iterdir():
            p.unlink()
    got, lines = _gate().decide(results, logs, queue=queue, accounting=accounting)
    assert got == code and any(words in l for l in lines), (got, lines)


def test_gate_budget_is_the_charge_plus_336(tmp_path):
    gate = _gate()
    results, logs, _ = _wave1_done(tmp_path)
    for charged, code in ((514.0, 0), (514.1, 1)):                               # 514 + 336 = 850
        sacct = ''.join(f'{j}|{n}|cpu=48|0|COMPLETED\n' for j, n in WAVE1.items() if j != '305') + \
            f'305|lilq-p3-beltrami-b3-l1|cpu=48|{round(charged / 48 * 3600)}|COMPLETED\n'
        got, lines = gate.decide(results, logs, queue=lambda: [], accounting=lambda ids: sacct)
        assert got == code, (charged, lines)


def test_gate_reads_squeue_and_sacct(tmp_path, monkeypatch):
    gate = _gate()
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = '305|lilq-p3-beltrami-b3-l1|RUNNING\n' if cmd[0] == 'squeue' else '302|lilq-p3-k0|cpu=48|3600|COMPLETED\n'
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr='')
    monkeypatch.setattr(gate.subprocess, 'run', fake_run)
    monkeypatch.setenv('USER', 'someone')
    assert gate.squeue() == [('305', 'lilq-p3-beltrami-b3-l1', 'RUNNING')]
    assert calls[0][:4] == ['squeue', '-h', '-u', 'someone']
    assert gate.sacct(['302', '303']) == '302|lilq-p3-k0|cpu=48|3600|COMPLETED\n'
    assert '302,303' in calls[1] and 'JobID,JobName,AllocTRES,ElapsedRaw,State' in calls[1]

    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(gate.subprocess, 'run', missing)
    assert gate.squeue() is None and gate.sacct(['1']) is None


# ---------------------------------------------------------------- the submission scripts (bash)

def _fake_cluster(tmp_path, dry, squeue_out='', sacct_out=''):
    """sbatch (as Package 2's tests), squeue, sacct and a `python` that is this one."""
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    if dry:
        (bin_ / 'sbatch').write_text('#!/bin/bash\nfor a in "$@"; do [[ "$a" == --test-only ]] && { echo "$*" >> '
                                     '"$(dirname "$0")/log"; echo "sbatch: Job 1 to start at soon" >&2; exit 0; }; done\nexit 1\n')
    else:
        (bin_ / 'sbatch').write_text('#!/bin/bash\nn=$(cat "$(dirname "$0")/n" 2>/dev/null || echo 100); '
                                     'echo $((n+1)) > "$(dirname "$0")/n"\necho "$*" >> "$(dirname "$0")/log"; echo $((n+1))\n')
    (bin_ / 'squeue').write_text(f"#!/bin/bash\nprintf '%s' '{squeue_out}'\n")
    (bin_ / 'sacct').write_text(f"#!/bin/bash\nprintf '%s' '{sacct_out}'\n")
    (bin_ / 'python').write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    for p in bin_.iterdir():
        p.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", LILQ_ACCOUNT='000000000000', YES='1',
               LILQ_RESULTS=str(tmp_path / 'results'), LILQ_LOGS=str(tmp_path / 'logs'), **({'DRY_RUN': '1'} if dry else {}))
    env.pop('CLUSTER', None)
    return env, bin_ / 'log'


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_wave1_dry_run_checks_all_9_jobs_and_submits_nothing(tmp_path):
    env, log = _fake_cluster(tmp_path, True)
    out = subprocess.run(['bash', str(SUBMIT)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'Package 3 (results/package3)' in out.stdout and '828 SU requested' in out.stdout
    lines = log.read_text().splitlines()
    assert len(lines) == 9 and all('--account=000000000000' in l and '--test-only' in l for l in lines)
    assert not any('--dependency' in l for l in lines)
    for l in lines:
        job = re.search(r'(p3_[A-Za-z0-9_]+)\.slurm', l).group(1)
        assert ('--exclusive' in l) == (CLASS[job][0] == 'timed-cpu'), l


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_wave1_chains_the_jobs(tmp_path):
    env, log = _fake_cluster(tmp_path, False)
    out = subprocess.run(['bash', str(SUBMIT)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 9
    ids = {re.search(r'(p3_[A-Za-z0-9_]+)\.slurm', l).group(1): 101 + i for i, l in enumerate(lines)}
    assert set(ids) == COMPUTE | {'p3_preflight', 'p3_report'}
    assert '--dependency' not in lines[0] and 'p3_preflight' in lines[0]
    for l in lines[1:-1]:
        assert f"--dependency=afterok:{ids['p3_preflight']} " in l, l
    assert 'p3_report' in lines[-1] and 'afterany' in lines[-1]
    assert all(f':{ids[j]}' in lines[-1] for j in COMPUTE) and f":{ids['p3_preflight']}" not in lines[-1]
    assert 'submit_p3_b3l2.sh' in out.stdout


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_wave2_submits_b3_level_2_when_the_gate_says_so(tmp_path):
    _, _, sacct = _wave1_done(tmp_path)
    env, log = _fake_cluster(tmp_path, False, sacct_out=sacct)
    out = subprocess.run(['bash', str(SUBMIT_L2)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'LAUNCH B3 level 2' in out.stdout, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and 'p3_beltrami_b3_l2.slurm' in lines[0] and '--dependency' not in lines[0]
    assert '--exclusive' in lines[0] and 'p3_report.slurm' in lines[1] and '--dependency=afterany:101 ' in lines[1]


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
@pytest.mark.parametrize('case, code', [('rule', 1), ('wait', 2)])
def test_wave2_submits_nothing_when_the_gate_refuses(tmp_path, case, code):
    _, _, sacct = _wave1_done(tmp_path, **({'s_per_it': 1500.0, 'kappa_s': 1000.0} if case == 'rule' else {}))
    queued = '305|lilq-p3-beltrami-b3-l1|RUNNING\n' if case == 'wait' else ''
    env, log = _fake_cluster(tmp_path, False, squeue_out=queued, sacct_out=sacct)
    out = subprocess.run(['bash', str(SUBMIT_L2)], env=env, capture_output=True, text=True)
    assert out.returncode == code and 'Nothing submitted.' in out.stdout, out.stdout + out.stderr
    assert not log.exists()


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_wave2_dry_run_checks_both_jobs_whatever_the_gate_says(tmp_path):
    env, log = _fake_cluster(tmp_path, True)                                     # nothing has run: the gate says wait
    out = subprocess.run(['bash', str(SUBMIT_L2)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'NOT YET' in out.stdout and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and 'p3_beltrami_b3_l2.slurm' in lines[0] and 'p3_report.slurm' in lines[1]
