"""Package 5's submission (items 1, 2 and 4): the job scripts (as generated, LF), their classes and
the SU plan, the inputs the preflight checks, the p5 stage in the cluster scripts and the report,
and the submission script (bash): every job once, in order, with its dependencies, and the dry run."""

import importlib.util
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
P5 = REPO / 'scripts' / 'cluster' / 'package5'
SUBMIT = P5 / 'submit_p5.sh'
RETIME = P5 / 'submit_p5_retime.sh'
CLASS = {'p5_preflight': ('cpu-mem', '00:30:00'), 'p5_ref_viscous': ('cpu-mem', '02:00:00'),
         'p5_ref_gravity': ('cpu-mem', '02:00:00'), 'p5_item1': ('cpu-mem', '01:15:00'),
         'p5_beltrami': ('cpu-mem', '03:00:00'), 'p5_timed': ('timed-cpu', '01:00:00'),
         'p5_item4': ('cpu-mem', '02:30:00'), 'p5_report': ('cpu', '00:30:00')}
AFTER_PREFLIGHT = {'p5_ref_viscous', 'p5_ref_gravity', 'p5_beltrami', 'p5_timed', 'p5_item4'}
RATE = {'cpu': 24, 'cpu-mem': 24, 'timed-cpu': 48}        # SU per hour on Grace (cpu-mem: checked by the preflight)
REQUESTED = 330


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hours(wall):
    h, m, s = map(int, wall.split(':'))
    return h + m / 60 + s / 3600


def _text(name):
    return (P5 / f'{name}.slurm').read_text()


def test_job_scripts_their_classes_and_the_su_plan():
    assert {p.stem for p in P5.glob('*.slurm')} == set(CLASS) | {'p5_retime'}
    for name, (cls, wall) in CLASS.items():
        t = _text(name)
        assert re.search(rf'^# lilq-resources: {cls} ', t, re.M), name
        assert f'#SBATCH --time={wall}\n' in t, name
        assert re.search(r'#SBATCH --job-name=lilq-p5-[a-z0-9-]+\n', t), name     # the report finds the logs by it
        assert 'source scripts/cluster/env.sh\nset -euo pipefail\n' in t, name
    requested = sum(RATE[c] * _hours(w) for c, w in CLASS.values())
    assert requested == pytest.approx(REQUESTED)
    assert f'{REQUESTED} SU requested' in SUBMIT.read_text()


def test_job_scripts_are_as_generated_and_lf(tmp_path):
    gen = _module('p5_generate_jobs', P5 / '_generate_jobs.py')
    gen.HERE = tmp_path
    gen.main()
    for p in tmp_path.glob('*.slurm'):
        assert p.read_bytes() == (P5 / p.name).read_bytes(), p.name
    for p in P5.iterdir():
        assert not p.is_file() or b'\r' not in p.read_bytes(), p.name                              # bash on the cluster


def test_every_run_is_in_a_job_and_every_input_is_checked():
    text = {name: _text(name) for name in CLASS}
    everything = ''.join(text.values())
    # item 1: both references with the advisor's sizes and the fallback choice, the six families, Beltrami
    for case in ('viscous', 'gravity'):
        t = text[f'p5_ref_{case}']
        assert 'ref 40m 80\n' in t and 'ref 60m 88\n' in t
        assert f'ref-ok --case {case} --p-ref 80 ' in t and f'choose --case {case} --p-ref 88 ' in t
    assert 'for family in bratu burgers bl_cheb_viscous bl_cheb_gravity bl_sine_viscous bl_sine_gravity; do' in \
        text['p5_item1'] and 'p5_1_weights.py runs --only "$family"' in text['p5_item1']
    assert 'for size in B1 B2; do\n    for lam in 1e3 1e4 1e6; do' in text['p5_beltrami']
    assert '--lambda-bc "$lam" --lambda-ic "$lam"' in text['p5_beltrami'] and '--tag "_lam$lam"' in text['p5_beltrami']
    assert '--level 1 ' in text['p5_beltrami'] and '--pin-scale' not in text['p5_beltrami']      # pin_scale 1
    # item 2 and the 4a timings on the whole node
    assert 'p5_2_nlls.py run --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference" --points data/package4' in \
        text['p5_timed']
    assert 'p5_4_pb.py time --out "$PKG" --points data/package5' in text['p5_timed']
    assert 'p4_checks.py provenance --out "$PKG" --name timed' in text['p5_timed']
    # item 4: each seed's 4a and 2c, 4b up to 800 and at 1,600 by sigma, delta_P of every space
    t = text['p5_item4']
    for seed in (0, 1, 2):
        assert f'lilq --seeds {seed} ' in t and f'under --seeds {seed} ' in t
    assert 'rf --P 100 200 400 800 ' in t and 'for sigma in 2 4 8; do' in t and 'rf --P 1600 --sigma "$sigma"' in t
    assert 'delta --which p4 4a ' in t and 'delta --which 4b ' in t
    assert t.count('& pids+=($!)') == 4 and 'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4' in t
    # the report: every summary, then the tarball
    rep = text['p5_report']
    for driver in ('p5_1_weights.py', 'p3_1_beltrami_certified.py', 'p5_2_nlls.py', 'p5_4_pb.py'):
        assert f'{driver} summarize ' in rep
    assert 'stage_report.py --stage p5' in rep
    # every input a job reads is checked by the preflight, and is in the bundle where shipped
    pre = text['p5_preflight']
    assert 'MISSING INPUT' in pre and 'pytest' in pre and 'compare-points' in pre
    for job, tt in text.items():
        for path in re.findall(r'"\$RESULTS/([^"]+)"', tt):
            assert path in pre, (job, path)
    for k in (4, 5):
        for seed in (0, 1, 2):
            assert f'data/package{k}/halton_seed{seed}.npz' in pre
            assert (REPO / 'data' / f'package{k}' / f'halton_seed{seed}.npz').exists()
    assert everything.count('python scripts/cluster/package2/stage_report.py') == 1


def test_p5_is_a_stage_of_the_cluster_scripts():
    cluster = REPO / 'scripts' / 'cluster'
    assert 'p5)   export PKG="$RESULTS/package5"' in (cluster / 'env.sh').read_text()
    sb = (cluster / 'sbatch.sh').read_text()
    assert '|p3|p4|p5|p5t)$' in sb
    assert 'cpu-mem)' in sb and '--cpus-per-task=24 --mem=96G' in sb
    assert 'LILQ_WAVE" == p5' in (cluster / 'submit_lib.sh').read_text()
    assert _module('stage_report', cluster / 'package2' / 'stage_report.py').STAGES['p5'] == 'package5'


def test_stage_report_packs_package5(tmp_path, monkeypatch):
    sr = _module('stage_report', REPO / 'scripts' / 'cluster' / 'package2' / 'stage_report.py')
    pkg = tmp_path / 'results' / 'package5' / 'P5_1_weights'
    pkg.mkdir(parents=True)
    (pkg / 'terminal.csv').write_text('family\n')
    logs = tmp_path / 'logs'
    logs.mkdir()
    for name in ('lilq-p5-timed.501.out', 'lilq-p5-report.599.out', 'lilq-p4-item1.401.out'):
        (logs / name).write_text('log\n')
    monkeypatch.setenv('SLURM_JOB_ID', '599')
    monkeypatch.delenv('CLUSTER', raising=False)
    tar_path = sr.report('p5', tmp_path / 'results', logs_dir=logs, sacct=False)
    assert tar_path.name == 'package5.tar.gz'
    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert {'package5/P5_1_weights/terminal.csv', 'package5/slurm_logs/lilq-p5-timed.501.out'} <= names
    assert not any('report.599' in n or 'p4' in n for n in names)


# ---------------------------------------------------------------- the submission script (bash)

def _fake_cluster(tmp_path, dry):
    """sbatch (as Package 4's tests) and a `python` that is this one."""
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
    return re.search(r'(p5_[A-Za-z0-9_]+)\.slurm', line).group(1)


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_dry_run_checks_all_8_jobs_and_submits_nothing(tmp_path):
    env, log = _fake_cluster(tmp_path, True)
    out = subprocess.run(['bash', str(SUBMIT)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'Package 5, items 1, 2 and 4 (results/package5)' in out.stdout and f'{REQUESTED} SU requested' in out.stdout
    lines = log.read_text().splitlines()
    assert len(lines) == 8 and all('--account=000000000000' in l and '--test-only' in l for l in lines)
    assert not any('--dependency' in l for l in lines)
    for l in lines:
        cls = CLASS[_job(l)][0]
        assert ('--exclusive' in l) == (cls == 'timed-cpu'), l
        assert ('--mem=96G' in l) == (cls == 'cpu-mem'), l


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_submission_chains_the_jobs(tmp_path):
    env, log = _fake_cluster(tmp_path, False)
    out = subprocess.run(['bash', str(SUBMIT)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 8
    ids = {_job(l): 101 + i for i, l in enumerate(lines)}
    assert set(ids) == set(CLASS)
    by_job = {_job(l): l for l in lines}
    assert '--dependency' not in lines[0] and _job(lines[0]) == 'p5_preflight'
    for job in AFTER_PREFLIGHT:
        assert f"--dependency=afterok:{ids['p5_preflight']} " in by_job[job], job
    assert (f"--dependency=afterok:{ids['p5_preflight']},afterany:{ids['p5_ref_viscous']}:{ids['p5_ref_gravity']} "
            in by_job['p5_item1'])
    rep = by_job['p5_report']
    assert _job(lines[-1]) == 'p5_report' and 'afterany' in rep
    assert all(f':{ids[j]}' in rep for j in AFTER_PREFLIGHT | {'p5_item1'})
    assert f":{ids['p5_preflight']}" not in rep


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash job script')
def test_the_reference_falls_back_when_p88_hits_its_limit(tmp_path):
    """p5_ref_viscous with a stand-in `python`: p = 80 stops by the rule, p = 88 is stopped at its
    limit (timeout's exit 124), so `choose` runs (and decides) and the job goes on."""
    t = _text('p5_ref_viscous')
    body = t.split('set -euo pipefail\n', 1)[1]
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    (bin_ / 'python').write_text('#!/bin/bash\necho "python $*" >> "$LOG"\n'
                                 'case "$*" in *"reference --case viscous --p-ref 88"*) exit 124;; esac\nexit 0\n')
    (bin_ / 'python').chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", LOG=str(tmp_path / 'log'), PKG='pkg', RESULTS='res')
    out = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + body], env=env, capture_output=True, text=True,
                         timeout=30)
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'REFERENCE viscous p = 88: STOPPED OR FAILED (exit 124)' in out.stdout
    calls = (tmp_path / 'log').read_text().splitlines()
    assert [c.split()[2] for c in calls] == ['reference', 'ref-ok', 'reference', 'choose']


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash job script')
def test_the_preflight_stops_if_cpu_mem_is_billed_above_24(tmp_path):
    t = _text('p5_preflight')
    lines = t.split('set -euo pipefail\n', 1)[1].splitlines()
    guard = '\n'.join(l for l in lines if 'billing' in l)
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    for billing, ok in (('24', True), ('30', False)):
        (bin_ / 'scontrol').write_text(f'#!/bin/bash\necho "TRES=cpu=24,mem=96G,node=1,billing={billing}"\n')
        (bin_ / 'scontrol').chmod(0o755)
        env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", SLURM_JOB_ID='1')
        out = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + guard + '\necho PASSED'], env=env,
                             capture_output=True, text=True, timeout=30)
        assert ('PASSED' in out.stdout) == ok, out.stdout + out.stderr
        assert ok or 'NOT 24: STOPPING' in out.stdout


# ---------------------------------------------------------------- the re-time (stage p5t)

def test_the_retime_job_runs_the_one_run_on_a_whole_node():
    t = _text('p5_retime')
    assert re.search(r'^# lilq-resources: timed-cpu ', t, re.M) and '#SBATCH --time=00:15:00\n' in t
    assert '#SBATCH --job-name=lilq-p5t-retime\n' in t
    body = t.split('source scripts/cluster/env.sh\nset -euo pipefail\n', 1)[1]
    runs = [l for l in body.splitlines() if l.startswith('python ')]
    assert runs == ['python experiments/p4_checks.py provenance --out "$PKG" --name timed',
                    'python experiments/p5_2_nlls.py run --out "$PKG" --reference-dir '
                    '"$RESULTS/package2_stage1/reference" --points data/package4 --problem burgers --p 10 '
                    '--solver lbfgs',
                    'python scripts/cluster/package2/stage_report.py --stage p5t --results "$RESULTS"']
    assert 48 * _hours('00:15:00') == 12 and '12 SU requested' in RETIME.read_text()
    cluster = REPO / 'scripts' / 'cluster'
    assert 'p5t)  export PKG="$RESULTS/package5_retime"' in (cluster / 'env.sh').read_text()
    assert _module('stage_report', cluster / 'package2' / 'stage_report.py').STAGES['p5t'] == 'package5_retime'


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_the_retime_dry_run_checks_its_one_job(tmp_path):
    env, log = _fake_cluster(tmp_path, True)
    out = subprocess.run(['bash', str(RETIME)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'results/package5_retime' in out.stdout and '12 SU requested' in out.stdout
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and _job(lines[0]) == 'p5_retime'
    assert '--exclusive' in lines[0] and '--test-only' in lines[0] and '--dependency' not in lines[0]


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_the_retime_submits_its_one_job(tmp_path):
    env, log = _fake_cluster(tmp_path, False)
    out = subprocess.run(['bash', str(RETIME)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and _job(lines[0]) == 'p5_retime' and '--account=000000000000' in lines[0]
