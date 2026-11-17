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
A2 = P5 / 'submit_p5_a2.sh'
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
    assert {p.stem for p in P5.glob('p5_*.slurm')} == set(CLASS) | {'p5_retime', 'p5_a2'}
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
    assert '|p3|p4|p5|p5t|p5a|p5c|p5g|p5o)$' in sb
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
    return re.search(r'(p5[a-z]?_[A-Za-z0-9_]+)\.slurm', line).group(1)


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


# ---------------------------------------------------------------- Addendum 2 (stage p5a)

def test_the_a2_job_runs_parts_a_to_c_at_4_threads_and_d_at_24():
    t = _text('p5_a2')
    assert re.search(r'^# lilq-resources: cpu-mem ', t, re.M) and '#SBATCH --time=01:00:00\n' in t
    assert '#SBATCH --job-name=lilq-p5a-a2\n' in t
    body = t.split('source scripts/cluster/env.sh\nset -euo pipefail\n', 1)[1]
    four, d = body.index('export OMP_NUM_THREADS=4 '), body.index('export OMP_NUM_THREADS=24 ')
    a2 = 'python experiments/p5_a2_plateau.py '
    before = [l for l in body[four:d].splitlines() if a2 in l]
    assert len(before) == 6 and all(l.endswith('& pids+=($!)') for l in before)
    parts = [re.search(r'run --part ([A-D])(?: --space (\w+))?', l) for l in before[:5]]
    assert [(m.group(1), m.group(2)) for m in parts] == [('A', 'cheb'), ('A', 'sine'), ('B', 'cheb'), ('B', 'sine'), ('C', None)]
    assert a2 + 'k1 ' in before[5]
    after = [l for l in body[d:].splitlines() if l.startswith(('until_end python', 'python'))]
    assert after[0] == 'until_end ' + a2 + 'run --part D "${A2[@]}"'
    assert after[1].startswith(a2 + 'summarize ') and 'P2_9_bases/bl_gravity_paper_P1024_ic' in after[1]
    assert after[2] == 'python scripts/cluster/package2/stage_report.py --stage p5a --results "$RESULTS"'
    # every input is checked before anything runs
    for f in ('bl_fd_ref_gravity_nu0.05.npz', 'bl_fd_ref_gravity_nu0.02.npz', 'chosen_gravity.json',
              'gravity_p88/beta_reference.npy', 'bl_sine_gravity/P1024_lam10000/errors_by_iterate.csv'):
        assert f in body[:four], f
    assert 24 * _hours('01:00:00') == 24 and '24 SU requested' in A2.read_text()
    cluster = REPO / 'scripts' / 'cluster'
    assert 'p5a)  export PKG="$RESULTS/package5_a2"' in (cluster / 'env.sh').read_text()
    assert _module('stage_report', cluster / 'package2' / 'stage_report.py').STAGES['p5a'] == 'package5_a2'
    assert 'LILQ_WAVE" == p5a' in (cluster / 'submit_lib.sh').read_text()


def test_the_a2_job_covers_the_18_runs():
    import experiments.p5_a2_plateau as a2
    groups = [('A', 'cheb'), ('A', 'sine'), ('B', 'cheb'), ('B', 'sine'), ('C', None), ('D', None)]
    covered = [a2.run_name(*r) for r in a2.runs() for p, s in groups if r[0] == p and (s is None or r[1] == s)]
    assert sorted(covered) == sorted(a2.run_name(*r) for r in a2.runs())


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_the_a2_dry_run_checks_its_one_job(tmp_path):
    env, log = _fake_cluster(tmp_path, True)
    out = subprocess.run(['bash', str(A2)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'results/package5_a2' in out.stdout and '24 SU requested' in out.stdout
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and _job(lines[0]) == 'p5_a2'
    assert '--cpus-per-task=24 --mem=96G' in lines[0] and '--test-only' in lines[0]


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_the_a2_submits_its_one_job(tmp_path):
    env, log = _fake_cluster(tmp_path, False)
    out = subprocess.run(['bash', str(A2)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and _job(lines[0]) == 'p5_a2' and '--account=000000000000' in lines[0]


# ---------------------------------------------------------------- item 3 (stages p5c and p5g)

CPU3 = P5 / 'submit_p5_item3_cpu.sh'
GPU3 = P5 / 'submit_p5_item3_gpu.sh'
CPU3_JOBS = {'p5c_preflight', 'p5c_protocol', 'p5c_bratu', 'p5c_burgers_a', 'p5c_burgers_b', 'p5c_bl_a', 'p5c_bl_b',
             'p5c_report'}
GPU3_JOBS = {'p5g_preflight', 'p5g_bratu', 'p5g_burgers', 'p5g_bl', 'p5g_report'}
RATE3 = {'cpu': 24, 'timed-cpu': 48, 'shared-gpu': 80, 'gpu-pair': 192}    # SU per hour on Grace


def _gen():
    return _module('p5_generate_jobs', P5 / '_generate_jobs.py')


def _body(name):
    return _text(name).split('source scripts/cluster/env.sh\nset -euo pipefail\n', 1)[1]


def _calls(name, driver):
    """(the driver's arguments, the CUDA device or None) of each call of a driver in a job."""
    out, device = [], None
    for line in _body(name).replace('\\\n', ' ').splitlines():
        m = re.search(r'CUDA_VISIBLE_DEVICES=(\d)', line)
        if m:
            device = int(m.group(1))
        if f'python {driver} ' in line:
            out.append((line.split(f'python {driver} ', 1)[1], device))
    return out


def _arg(args, flag):
    """The values after ``flag`` (up to the next option or quoted word)."""
    words = args.split()
    if flag not in words:
        return []
    out = []
    for w in words[words.index(flag) + 1:]:
        if w.startswith(('-', '"${')):
            break
        out.append(w)
    return out


def test_the_generator_has_the_drivers_sizes_methods_and_seeds():
    import experiments.p5_3_lil as lil
    import experiments.p5_3_networks as net
    gen = _gen()
    for bench in gen.BENCHES:
        assert tuple(net.sizes(bench)) == gen.NET_SIZES[bench]
        assert tuple(lil.SIZES[bench]) == gen.LIL_SIZES[bench]
        runs = [(P, m, s) for P in net.sizes(bench) for m in net.METHODS for s in net.SEEDS]
        assert gen.net_runs(bench) == runs
        for i, n in ((0, 1), (0, 2), (1, 2)):
            assert gen.part(runs, i, n) == net.part_of(runs, f'{i}/{n}')
    assert gen.NET_METHODS == net.METHODS and gen.NET_SEEDS == net.SEEDS


@pytest.mark.parametrize('stage', ['p5c', 'p5g'])
def test_every_budget_run_is_in_one_job_and_every_process_has_its_own_records(stage):
    import experiments.p5_3_networks as net
    gen = _gen()
    jobs = sorted(p.stem for p in P5.glob(f'{stage}_*.slurm'))
    assert set(jobs) == (CPU3_JOBS if stage == 'p5c' else GPU3_JOBS)
    device = 'cpu' if stage == 'p5c' else 'cuda'
    nets, lils = {b: [] for b in gen.BENCHES}, {b: [] for b in gen.BENCHES}
    records = {'networks': [], 'lil': []}        # (bench, tag): one warm-up and provenance record each
    for job in jobs:
        for args, _ in _calls(job, 'experiments/p5_3_networks.py'):
            if args.startswith(('pairing ', 'gate ')):
                continue                                  # the comparisons: write checks/*.json only
            bench = _arg(args, '--bench')[0]
            assert f'--device {device}' in args or '"${NET[@]}"' in args
            records['networks'].append((bench, _arg(args, '--tag')[0]))
            if args.startswith('run '):
                nets[bench] += net.part_of(gen.net_runs(bench), _arg(args, '--part')[0])
        for args, _ in _calls(job, 'experiments/p5_3_lil.py'):
            bench = _arg(args, '--bench')[0]
            records['lil'].append((bench, _arg(args, '--tag')[0]))
            if _arg(args, '--methods') == ['lbfgs']:
                lils[bench] += [int(p) for p in _arg(args, '--P')]
            else:
                assert stage == 'p5c' and _arg(args, '--methods') == ['lilq', 'trf', 'square']
    for bench in gen.BENCHES:
        assert sorted(nets[bench]) == sorted(gen.net_runs(bench)), bench          # each run once
        assert sorted(lils[bench]) == sorted(gen.LIL_SIZES[bench]), bench
    # the protocol and checks loops call each driver once per problem ("$bench")
    for driver, rec in records.items():
        expanded = [(b2, t) for b, t in rec for b2 in (gen.BENCHES if b == '"$bench"' else (b,))]
        assert len(expanded) == len(set(expanded)), (driver, expanded)
    names = [re.search(r'provenance --out "\$PKG" --name (\S+)', _text(j)) for j in jobs]
    names = [m.group(1) for m in names if m]
    assert len(names) == len(set(names))


def test_the_protocol_job_runs_the_lil_methods_and_c2_c4_on_the_three_problems():
    lil = _calls('p5c_protocol', 'experiments/p5_3_lil.py')
    assert len(lil) == 1 and '"$bench"' in lil[0][0] and '--device cpu' in lil[0][0]
    checks = _calls('p5c_protocol', 'experiments/p5_3_networks.py')
    assert len(checks) == 1 and checks[0][0].startswith('checks ') and '--device cpu' in checks[0][0]
    assert _body('p5c_protocol').count('for bench in bratu burgers bl; do') == 2


def test_the_pair_runs_one_process_per_gpu_and_the_solo_check_on_its_runs_gpu():
    import experiments.p5_3_networks as net
    gen = _gen()
    for bench in gen.BENCHES:
        name = f'p5g_{bench}'
        body = _body(name)
        assert 'export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24' in body
        assert '[[ "$gpus" == 2 ]]' in body
        calls = _calls(name, 'experiments/p5_3_networks.py')
        solo = [(a, d) for a, d in calls if a.startswith('solo ') and '--gate' not in a]
        gates = [(a, d) for a, d in calls if a.startswith('solo ') and '--gate' in a]
        decide = [a for a, _ in calls if a.startswith('gate ')]
        runs = [(a, d) for a, d in calls if a.startswith('run ')]
        assert len(solo) == 1 and len(runs) == 2 and len(gates) == 2 and len(decide) == 1
        assert body.index('solo --bench') < body.index('pids=()')                    # alone, before the pair
        args, gpu = solo[0]
        key = (int(_arg(args, '--P')[0]), _arg(args, '--method')[0], int(_arg(args, '--seed')[0]))
        assert key == (gen.NET_SIZES[bench][-1], 'nil_n', 0)
        # the pairing gate: the solo's run on both GPUs at once, judged on the solo's GPU, before the pair
        for a, d in gates:
            assert (int(_arg(a, '--P')[0]), _arg(a, '--method')[0], int(_arg(a, '--seed')[0])) == key
            assert _arg(a, '--gate') == [str(d)] and _arg(a, '--tag') == [f'gpu-{bench}-gate{d}']
        assert sorted(d for _, d in gates) == [0, 1]
        g = decide[0]
        assert (int(_arg(g, '--P')[0]), _arg(g, '--method')[0], int(_arg(g, '--seed')[0])) == key
        assert _arg(g, '--gpu') == [str(gpu)] and '--device cuda' in g
        lines = body.splitlines()
        stop = next(i for i, l in enumerate(lines) if 'STOPPING BEFORE THE BUDGET RUNS' in l)
        assert lines[stop].lstrip().startswith('|| {') and 'exit 1' in lines[stop]
        assert body.index('--gate 1') < body.index(' gate --bench') < body.index('run --bench')
        paired = [d for a, d in runs if key in net.part_of(gen.net_runs(bench), _arg(a, '--part')[0])]
        assert paired == [gpu]
        assert {d for _, d in runs} == {0, 1} and all(_arg(a, '--part')[0] == f'{d}/2' for a, d in runs)
        for a, d in _calls(name, 'experiments/p5_3_lil.py'):
            assert _arg(a, '--tag')[0] == f'gpu-{bench}-{d}'


def test_item3_classes_walltimes_and_the_su_requested():
    gen = _gen()
    for jobs, stage, script in ((gen.cpu_jobs(), 'p5c', CPU3), (gen.gpu_jobs(), 'p5g', GPU3)):
        requested = 0
        for name, (short, cls, wall, _, _) in jobs.items():
            t = (P5 / name).read_text()
            assert re.search(rf'^# lilq-resources: {cls} ', t, re.M) and f'#SBATCH --time={wall}\n' in t, name
            assert f'#SBATCH --job-name=lilq-{stage}-{short}\n' in t
            requested += RATE3[cls] * _hours(wall)
        assert f'{round(requested):,} SU requested' in script.read_text(), (stage, requested)
    # the budget jobs' walltimes cover their runs: each run's 600 s, its error log and a fixed part
    assert gen.run_wall_s('bratu', 'lm') == 600 * 1.22 + gen.FIXED_S
    assert gen.run_wall_s('bl', 'nil_n') == gen.run_wall_s('bl', 'lbfgs') == 600 * 1.02 + gen.FIXED_S
    for job in gen.CPU_SPLIT:
        wall = gen.cpu_jobs()[f'p5c_{job.replace("-", "_")}.slurm'][2]
        nets, lil = gen.cpu_runs(job)
        floor = sum(gen.run_wall_s(gen.CPU_SPLIT[job][0], m) for _, m, _ in nets) + len(lil) * (600 * 1.02 + gen.FIXED_S)
        assert _hours(wall) * 3600 >= floor * gen.MARGIN
    for bench in gen.BENCHES:
        wall = gen.gpu_jobs()[f'p5g_{bench}.slurm'][2]
        n = max(gen.gpu_slots(bench, 0), gen.gpu_slots(bench, 1))
        assert _hours(wall) * 3600 >= (n * (600 * 1.02 + gen.FIXED_S) + gen.SOLO_EXTRA_S + gen.GATE_EXTRA_S) * gen.MARGIN


def test_item3_reports_guard_each_step_and_pack_the_stage():
    for stage in ('p5c', 'p5g'):
        body = _body(f'{stage}_report')
        steps = [l for l in body.splitlines() if 'python ' in l]
        assert steps and all(l.startswith('timeout ') for l in steps), steps
        assert 'experiments/p5_3_table.py --out "$PKG"' in body
        assert f'stage_report.py --stage {stage} --results "$RESULTS"' in body
        assert ('p5_3_networks.py pairing' in body) == (stage == 'p5g')
    cluster = REPO / 'scripts' / 'cluster'
    env = (cluster / 'env.sh').read_text()
    assert 'p5c)  export PKG="$RESULTS/package5_item3_cpu"' in env and 'p5g)  export PKG="$RESULTS/package5_item3_gpu"' in env
    stages = _module('stage_report', cluster / 'package2' / 'stage_report.py').STAGES
    assert stages['p5c'] == 'package5_item3_cpu' and stages['p5g'] == 'package5_item3_gpu'
    lib = (cluster / 'submit_lib.sh').read_text()
    assert 'LILQ_WAVE" == p5c' in lib and 'LILQ_WAVE" == p5g' in lib
    assert 'GPU_PAIR_GRES=gpu:a100:2' in (cluster / 'profiles' / 'grace.sh').read_text()


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
@pytest.mark.parametrize('which', ['cpu', 'gpu'])
def test_item3_dry_run_checks_every_job_with_its_resources(tmp_path, which):
    script, jobs = (CPU3, CPU3_JOBS) if which == 'cpu' else (GPU3, GPU3_JOBS)
    env, log = _fake_cluster(tmp_path, True)
    out = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert sorted(_job(l) for l in lines) == sorted(jobs) and not any('--dependency' in l for l in lines)
    gen = _gen()
    classes = {Path(n).stem: c for n, (_, c, _, _, _) in {**gen.cpu_jobs(), **gen.gpu_jobs()}.items()}
    for l in lines:
        cls = classes[_job(l)]
        assert ('--exclusive' in l) == (cls == 'timed-cpu'), l
        assert ('--gres=gpu:a100:2' in l) == (cls == 'gpu-pair'), l
        assert ('--gres=gpu:a100:1' in l) == (cls == 'shared-gpu'), l
        if cls == 'gpu-pair':
            assert '--cpus-per-task=48 --mem=360G' in l


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
@pytest.mark.parametrize('which', ['cpu', 'gpu'])
def test_item3_submission_chains_the_jobs(tmp_path, which):
    script, jobs = (CPU3, CPU3_JOBS) if which == 'cpu' else (GPU3, GPU3_JOBS)
    env, log = _fake_cluster(tmp_path, False)
    out = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    ids = {_job(l): 101 + i for i, l in enumerate(lines)}
    assert set(ids) == jobs
    stage = _job(lines[0]).split('_')[0]
    pre, rep = f'{stage}_preflight', f'{stage}_report'
    assert _job(lines[0]) == pre and '--dependency' not in lines[0] and _job(lines[-1]) == rep
    for l in lines[1:-1]:
        assert f'--dependency=afterok:{ids[pre]} ' in l, l
    middle = sorted(jobs - {pre, rep}, key=ids.get)
    assert f"--dependency=afterany:{':'.join(str(ids[j]) for j in middle)} " in lines[-1]


# ---------------------------------------------------------------- the OpenMP library diagnostic (stage p5o)

OPENMP = P5 / 'submit_p5_item3_openmp.sh'


def test_the_openmp_job_alternates_the_two_orders_on_a_whole_node():
    t = _text('p5o_openmp')
    assert re.search(r'^# lilq-resources: timed-cpu ', t, re.M) and '#SBATCH --time=00:30:00\n' in t
    assert '#SBATCH --job-name=lilq-p5o-openmp\n' in t
    runs = [(a, d) for a, d in _calls('p5o_openmp', 'experiments/p5_3_openmp.py')]
    order = [(int(_arg(a, '--round')[0]), _arg(a, '--first')[0]) for a, _ in runs]
    assert order == [(0, 'numpy'), (0, 'torch'), (1, 'torch'), (1, 'numpy'), (2, 'numpy'), (2, 'torch')]
    assert all('--require-library' in a and a.startswith('--first') for a, _ in runs)
    body = _body('p5o_openmp')
    assert all(l.startswith('until_end python experiments/p5_3_openmp.py') for l in body.splitlines()
               if 'p5_3_openmp.py' in l and not l.startswith('#'))
    assert 'stage_report.py --stage p5o --results "$RESULTS"' in body
    assert 48 * _hours('00:30:00') == 24 and '24 SU requested' in OPENMP.read_text()
    cluster = REPO / 'scripts' / 'cluster'
    assert 'p5o)  export PKG="$RESULTS/package5_item3_openmp"' in (cluster / 'env.sh').read_text()
    assert _module('stage_report', cluster / 'package2' / 'stage_report.py').STAGES['p5o'] == 'package5_item3_openmp'
    assert 'LILQ_WAVE" == p5o' in (cluster / 'submit_lib.sh').read_text()


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_the_openmp_dry_run_checks_its_one_job(tmp_path):
    env, log = _fake_cluster(tmp_path, True)
    out = subprocess.run(['bash', str(OPENMP)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'results/package5_item3_openmp' in out.stdout and '24 SU requested' in out.stdout
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and _job(lines[0]) == 'p5o_openmp'
    assert '--exclusive' in lines[0] and '--test-only' in lines[0] and '--dependency' not in lines[0]


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_the_openmp_diagnostic_submits_its_one_job(tmp_path):
    env, log = _fake_cluster(tmp_path, False)
    out = subprocess.run(['bash', str(OPENMP)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and _job(lines[0]) == 'p5o_openmp' and '--account=000000000000' in lines[0]
