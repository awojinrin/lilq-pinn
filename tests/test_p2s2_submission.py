"""Package 2, Stage 2's submission: every job submitted once, in order, with its
dependencies; the dry run; the references job; and the stage report."""

import importlib.util
import os
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
P2 = REPO / 'scripts' / 'cluster' / 'package2'
SUBMIT = P2 / 'submit_p2s2.sh'
CONTINGENT = {'p2s2_lm_networks_gpu'}          # submitted by hand, only if gpu-list names something
FASTER = {'p2s2_darcy_hardbc'}                 # item 5, submitted on FASTER (submit_p2s2_faster.sh)
COMPUTE = {'p2s2_classical_bratu', 'p2s2_classical_kovasznay', 'p2s2_classical_burgers', 'p2s2_lm_networks_bratu',
           'p2s2_lm_networks_burgers', 'p2s2_lm_networks_bl', 'p2s2_certified', 'p2s2_elm', 'p2s2_nu_refinement',
           'p2s2_bases', 'p2s2_scaling_cpu_a', 'p2s2_scaling_cpu_b', 'p2s2_scaling_gpu',
           'p2s2_elasticity_manufactured', 'p2s2_darcy_softbc_lm'}


def test_every_stage_2_job_is_submitted_or_contingent():
    jobs = {p.stem for p in P2.glob('p2s2_*.slurm')}
    assert jobs == COMPUTE | CONTINGENT | FASTER | {'p2s2_preflight', 'p2s2_references', 'p2s2_report'}
    text = SUBMIT.read_text()
    for job in jobs - CONTINGENT - FASTER:
        assert text.count(f'/{job}.slurm') == 1, job
    assert 'p2s2_lm_networks_gpu.slurm' not in text.split('LILQ_WAVE=p2s2')[1]
    assert 'p2s2_darcy_hardbc.slurm' not in text.split('LILQ_WAVE=p2s2')[1]
    for p in P2.glob('p2s2_*.slurm'):
        t = p.read_text()
        assert re.search(r'#SBATCH --job-name=lilq-p2s2-', t), p.name          # the report finds the logs by it
        assert re.search(r'# lilq-resources: (timed|timed-cpu|cpu|shared-gpu) ', t), p.name


def test_references_job_makes_every_reference():
    t = (P2 / 'p2s2_references.slurm').read_text()
    for cmd in ('p2_3_classical.py reference', 'p2_references.py burgers', 'p2_2_nu_refinement.py reference',
                'p2_9_bases.py --references-only'):
        assert cmd in t
    assert 'python experiments/p2_3_classical.py reference' not in (P2 / 'p2s2_classical_bratu.slurm').read_text().split(
        '|| python experiments/p2_3_classical.py reference')[0].split('set -euo pipefail')[1]   # only if missing


def _fake_sbatch(tmp_path, dry):
    fake = tmp_path / 'sbatch'
    if dry:
        fake.write_text('#!/bin/bash\nfor a in "$@"; do [[ "$a" == --test-only ]] && { echo "$*" >> "$(dirname "$0")/log"; '
                        'echo "sbatch: Job 1 to start at soon" >&2; exit 0; }; done\nexit 1\n')
    else:
        fake.write_text('#!/bin/bash\nn=$(cat "$(dirname "$0")/n" 2>/dev/null || echo 100); echo $((n+1)) > "$(dirname "$0")/n"\n'
                        'echo "$*" >> "$(dirname "$0")/log"; echo $((n+1))\n')
    fake.chmod(0o755)
    return dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", LILQ_ACCOUNT='000000000000', YES='1',
                LILQ_RESULTS=str(tmp_path / 'results'), **({'DRY_RUN': '1'} if dry else {}))


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_dry_run_checks_all_18_jobs_and_submits_nothing(tmp_path):
    out = subprocess.run(['bash', str(SUBMIT)], env=_fake_sbatch(tmp_path, True), capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'Package 2, stage 2 (results/package2_stage2)' in out.stdout
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 18 and all('--account=000000000000' in l for l in log)
    assert not any('--dependency' in l for l in log)
    assert all('--exclusive' in l for l in log if 'scaling_cpu' in l or 'classical' in l or 'lm_networks' in l
               or 'nu_refinement' in l)


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_submission_chains_the_jobs(tmp_path):
    out = subprocess.run(['bash', str(SUBMIT)], env=_fake_sbatch(tmp_path, False), capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 18
    ids = {re.search(r'(p2s2_[a-z_]+)\.slurm', l).group(1): 101 + i for i, l in enumerate(log)}
    assert '--dependency' not in log[0] and 'p2s2_preflight' in log[0]
    assert f"afterok:{ids['p2s2_preflight']}" in log[1] and 'p2s2_references' in log[1]
    for l in log[2:-1]:
        assert f"--dependency=afterok:{ids['p2s2_references']}" in l
    assert 'p2s2_report' in log[-1]
    assert all(f':{ids[j]}' in log[-1] for j in COMPUTE) and 'afterany' in log[-1]
    assert not any('lm_networks_gpu' in l for l in log)


def _stage_report():
    spec = importlib.util.spec_from_file_location('stage_report', P2 / 'stage_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_report_packs_the_stage_and_its_logs(tmp_path, monkeypatch):
    sr = _stage_report()
    stage = tmp_path / 'results' / 'package2_stage2'
    (stage / 'P2_16_certified').mkdir(parents=True)
    (stage / 'P2_16_certified' / 'terminal.csv').write_text('x\n')
    logs = tmp_path / 'logs'
    logs.mkdir()
    for name in ('lilq-p2s2-certified.201.out', 'lilq-p2s2-report.299.out', 'lilq-p2s1-scalar.5.out'):
        (logs / name).write_text('log\n')
    monkeypatch.setenv('SLURM_JOB_ID', '299')
    assert set(sr.stage_logs('p2s2', logs, exclude_job='299')) == {'201'}
    tar_path = sr.report('p2s2', tmp_path / 'results', logs_dir=logs, sacct=False)
    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert 'package2_stage2/P2_16_certified/terminal.csv' in names
    assert 'package2_stage2/slurm_logs/lilq-p2s2-certified.201.out' in names
    assert not any('report.299' in n or 'p2s1' in n for n in names) and 'package2_stage2/report_log.txt' in names
    monkeypatch.setenv('CLUSTER', 'faster')                     # FASTER's part: its own tarball name
    assert sr.report('p2s2', tmp_path / 'results', logs_dir=logs, sacct=False).name == 'package2_stage2_faster.tar.gz'


# ---------------------------------------------------------------- the local assembly (p2_assemble.py stage2)

def _fake_stage(tmp_path, commit):
    import json
    import numpy as np
    stage = tmp_path / 'package2_stage2'
    (stage / 'P2_16_certified').mkdir(parents=True)
    (stage / 'P2_16_certified' / 'terminal.csv').write_text('benchmark\nbratu\n')
    (stage / 'reference').mkdir()
    np.savez(stage / 'reference' / 'bratu_ref_p48.npz', u=np.ones(3) * (1 + 1e-14))        # differs from Stage 1's
    np.savez(stage / 'reference' / 'burgers_cole_hopf.npz', u=np.zeros(2))                # identical
    np.savez(stage / 'reference' / 'bl_fd_ref_viscous_nu0.1.npz', u=np.zeros(2))          # new
    (stage / 'COMMIT').write_text(json.dumps({'commit': commit}))
    (stage / 'su_per_job.csv').write_text('job_id,su\n1,2.0\n')
    out = tmp_path / 'package2_results'
    (out / 'reference').mkdir(parents=True)
    np.savez(out / 'reference' / 'bratu_ref_p48.npz', u=np.ones(3))
    np.savez(out / 'reference' / 'burgers_cole_hopf.npz', u=np.zeros(2))
    (out / 'P2_12_reference_errors').mkdir()
    (out / 'P2_12_reference_errors' / 'COMMIT').write_text(json.dumps({'commit': 'stage1commit'}))
    return stage, out


def test_stage2_assembly(tmp_path):
    import json
    import subprocess
    import experiments.p2_assemble as pa
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    if not head:
        pytest.skip('no git checkout')
    stage, out = _fake_stage(tmp_path, head)
    r = pa.stage2(stage, out, summaries=False)
    assert (out / 'P2_16_certified' / 'terminal.csv').exists() and (out / 'su_per_job.csv').exists()
    assert r['references']['bratu_ref_p48.npz']['max_abs_difference_to_stage1']['u'] == pytest.approx(1e-14)
    assert (out / 'reference' / 'stage1' / 'bratu_ref_p48.npz').exists()                 # Stage 1's kept
    assert r['references']['burgers_cole_hopf.npz'] == {'identical_to_stage1': True}
    assert r['references']['bl_fd_ref_viscous_nu0.1.npz'] == {'new_in_stage2': True}
    assert r['stage1_commit'] == 'stage1commit' and r['stage2_commit'] == head
    assert (out / 'code' / 'COMMIT_stage2').exists() and (out / 'code' / 'COMMIT_stage1').exists()
    assert (out / 'code' / 'experiments' / 'p2_16_certified.py').exists()                # added after v2.0.0
    assert 'experiments/p2_16_certified.py' in (out / 'code' / 'FILES_CHANGED_SINCE_v2.0.0.txt').read_text()
    assert json.loads((out / 'provenance.json').read_text())['items'] == ['P2_16_certified']
    # assembling again gives the same record and replaces nothing (batch 9a)
    r2 = pa.stage2(stage, out, summaries=False)
    assert r2['references'] == r['references'] and r2['replaced'] == []
    assert (out / 'reference' / 'stage1' / 'bratu_ref_p48.npz').read_bytes() != (stage / 'reference' / 'bratu_ref_p48.npz').read_bytes()


def test_stage2_assembly_refuses_another_commit(tmp_path):
    import experiments.p2_assemble as pa
    stage, out = _fake_stage(tmp_path, '0' * 40)
    with pytest.raises(SystemExit, match='locked to'):
        pa.stage2(stage, out, summaries=False)


def test_stage2_assembly_refuses_without_git(tmp_path, monkeypatch):
    """No git (as in the cluster bundle): the checkout's commit is unknown, and the
    assembly refuses, with or without the GPU addendum (batch 8's check once let it pass)."""
    import json
    import experiments.p2_assemble as pa
    monkeypatch.setattr(pa, '_git', lambda *args: None)
    stage, out = _fake_stage(tmp_path, '0' * 40)
    with pytest.raises(SystemExit, match='locked to'):
        pa.stage2(stage, out, summaries=False)
    gpu = tmp_path / 'gpu'
    gpu.mkdir()
    (gpu / 'COMMIT').write_text(json.dumps({'commit': '1' * 40}))
    with pytest.raises(SystemExit, match='GPU addendum'):
        pa.stage2(stage, tmp_path / 'out2', summaries=False, gpu=gpu)
    with pytest.raises(SystemExit, match='does not descend'):           # nor with --later-commit
        pa.stage2(stage, tmp_path / 'out3', summaries=False, gpu=gpu, later_commit=True)


def test_stage2_assembly_at_a_later_commit(tmp_path):
    """--later-commit: a descendant of the locked commits is accepted, and the
    files changed since are listed; a commit off that line is still refused."""
    import json
    import subprocess
    import experiments.p2_assemble as pa
    git = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    head, parent = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD~1')
    if not (head and parent):
        pytest.skip('no git checkout')
    stage, out = _fake_stage(tmp_path, parent)
    gpu = tmp_path / 'gpu' / 'package2_stage2_gpu'
    gpu.mkdir(parents=True)
    (gpu / 'COMMIT').write_text(json.dumps({'commit': parent}))
    with pytest.raises(SystemExit, match='locked to'):
        pa.stage2(stage, out, summaries=False, gpu=gpu)
    r = pa.stage2(stage, out, summaries=False, gpu=gpu, later_commit=True)
    assert r['assembled_at'] == head
    assert r['files_changed_for_assembly'] == git('diff', '--name-only', f'{parent}..{head}').splitlines()
    assert (out / 'code' / 'FILES_CHANGED_FOR_ASSEMBLY.txt').read_text().split() == r['files_changed_for_assembly']
    (gpu / 'COMMIT').write_text(json.dumps({'commit': 'a' * 40}))       # not an ancestor
    with pytest.raises(SystemExit, match='does not descend'):
        pa.stage2(stage, tmp_path / 'out2', summaries=False, gpu=gpu, later_commit=True)
    # at a locked commit nothing is listed
    (gpu / 'COMMIT').write_text(json.dumps({'commit': head}))
    r = pa.stage2(stage, tmp_path / 'out3', summaries=False, gpu=gpu, later_commit=True)
    assert r['files_changed_for_assembly'] is None and not (tmp_path / 'out3' / 'code' / 'FILES_CHANGED_FOR_ASSEMBLY.txt').exists()


def test_stage2_assembly_merges_fasters_part(tmp_path):
    import json
    import subprocess
    import experiments.p2_assemble as pa
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    if not head:
        pytest.skip('no git checkout')
    stage, out = _fake_stage(tmp_path, head)
    faster = tmp_path / 'faster' / 'package2_stage2'
    (faster / 'P2_15_darcy_hardbc' / 'S1_seed0').mkdir(parents=True)
    (faster / 'P2_15_darcy_hardbc' / 'S1_seed0' / 'run.json').write_text('{}')
    (faster / 'P2_16_certified').mkdir()                         # a snapshot of a Grace folder (shared $SCRATCH)
    (faster / 'P2_16_certified' / 'terminal.csv').write_text('partial\n')
    (faster / 'sacct.txt').write_text('faster sacct\n')
    (faster / 'slurm_logs').mkdir()
    (faster / 'slurm_logs' / 'lilq-p2s2-darcy-hardbc.7_0.out').write_text('log\n')
    (faster / 'COMMIT').write_text(json.dumps({'commit': head}))
    r = pa.stage2(stage, out, summaries=False, faster=faster)
    assert r['faster_commit'] == head and 'P2_15_darcy_hardbc' in r['items']
    assert (out / 'P2_15_darcy_hardbc' / 'S1_seed0' / 'run.json').exists()
    assert (out / 'P2_16_certified' / 'terminal.csv').read_text() == 'benchmark\nbratu\n'   # Grace's copy stands
    assert (out / 'stage2_faster_sacct.txt').read_text() == 'faster sacct\n'
    assert (out / 'slurm_logs' / 'stage2_faster' / 'lilq-p2s2-darcy-hardbc.7_0.out').exists()
    (faster / 'COMMIT').write_text(json.dumps({'commit': '1' * 40}))
    with pytest.raises(SystemExit, match='one commit'):
        pa.stage2(stage, tmp_path / 'out2', summaries=False, faster=faster)


def test_stage2_assembly_merges_the_gpu_addendum(tmp_path):
    """The main run's GPU scaling runs move to old_environment/ (kept for C8'
    against Package 1); the addendum's take their place; the CPU runs stay."""
    import json
    import subprocess
    import experiments.p2_assemble as pa
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    if not head:
        pytest.skip('no git checkout')
    stage, out = _fake_stage(tmp_path, head)
    sc = stage / 'P2_1_scaling' / 'kovasznay'
    for name, body in (('kovasznay_P_25_cuda', {'series': 'kovasznay_P', 'size': 25, 'env': 'old'}),
                       ('kovasznay_P_35_cuda', None),                                  # the failed one: no run.json
                       ('kovasznay_P_25_cpu', {'series': 'kovasznay_P', 'size': 25, 'device': 'cpu'})):
        (sc / name).mkdir(parents=True)
        if body:
            (sc / name / 'run.json').write_text(json.dumps(body))
    (stage / 'P2_1_scaling' / 'provenance_cuda_20017337').mkdir()
    (stage / 'P2_1_scaling' / 'provenance_cuda_20017337' / 'hardware.json').write_text('{}')
    gpu = tmp_path / 'gpu' / 'package2_stage2_gpu'
    for name in ('kovasznay_P_25_cuda', 'kovasznay_P_35_cuda'):
        (gpu / 'P2_1_scaling' / 'kovasznay' / name).mkdir(parents=True)
        (gpu / 'P2_1_scaling' / 'kovasznay' / name / 'run.json').write_text(json.dumps({'env': 'new', 'run': name}))
    (gpu / 'P2_1_scaling' / 'provenance_cuda_20030000').mkdir()
    (gpu / 'P2_1_scaling' / 'provenance_cuda_20030000' / 'hardware.json').write_text('{}')
    (gpu / 'cuda_libraries.json').write_text('{}')
    (gpu / 'cuda_libraries_module_first.json').write_text('{}')
    (gpu / 'COMMIT').write_text(json.dumps({'commit': head}))
    r = pa.stage2(stage, out, summaries=False, gpu=gpu)
    s = out / 'P2_1_scaling'
    assert json.loads((s / 'kovasznay' / 'kovasznay_P_25_cuda' / 'run.json').read_text())['env'] == 'new'
    assert json.loads((s / 'kovasznay' / 'kovasznay_P_35_cuda' / 'run.json').read_text())['env'] == 'new'
    assert json.loads((s / 'old_environment' / 'kovasznay' / 'kovasznay_P_25_cuda' / 'run.json').read_text())['env'] == 'old'
    assert (s / 'kovasznay' / 'kovasznay_P_25_cpu' / 'run.json').exists()
    assert (s / 'old_environment' / 'provenance_cuda_20017337').is_dir() and (s / 'provenance_cuda_20030000').is_dir()
    assert (s / 'gpu_addendum' / 'cuda_libraries_module_first.json').exists()
    assert (out / 'code' / 'COMMIT_stage2_gpu').exists() and (out / 'code' / 'FILES_CHANGED_IN_GPU_ADDENDUM.txt').exists()
    assert r['gpu_addendum']['runs'] == ['kovasznay_P_25_cuda', 'kovasznay_P_35_cuda']
    assert r['gpu_addendum']['old_environment_runs'] == ['kovasznay_P_25']
    assert r['gpu_addendum']['did_not_fit_from_logs'] == []
    assert (s / 'old_environment' / 'kovasznay' / 'kovasznay_P_35_cuda').is_dir()       # the failed run, kept empty
    assert not any('cuda' in str(f) for f in r['replaced'])
    # assembling again changes nothing (batch 9a: the first version filed the addendum as the old environment)
    r2 = pa.stage2(stage, out, summaries=False, gpu=gpu)
    assert r2['replaced'] == [] and r2['gpu_addendum']['old_environment_runs'] == ['kovasznay_P_25']
    assert json.loads((s / 'old_environment' / 'kovasznay' / 'kovasznay_P_25_cuda' / 'run.json').read_text())['env'] == 'old'
    assert json.loads((s / 'kovasznay' / 'kovasznay_P_25_cuda' / 'run.json').read_text())['env'] == 'new'
    # a checkout at neither commit is refused
    (gpu / 'COMMIT').write_text(json.dumps({'commit': '2' * 40}))
    (stage / 'COMMIT').write_text(json.dumps({'commit': '3' * 40}))
    with pytest.raises(SystemExit, match='GPU addendum'):
        pa.stage2(stage, tmp_path / 'out3', summaries=False, gpu=gpu)


# the end of the GPU addendum's scaling log (job 20030356, 6 October), shortened
ADDENDUM_LOG = """\
  beltrami_7_cuda: P = 13764, N = 52225 (N/P 3.79), 4 iterations, 48.82 s (assembly 30.42, solve 14.64), host peak 27.47 GB, GPU peak 18.83 GB, kappa 1.0e+05, rank 13764, u(t=1) 1.52e-05, p(t=1) 8.86e-04
Traceback (most recent call last):
  File "$SCRATCH/lilq-run/lilq-pinn/problems/kovasznay.py", line 404, in _lstsq_gpu_qr
    qtb = torch.ormqr(reflectors, tau, bt.unsqueeze(1), left=True, transpose=True)   # Q^T b
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE, when calling `cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)`. This error may appear if the input matrix contains NaN.
Traceback (most recent call last):
  File "$SCRATCH/lilq-run/lilq-pinn/experiments/p2_1_scaling.py", line 299, in run_series
    subprocess.run([sys.executable, __file__, 'one', '--series', series, '--size', str(size),
  File "/sw/eb/sw/Python/3.12.3-GCCcore-13.3.0/lib/python3.12/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['$SCRATCH/lilq-run/venv/bin/python', '$SCRATCH/lilq-run/lilq-pinn/experiments/p2_1_scaling.py', 'one', '--series', 'beltrami', '--size', '8', '--device', 'cuda', '--out', '$SCRATCH/lilq-run/lilq-pinn/results/package2_stage2_gpu']' returned non-zero exit status 1.
"""


def test_gpu_did_not_fit_from_logs(tmp_path):
    """The addendum's Beltrami 22,288 raised on cuSOLVER's ormqr workspace: the
    assembly writes its did_not_fit record from the log; any other failure, and
    a run that has its run.json, are left alone."""
    import json
    import experiments.p2_assemble as pa
    gpu, out = tmp_path / 'package2_stage2_gpu', tmp_path / 'out'
    (gpu / 'slurm_logs').mkdir(parents=True)
    (gpu / 'P2_1_scaling' / 'beltrami' / 'beltrami_7_cuda').mkdir(parents=True)
    (gpu / 'P2_1_scaling' / 'beltrami' / 'beltrami_7_cuda' / 'run.json').write_text(
        json.dumps({'gpu_name': 'NVIDIA A100-PCIE-40GB', 'gpu_total_bytes': 42404806656}))
    (gpu / 'COMMIT').write_text(json.dumps({'commit': 'e' * 40}))
    log = gpu / 'slurm_logs' / 'lilq-p2s2g-scaling-gpu.20030356.out'
    log.write_text(ADDENDUM_LOG)
    assert pa.gpu_did_not_fit_from_logs(gpu, out) == ['beltrami_8_cuda']
    rec = json.loads((out / 'P2_1_scaling' / 'beltrami' / 'beltrami_8_cuda' / 'run.json').read_text())
    assert (rec['status'], rec['P'], rec['N'], rec['A_bytes']) == ('did_not_fit', 22288, 75689, 13495651456)
    assert (rec['slurm_job'], rec['commit'], rec['gpu_name']) == ('20030356', 'e' * 40, 'NVIDIA A100-PCIE-40GB')
    assert rec['error'].startswith('torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE')
    assert rec['recorded_from'] == 'slurm_logs/stage2_gpu/lilq-p2s2g-scaling-gpu.20030356.out'
    # written once: a second assembly leaves it alone
    assert pa.gpu_did_not_fit_from_logs(gpu, out) == []
    # a failure that is not an A100 limit is not turned into did_not_fit
    log.write_text(ADDENDUM_LOG.replace('Dormqr_bufferSize', 'Dgeqrf'))
    assert pa.gpu_did_not_fit_from_logs(gpu, tmp_path / 'out2') == []
    other = [ln if not ln.startswith('torch._C') else 'ValueError: something else' for ln in ADDENDUM_LOG.splitlines()]
    log.write_text('\n'.join(other))
    assert pa.gpu_did_not_fit_from_logs(gpu, tmp_path / 'out3') == []


def test_stage2_assembly_runs_as_a_script(tmp_path):
    """The command line as the user runs it: ``python experiments/p2_assemble.py``
    from another folder, with no PYTHONPATH. Its imports of experiments.* must
    resolve (batch 9a's first real assembly stopped on them)."""
    import json
    import sys
    git = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    head, parent = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD~1')
    if not (head and parent):
        pytest.skip('no git checkout')
    stage, out = _fake_stage(tmp_path, parent)
    gpu = tmp_path / 'gpu' / 'package2_stage2_gpu'
    (gpu / 'slurm_logs').mkdir(parents=True)
    (gpu / 'slurm_logs' / 'lilq-p2s2g-scaling-gpu.20030356.out').write_text(ADDENDUM_LOG)
    (gpu / 'COMMIT').write_text(json.dumps({'commit': parent}))
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    rel = lambda p: str(p.relative_to(tmp_path))  # noqa: E731  (relative paths: the steps run from the repository)
    r = subprocess.run([sys.executable, str(REPO / 'experiments' / 'p2_assemble.py'), 'stage2', '--stage', rel(stage),
                        '--out', rel(out), '--gpu', rel(gpu), '--later-commit'],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / 'P2_1_scaling' / 'scaling.csv').exists(), (out / 'assembly_log.txt').read_text()[-3000:]
    rec = json.loads((out / 'P2_1_scaling' / 'beltrami' / 'beltrami_8_cuda' / 'run.json').read_text())
    assert rec['status'] == 'did_not_fit' and rec['P'] == 22288
    assert 'assembled at' in r.stdout
