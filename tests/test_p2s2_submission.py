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
COMPUTE = {'p2s2_classical_bratu', 'p2s2_classical_kovasznay', 'p2s2_classical_burgers', 'p2s2_lm_networks_bratu',
           'p2s2_lm_networks_burgers', 'p2s2_lm_networks_bl', 'p2s2_certified', 'p2s2_elm', 'p2s2_nu_refinement',
           'p2s2_bases', 'p2s2_scaling_cpu_a', 'p2s2_scaling_cpu_b', 'p2s2_scaling_gpu',
           'p2s2_elasticity_manufactured'}


def test_every_stage_2_job_is_submitted_or_contingent():
    jobs = {p.stem for p in P2.glob('p2s2_*.slurm')}
    assert jobs == COMPUTE | CONTINGENT | {'p2s2_preflight', 'p2s2_references', 'p2s2_report'}
    text = SUBMIT.read_text()
    for job in jobs - CONTINGENT:
        assert text.count(f'/{job}.slurm') == 1, job
    assert 'p2s2_lm_networks_gpu.slurm' not in text.split('LILQ_WAVE=p2s2')[1]
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
def test_dry_run_checks_all_17_jobs_and_submits_nothing(tmp_path):
    out = subprocess.run(['bash', str(SUBMIT)], env=_fake_sbatch(tmp_path, True), capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert 'Package 2, stage 2 (results/package2_stage2)' in out.stdout
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 17 and all('--account=000000000000' in l for l in log)
    assert not any('--dependency' in l for l in log)
    assert all('--exclusive' in l for l in log if 'scaling_cpu' in l or 'classical' in l or 'lm_networks' in l
               or 'nu_refinement' in l)


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_submission_chains_the_jobs(tmp_path):
    out = subprocess.run(['bash', str(SUBMIT)], env=_fake_sbatch(tmp_path, False), capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 17
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


def test_stage2_assembly_refuses_another_commit(tmp_path):
    import experiments.p2_assemble as pa
    stage, out = _fake_stage(tmp_path, '0' * 40)
    with pytest.raises(SystemExit, match='locked to'):
        pa.stage2(stage, out, summaries=False)
