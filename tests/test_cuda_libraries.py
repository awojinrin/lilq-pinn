"""Package 2, Stage 2, batch 8: the torch wheel's NVIDIA libraries first on
LD_LIBRARY_PATH, the library record, and the GPU addendum's submission."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lilq import cuda_libraries as cl

REPO = Path(__file__).resolve().parents[1]
CLUSTER = REPO / 'scripts' / 'cluster'


def test_wheel_library_dirs(tmp_path):
    for name in ('cublas', 'cusolver', 'cuda_runtime'):
        (tmp_path / 'nvidia' / name / 'lib').mkdir(parents=True)
    (tmp_path / 'nvidia' / 'not_a_lib_package').mkdir()
    dirs = cl.wheel_library_dirs(str(tmp_path))
    assert [Path(d).parent.name for d in dirs] == ['cublas', 'cuda_runtime', 'cusolver']
    assert cl.wheel_library_dirs(str(tmp_path / 'empty')) == []


def test_ld_path_runs_without_torch_and_prints_a_path_list():
    out = subprocess.run([sys.executable, '-c', 'import sys, lilq.cuda_libraries; '
                          'print("torch" in sys.modules)'], cwd=REPO, capture_output=True, text=True)
    assert out.stdout.strip() == 'False'                       # env.sh runs it in every job, CPU-only ones too
    out = subprocess.run([sys.executable, '-m', 'lilq.cuda_libraries', '--ld-path'], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == ':'.join(cl.wheel_library_dirs())


def test_loaded_libraries_from_maps():
    maps = '\n'.join([
        '7f00-7f01 r-xp 00000000 08:02 1 /venv/site-packages/nvidia/cusolver/lib/libcusolver.so.11',
        '7f01-7f02 r--p 00001000 08:02 1 /venv/site-packages/nvidia/cusolver/lib/libcusolver.so.11',
        '7f02-7f03 r-xp 00000000 08:02 2 /sw/eb/sw/CUDA/12.6.0/lib64/libcublas.so.12',
        '7f03-7f04 r-xp 00000000 08:02 3 /usr/lib64/libc.so.6',
        '7f04-7f05 rw-p 00000000 00:00 0 ',
        '7f05-7f06 r-xp 00000000 08:02 4 /venv/site-packages/torch/lib/libtorch_cuda.so'])
    libs = cl.loaded_libraries(maps)
    assert [l['path'] for l in libs] == ['/sw/eb/sw/CUDA/12.6.0/lib64/libcublas.so.12',
                                         '/venv/site-packages/nvidia/cusolver/lib/libcusolver.so.11']


def test_report_records_the_environment():
    pytest.importorskip('torch')
    r = cl.report()
    assert {'torch', 'torch_cuda', 'LD_LIBRARY_PATH', 'wheel_library_dirs', 'wheel_versions', 'gpu'} <= set(r)
    import torch
    if torch.cuda.is_available():
        r = cl.report(probe=[(64, 16)])
        assert r['geqrf_probe'] == {'64x16': 'ok'} and 'loaded' in r


def test_env_sh_puts_the_wheel_libraries_first():
    text = (CLUSTER / 'env.sh').read_text()
    venv = text.index('source "$SCRATCH/lilq-run/venv/bin/activate"')
    cd = text.index('cd "$SCRATCH/lilq-run/lilq-pinn"')
    keep = text.index('export LILQ_MODULE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"')
    first = text.index('export LD_LIBRARY_PATH="$_lilq_nv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"')
    lock = text.index('python -m lilq.source_lock')
    assert venv < cd < keep < first < lock                     # after the venv and the cd; before any job's python work
    assert 'p2s2g) export PKG="$RESULTS/package2_stage2_gpu"' in text


def test_the_preflight_records_and_probes_the_libraries():
    text = (CLUSTER / 'package2' / 'p2s2_preflight.slurm').read_text()
    assert 'python -m lilq.cuda_libraries --json "$PKG/cuda_libraries.json" --probe 11036x3675' in text
    assert 'LD_LIBRARY_PATH="$LILQ_MODULE_LD_LIBRARY_PATH" python -m lilq.cuda_libraries' in text
    assert text.index('cuda_libraries') < text.index('python -m pytest')


def test_the_report_names_the_addendums_tarball(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('stage_report', CLUSTER / 'package2' / 'stage_report.py')
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    (tmp_path / 'results' / 'package2_stage2_gpu' / 'P2_1_scaling').mkdir(parents=True)
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs' / 'lilq-p2s2g-scaling-gpu.9.out').write_text('log\n')
    (tmp_path / 'logs' / 'lilq-p2s2-scaling-gpu.8.out').write_text('log\n')     # the main stage's: not this one's
    tar = sr.report('p2s2g', tmp_path / 'results', logs_dir=tmp_path / 'logs', sacct=False)
    assert tar.name == 'package2_stage2_gpu.tar.gz'
    import tarfile
    with tarfile.open(tar) as t:
        names = t.getnames()
    assert 'package2_stage2_gpu/slurm_logs/lilq-p2s2g-scaling-gpu.9.out' in names
    assert not any('lilq-p2s2-scaling-gpu.8' in n for n in names)
    assert 'stage_report.py --stage "$LILQ_WAVE"' in (CLUSTER / 'package2' / 'p2s2_report.slurm').read_text()


def _fake_sbatch(tmp_path):
    fake = tmp_path / 'sbatch'
    fake.write_text('#!/bin/bash\nn=$(cat "$(dirname "$0")/n" 2>/dev/null || echo 100); echo $((n+1)) > "$(dirname "$0")/n"\n'
                    'echo "$*" >> "$(dirname "$0")/log"; echo $((n+1))\n')
    fake.chmod(0o755)
    return dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", LILQ_ACCOUNT='000000000000', YES='1',
                LILQ_RESULTS=str(tmp_path / 'results'))


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_the_gpu_addendum_chain(tmp_path):
    out = subprocess.run(['bash', str(CLUSTER / 'package2' / 'submit_p2s2g.sh')], env=_fake_sbatch(tmp_path),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'GPU addendum (results/package2_stage2_gpu)' in out.stdout
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 3
    assert '--job-name=lilq-p2s2g-preflight' in log[0] and 'p2s2_preflight.slurm' in log[0]
    assert '--job-name=lilq-p2s2g-scaling-gpu' in log[1] and '--dependency=afterok:101' in log[1]
    assert '--cpus-per-task=48' in log[1] and '--gres=gpu:a100:1' in log[1]          # timed: a whole A100 node
    assert '--job-name=lilq-p2s2g-report' in log[2] and '--dependency=afterany:102' in log[2]


@pytest.mark.skipif(os.name == 'nt', reason='runs bash')
def test_sbatch_accepts_the_addendum_stage(tmp_path):
    env = dict(_fake_sbatch(tmp_path), LILQ_WAVE='p2s2g')
    out = subprocess.run(['bash', str(CLUSTER / 'sbatch.sh'), str(CLUSTER / 'package2' / 'p2s2_scaling_gpu.slurm')],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    out = subprocess.run(['bash', str(CLUSTER / 'sbatch.sh'), str(CLUSTER / 'package2' / 'p2s2_scaling_gpu.slurm')],
                         env=dict(env, LILQ_WAVE='p2s3'), capture_output=True, text=True)
    assert out.returncode != 0 and 'p2s2g' in out.stderr


def test_a_gpu_scaling_run_records_its_libraries(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no GPU')
    import experiments.p2_1_scaling as m
    rec = m.run_one('kovasznay_P', 5, 'cuda', tmp_path)
    assert 'cuda_libraries' in rec and 'ld_library_path' in rec
    assert json.loads((tmp_path / 'P2_1_scaling' / 'kovasznay' / 'kovasznay_P_5_cuda' / 'run.json').read_text())[
        'status'] == 'ok'
