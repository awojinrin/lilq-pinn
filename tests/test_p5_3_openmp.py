"""Package 5, item 3: the OpenMP library diagnostic (``experiments/p5_3_openmp.py``)."""
import csv
import json

import numpy as np
import pytest

import experiments.p5_3_networks as n3
import experiments.p5_3_openmp as om


def test_library_kind():
    assert om.library_kind(['/scratch/u/lilq-run/venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1']) == 'torch'
    assert om.library_kind(['/sw/eb/sw/GCCcore/13.3.0/lib64/libgomp.so.1.0.0']) == 'module'
    assert om.library_kind([]) is None


@pytest.fixture
def ref(tmp_path):
    x = np.linspace(0, 1, 11)
    np.savez(tmp_path / 'bratu_ref_p48.npz', x=x, y=x, u=np.outer(np.sin(np.pi * x), np.sin(np.pi * x)))
    xb, t = np.linspace(-1, 1, 11), np.linspace(0, 1, 11)
    np.savez(tmp_path / 'burgers_cole_hopf.npz', x=xb, t=t, u=-np.outer(np.sin(np.pi * xb), np.exp(-t)))
    return tmp_path


def test_a_process_writes_its_c1_runs_its_timing_and_its_library(ref, tmp_path, monkeypatch):
    monkeypatch.setattr(om, 'C1_BENCHES', ('burgers',))
    monkeypatch.setattr(om, 'TIMING_BENCHES', ('burgers',))
    monkeypatch.setattr(n3, 'sizes', lambda bench: (25,))
    monkeypatch.setattr(om, 'C1_RUN_ITERATIONS', 12)
    out = tmp_path / 'out'
    om.main(['--first', 'numpy', '--round', '0', '--out', str(out), '--reference-dir', str(ref), '--budget', '0.3'])
    root = out / n3.ITEM / 'openmp'
    c = root / 'numpy-first' / 'c1' / 'burgers_P25_nil_n_s0_cpu'
    rec = json.loads((c / 'run.json').read_text())
    assert rec['iterations'] == 12 and rec['first'] == 'numpy' and rec['method_key'] == 'nil_n'
    assert (c / 'metrics.csv').exists()
    with open(root / 'timing_numpy-first_r0.csv') as fh:
        rows = list(csv.DictReader(fh))
    assert [(r['benchmark'], r['method']) for r in rows] == [('burgers', 'nil_n'), ('burgers', 'lm')]
    assert all(r['stopping_reason'] == 'budget' and int(r['iterations']) > 0 for r in rows)
    proc = json.loads((root / 'numpy-first' / 'process_r0.json').read_text())
    assert proc['expected_library'] == 'module' and 'openmp_libraries' in proc
    om.main(['--first', 'torch', '--round', '1', '--out', str(out), '--reference-dir', str(ref), '--budget', '0.3'])
    assert not (root / 'torch-first' / 'c1').exists()                    # the C1 runs: round 0 only
    assert (root / 'timing_torch-first_r1.csv').exists()


def test_require_library_stops_a_process_with_the_other_library(ref, tmp_path, monkeypatch):
    monkeypatch.setattr(om, 'openmp_libraries', lambda: ['/sw/eb/sw/GCCcore/13.3.0/lib64/libgomp.so.1.0.0'])
    with pytest.raises(SystemExit, match='not the torch one'):
        om.main(['--first', 'torch', '--round', '0', '--out', str(tmp_path / 'out'), '--reference-dir', str(ref),
                 '--require-library'])
    assert not (tmp_path / 'out' / n3.ITEM / 'openmp' / 'timing_torch-first_r0.csv').exists()
