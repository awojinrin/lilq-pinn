"""Package 5, item 3: the OpenMP library diagnostic (``experiments/p5_3_openmp.py``)."""
import csv
import json

import numpy as np
import pytest

import experiments.p5_3_networks as n3
import experiments.p5_3_openmp as om
import lilq.provenance as prov


TORCH_LIB = '/scratch/u/lilq-run/venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1'
MODULE_LIB = '/sw/eb/sw/GCCcore/13.3.0/lib64/libgomp.so.1.0.0'


def test_library_kind():
    assert prov.openmp_library_kind([TORCH_LIB]) == 'torch'
    assert prov.openmp_library_kind([MODULE_LIB]) == 'module'
    assert prov.openmp_library_kind([MODULE_LIB, TORCH_LIB]) == 'torch'     # both loaded: the wheel's is in use
    assert prov.openmp_library_kind([]) is None


def test_require_openmp_stops_on_any_other_library(monkeypatch, capsys):
    monkeypatch.setattr(prov, 'openmp_libraries', lambda: [MODULE_LIB])
    assert prov.require_openmp('module') == [MODULE_LIB]
    assert 'OpenMP library: module' in capsys.readouterr().out
    with pytest.raises(SystemExit, match='OPENMP LIBRARY CHECK FAILED: .*the module OpenMP library.*not the torch one'):
        prov.require_openmp('torch')
    monkeypatch.setattr(prov, 'openmp_libraries', lambda: [TORCH_LIB])
    with pytest.raises(SystemExit, match='loaded the torch OpenMP library'):
        prov.require_openmp('module')
    monkeypatch.setattr(prov, 'openmp_libraries', lambda: [])
    with pytest.raises(SystemExit, match='loaded the None OpenMP library'):
        prov.require_openmp('module')
    with pytest.raises(ValueError):
        prov.require_openmp('gcc')


def test_this_process_check_matches_what_threadpoolctl_reports():
    """The real libraries of the test process: whatever they are, the check agrees with the classification."""
    kind = prov.openmp_library_kind(prov.openmp_libraries())
    if kind is None:
        pytest.skip('no OpenMP runtime loaded in this process')
    assert prov.require_openmp(kind)


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
    monkeypatch.setattr(prov, 'openmp_libraries', lambda: [MODULE_LIB])   # main imports it from lilq.provenance
    with pytest.raises(SystemExit, match='not the torch one'):
        om.main(['--first', 'torch', '--round', '0', '--out', str(tmp_path / 'out'), '--reference-dir', str(ref),
                 '--require-library'])
    assert not (tmp_path / 'out' / n3.ITEM / 'openmp' / 'timing_torch-first_r0.csv').exists()


_FIRST_OF = r'''
import sys, importlib.abc
first = []
class Spy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in ('numpy', 'torch') and not first:
            first.append(name)
        return None
sys.meta_path.insert(0, Spy())
sys.path.insert(0, sys.argv[1])
importlib.import_module(sys.argv[2])
print(first[0] if first else 'none')
'''


@pytest.mark.parametrize('module', ['experiments.p5_3_networks', 'experiments.p5_3_lil', 'experiments.p5_3_table',
                                    'experiments.p4_checks', 'lilq.provenance'])
def test_every_item3_entry_point_imports_numpy_before_torch(module):
    """In a fresh process, numpy is the first of the two loaded: on Grace that gives the GCC module's OpenMP
    library, as Package 1 (DECISIONS.md, 17 November 2026)."""
    import subprocess
    import sys
    from pathlib import Path
    repo = str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, '-c', _FIRST_OF, repo, module], capture_output=True, text=True, cwd=repo,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == 'numpy'


@pytest.mark.parametrize('driver', ['networks', 'lil'])
def test_a_driver_with_require_openmp_stops_before_any_run_or_record(driver, ref, tmp_path, monkeypatch):
    import experiments.p5_3_lil as lil
    monkeypatch.setattr(prov, 'openmp_libraries', lambda: [TORCH_LIB])
    out = tmp_path / 'out'
    main = n3.main if driver == 'networks' else lil.main
    with pytest.raises(SystemExit, match='OPENMP LIBRARY CHECK FAILED'):
        main(['run', '--bench', 'burgers', '--out', str(out), '--reference-dir', str(ref), '--P', '25',
              '--require-openmp', 'module'])
    assert not out.exists()                                   # no provenance, no run folder
    if driver == 'networks':
        for cmd in (['solo', '--bench', 'burgers', '--P', '25', '--method', 'nil_n'], ['checks', '--bench', 'burgers']):
            with pytest.raises(SystemExit, match='OPENMP LIBRARY CHECK FAILED'):
                n3.main(cmd + ['--out', str(out), '--reference-dir', str(ref), '--require-openmp', 'module'])
        assert not out.exists()


def test_require_openmp_takes_only_the_two_kinds(ref, tmp_path):
    with pytest.raises(SystemExit):
        n3.main(['run', '--bench', 'burgers', '--out', str(tmp_path), '--reference-dir', str(ref), '--require-openmp', 'gcc'])
