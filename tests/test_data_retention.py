"""Addendum v2.2 Section 2.8: histories next to the saved models, the
partial B8 log, the commit in every record, and the provenance lock."""

import csv
import json
import shutil
import tarfile

import pytest
import torch

import experiments.b8_initial_guess as b8
import experiments.four_method_tables as fmt
from lilq import source_lock
from lilq.four_method_log import FourMethodLogger
from lilq.source_lock import REPO_ROOT, check_lock, normalized_bytes, source_identity, tree_hash

# The cluster upload bundle ships without .git: tests that need git skip there.
requires_git_checkout = pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="needs a git checkout")


# ── histories ────────────────────────────────────────────────────────────────

def test_four_method_rows_keep_full_history_and_commit(tmp_path):
    from problems.bratu import BratuConfig, BratuOptConfig, run_nil_n
    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = fmt._apply_quick_budgets(BratuOptConfig())
    logger, csv_path = FourMethodLogger(), tmp_path / 'four_method_tables.csv'
    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'NiL-N', run_nil_n, seeds=[0],
                     devices=[torch.device('cpu')], verbose=False, csv_path=csv_path)
    row = logger.rows[0]
    assert row['commit'] == source_lock.current_commit()
    assert all(len(h) == 3 for h in row['loss_history_every_10'])          # iteration, loss, wall time
    with open(tmp_path / 'models' / 'bratu_P9_NiL-N_s0_cpu' / 'history.csv', newline='') as f:
        hist = list(csv.DictReader(f))
    assert len(hist) == row['total_iterations'] + 1
    assert set(hist[0]) == {'iteration', 'n_func_evals', 'loss', 'pde_loss', 'ic_loss', 'bc_loss', 'wall_time'}
    assert int(hist[-1]['n_func_evals']) == row['total_line_searches']


def test_b8_keeps_partial_lilq_log_when_the_run_raises(tmp_path, monkeypatch):
    real = b8.run_lil_q

    def failing(config, opt, verbose, iteration_logger, run_json_path):
        iteration_logger.record(k=0, norm_R_h=1.0)
        raise FloatingPointError("diverged")
    monkeypatch.setattr(b8, "run_lil_q", failing)
    with pytest.raises(FloatingPointError):
        b8.run_one('gravity', 'zero', 8, 'LiL-Q', '', quick=True, log_root=tmp_path)
    assert (tmp_path / 'gravity_zero_P64' / 'iterations.csv').exists()
    monkeypatch.setattr(b8, "run_lil_q", real)


def test_b8_saves_histories(tmp_path):
    b8.run_b8(tmp_path, quick=True, verbose=False, cases=('viscous',), guesses=('zero',), sizes=(8,),
              methods=('NiL-N',), seeds=(0,))
    assert (tmp_path / 'models' / 'viscous_zero_P64_NiL-N_s0' / 'history.csv').exists()
    with open(tmp_path / 'b8_initial_guess.csv', newline='') as f:
        assert next(csv.DictReader(f))['commit'] == source_lock.current_commit()


# ── provenance lock ──────────────────────────────────────────────────────────

def test_tree_hash_ignores_line_endings(tmp_path):
    (tmp_path / 'a.py').write_bytes(b'x = 1\r\ny = 2\r\n')
    (tmp_path / 'b.py').write_bytes(b'x = 1\ny = 2\n')
    assert normalized_bytes(tmp_path / 'a.py') == normalized_bytes(tmp_path / 'b.py')
    assert tree_hash(tmp_path, ['a.py']) != tree_hash(tmp_path, ['b.py'])       # the path is part of it
    (tmp_path / 'c').mkdir()
    shutil.copy(tmp_path / 'a.py', tmp_path / 'c' / 'b.py')
    assert tree_hash(tmp_path / 'c', ['b.py']) == tree_hash(tmp_path, ['b.py'])


def test_lock_is_created_then_enforced(tmp_path, fake_bundle):
    root = fake_bundle
    pkg = tmp_path / 'pkg'
    first = check_lock(pkg, root)
    assert first['commit'] == 'abc123' and (pkg / 'COMMIT').exists()
    assert check_lock(pkg, root)['tree_hash'] == first['tree_hash']       # later jobs, same code: fine
    with open(root / 'lilq' / 'source_lock.py', 'a') as f:                  # edited on the cluster
        f.write('\n# edit\n')
    with pytest.raises(RuntimeError, match='differ from the bundle'):
        check_lock(pkg, root)


def test_lock_refuses_a_different_version(tmp_path, fake_bundle):
    root = fake_bundle
    pkg = tmp_path / 'pkg'
    check_lock(pkg, root)
    record = json.loads((root / 'PROVENANCE.json').read_text())
    record['commit'] = 'def456'
    (root / 'PROVENANCE.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='locked to commit abc123'):
        check_lock(pkg, root)


def test_lock_refuses_unknown_or_dirty_provenance(tmp_path, fake_bundle):
    empty = tmp_path / 'nothing'
    empty.mkdir()
    with pytest.raises(RuntimeError, match='unknown'):
        check_lock(tmp_path / 'pkg', empty)
    root = fake_bundle
    record = json.loads((root / 'PROVENANCE.json').read_text())
    record['dirty'] = True
    (root / 'PROVENANCE.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='uncommitted'):
        check_lock(tmp_path / 'pkg2', root)


@requires_git_checkout      # building a bundle needs git; the cluster copy has no .git
def test_bundle_records_a_tree_hash_its_files_reproduce(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('bundle', REPO_ROOT / 'scripts' / 'make_hprc_bundle.py')
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)
    out = tmp_path / 'b.tar.gz'
    result = bundle.build_bundle(out)
    with tarfile.open(out) as tar:
        tar.extractall(tmp_path / 'x', filter='data')
    extracted = tmp_path / 'x' / 'lilq-pinn'
    record = json.loads((extracted / 'PROVENANCE.json').read_text())
    assert record['tree_hash'] == result['record']['tree_hash'] == tree_hash(extracted, record['files'])
    ident = source_identity(extracted)
    assert ident['source'] == 'bundle' and ident['tree_hash'] == record['tree_hash']
