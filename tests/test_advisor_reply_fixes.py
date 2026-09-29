"""The advisor's reply to our Addendum v2.2 response, Section 2: fixes for
waves 2 and 3, made before wave 1 so all three waves share one commit."""

import json
import math
import threading
import time

import numpy as np
import pytest
import torch

from lilq.solvers import solve_lil_n


def test_non_finite_selection_losses_rank_last():
    """Item 2.1: a diverged F1 run's NaN loss must not enter the top 3."""
    import experiments.component_a as ca
    assert ca._final_loss({'final_loss_unweighted': float('nan')}) == math.inf
    assert ca._final_loss({'final_loss_unweighted': float('inf')}) == math.inf
    assert ca._final_loss({'final_loss': None}) == math.inf
    assert ca._final_loss({'final_loss_unweighted': 0.5, 'final_loss': 9.0}) == 0.5
    ranked = sorted([('a', float('nan')), ('b', 0.2), ('c', 0.1)], key=lambda r: ca._final_loss(
        {'final_loss_unweighted': r[1]}))
    assert [r[0] for r in ranked] == ['c', 'b', 'a']


def test_lock_is_created_once_by_concurrent_jobs(tmp_path, fake_bundle):
    """Item 2.2: jobs starting together each write their own temporary file;
    exactly one lock is created, and every job reads a whole record."""
    from lilq.source_lock import check_lock
    root, pkg = fake_bundle, tmp_path / 'pkg'
    results, errors = [], []

    def job():
        try:
            results.append(check_lock(pkg, root))
        except Exception as e:           # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=job) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 8
    assert len({json.dumps(r, sort_keys=True) for r in results}) == 1
    assert sorted(p.name for p in pkg.iterdir()) == ['COMMIT']          # no temporary files left


def test_lock_is_created_once_by_concurrent_processes(tmp_path, fake_bundle):
    """The cluster case: separate job processes checking one package root at
    once all succeed with the same record, and leave only COMMIT behind."""
    import subprocess
    import sys
    from lilq.source_lock import REPO_ROOT
    root, pkg = fake_bundle, tmp_path / 'pkg'
    code = (f"import sys, json; sys.path.insert(0, {str(REPO_ROOT)!r}); from lilq.source_lock import check_lock; "
            f"print(json.dumps(check_lock({str(pkg)!r}, {str(root)!r}), sort_keys=True))")
    procs = [subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(8)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), [err for _, err in outs]
    assert len({out.strip() for out, _ in outs}) == 1
    assert sorted(p.name for p in pkg.iterdir()) == ['COMMIT']


def test_lock_read_waits_for_a_half_written_file(tmp_path):
    from lilq.source_lock import _read_lock
    lock = tmp_path / 'COMMIT'
    lock.write_text('{"commit": "ab')                                   # half-written

    def finish():
        time.sleep(0.8)
        lock.write_text('{"commit": "abc"}')
    threading.Thread(target=finish).start()
    assert _read_lock(lock)['commit'] == 'abc'


def test_every_job_checks_the_torch_version():
    """Item 2.3: env.sh checks it after the lock, so waves that skip the
    preflight still verify PyTorch 2.10.0."""
    from lilq.source_lock import REPO_ROOT
    env = (REPO_ROOT / 'scripts' / 'cluster' / 'env.sh').read_text()
    lock_at, torch_at = env.index('lilq.source_lock'), env.index('assert_torch_version()')
    assert lock_at < torch_at


def test_b8_lilq_non_finite_final_loss_is_a_failure(tmp_path, monkeypatch):
    """Item 2.4: B8's LiL-Q rows follow the 2.3 rule."""
    import experiments.b8_initial_guess as b8
    from lilq.iteration_log import IterationLogger

    def diverged(config, opt, verbose, iteration_logger, run_json_path):
        iteration_logger.record(k=0, norm_R_h=1.0, norm_Rlin_h=1.0, chi=0.5, stall_flag=False)
        iteration_logger.record(k=1, norm_R_h=float('nan'))
        return None, np.zeros(3), None, {'total_iterations': 1, 'final_loss': float('nan'), 'converged': False}
    monkeypatch.setattr(b8, 'run_lil_q', diverged)
    row = b8.run_one('gravity', 'zero', 8, 'LiL-Q', '', quick=True)
    assert row['stopping_reason'] == 'failure'


def test_diverged_lbfgs_run_stops_and_is_a_failure():
    """Item 2.5: NaN parameters never compare equal, so the stall test cannot
    fire; the run stops at the first non-finite loss and is classified
    'failure' instead of spending its iteration budget."""
    from types import SimpleNamespace
    from lilq.four_method_log import stopping_fields
    start = np.array([1.0, -2.0])

    def loss_fn(beta):
        r = ((beta - 3.0) ** 2).sum()
        moved = not torch.equal(beta.detach(), torch.from_numpy(start))
        r = r * float('nan') if moved else r                     # diverges as soon as it moves
        return r, r, 0.0, r * 0
    _, _, summary = solve_lil_n(loss_fn, start.copy(), torch.device('cpu'), max_iterations=500,
                                R_tol=-1.0, verbose=False)
    assert summary['total_iterations'] <= 2 and not math.isfinite(summary['final_loss'])
    opt = SimpleNamespace(max_iterations=500, max_line_searches=None)
    assert stopping_fields('LiL-N', summary, opt)['stopping_reason'] == 'failure'


def test_b9_rows_carry_commit_and_parameter_count(tmp_path):
    """Item 2.6: the B9 CSV records the commit and each method's size."""
    import experiments.darcy_fv_comparison as b9
    from lilq.source_lock import current_commit
    rows = b9.compare_field('S1', order=6, nil_seeds=(0,), nil_epochs=2, verbose=False, model_root=tmp_path,
                            nil_dtypes=('float32',))
    assert all(r['commit'] == current_commit() for r in rows)
    assert rows[1]['n_params'] == 3555 and rows[0]['n_params'] > 0
    assert set(b9.COLUMNS) >= {'commit', 'n_params'}


def test_pinned_beltrami_skips_a_completed_run(tmp_path):
    """A resubmitted job 10a does not redo a finished pinned run."""
    import experiments.run_beltrami_pinned as rbp
    (tmp_path / 'report.json').write_text('{"done": true}')
    assert rbp.run_beltrami_pinned(verbose=False, out_dir=tmp_path) == {'done': True}


def test_no_test_file_imports_another():
    """Test files share helpers through conftest.py fixtures, never by
    importing each other: ``from tests.x import ...`` depends on how the
    interpreter resolves the name ``tests``, and on Grace it did not resolve
    to this folder (the wave-1 preflight failed on it)."""
    import re
    from pathlib import Path
    offenders = [p.name for p in Path(__file__).parent.glob('test_*.py')
                 if re.search(r'^\s*(from|import)\s+tests[.\s]', p.read_text(encoding='utf-8'), re.M)]
    assert offenders == []
