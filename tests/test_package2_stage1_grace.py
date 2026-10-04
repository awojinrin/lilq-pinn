"""Package 2, Stage 1, the Grace job: the scalar LiL-Q reruns log eps_ref
without changing the solve; check C4 compares norm_R_h with package1 row by
row; the stage names p2s1 and p2s2 have their own results folders. (Rehearsed
on this laptop against Grace's wave 2 logs: every run within 9.1e-11.)"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_eps_ref_is_logged_and_the_solve_is_unchanged(tmp_path):
    import dataclasses
    from lilq.iteration_log import IterationLogger
    from problems.bratu import BratuConfig, BratuOptConfig, TEST_GRID, run_lil_q
    cfg = BratuConfig(N_x=4, N_y=4)
    opt = dataclasses.replace(BratuOptConfig(), max_quasi_iters_lil=3)
    x = y = np.linspace(0, 1, TEST_GRID[0])
    np.savez_compressed(tmp_path / 'ref.npz', x=x, y=y, u=0.1 + np.outer(np.sin(np.pi * x), np.sin(np.pi * y)),
                        meta=np.array('{}'))
    plain, with_ref = IterationLogger(), IterationLogger()
    _, c0, _, _ = run_lil_q(cfg, opt, verbose=False, iteration_logger=plain)
    _, c1, _, _ = run_lil_q(cfg, opt, verbose=False, iteration_logger=with_ref, reference_npz=tmp_path / 'ref.npz')
    assert np.array_equal(c0, c1)
    assert all(r['eps_ref'] is None for r in plain.rows)
    assert all(isinstance(r['eps_ref'], float) for r in with_ref.rows)
    assert [r['norm_R_h'] for r in plain.rows] == [r['norm_R_h'] for r in with_ref.rows]


def test_c4_compares_norm_R_h_row_by_row(tmp_path):
    import experiments.p2_c4_check as c4

    def log(path, norms, eps=''):
        path.mkdir(parents=True)
        with open(path / 'iterations.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['k', 'norm_R_h', 'eps_ref'])
            w.writeheader()
            w.writerows({'k': k, 'norm_R_h': n, 'eps_ref': eps} for k, n in enumerate(norms))
    log(tmp_path / 'new' / 'a', [1.0, 0.1, 0.01], eps=0.5)
    log(tmp_path / 'old' / 'a', [1.0, 0.1, 0.01 * (1 + 5e-11)])
    log(tmp_path / 'new' / 'b', [1.0, 0.1, 0.01], eps=0.5)
    log(tmp_path / 'old' / 'b', [1.0, 0.1, 0.0101])
    a, b = c4.compare(tmp_path / 'new', tmp_path / 'old', runs=['a', 'b'])
    assert a['within_1e-10'] and a['eps_ref_logged'] and a['eps_ref_final'] == 0.5
    assert not b['within_1e-10'] and b['max_rel_diff_norm_R_h'] == pytest.approx(0.0099, rel=1e-2)


def test_stage_names_and_their_results_folders():
    env = (REPO / 'scripts/cluster/env.sh').read_text()
    assert 'p2s1) export PKG="$RESULTS/package2_stage1"' in env and 'p2s2) export PKG="$RESULTS/package2_stage2"' in env
    assert '^([1234]|p2s[12])$' in (REPO / 'scripts/cluster/sbatch.sh').read_text()
    job = (REPO / 'scripts/cluster/package2/p2s1_scalar_reruns.slurm').read_text()
    assert '# lilq-resources: timed-cpu' in job and '--reference-dir "$PKG/reference"' in job
    assert '--package1 "$RESULTS/wave2/B_instrumentation"' in job
