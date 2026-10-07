"""Beltrami's CPU path is bit-identical to the release (ab0484d, the release
check on main; problems/beltrami.py is unchanged from v2.0.0 to there). The
GPU path was repaired in Package 2, Stage 2, batch 4 (``_lstsq``), and the
advisor accepted the repair on this statement (reply of 5 October,
Section 2, 3.6).

The release tree is taken from git (``git archive``) and run beside this
checkout, on the same machine, in two subprocesses, so the comparison is
exact on any machine. Without git or the commit (the cluster bundle has no
history), the test is skipped."""

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RELEASE = 'ab0484d'

# The paper's pinned configuration (run_beltrami_pinned.pinned_config) at a small
# size: the same basis, pins and collocation rule, three iterations. With and
# without the diagnostics, the coefficients after every run, and the logged rows.
SCRIPT = '''
import dataclasses, sys
sys.path.insert(0, sys.argv[1])
import lilq.blas_threads  # noqa: F401
import numpy as np
from experiments.run_beltrami_pinned import pinned_config
from lilq.iteration_log import IterationLogger
from problems.beltrami import solve_beltrami
config = dataclasses.replace(pinned_config(), N_vel=3, N_p=4, N_x=5, N_y=5, N_z=5, N_t=5, N_bc=3, N_t_bc=3, N_ic=5,
                             max_iter=3, tol=0.0)
out = {}
r = solve_beltrami(config, verbose=False, diagnostics=False)
for k in ('theta_u', 'theta_v', 'theta_w', 'theta_p'):
    out['plain_' + k] = np.asarray(r[k])
out['plain_coeff_change'] = np.asarray(r['history']['coeff_change'])
logger = IterationLogger()
r = solve_beltrami(config, verbose=False, iteration_logger=logger)
for k in ('theta_u', 'theta_v', 'theta_w', 'theta_p'):
    out['diag_' + k] = np.asarray(r[k])
for col in ('norm_R_h', 'norm_Rlin_h', 'kappa', 'num_rank_gelsy', 'eps_u', 'norm_beta'):
    out['log_' + col] = np.array([np.nan if row.get(col) in (None, '') else float(row[col]) for row in logger.rows])
np.savez(sys.argv[2], **out)
'''


def _release_tree(tmp_path):
    try:
        ok = subprocess.run(['git', 'cat-file', '-e', f'{RELEASE}^{{commit}}'], cwd=REPO, capture_output=True)
    except OSError:
        pytest.skip('no git')
    if ok.returncode != 0:
        pytest.skip(f'the release commit {RELEASE} is not in this checkout')
    archive = tmp_path / 'release.zip'
    subprocess.run(['git', 'archive', '--format=zip', '-o', str(archive), RELEASE], cwd=REPO, check=True)
    root = tmp_path / 'release'
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    return root


def _run(root, out, tmp_path):
    script = tmp_path / 'beltrami_identity.py'
    script.write_text(SCRIPT)
    subprocess.run([sys.executable, str(script), str(root), str(out)], check=True, cwd=root,
                   capture_output=True, text=True)
    with np.load(out) as f:
        return {k: f[k] for k in f.files}


def test_beltrami_cpu_path_is_bit_identical_to_the_release(tmp_path):
    release = _run(_release_tree(tmp_path), tmp_path / 'release.npz', tmp_path)
    current = _run(REPO, tmp_path / 'current.npz', tmp_path)
    assert set(release) == set(current)
    for k in release:
        assert np.array_equal(release[k], current[k], equal_nan=True), k
    assert release['plain_theta_u'].size > 0 and len(release['plain_coeff_change']) == 3
