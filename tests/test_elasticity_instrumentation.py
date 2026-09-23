"""Section 3.1 instrumentation for linear elasticity (Section 3.3 asks for
logged elasticity runs at all five sizes). Linear problem, single solve:
the same two-row layout as Darcy -- k=0 (the solve, assembled at zero)
and the terminal k=1 (residual at the solution).
"""

import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger, solve_rows
from lilq.run_metadata import first_stall_iteration
from problems.elasticity import ElasticityConfig, solve_elasticity

REPO_ROOT = Path(__file__).resolve().parent.parent

# The cluster upload bundle ships without .git (scripts/make_hprc_bundle.py):
# tests that read git history skip there instead of failing.
requires_git_checkout = pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / ".git").exists(),
    reason="needs a git checkout",
)


def _small_config(**overrides):
    kwargs = dict(N_x=5, N_y=5)
    kwargs.update(overrides)
    return ElasticityConfig(**kwargs)


def test_logger_does_not_change_the_solve():
    config = _small_config()
    plain = solve_elasticity(config, verbose=False)
    logged = solve_elasticity(config, verbose=False, iteration_logger=IterationLogger())
    assert np.array_equal(plain['theta_u'], logged['theta_u'])
    assert np.array_equal(plain['theta_v'], logged['theta_v'])


@requires_git_checkout
def test_solution_bit_identical_to_the_committed_pre_instrumentation_solver(tmp_path):
    """The only solver change is making lstsq's cond=EPS_MACH explicit
    (scipy's default for gelsy): the coefficients must not move by a bit."""
    old_src = subprocess.run(
        ["git", "show", "2cf787e:problems/elasticity.py"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    old_path = tmp_path / "elasticity_old.py"
    old_path.write_text(old_src)
    spec = importlib.util.spec_from_file_location("elasticity_old", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

    for bc_mode in ("paper", "exact"):
        new_res = solve_elasticity(_small_config(bc_mode=bc_mode), verbose=False)
        old_res = old.solve_elasticity(old.ElasticityConfig(N_x=5, N_y=5, bc_mode=bc_mode), verbose=False)
        assert np.array_equal(new_res['theta_u'], old_res['theta_u'])
        assert np.array_equal(new_res['theta_v'], old_res['theta_v'])


def test_rows_follow_the_single_solve_layout():
    config = _small_config()
    logger = IterationLogger()
    result = solve_elasticity(config, verbose=False, iteration_logger=logger)

    row, terminal = logger.rows
    assert (row["k"], terminal["k"]) == (0, 1)
    assert len(solve_rows(logger.rows)) == 1
    # R^(0) at the zero vector is -f: its norm is ||f||.
    assert row["norm_R_h"] == pytest.approx(row["norm_f_h"], rel=1e-14)
    # Linear system: the operator and its linearization coincide.
    assert row["chi"] == 0.0 or row["chi"] < 1e-12
    assert row["num_rank_gelsy"] == result['n_params']
    assert row["kappa_method"] == "svd" and row["kappa"] > 1
    assert not math.isnan(row["norm_Rlin_interior"]) and not math.isnan(row["norm_R_interior"])
    assert terminal["norm_R_h"] == pytest.approx(row["norm_Rlin_h"], rel=1e-12)


def test_run_json_written_and_self_consistent(tmp_path):
    config = _small_config()
    logger = IterationLogger()
    out_path = tmp_path / "run.json"
    result = solve_elasticity(config, verbose=False, iteration_logger=logger, run_json_path=out_path)

    meta = json.loads(out_path.read_text())
    assert meta["N_total"] == sum(meta["N_composition"].values())
    assert meta["P_total"] == sum(meta["P_composition"].values()) == result['n_params']
    assert meta["K_max"] == 1
    assert meta["stopping_reason"] == "direct_solve"
    assert meta["solver_driver"] == "gelsy"
    assert meta["first_stall_iteration"] == first_stall_iteration(logger.rows) is None
    assert meta["b2_check"]["rel_err"] < 1e-12


def test_run_json_requires_iteration_logger():
    with pytest.raises(ValueError):
        solve_elasticity(_small_config(), verbose=False, run_json_path="unused.json")
