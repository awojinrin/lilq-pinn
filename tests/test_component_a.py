"""Component A driver: every stage end to end on tiny configurations and
second-long budgets (the real budgets are 600 s and 3,600 s)."""

import json

import pytest
import torch

import baselines.f1_pinn as f1
import experiments.component_a as ca

F1_TINY = [dict(id=f"F1_{i:02d}", family="F1", width=12, depth=2, trunk="shared", m=6, sigma_ff=1.0,
                bc=bc, lambda_bc=lam, balancing=False, eta=1e-3, t_adam=50000, n_int=48, resample=False)
           for i, (bc, lam) in enumerate([("hard", None), ("soft", 10.0)])]
F2_TINY = [dict(id=f"F2_{i:02d}", family="F2", width=10, depth=2, m=6, sigma_ff=s, n_int=120)
           for i, s in enumerate([1.0, 2.0])]


@pytest.fixture(autouse=True)
def restore_default_dtype():
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "search").mkdir()
    for fam, cfgs in (("F1", F1_TINY), ("F2", F2_TINY)):
        (tmp_path / "search" / f"{fam}_configs.json").write_text(json.dumps({"configs": cfgs}))
    monkeypatch.setattr(ca, "TOP_K", 2)
    return tmp_path


def test_every_stage_end_to_end(root, monkeypatch):
    for fam in ca.FAMILIES:
        runs = ca.screen(root, fam, "cpu", budget_s=2)
        assert len(runs) == 2 and all("final_loss" in r for r in runs.values())
        # Every run made by this code records its validation residual in run.json.
        assert all(isinstance(r["val_residual"], float) for r in runs.values())
        sel = ca.select(root, fam)
        assert len(sel["top"]) == 2 and sel["ranking"][0]["val_residual"] <= sel["ranking"][1]["val_residual"]
        assert sel["selection_criterion"] == "validation_residual" and len(sel["ranking_by_training_loss"]) == 2
        summary = ca.full(root, fam, "cpu", budget_s=2, seeds=(0, 1))
        assert summary["representative"] in sel["top"]
        assert summary["representative"] == min(summary["median_val_residual"], key=summary["median_val_residual"].get)
        # F1: pressure only mean-free (Addendum v2.2 Section 1, item 9); F2 keeps the pinned one too.
        expected = {"eps_u", "eps_v", "eps_p_meanfree"} | ({"eps_p"} if fam == "F2" else set())
        assert set(summary["best_test_errors_over_all_full_runs"]) == expected
        assert len(ca.cpu_reruns(root, fam, budget_s=2, seeds=(0,))) == 1
    assert ca.float32_run(root, "cpu", budget_s=2)["precision"] == "adam32"
    monkeypatch.setattr(f1, "A1_CONFIG", dict(F1_TINY[1], id="A1_tiny"))
    a1 = ca.check_a1(root, "cpu", budget_s=2)
    assert set(a1) == {"eps_u", "target", "budget_s", "passed"}

    # Resume: nothing is rerun, so the log does not grow.
    log = (root / "tuning_log.md").read_text(encoding="utf-8")
    n_runs = log.count("\n- ")
    assert n_runs == 2 * (2 + 4 + 1) + 2
    ca.screen(root, "F1", "cpu", budget_s=2)
    assert (root / "tuning_log.md").read_text(encoding="utf-8").count("\n- ") == n_runs


def test_f2_jacobian_check(tmp_path):
    result = ca.check_f2_jacobian(tmp_path, "cpu")
    assert result["passed"] and result["max_relative_fd_error"] < 1e-6
    assert json.loads((tmp_path / "checks" / "f2_jacobian.json").read_text())["passed"]


def test_check_a2(tmp_path, monkeypatch):
    result = ca.check_a2(tmp_path, "cpu", steps=5)
    assert result["passed"] and result["steps"] == [5, 5] and result["max_relative_difference"] <= 1e-12


def test_a1_is_a_gate(tmp_path, monkeypatch):
    """Addendum v2.2 Section 2.6: ``component_a.py a1`` exits non-zero when
    the plain PINN misses eps_u <= 1e-3 (and zero when it reaches it)."""
    import sys
    for passed, code in ((False, 1), (True, 0)):
        monkeypatch.setattr(ca, "check_a1", lambda root, device, budget_s, p=passed: {"passed": p})
        monkeypatch.setattr(sys, "argv", ["component_a.py", "a1", "--root", str(tmp_path)])
        with pytest.raises(SystemExit) as exit_info:
            ca.main()
        assert exit_info.value.code == code


def test_component_a_test_list_names_real_files():
    """env.sh's list of Component A-only tests (run by 29_A1_gate, skipped by
    the preflight) must name files that exist."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = (root / "scripts" / "cluster" / "env.sh").read_text()
    files = re.search(r'LILQ_COMPONENT_A_TESTS="([^"]+)"', env).group(1).split()
    assert files and all((root / f).is_file() for f in files)
    assert "tests/test_component_a.py" in files
