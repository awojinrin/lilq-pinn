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
        sel = ca.select(root, fam)
        assert len(sel["top"]) == 2 and sel["ranking"][0]["final_loss"] <= sel["ranking"][1]["final_loss"]
        summary = ca.full(root, fam, "cpu", budget_s=2, seeds=(0, 1))
        assert summary["representative"] in sel["top"]
        assert set(summary["best_test_errors_over_all_full_runs"]) == {"eps_u", "eps_v", "eps_p", "eps_p_meanfree"}
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
