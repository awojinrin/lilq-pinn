"""Component A, family F1 (Package 1 v2.0 Section 4.2)."""

import csv
import json

import pytest
import torch

import baselines.f1_pinn as f1
import baselines.lm_kovasznay as lm

TINY = dict(id="tiny", family="F1", width=16, depth=2, trunk="shared", m=8, sigma_ff=1.0, bc="hard",
            lambda_bc=None, balancing=False, eta=1e-3, t_adam=150, n_int=64, resample=True)


@pytest.fixture(autouse=True)
def float64_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_learning_rate_schedule():
    eta = 1e-3
    assert f1.adam_lr(0, eta) == pytest.approx(eta / 1000)
    assert f1.adam_lr(999, eta) == pytest.approx(eta)
    assert f1.adam_lr(1999, eta) == pytest.approx(eta)
    assert f1.adam_lr(2000, eta) == pytest.approx(0.9 * eta)
    assert f1.adam_lr(4999, eta) == pytest.approx(0.81 * eta)


def test_balancing_update():
    w = f1.balanced_weights({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 3.0})
    assert w["a"] == pytest.approx(0.9 + 0.1 * 4.0)
    assert w["b"] == pytest.approx(0.9 + 0.1 * 4.0 / 3.0)


def test_exact_solution_has_zero_residual():
    xy = torch.rand(20, 2) * torch.tensor([1.5, 2.0]) + torch.tensor([lm.X0, lm.Y0])
    r = f1.ns_residuals(lambda z: torch.stack(lm.exact(z[:, 0], z[:, 1]), dim=1), xy)
    assert max(t.abs().max().item() for t in r) < 1e-10


def test_hard_boundary_values():
    model = f1.F1Model(16, 2, 8, 1.0, "separate", "hard")
    b = f1.boundary_points(7, torch.float64, "cpu")
    out = model(b)
    ue, ve, _ = lm.exact(b[:, 0], b[:, 1])
    assert (out[:, 0] - ue).abs().max() < 1e-14 and (out[:, 1] - ve).abs().max() < 1e-14


def test_residual_matches_the_f2_reference_on_the_same_network():
    """Same weights in F1's shared trunk and in the advisor's F2 network:
    the batched F1 residual must equal the reference's per-point one."""
    ref = lm.FourierMLP(12, 2, 8, 1.0, 0)
    model = f1.F1Model(12, 2, 8, 1.0, "shared", "hard")
    model.nets[0].load_state_dict(ref.state_dict())
    xy = torch.tensor([[0.1, 0.3], [0.7, -0.2], [-0.3, 1.1]])
    res = lm.Residual(ref, xy, torch.tensor([lm.X0, lm.Y0]))
    theta = torch.cat([p.detach().reshape(-1) for p in ref.parameters()])
    ours = torch.stack(f1.ns_residuals(model, xy), dim=1)
    theirs = torch.stack([res.point_residual(theta, z) for z in xy])
    assert (ours - theirs).abs().max() < 1e-12


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_short_run_logs_both_phases_and_is_deterministic(tmp_path):
    runs = [f1.f1_train(TINY, 0, 120, tmp_path / name, max_adam_iters=150, max_lbfgs_calls=1)
            for name in ("a", "b")]
    rows = _read(tmp_path / "a" / "log.csv")
    assert tuple(rows[0]) == f1.LOG_COLUMNS
    assert [r["phase"] for r in rows] == ["adam", "adam", "lbfgs"]
    assert rows[0]["eps_u"] != "" and rows[0]["loss_bc"] == ""
    assert [r["loss_total"] for r in rows] == [r["loss_total"] for r in _read(tmp_path / "b" / "log.csv")]
    run = json.loads((tmp_path / "a" / "run.json").read_text())
    assert run["n_theta"] == sum(p.numel() for p in f1.F1Model(16, 2, 8, 1.0).parameters())
    assert run["adam_iters"] == 150 and run["lbfgs_calls"] == 1 and runs[0]["eps_u"] is not None


def test_soft_balanced_adam32_run(tmp_path):
    cfg = dict(TINY, bc="soft", lambda_bc=10.0, balancing=True, trunk="separate", t_adam=120)
    run = f1.f1_train(cfg, 1, 120, tmp_path, precision="adam32", max_adam_iters=120, max_lbfgs_calls=1)
    rows = _read(tmp_path / "log.csv")
    assert rows[0]["loss_bc"] != "" and float(rows[1]["w_bc"]) != 10.0      # balancing moved it
    assert run["precision"] == "adam32" and run["end_reason"] == "budget"
    state = torch.load(tmp_path / "model.pt")
    assert all(v.dtype == torch.float64 for v in state.values())         # L-BFGS ran in float64
