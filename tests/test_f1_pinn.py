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
    phases = [r["phase"] for r in rows]
    assert phases[:2] == ["adam", "adam"] and set(phases[2:]) == {"lbfgs"}
    # One L-BFGS row every 10 iterations inside a call (Addendum v2.2 2.5).
    iters = [int(r["iter"]) for r in rows[2:]]
    assert len(iters) > 1 and all(b - a == f1.LBFGS_LOG_EVERY for a, b in zip(iters, iters[1:-1]))
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


def test_budget_holds_inside_an_lbfgs_call(tmp_path):
    """A 500-iteration L-BFGS call must not run past the budget: the run
    stops at the next evaluation and keeps the lowest-loss point."""
    cfg = dict(TINY, t_adam=20, n_int=400, width=32)
    run = f1.f1_train(cfg, 0, 3.0, tmp_path, test_every=10 ** 6)
    rows = _read(tmp_path / "log.csv")
    assert run["end_reason"] == "budget" and rows[-1]["phase"] == "lbfgs"
    assert run["wall_s"] < 3.0 + 1.0
    assert run["final_loss"] <= float(rows[-1]["loss_total"]) + 1e-15


# ── Addendum v2.2 Section 2.5 ────────────────────────────────────────────────

def test_logging_every_10_iterations_leaves_the_trajectory_unchanged(tmp_path, monkeypatch):
    """A call run in 10-iteration pieces (state carried over, each point
    evaluated once) ends at the same parameters, bit for bit, as one piece."""
    cfg = dict(TINY, n_int=128)
    states = {}
    for piece in (500, f1.LBFGS_LOG_EVERY):
        monkeypatch.setattr(f1, "LBFGS_LOG_EVERY", piece)
        run = f1.f1_train(cfg, 0, 600, tmp_path / str(piece), max_adam_iters=50, max_lbfgs_calls=1,
                          test_every=10 ** 6)
        states[piece] = (torch.load(tmp_path / str(piece) / "model.pt"), run)
    (a, run_a), (b, run_b) = states[500], states[f1.LBFGS_LOG_EVERY]
    assert run_a["lbfgs_iters"] == run_b["lbfgs_iters"] > 0
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert run_a["lbfgs_evaluations"] == run_b["lbfgs_evaluations"]


def test_interrupted_call_logs_its_best_point_inside_the_budget(tmp_path):
    cfg = dict(TINY, n_int=4096, width=64)
    run = f1.f1_train(cfg, 0, 3.0, tmp_path, max_adam_iters=20, test_every=10 ** 6)
    rows = _read(tmp_path / "log.csv")
    assert run["end_reason"] == "budget" and rows[-1]["phase"] == "lbfgs"
    assert float(rows[-1]["t_cum_s"]) <= 3.0                 # logged when evaluated, not after the budget
    assert float(rows[-1]["eps_u"]) == pytest.approx(run["eps_u"], rel=1e-12)   # the restored point


def test_no_progress_restarts_once_then_ends(tmp_path, monkeypatch):
    """A loss with zero gradient: L-BFGS can never lower it. The second call
    makes no progress, a fresh optimizer is tried once, and the run ends."""
    real = f1.loss_terms
    monkeypatch.setattr(f1, "loss_terms", lambda m, a, b: {k: 0.0 * v + 1.0 for k, v in real(m, a, b).items()})
    run = f1.f1_train(TINY, 0, 60, tmp_path, max_adam_iters=5, test_every=10 ** 6)
    assert run["end_reason"].startswith("criterion") and run["lbfgs_restarts"] == 1
    assert run["lbfgs_calls"] == 3


def test_lbfgs_tolerances_are_off():
    opt = f1.new_lbfgs([torch.nn.Parameter(torch.zeros(2))])
    group = opt.param_groups[0]
    assert group["tolerance_grad"] == 0.0 and group["tolerance_change"] == 0.0


def test_unweighted_final_loss_is_stored(tmp_path):
    cfg = dict(TINY, bc="soft", lambda_bc=10.0)
    run = f1.f1_train(cfg, 0, 60, tmp_path, max_adam_iters=20, max_lbfgs_calls=1, test_every=10 ** 6)
    model = f1.F1Model(8, 2, 8, 1.0, "shared", "soft")
    del model
    terms = run["final_loss"] - run["final_loss_unweighted"]      # weights 1, 1, 1, 10 without balancing
    assert run["final_loss_unweighted"] > 0 and terms > 0
    rows = _read(tmp_path / "log.csv")
    last = {k: float(rows[-1][f"loss_{k}"]) for k in ("xmom", "ymom", "cont", "bc")}
    assert terms == pytest.approx(9 * last["bc"], rel=1e-9)
    assert run["final_loss_unweighted"] == pytest.approx(sum(last.values()), rel=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_peak_memory_is_per_run(tmp_path):
    big = torch.empty(64 * 2 ** 20, device="cuda")          # 256 MB before the run
    del big
    run = f1.f1_train(TINY, 0, 60, tmp_path, device="cuda", max_adam_iters=5, max_lbfgs_calls=1,
                      test_every=10 ** 6)
    assert run["peak_gpu_bytes"] < 200 * 2 ** 20
