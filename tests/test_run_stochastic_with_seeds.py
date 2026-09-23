"""Tests for experiments.exp_utils.run_stochastic_with_seeds -- the
multi-seed wiring helper (Computational_Package_1_v2.md Section 2/3.4)
used by run_bratu.py/run_burgers.py/run_bl.py's NiL-N/NiL-Q branches.
"""

import dataclasses
from typing import Optional

import pytest

from lilq.utils import nn_init_seed

from experiments.exp_utils import run_stochastic_with_seeds


@dataclasses.dataclass
class _FakeConfig:
    seed: int = 42
    N: int = 10
    init_seed: Optional[int] = None


def _fake_runner(config, opt, device=None, verbose=True):
    # final_loss deterministic in seed so the "median seed" is predictable.
    loss_by_seed = {0: 0.30, 1: 0.10, 2: 0.20}
    loss = loss_by_seed.get(nn_init_seed(config), 1.0)
    model = f"model-seed-{nn_init_seed(config)}"
    metrics = f"metrics-seed-{nn_init_seed(config)}"
    return model, metrics, {
        "final_loss": loss, "converged": True,
        "total_iterations": 100 + nn_init_seed(config),
        "training_time": 1.0 + nn_init_seed(config),
    }


def test_no_seeds_calls_runner_once_directly():
    calls = []

    def runner(config, opt, device=None, verbose=True):
        calls.append(nn_init_seed(config))
        return "model", "metrics", {"final_loss": 1.0, "converged": True}

    model, metrics, summary = run_stochastic_with_seeds(
        runner, _FakeConfig(seed=999), opt=None, seeds=None, device="cpu",
    )

    assert calls == [999]  # config's own seed, unchanged
    assert model == "model"
    assert "multiseed_aggregate" not in summary


def test_empty_seed_list_behaves_like_no_seeds():
    calls = []

    def runner(config, opt, device=None, verbose=True):
        calls.append(nn_init_seed(config))
        return "model", "metrics", {"final_loss": 1.0, "converged": True}

    run_stochastic_with_seeds(runner, _FakeConfig(seed=7), opt=None, seeds=[], device="cpu")
    assert calls == [7]


def test_seeds_given_runs_all_and_picks_median_seed_as_representative():
    model, metrics, summary = run_stochastic_with_seeds(
        _fake_runner, _FakeConfig(seed=999), opt=None, seeds=[0, 1, 2], device="cpu",
    )

    # seed 2 has final_loss=0.20, the median of {0.30, 0.10, 0.20}.
    assert model == "model-seed-2"
    assert metrics == "metrics-seed-2"
    assert summary["final_loss"] == 0.20
    assert summary["multiseed_representative_seed"] == 2
    assert summary["multiseed_seeds"] == [0, 1, 2]


def test_seeds_given_summary_carries_full_aggregate():
    _model, _metrics, summary = run_stochastic_with_seeds(
        _fake_runner, _FakeConfig(seed=999), opt=None, seeds=[0, 1, 2], device="cpu",
    )

    agg = summary["multiseed_aggregate"]
    assert agg["final_loss"]["median"] == pytest.approx(0.20)
    assert agg["final_loss"]["min"] == pytest.approx(0.10)
    assert agg["final_loss"]["max"] == pytest.approx(0.30)
    assert sorted(agg["final_loss"]["values"]) == pytest.approx([0.10, 0.20, 0.30])
    assert agg["converged"]["rate"] == 1.0


def test_seeds_given_does_not_mutate_original_config():
    original = _FakeConfig(seed=999, N=10)
    run_stochastic_with_seeds(_fake_runner, original, opt=None, seeds=[0, 1, 2], device="cpu")
    assert original.seed == 999


def test_forwards_opt_and_device_and_verbose():
    seen = []

    def runner(config, opt, device=None, verbose=True):
        seen.append((opt, device, verbose))
        return "m", "me", {"final_loss": nn_init_seed(config) / 10.0, "converged": True}

    run_stochastic_with_seeds(runner, _FakeConfig(), opt="OPT", seeds=[0, 1], device="DEV", verbose=False)
    assert seen == [("OPT", "DEV", False), ("OPT", "DEV", False)]


def test_multiseed_keeps_collocation_seed_fixed():
    seen = []

    def runner(config, opt, device=None, verbose=True):
        seen.append(config.seed)
        return "m", "me", {"final_loss": 0.1, "converged": True}

    run_stochastic_with_seeds(runner, _FakeConfig(seed=42), opt=None, seeds=[0, 1, 2], device="cpu")
    assert seen == [42, 42, 42]
