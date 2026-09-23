"""Tests for lilq.multiseed -- the seed-sweep/aggregation harness
implementing Computational_Package_1_v2.md Section 2's "run every
stochastic method at multiple seeds, report median/min/max, never a
single run" requirement.
"""

import dataclasses
from typing import Optional
import statistics

import pytest

from lilq.utils import nn_init_seed

from lilq.multiseed import aggregate_summaries, run_multiseed


@dataclasses.dataclass
class _FakeConfig:
    seed: int = 42
    N: int = 10
    init_seed: Optional[int] = None


def test_run_multiseed_threads_seed_into_each_call():
    seen_seeds = []

    def fake_runner(config):
        seen_seeds.append(nn_init_seed(config))
        return ({}, {}, {"final_loss": nn_init_seed(config) / 10.0, "converged": True})

    result = run_multiseed(fake_runner, _FakeConfig(seed=999), seeds=[0, 1, 2])

    assert seen_seeds == [0, 1, 2]
    assert set(result["per_seed"].keys()) == {0, 1, 2}


def test_run_multiseed_does_not_mutate_original_config():
    original = _FakeConfig(seed=999, N=10)

    def fake_runner(config):
        return ({}, {"final_loss": 1.0, "converged": True})

    run_multiseed(fake_runner, original, seeds=[0, 1, 2])

    assert original.seed == 999  # unchanged


def test_run_multiseed_forwards_extra_args_and_kwargs():
    calls = []

    def fake_runner(config, positional_arg, keyword_arg=None):
        calls.append((positional_arg, keyword_arg))
        return ({"final_loss": 1.0, "converged": True},)

    run_multiseed(fake_runner, _FakeConfig(), [0], "pos", keyword_arg="kw")

    assert calls == [("pos", "kw")]


def test_run_multiseed_rejects_empty_seed_list():
    with pytest.raises(ValueError):
        run_multiseed(lambda config: ({},), _FakeConfig(), seeds=[])


def test_run_multiseed_preserves_last_element_as_summary_for_nil_and_lil_shapes():
    # NiL-style: (model, metrics, summary). LiL-style: (basis, coeffs, metrics, summary).
    def nil_style_runner(config):
        return ("model", "metrics", {"final_loss": 0.1, "converged": True})

    def lil_style_runner(config):
        return ("basis", "coeffs", "metrics", {"final_loss": 0.05, "converged": True})

    nil_result = run_multiseed(nil_style_runner, _FakeConfig(), seeds=[0])
    lil_result = run_multiseed(lil_style_runner, _FakeConfig(), seeds=[0])

    assert nil_result["per_seed"][0] == {"final_loss": 0.1, "converged": True}
    assert lil_result["per_seed"][0] == {"final_loss": 0.05, "converged": True}


def test_aggregate_summaries_computes_median_min_max():
    per_seed = {
        0: {"final_loss": 0.30, "total_iterations": 100, "converged": True},
        1: {"final_loss": 0.10, "total_iterations": 300, "converged": True},
        2: {"final_loss": 0.20, "total_iterations": 200, "converged": False},
    }

    agg = aggregate_summaries(per_seed)

    assert agg["final_loss"]["median"] == 0.20
    assert agg["final_loss"]["min"] == 0.10
    assert agg["final_loss"]["max"] == 0.30
    assert sorted(agg["final_loss"]["values"]) == [0.10, 0.20, 0.30]

    assert agg["total_iterations"]["median"] == 200
    assert agg["total_iterations"]["min"] == 100
    assert agg["total_iterations"]["max"] == 300


def test_aggregate_summaries_reports_convergence_rate_not_a_median_of_booleans():
    per_seed = {
        0: {"final_loss": 1.0, "converged": True},
        1: {"final_loss": 1.0, "converged": True},
        2: {"final_loss": 1.0, "converged": False},
    }

    agg = aggregate_summaries(per_seed)

    assert agg["converged"] == {"n_converged": 2, "n_total": 3, "rate": pytest.approx(2 / 3)}


def test_aggregate_summaries_skips_fields_absent_from_every_summary():
    per_seed = {0: {"final_loss": 1.0}, 1: {"final_loss": 2.0}}
    agg = aggregate_summaries(per_seed)
    assert "final_pde_loss" not in agg
    assert "converged" not in agg


def test_aggregate_summaries_matches_stdlib_statistics_median_on_even_count():
    per_seed = {i: {"final_loss": float(i)} for i in range(4)}  # 0,1,2,3
    agg = aggregate_summaries(per_seed)
    assert agg["final_loss"]["median"] == statistics.median([0.0, 1.0, 2.0, 3.0])


def test_run_multiseed_varies_only_the_init_seed_not_the_collocation_seed():
    """config.seed also fixes the collocation set, which must stay at the
    paper's 42 for every seed -- only the network init seed varies."""
    seen = []

    def fake_runner(config):
        seen.append((config.seed, config.init_seed))
        return ({"final_loss": 0.0, "converged": True},)

    run_multiseed(fake_runner, _FakeConfig(seed=42), seeds=[0, 1, 2])

    assert seen == [(42, 0), (42, 1), (42, 2)]
