"""The NiL runners must seed the network initialization from
``init_seed`` while keeping the collocation set on ``config.seed`` (42).
Previously both came from ``config.seed``, so the seeds-0/1/2 runs of the
four-method tables trained NiL-N/NiL-Q on different collocation grids
than the seed-42 grid every other method used (DECISIONS.md).
"""

import dataclasses

import pytest

import problems.bratu as bratu
import problems.burgers as burgers
import problems.buckley_leverett as bl

CASES = [
    (bratu, bratu.BratuConfig(N_x=3, N_y=3),
     bratu.BratuOptConfig(max_iterations=1, max_quasi_iters_nn=1, max_inner_iters_nn=1, pretrain_epochs=1)),
    (burgers, burgers.BurgersConfig(N_x=3, N_t=3),
     burgers.BurgersOptConfig(max_iterations=1, max_quasi_iters_nn=1, max_inner_iters_nn=1, pretrain_epochs=1)),
    (bl, bl.BLConfig(N_x=3, N_t=3),
     bl.BLOptConfig(max_iterations=1, max_quasi_iters_nn=1, max_inner_iters_nn=1, pretrain_epochs=1)),
]


def _record_seeds(monkeypatch, module):
    seen = {"set_seed": [], "collocation": []}
    real_set_seed = module.set_seed
    real_colloc = module.generate_collocation_points_2d

    def fake_set_seed(seed, *a, **kw):
        seen["set_seed"].append(seed)
        return real_set_seed(seed, *a, **kw)

    def fake_colloc(*a, **kw):
        seen["collocation"].append(kw["seed"])
        return real_colloc(*a, **kw)

    monkeypatch.setattr(module, "set_seed", fake_set_seed)
    monkeypatch.setattr(module, "generate_collocation_points_2d", fake_colloc)
    return seen


@pytest.mark.parametrize("module,config,opt", CASES, ids=["bratu", "burgers", "bl"])
@pytest.mark.parametrize("runner_name", ["run_nil_n", "run_nil_q"])
def test_init_seed_varies_network_only(monkeypatch, module, config, opt, runner_name):
    seen = _record_seeds(monkeypatch, module)
    getattr(module, runner_name)(dataclasses.replace(config, init_seed=1), opt,
                                 device="cpu", verbose=False)
    assert seen["set_seed"] == [1]
    assert seen["collocation"] == [42]


@pytest.mark.parametrize("module,config,opt", CASES, ids=["bratu", "burgers", "bl"])
def test_default_init_seed_is_config_seed(monkeypatch, module, config, opt):
    seen = _record_seeds(monkeypatch, module)
    module.run_nil_n(config, opt, device="cpu", verbose=False)
    assert seen["set_seed"] == [42]
    assert seen["collocation"] == [42]


def test_different_init_seeds_give_different_networks_same_points():
    config, opt = CASES[0][1], CASES[0][2]
    runs = [bratu.run_nil_n(dataclasses.replace(config, init_seed=s), opt, device="cpu", verbose=False)
            for s in (0, 1)]
    w0 = next(runs[0][0].parameters()).detach()
    w1 = next(runs[1][0].parameters()).detach()
    assert not bool((w0 == w1).all())
