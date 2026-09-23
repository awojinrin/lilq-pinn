"""Regression guard: the three run_*.py scripts' NiL-N/NiL-Q branches
must go through run_stochastic_with_seeds (the Section 2/3.4 multi-seed
wiring), not call their runner directly -- otherwise `--seeds` would be
silently ignored. LiL-N/LiL-Q must NOT go through it (the spec's own
exemption: deterministic, no random init to average over).

Checked at the source level, same as test_bl_experiment_runner_config.py
-- run_experiment_for_N does real file I/O (make_experiment_dir) and,
for NiL-N/NiL-Q without mocking, real gradient-descent training, neither
of which a unit test should trigger just to check which function a
branch calls. The actual multi-seed wiring correctness (does it run
every seed, pick the median, populate the aggregate) is covered by
tests/test_run_stochastic_with_seeds.py against a real Bratu NiL-N run,
and directly in this session against a real 3-seed Bratu NiL-N run
before this file was written.
"""

import inspect

import experiments.run_bratu as run_bratu_module
import experiments.run_burgers as run_burgers_module
import experiments.run_bl as run_bl_module


def _non_lil_branch_source(module):
    source = inspect.getsource(module.run_experiment_for_N)
    # Every script's structure is "if is_lil: ... else: <non-LiL branch>"
    # -- isolate the else branch so a run_stochastic_with_seeds mention
    # inside the LiL branch (there shouldn't be one) wouldn't pass this
    # by accident.
    else_idx = source.index("else:")
    return source[else_idx:]


def _lil_branch_source(module):
    source = inspect.getsource(module.run_experiment_for_N)
    if_idx = source.index("if is_lil:")
    else_idx = source.index("else:")
    return source[if_idx:else_idx]


def test_bratu_seeds_parameter_exists_and_defaults_to_none():
    sig = inspect.signature(run_bratu_module.run_experiment_for_N)
    assert "seeds" in sig.parameters
    assert sig.parameters["seeds"].default is None


def test_bratu_non_lil_branch_uses_run_stochastic_with_seeds():
    assert "run_stochastic_with_seeds" in _non_lil_branch_source(run_bratu_module)


def test_bratu_lil_branch_does_not_use_run_stochastic_with_seeds():
    assert "run_stochastic_with_seeds" not in _lil_branch_source(run_bratu_module)


def test_burgers_seeds_parameter_exists_and_defaults_to_none():
    sig = inspect.signature(run_burgers_module.run_experiment_for_N)
    assert "seeds" in sig.parameters
    assert sig.parameters["seeds"].default is None


def test_burgers_non_lil_branch_uses_run_stochastic_with_seeds():
    assert "run_stochastic_with_seeds" in _non_lil_branch_source(run_burgers_module)


def test_burgers_lil_branch_does_not_use_run_stochastic_with_seeds():
    assert "run_stochastic_with_seeds" not in _lil_branch_source(run_burgers_module)


def test_bl_seeds_parameter_exists_and_defaults_to_none():
    sig = inspect.signature(run_bl_module.run_experiment_for_N)
    assert "seeds" in sig.parameters
    assert sig.parameters["seeds"].default is None


def test_bl_non_lil_branch_uses_run_stochastic_with_seeds():
    assert "run_stochastic_with_seeds" in _non_lil_branch_source(run_bl_module)


def test_bl_lil_branch_does_not_use_run_stochastic_with_seeds():
    assert "run_stochastic_with_seeds" not in _lil_branch_source(run_bl_module)


def test_all_three_scripts_expose_seeds_cli_flag():
    for module in (run_bratu_module, run_burgers_module, run_bl_module):
        main_source = inspect.getsource(module.main)
        assert "'--seeds'" in main_source or '"--seeds"' in main_source
