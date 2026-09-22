"""Regression guard: no module should mutate torch's process-wide default
dtype as an import side effect; float64 must come from explicit,
per-construction dtype arguments instead.

Context (DECISIONS.md): lilq/solvers.py used to call
torch.set_default_dtype(torch.float64) at import time, silently changing
every later nn.Module construction anywhere in the process -- regardless
of which file did the constructing -- based on import order rather than
an explicit choice at the call site. This is the same class of problem as
set_seed()'s old unconditional CUDA touching: a shared setup step
mutating global state instead of taking explicit configuration.

Fixed by giving MLP its own dtype=torch.float64 default (matching every
current NiL-N/NiL-Q call site's existing behavior with no call-site
changes needed) and making every bare torch.tensor/linspace/zeros/full
construction across problems/*.py explicit about its dtype.
"""

import subprocess
import sys

import torch

from lilq.nn import MLP


def test_importing_solvers_does_not_change_global_default_dtype():
    # Run in a fresh subprocess: torch's default dtype is process-global,
    # so this must not be checked in a process where something else may
    # have already touched it (e.g. an earlier test in the same session).
    code = (
        "import torch; "
        "before = torch.get_default_dtype(); "
        "import lilq.solvers; "
        "after = torch.get_default_dtype(); "
        "assert before == after, (before, after); "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_mlp_defaults_to_float64_without_relying_on_global_state():
    saved = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)  # deliberately hostile global state
        model = MLP(hidden_dim=8, num_layers=2)
        for p in model.parameters():
            assert p.dtype == torch.float64
    finally:
        torch.set_default_dtype(saved)


def test_mlp_float32_still_available_for_darcy():
    model = MLP(hidden_dim=8, num_layers=2, dtype=torch.float32)
    for p in model.parameters():
        assert p.dtype == torch.float32
