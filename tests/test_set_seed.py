"""Regression guard for set_seed()'s CUDA/cuDNN behavior.

Context (DECISIONS.md, Q6 finding + this session's Phase 1 follow-up):
set_seed() used to unconditionally call torch.cuda.manual_seed_all() and
force cudnn.deterministic=True/benchmark=False, applied identically to
every problem regardless of whether it uses CUDA at all. Confirmed to
cause a ~1.7x wall-clock regression on Beltrami (pure CPU
scipy.linalg.lstsq, no PyTorch beyond a capability check) simply from the
unnecessary CUDA context initialization. Also confirmed unnecessary for
every GPU-using problem in this codebase: no Dropout/BatchNorm/Conv
layers exist anywhere (grep-verified), so there is no GPU-side random
operation and no convolution for cuDNN to benchmark -- CUDA-level
determinism has zero observable effect on any current result.

Fixed by making it opt-in (deterministic_cuda=False by default).
"""

import torch

from lilq.utils import set_seed


def test_default_does_not_force_cudnn_determinism():
    # Perturb the flags first so we can tell set_seed left them alone.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(42)

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_explicit_opt_in_forces_cudnn_determinism():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(42, deterministic_cuda=True)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    # Restore, so this test doesn't leak state into whichever test runs next.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def test_cpu_seed_reproducibility_unaffected():
    """The one thing that actually matters for this codebase's weight
    init -- CPU-side reproducibility -- must be unchanged by this fix."""
    set_seed(123)
    a = torch.nn.Linear(4, 4).weight.clone()
    set_seed(123)
    b = torch.nn.Linear(4, 4).weight.clone()
    assert torch.equal(a, b)
