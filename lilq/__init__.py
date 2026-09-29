"""
LiL-Q: Linear-in-Learnables Quasilinearized Solvers
=====================================================

A unified framework for solving PDEs using four numerical methods:
    - NiL-N: Standard PINN (Nonlinear-in-Learnables, Nonlinear PDE)
    - NiL-Q: Quasilinear PINN (Nonlinear-in-Learnables, Quasilinearized PDE)
    - LiL-N: Nonlinear LiL (Linear-in-Learnables, Nonlinear PDE)
    - LiL-Q: Quasilinear LiL (Linear-in-Learnables, Quasilinearized PDE)

Submodules:
    basis        - 1D/2D/ND basis functions (Chebyshev, Fourier, ELM, tensor products)
    nn           - Neural network architectures (MLP)
    metrics      - Experiment metric trackers
    collocation  - Collocation point generation
    pretraining  - NN and basis coefficient pretraining
    solvers      - Generic solver templates for all four methods
    style        - Publication-quality matplotlib styling (CMAME/Elsevier)
    plotting     - Convergence and solution field visualization
    analysis     - SVD, condition number studies (opt-in)
    properties   - Theorem 2 residual bounds validation
    utils        - Reproducibility, GPU management, checkpointing
"""

__version__ = "0.1.0"

# Submodules load on first use (PEP 562), not here: importing any lilq
# module -- above all ``lilq.blas_threads``, which must set the BLAS thread
# counts before NumPy, SciPy or PyTorch load -- must not pull those libraries
# in first (Addendum v2.2 Section 2.12). ``lilq.basis`` and
# ``from lilq import basis`` work as before.
import importlib as _importlib

__all__ = [
    "basis", "nn", "metrics", "collocation", "pretraining",
    "solvers", "style", "plotting", "analysis", "properties", "utils",
]


def __getattr__(name):
    if name in __all__:
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'lilq' has no attribute {name!r}")
