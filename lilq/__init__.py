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

from . import basis
from . import nn
from . import metrics
from . import collocation
from . import pretraining
from . import solvers
from . import style
from . import plotting
from . import analysis
from . import properties
from . import utils

__all__ = [
    "basis", "nn", "metrics", "collocation", "pretraining",
    "solvers", "style", "plotting", "analysis", "properties", "utils",
]
