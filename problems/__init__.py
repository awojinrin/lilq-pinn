"""
Problems Package for LiL-Q
============================

Each module defines the PDE-specific physics (residuals, linearisation,
boundary/initial conditions) that plug into the generic solver templates
in ``lilq.solvers``.
"""

from . import bratu
from . import burgers
from . import buckley_leverett
from . import kovasznay
from . import beltrami
from . import elasticity
from . import darcy

__all__ = [
    "bratu", "burgers", "buckley_leverett",
    "kovasznay", "beltrami", "elasticity", "darcy",
]
