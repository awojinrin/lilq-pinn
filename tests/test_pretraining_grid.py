"""Regression guard for the pretraining fit grid (DECISIONS.md, 2026-09-24).

Every problem pretrains on one fixed 50 x 50 grid, for the NN fit and the
LiL coefficient fit alike, at every size -- the GitHub reference. (A
per-problem, size-scaled grid restored from the pre-GitHub code was
reverted: it was sparser at every P used and differed across problems for
no stated reason.)
"""

import math

import numpy as np
import pytest

from lilq.basis import Chebyshev1D, TensorProductBasis2D
from lilq.pretraining import pretrain_lil
from problems.bratu import BratuOptConfig
from problems.buckley_leverett import BLOptConfig
from problems.burgers import BurgersOptConfig


@pytest.mark.parametrize("opt_cls", [BratuOptConfig, BurgersOptConfig, BLOptConfig])
def test_every_problem_pretrains_on_the_same_50_grid(opt_cls):
    assert opt_cls().pretrain_grid == 50


@pytest.mark.parametrize("n_per_dim", [5, 15, 32])  # P = 25, 225, 1024
def test_pretrain_lil_fits_on_50x50_at_every_package_size(n_per_dim):
    basis = TensorProductBasis2D(Chebyshev1D(order=n_per_dim - 1), Chebyshev1D(order=n_per_dim - 1))
    call_sizes = []

    def initial_guess_fn(x, y):
        call_sizes.append(len(x))
        return np.zeros_like(x)

    pretrain_lil(basis, initial_guess_fn, (-1.0, 1.0), (-1.0, 1.0), n_grid=50, verbose=False)
    # max(50, ceil(sqrt(2P))) points per direction; 50 for every P <= 1,250.
    assert math.ceil(math.sqrt(2 * basis.n_basis)) <= 50
    assert call_sizes == [50 * 50]
