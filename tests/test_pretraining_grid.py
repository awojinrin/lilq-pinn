"""Regression guard for the pretraining fit-grid density.

Context (Codebase_v3_Proposal.md S5.3(a), DECISIONS.md): the pre-GitHub
codebase sized the pretraining fit grid per-problem, scaling with the basis
DOF count and floored at a per-problem minimum. The June 2026 consolidation
replaced this with one fixed grid (50x50) for NN pretraining, and separately
introduced a structurally different (and smaller, at every P used in this
package) formula for LiL pretraining. Both are reverted here to their exact
historical values -- this test locks those values in against every P this
package's Component B actually reruns (Section 3.3).
"""

import math

import numpy as np

from lilq.basis import Chebyshev1D, TensorProductBasis2D
from lilq.pretraining import nn_pretrain_grid_side, pretrain_lil


# (floor, target_dof, expected_side) -- hand-computed from each problem's
# pre-GitHub formula, at every P this package's Component B reruns.
NN_PRETRAIN_CASES = [
    # Bratu: floor=50, target_dof = N_x*N_y = P
    (50, 25, 7),     # P=25:  max(50,25)=50  -> floor(sqrt(50))=7
    (50, 100, 10),   # P=100: max(50,100)=100 -> floor(sqrt(100))=10
    (50, 225, 15),   # P=225: max(50,225)=225 -> floor(sqrt(225))=15
    # Burgers: floor=100, target_dof = N_x*N_t = P
    (100, 625, 25),  # P=625: max(100,625)=625 -> floor(sqrt(625))=25
    # Buckley-Leverett: floor=50, target_dof = N_x*N_t = P
    (50, 1024, 32),  # P=1024: max(50,1024)=1024 -> floor(sqrt(1024))=32
]


def test_nn_pretrain_grid_side_matches_pre_github_values():
    for floor, target_dof, expected_side in NN_PRETRAIN_CASES:
        assert nn_pretrain_grid_side(target_dof, floor=floor) == expected_side


def test_nn_pretrain_grid_side_respects_floor_for_small_problems():
    # A tiny problem must still get the floor's worth of fit points, not be
    # starved down to something smaller.
    assert nn_pretrain_grid_side(4, floor=50) == int(math.sqrt(50))


def test_pretrain_lil_uses_historical_total_point_formula():
    """End-to-end: pretrain_lil's fit-point count must be
    floor(sqrt(max(100, 2*n_basis)))**2, matching every pre-GitHub problem
    file identically (Bratu/Burgers/BL all used this exact formula for LiL
    pretraining -- only the NN-pretrain path differed per problem)."""
    basis = TensorProductBasis2D(Chebyshev1D(order=4), Chebyshev1D(order=4))
    assert basis.n_basis == 25  # (4+1) * (4+1)

    call_sizes = []

    def initial_guess_fn(x, y):
        call_sizes.append(len(x))
        return np.zeros_like(x)

    pretrain_lil(basis, initial_guess_fn, (-1.0, 1.0), (-1.0, 1.0), verbose=False)

    expected_side = int(math.sqrt(max(100, 2 * 25)))  # max(100,50)=100 -> 10
    assert call_sizes == [expected_side ** 2]
