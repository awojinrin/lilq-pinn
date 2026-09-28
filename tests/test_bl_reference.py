"""The finite-difference reference behind Buckley-Leverett's test errors
(``problems.buckley_leverett.reference_solution``)."""

import numpy as np
import pytest

from problems.buckley_leverett import BLConfig, BLPhysics, TEST_GRID, reference_solution

CASES = {"viscous": BLConfig(), "gravity": BLConfig.with_gravity()}


@pytest.mark.parametrize("case", CASES)
def test_reference_is_converged(case):
    config = CASES[case]
    fine, coarse = reference_solution(config, 4000), reference_solution(config, 2000)
    assert np.linalg.norm(coarse - fine) / np.linalg.norm(fine) < 1e-5


@pytest.mark.parametrize("case", CASES)
def test_reference_meets_the_initial_and_boundary_data(case):
    config = CASES[case]
    S = reference_solution(config)
    assert S.shape == TEST_GRID
    x = np.linspace(*config.x_domain, TEST_GRID[0])
    np.testing.assert_allclose(S[:, 0], BLPhysics(config).initial_condition(x), rtol=0, atol=1e-15)
    np.testing.assert_allclose(S[0, 1:], config.S_left, rtol=0, atol=1e-15)
    np.testing.assert_allclose(S[-1, 1:], config.S_right, rtol=0, atol=1e-15)
    assert S.min() > -1e-6 and S.max() < 1 + 1e-6      # saturation stays in [0, 1]


def test_reference_rejects_a_grid_that_misses_the_test_points():
    with pytest.raises(ValueError):
        reference_solution(BLConfig(), 3001)
