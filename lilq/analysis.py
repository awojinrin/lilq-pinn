"""
Opt-In Analysis Module for LiL-Q
==================================

SVD, condition number, and predictive bounds analysis that can be
performed AFTER or ALONGSIDE experiments — never embedded in the
core solver timing.

Usage::

    # As a diagnostics callback passed to solve_lil_q
    from lilq.analysis import make_svd_callback
    callback = make_svd_callback(verbose=True)
    coeffs, metrics, summary = solve_lil_q(
        ..., diagnostics_callback=callback
    )

    # As a standalone post-analysis
    from lilq.analysis import svd_analysis
    results = svd_analysis(A_stacked)
"""

import numpy as np
from typing import Dict, Optional, List


def svd_analysis(A: np.ndarray) -> Dict:
    """Perform SVD analysis on a system matrix.

    Computes SVD **once** and derives all metrics from it,
    avoiding the double-SVD pattern (np.linalg.cond + np.linalg.svd)
    that was previously baked into the solver.

    Parameters
    ----------
    A : np.ndarray of shape (m, n)
        The system matrix to analyze.

    Returns
    -------
    dict with keys:
        sigma : np.ndarray
            All singular values (descending).
        kappa : float
            Condition number (σ_max / σ_min).
        q4, q3, q2, q1, q0 : float
            Singular value quartiles (max, 75th, median, 25th, min).
        epsilon : float
            Machine epsilon for the matrix dtype.
        tolerance : float
            Numerical rank tolerance.
        numerical_rank : int
            Number of singular values above tolerance.
        total_cols : int
            Total number of columns (= n_basis).
    """
    U, sigma, Vh = np.linalg.svd(A, full_matrices=False)

    epsilon = np.finfo(A.dtype).eps
    tolerance = max(A.shape) * sigma[0] * epsilon
    numerical_rank = int(np.sum(sigma > tolerance))

    kappa = float(sigma[0] / (sigma[-1] + 1e-30))
    q4, q3, q2, q1, q0 = np.percentile(sigma, [100, 75, 50, 25, 0])

    return {
        'sigma': sigma,
        'kappa': kappa,
        'q4': float(q4),
        'q3': float(q3),
        'q2': float(q2),
        'q1': float(q1),
        'q0': float(q0),
        'epsilon': float(epsilon),
        'tolerance': float(tolerance),
        'numerical_rank': numerical_rank,
        'total_cols': A.shape[1],
    }


def make_svd_callback(verbose: bool = True, store: Optional[List] = None):
    """Create a diagnostics callback for LiL-Q that runs SVD at each iteration.

    Parameters
    ----------
    verbose : bool
        If True, prints SVD summary at each iteration.
    store : list, optional
        If provided, appends SVD results dict at each iteration.

    Returns
    -------
    callable
        ``(A_stacked, beta, quasi_iter) -> None``
    """
    if store is None:
        store = []

    def callback(A_stacked, beta, quasi_iter):
        result = svd_analysis(A_stacked)
        result['quasi_iter'] = quasi_iter
        store.append(result)

        if verbose:
            print(f"  SVD analysis (iter {quasi_iter + 1}):\n"
                  f"    σ_max = {result['q4']:.4e}, Q3 = {result['q3']:.4e}, "
                  f"median = {result['q2']:.4e}, Q1 = {result['q1']:.4e}, "
                  f"σ_min = {result['q0']:.4e}\n"
                  f"    κ = {result['kappa']:.4e}, "
                  f"rank = {result['numerical_rank']}/{result['total_cols']}")

    return callback


def condition_number_study(
    problem_runner,
    config,
    opt,
    N_values: List[int],
    verbose: bool = True,
) -> Dict:
    """Run LiL-Q across multiple N values and collect conditioning data.

    Parameters
    ----------
    problem_runner : callable
        ``run_lil_q(config, opt, verbose, diagnostics_callback)`` function
        from the appropriate problem module (e.g., problems.bratu.run_lil_q).
    config : dataclass
        Problem configuration. ``N_x`` and ``N_y`` (or ``N_t``) will be
        overwritten for each N.
    opt : dataclass
        Optimization configuration.
    N_values : list of int
        Basis sizes to test.
    verbose : bool
        Print progress.

    Returns
    -------
    dict mapping N -> list of per-iteration SVD analysis results.
    """
    all_results = {}

    for N in N_values:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Condition number study: N = {N}")
            print('=' * 60)

        # Set both dimensions to N
        config.N_x = N
        if hasattr(config, 'N_y'):
            config.N_y = N
        elif hasattr(config, 'N_t'):
            config.N_t = N

        svd_store = []
        callback = make_svd_callback(verbose=verbose, store=svd_store)

        _, _, _, summary = problem_runner(
            config, opt, verbose=verbose, diagnostics_callback=callback,
        )

        all_results[N] = svd_store

        if verbose:
            print(f"  N={N}: {len(svd_store)} iterations, "
                  f"final κ = {svd_store[-1]['kappa']:.4e}")

    return all_results


def predictive_bounds_analysis(
    problem_runner,
    config,
    opt,
    N_values: List[int],
    verbose: bool = True,
) -> Dict:
    """Analyze predictive bounds (singular value distribution across N).

    This replicates the analysis from ``LiLQ_Properties/predictive_bounds/``
    but as a clean, callable function.

    Parameters
    ----------
    problem_runner, config, opt : as in condition_number_study.
    N_values : list of int
        Basis sizes to test.

    Returns
    -------
    dict with keys:
        N_values : list
        final_kappa : list of float
        final_sigma_max : list of float
        final_sigma_min : list of float
        final_rank : list of int
        convergence_iters : list of int
        per_N : dict mapping N -> full SVD history
    """
    per_N = condition_number_study(problem_runner, config, opt, N_values, verbose)

    summary = {
        'N_values': list(N_values),
        'final_kappa': [],
        'final_sigma_max': [],
        'final_sigma_min': [],
        'final_rank': [],
        'convergence_iters': [],
        'per_N': per_N,
    }

    for N in N_values:
        svd_data = per_N[N]
        last = svd_data[-1]
        summary['final_kappa'].append(last['kappa'])
        summary['final_sigma_max'].append(last['q4'])
        summary['final_sigma_min'].append(last['q0'])
        summary['final_rank'].append(last['numerical_rank'])
        summary['convergence_iters'].append(len(svd_data))

    return summary
