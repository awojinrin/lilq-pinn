"""
Per-iteration convergence diagnostics for LiL-Q (Component B instrumentation)
================================================================================

Implements the four scalar diagnostics Computational_Package_1_v2.md
Section 3.1 (items 4-7) requires logged every outer quasilinear
iteration, exactly as the manuscript defines them:

- :func:`phase_indicator` -- Eq. (phase_indicator), $\\chi_k$.
- :func:`observed_order` -- Eq. (observed_order), $o_k$.
- :func:`stall_flag` -- Eq. (stall_detector), with $\\tau_\\chi=\\tau_r=0.1$.
- :func:`roundoff_comparison` -- the round-off comparison of Algorithm 1.

These are pure functions of already-computed residual norms (and, for
:func:`phase_indicator` only, the two underlying weighted residual
vectors -- $\\|\\mathbf{a}-\\mathbf{b}\\|$ is not derivable from
$\\|\\mathbf{a}\\|$ and $\\|\\mathbf{b}\\|$ alone). None of them touch a
solver, a basis, or a PDE -- they're deliberately solver-agnostic so they
can be unit-tested against synthetic sequences with a known correct
answer, independent of whether any particular problem's solver
integration (a later step) is correct. All norms here are $\\|\\cdot\\|_h$,
the weighted collocation norm already used everywhere in this
codebase's solvers -- i.e. the caller passes vectors/norms that are
already row-weighted; this module does no weighting of its own.

Every function returns ``float('nan')`` on a degenerate input (a zero
denominator) rather than raising -- these are logged diagnostics, and a
run must never crash because a diagnostic was momentarily undefined
(e.g. a perfect linear solve making $\\|\\mathbf{R}_{\\mathrm{lin}}\\|_h=0$).
"""

import numpy as np

# The manuscript states this as "2.2 x 10^-16"; that's IEEE double
# precision's true machine epsilon to the precision it gives -- use the
# exact value rather than the rounded literal.
EPS_MACH = float(np.finfo(np.float64).eps)

DEFAULT_TAU_CHI = 0.1
DEFAULT_TAU_R = 0.1


def phase_indicator(R_next: np.ndarray, R_lin_k: np.ndarray) -> float:
    r"""Eq. (phase_indicator): $\chi_k = \|\mathbf{R}^{(k+1)} -
    \mathbf{R}_{\mathrm{lin}}^{(k)}\|_h / \|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$.

    Measures how well the linearized step predicted the actual nonlinear
    residual after taking it: small $\chi_k$ means the quasilinear step
    behaved like a good local linear model (asymptotic/Newton-like
    regime); large $\chi_k$ means it didn't (pre-asymptotic regime).

    Parameters
    ----------
    R_next : array
        The weighted nonlinear residual $\mathbf{R}^{(k+1)}$, evaluated
        at the iterate produced by the current step.
    R_lin_k : array
        The weighted *linearized* residual $\mathbf{R}_{\mathrm{lin}}^{(k)}$
        from the linear solve that produced that step (i.e. $\mathbf{A}^{(k)}
        \boldsymbol\beta^{(k+1)} - \mathbf{f}^{(k)}$, not the nonlinear
        operator).

    Returns
    -------
    float
        NaN if $\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h = 0$.
    """
    denom = np.linalg.norm(R_lin_k)
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(R_next - R_lin_k) / denom)


def observed_order(norm_R_next: float, norm_R_k: float, norm_R_km1: float) -> float:
    r"""Eq. (observed_order), for $k \ge 1$:
    $o_k = \ln(\|\mathbf{R}^{(k+1)}\|_h / \|\mathbf{R}^{(k)}\|_h) /
           \ln(\|\mathbf{R}^{(k)}\|_h / \|\mathbf{R}^{(k-1)}\|_h)$.

    The empirically observed convergence order at step $k$: $o_k \to 2$
    for quadratic (Newton-like) convergence, $o_k \to 1$ for linear
    convergence -- see ``tests/test_instrumentation.py`` for a synthetic
    check that this formula actually recovers known orders, not just
    that it matches the written formula.

    Parameters
    ----------
    norm_R_next, norm_R_k, norm_R_km1 : float
        $\|\mathbf{R}^{(k+1)}\|_h$, $\|\mathbf{R}^{(k)}\|_h$,
        $\|\mathbf{R}^{(k-1)}\|_h$ respectively -- already-computed
        weighted residual norms, not vectors.

    Returns
    -------
    float
        NaN if either ratio is <= 0 (undefined log) or the denominator
        log is exactly zero (two consecutive equal norms -- order
        undefined, not infinite).
    """
    if norm_R_k <= 0.0 or norm_R_km1 <= 0.0 or norm_R_next <= 0.0:
        return float("nan")
    log_denom = np.log(norm_R_k / norm_R_km1)
    if log_denom == 0.0:
        return float("nan")
    return float(np.log(norm_R_next / norm_R_k) / log_denom)


def stall_flag(
    chi_k: float,
    norm_Rlin_k: float,
    norm_Rlin_km1: float,
    tau_chi: float = DEFAULT_TAU_CHI,
    tau_r: float = DEFAULT_TAU_R,
) -> bool:
    r"""Eq. (stall_detector), for $k \ge 1$: true if $\chi_k \le \tau_\chi$
    and $|\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h - \|\mathbf{R}_{\mathrm{lin}}^{(k-1)}\|_h|
    \le \tau_r\,\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$.

    Logged only -- per the spec, this flag never stops a run itself
    (Section 3.1 item 6: "The flag is logged; it does not stop the
    run."). The caller is responsible for only invoking this at $k \ge 1$
    (there is no $k{-}1$ term at $k=0$); this function takes the already
    -extracted scalars for exactly that iteration and has no notion of
    "which iteration" itself.

    Parameters
    ----------
    chi_k : float
        This iteration's phase indicator, from :func:`phase_indicator`.
    norm_Rlin_k, norm_Rlin_km1 : float
        $\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$ and
        $\|\mathbf{R}_{\mathrm{lin}}^{(k-1)}\|_h$.
    tau_chi, tau_r : float
        Both default to 0.1, the spec's stated value.
    """
    if np.isnan(chi_k):
        return False
    return bool(
        chi_k <= tau_chi
        and abs(norm_Rlin_k - norm_Rlin_km1) <= tau_r * norm_Rlin_k
    )


def roundoff_comparison(norm_Rlin_k: float, norm_f_k: float, kappa: float) -> tuple:
    r"""The round-off comparison of Algorithm 1:
    $\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h / \|\mathbf{f}^{(k)}\|_h$ against
    $\kappa(\mathbf{A}^{(k)})\,\varepsilon_{\mathrm{mach}}$.

    When the first (achieved relative residual) drops to the scale of
    the second (the round-off floor a system with this conditioning
    can't beat), the linear solve has converged as far as double
    precision allows -- further quasilinear iterations can't improve on
    it through this channel.

    Parameters
    ----------
    norm_Rlin_k : float
        $\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$.
    norm_f_k : float
        $\|\mathbf{f}^{(k)}\|_h$, the weighted right-hand side norm for
        this iteration's linear system.
    kappa : float
        $\kappa_2(\mathbf{A}^{(k)})$, computed separately (SVD for
        $P\le3{,}200$, pivoted QR at the final iterate for larger
        problems -- see the conditioning helper added in a later batch).

    Returns
    -------
    (relative_residual, roundoff_floor) : (float, float)
        ``relative_residual`` is NaN if ``norm_f_k == 0``.
    """
    relative_residual = float("nan") if norm_f_k == 0.0 else norm_Rlin_k / norm_f_k
    roundoff_floor = kappa * EPS_MACH
    return relative_residual, roundoff_floor
