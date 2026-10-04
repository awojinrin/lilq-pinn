"""
Per-iteration convergence diagnostics for LiL-Q (Component B instrumentation)
================================================================================

Implements the four scalar diagnostics Computational_Package_1_v2.md
Section 3.1 (items 4-7) requires logged every outer quasilinear
iteration, exactly as the manuscript defines them:

- :func:`phase_indicator` -- Eq. (phase_indicator), $\\chi_k$.
- :func:`observed_order` -- Eq. (observed_order), $o_k$.
- :func:`stall_flag` -- Eq. (stall_detector), with $\\tau_\\chi=0.1$,
  $\\tau_r=0.01$; :func:`stall_rule_index` -- the termination rule, both
  conditions at $n_s$ consecutive steps ($n_s=2$).
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
import scipy.linalg

# The manuscript states this as "2.2 x 10^-16"; that's IEEE double
# precision's true machine epsilon to the precision it gives -- use the
# exact value rather than the rounded literal.
EPS_MACH = float(np.finfo(np.float64).eps)

# Section 3.1 item 8: full-SVD conditioning every iteration only below
# this P; above it (Beltrami, P~8000), use conditioning_via_pivoted_qr
# at the final iterate only -- a full SVD at that scale is expensive
# enough to dominate total solve time for no benefit (confirmed
# empirically for Beltrami's *un-gated* per-iteration cond() call; see
# DECISIONS.md, "Beltrami's real slowdown cause").
DEFAULT_SVD_CONDITIONING_THRESHOLD = 3200

# The stall detector's tolerances and the termination rule's persistence:
# tau_r = 0.01 and n_s = 2 since the manuscript's revised rule (the advisor's
# reply to wave 4, item 2(b); tau_r was 0.1, the rule a single step).
DEFAULT_TAU_CHI = 0.1
DEFAULT_TAU_R = 0.01
DEFAULT_N_S = 2


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
        0.1 and 0.01 (``DEFAULT_TAU_CHI``, ``DEFAULT_TAU_R``), the
        manuscript's revised values; tau_r was 0.1 before the advisor's
        reply to wave 4 (item 2(b)), and logs written before then used it.
    """
    if np.isnan(chi_k):
        return False
    return bool(
        chi_k <= tau_chi
        and abs(norm_Rlin_k - norm_Rlin_km1) <= tau_r * norm_Rlin_k
    )


def stall_flags(chi, norm_Rlin, tau_chi: float = DEFAULT_TAU_CHI, tau_r: float = DEFAULT_TAU_R) -> np.ndarray:
    """:func:`stall_flag` at every step of a solve: ``chi[k]`` and
    ``norm_Rlin[k]`` are row k's; false at k = 0, which has no k - 1."""
    chi, norm_Rlin = np.asarray(chi, dtype=float), np.asarray(norm_Rlin, dtype=float)
    flags = np.zeros(len(chi), dtype=bool)
    for k in range(1, len(chi)):
        flags[k] = stall_flag(float(chi[k]), float(norm_Rlin[k]), float(norm_Rlin[k - 1]), tau_chi, tau_r)
    return flags


def stall_rule_index(chi, norm_Rlin, n_s: int = DEFAULT_N_S, tau_chi: float = DEFAULT_TAU_CHI,
                     tau_r: float = DEFAULT_TAU_R):
    r"""The manuscript's termination rule (the advisor's reply to wave 4,
    item 2(b)): the first step $k$ at which both conditions of the stall
    detector have held at the $n_s$ consecutive steps $k-n_s+1, \dots, k$
    (all $\ge 1$); the rule then returns $u^{(k+1)}$. ``None`` if it never
    holds within the given steps. $n_s = 1$ is the earlier single-step
    rule."""
    if n_s < 1:
        raise ValueError(f"n_s must be at least 1, not {n_s}")
    flags = stall_flags(chi, norm_Rlin, tau_chi, tau_r)
    run = 0
    for k in range(1, len(flags)):
        run = run + 1 if flags[k] else 0
        if run >= n_s:
            return k
    return None


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


def conditioning_via_svd(A: np.ndarray) -> dict:
    r"""Full-SVD conditioning: Section 3.1 item 8's default path, for
    $P \le$ :data:`DEFAULT_SVD_CONDITIONING_THRESHOLD`.

    Numerical rank matches the spec's own definition exactly: the count
    of singular values strictly above $\max(N,P)\,\sigma_{\max}\,
    \varepsilon_{\mathrm{mach}}$.

    Parameters
    ----------
    A : array, shape (N, P)
        The weighted system matrix $\mathbf{A}^{(k)}$ (or any matrix --
        this function has no PDE-specific assumptions).

    Returns
    -------
    dict with keys ``kappa`` (sigma_max / sigma_min), ``kappa_method``
    (always ``"svd"``), ``num_rank_svd``, ``kappa_raw`` (= ``kappa``) and
    ``kappa_retained`` = sigma_1 / sigma_r with r the numerical rank
    (Addendum v2.2 Section 2.7: at Buckley-Leverett P = 1,024 the full
    kappa is about 1e17, so kappa * eps_mach is about 22 and the round-off
    comparison would fire trivially; the retained part's is meaningful).
    """
    N, P = A.shape
    singular_values = np.linalg.svd(A, compute_uv=False)
    sigma_max, sigma_min = singular_values[0], singular_values[-1]
    kappa = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")

    rank_tol = max(N, P) * sigma_max * EPS_MACH
    num_rank_svd = int(np.sum(singular_values > rank_tol))
    kappa_retained = (float(sigma_max / singular_values[num_rank_svd - 1])
                      if num_rank_svd > 0 else float("inf"))

    return {"kappa": kappa, "kappa_method": "svd", "num_rank_svd": num_rank_svd,
            "kappa_raw": kappa, "kappa_retained": kappa_retained}


def conditioning_via_pivoted_qr(A: np.ndarray) -> dict:
    r"""Pivoted-QR conditioning: Section 3.1 item 8's required path for
    Beltrami (and any problem past the SVD threshold), computed **only at
    the final iterate**, not every iteration -- an $O(\min(N,P)^2\max(N,P))$
    pivoted QR is still real cost, just far cheaper than a full SVD at the
    same scale, and the spec asks for one number, not a per-iteration
    history, above the SVD threshold.

    The spec asks for two related quantities: the manuscript reports
    $\sigma_1/\sigma_{7977} = 1.2\times10^4$ "for the retained part" of a
    rank-deficient system -- i.e. $|R_{11}|/|R_{PP}|$ computed naively
    over *all* diagonal entries is not what's wanted (pivoted-out,
    near-zero trailing entries make that ratio meaningless, potentially
    infinite); discard those first via the same numerical-rank threshold
    :func:`conditioning_via_svd` uses, then take the ratio of the first
    to the last *retained* diagonal entry. Both are returned; ``kappa``
    (the schema column) is the retained-only ratio, since that's the one
    actually comparable to :func:`conditioning_via_svd`'s output.

    Parameters
    ----------
    A : array, shape (N, P)

    Returns
    -------
    dict with keys ``kappa`` (retained-only ratio), ``kappa_method``
    (always ``"qr_pivoted"``), ``kappa_raw_ratio`` (the naive, all-
    diagonal-entries ratio -- not a schema column, kept for anyone
    inspecting conditioning behavior directly), ``num_rank_svd`` (always
    ``None`` -- this path doesn't compute the SVD-based rank at all,
    that's the entire point of using it instead).
    """
    N, P = A.shape
    R, _pivots = scipy.linalg.qr(A, mode="r", pivoting=True)
    diag = np.abs(np.diag(R))

    raw_ratio = float(diag[0] / diag[-1]) if diag[-1] > 0 else float("inf")

    rank_tol = max(N, P) * diag[0] * EPS_MACH
    retained = diag[diag > rank_tol]
    if len(retained) == 0 or retained[-1] == 0:
        kappa = float("inf")
    else:
        kappa = float(retained[0] / retained[-1])

    return {
        "kappa": kappa,
        "kappa_method": "qr_pivoted",
        "kappa_raw_ratio": raw_ratio,
        "num_rank_svd": None,
        # The same two ratios under the names the log uses for both methods,
        # and the number of retained diagonal entries (Addendum v2.2 2.7).
        "kappa_raw": raw_ratio,
        "kappa_retained": kappa,
        "num_rank_qr": int(len(retained)),
    }
