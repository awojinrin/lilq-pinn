"""
``iterations.csv`` schema and logger (Component B instrumentation)
======================================================================

Computational_Package_1_v2.md Section 6 specifies the exact column list
and order for ``iterations.csv``; this module is the single source of
truth for that schema, plus a small accumulator class so every problem's
solver integration writes rows the same way instead of each hand-rolling
its own CSV writer.

Per the spec's own note: "``chi`` and ``order_obs`` refer to the current
$k$ and are empty where undefined; residual-MSE benchmarks put the test
residual in ``eps_u``." Any column a given problem/iteration doesn't
populate is written as an empty value (``NaN`` for floats, ``None`` for
everything else) rather than omitted -- every row has every column, so
the CSV stays uniform across problems that populate different subsets
(e.g. a scalar-residual benchmark has no ``eps_v``/``eps_p``).
"""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .instrumentation import (
    DEFAULT_SVD_CONDITIONING_THRESHOLD,
    EPS_MACH,
    conditioning_via_pivoted_qr,
    conditioning_via_svd,
    observed_order,
    phase_indicator,
    roundoff_comparison,
    stall_flag as _stall_flag,
)

# Section 6, verbatim order. Do not reorder without updating the spec
# reference -- downstream tooling (Section 3.5's residual-band figures,
# any reviewer script) is entitled to assume this exact column order.
ITERATION_CSV_COLUMNS = (
    "k",
    "t_assemble_s", "t_solve_s", "t_cum_s",
    "norm_R_h", "norm_R_interior",
    "norm_Rlin_h", "norm_Rlin_interior",
    "norm_f_h",
    "norm_dbeta", "rel_dbeta",
    "chi", "order_obs", "stall_flag",
    "roundoff_ratio", "kappa_eps",
    "kappa", "kappa_method",
    "num_rank_svd", "num_rank_gelsy", "rcond",
    "eps_u", "eps_v", "eps_p", "eps_p_meanfree",
    "maxerr_u", "maxerr_v", "maxerr_p",
    "solver_path", "gpu_mem_peak_bytes",
)


class IterationLogger:
    """Accumulates one row per outer iteration, matching
    :data:`ITERATION_CSV_COLUMNS` exactly.

    Usage::

        logger = IterationLogger()
        for k in range(max_iters):
            ...
            logger.record(k=k, norm_R_h=..., chi=..., ...)  # any subset of columns
        logger.to_csv(run_dir / "iterations.csv")
    """

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def record(self, **fields: Any) -> None:
        """Append one row. Any column in :data:`ITERATION_CSV_COLUMNS` not
        passed is filled with ``None`` (written as an empty CSV field).
        Raises on an unrecognized keyword -- a typo'd column name should
        fail loudly, not silently produce a 31st column or a dropped value.
        """
        unknown = set(fields) - set(ITERATION_CSV_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown iterations.csv column(s): {sorted(unknown)}")
        row = {col: fields.get(col) for col in ITERATION_CSV_COLUMNS}
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        """Read-only view of the recorded rows, in insertion order."""
        return list(self._rows)

    def to_csv(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ITERATION_CSV_COLUMNS)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(_stringify_row(row))


def solve_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows of outer iterations that performed a linear solve -- every
    row except the terminal ``k = K`` row (see :class:`LilQDiagnosticsTracker`)."""
    return [r for r in rows if r.get("norm_Rlin_h") is not None]


def last_solve_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The final linear solve's row: where the final-iterate conditioning
    (``kappa``, ranks) lives. ``rows[-1]`` is the terminal residual-only row."""
    return solve_rows(rows)[-1]


class LilQDiagnosticsTracker:
    """Turns ``solve_lil_q``'s raw per-iteration ingredients into a full
    :data:`ITERATION_CSV_COLUMNS` row, owning the history a single
    iteration's ingredients alone can't supply:
    :func:`~lilq.instrumentation.observed_order` needs
    $\\|\\mathbf{R}^{(k-1)}\\|_h$, and
    :func:`~lilq.instrumentation.stall_flag` needs
    $\\|\\mathbf{R}_{\\mathrm{lin}}^{(k-1)}\\|_h$. One instance per solve;
    construct once before the outer loop, call :meth:`step` once per
    iteration, then :meth:`finish` once after it.

    **Row numbering follows the package spec.** Row ``k`` (from 0)
    describes outer iteration $k$: the system $\\mathbf{A}^{(k)}$,
    $\\mathbf{f}^{(k)}$ assembled at $\\boldsymbol\\beta^{(k)}$ and solved
    for $\\boldsymbol\\beta^{(k+1)}$. It holds $\\|\\mathbf{R}^{(k)}\\|_h$
    (computed as $\\mathbf{A}^{(k)}\\boldsymbol\\beta^{(k)}-\\mathbf{f}^{(k)}$,
    the spec's one mat-vec), $\\|\\mathbf{R}_{\\mathrm{lin}}^{(k)}\\|_h$,
    $\\chi_k$, $o_k$, the stall flag, $\\kappa(\\mathbf{A}^{(k)})$ and
    $\\delta\\boldsymbol\\beta^{(k)}=\\boldsymbol\\beta^{(k+1)}-\\boldsymbol\\beta^{(k)}$.
    A run of $K$ solves ends with a terminal row ``k = K`` from
    :meth:`finish`, holding only $\\|\\mathbf{R}^{(K)}\\|_h$ (and its
    interior part) at the returned coefficients, where no system is
    assembled -- so a solve's rows number $K+1$.

    **Check B2 is measured on every iteration** when
    ``compute_residual_vector_fn`` is given: the mat-vec
    $\\mathbf{R}^{(k)}$ against the nonlinear operator evaluated directly
    at $\\boldsymbol\\beta^{(k)}$. :attr:`b2_check` reports it at ``k = 1``
    (the first iterate away from the starting point; the spec asks for one
    iterate per benchmark), plus the maximum over the run. Near
    convergence $\\mathbf{R}^{(k)}$ is a small difference of O(1)
    quantities, so cancellation alone lifts the *relative* difference
    there far above machine precision -- the maximum is kept for
    transparency, not as the check.

    Several columns require information ``solve_lil_q`` genuinely
    doesn't have access to unless the caller provides it, and are left
    ``NaN``/``None`` otherwise:

    - ``norm_R_interior``/``norm_Rlin_interior`` need the *unweighted*
      residual restricted to interior (PDE) rows. Every problem in this
      codebase weights each row block by one scalar,
      $\\sqrt{\\lambda_{\\mathrm{block}}/n_{\\mathrm{block}}}$ (this
      module's own docstring notation), and stacks interior rows first
      -- confirmed directly in Bratu's/Burgers'/BL's ``assemble_system_fn``
      implementations, not assumed. Given ``n_interior_rows`` (how many
      leading rows are interior) and ``interior_weight`` (that block's
      scalar weight), the interior-only unweighted norm is just
      ``norm(weighted_vector[:n_interior_rows]) / interior_weight`` --
      exact, not an approximation, since the weighting is a single scalar
      multiply with no information loss to invert. Left ``NaN`` if either
      is omitted (a problem whose blocks don't fit this one-scalar-per-
      block convention would need different handling, not currently
      needed by anything in this codebase).
    - ``chi``/``order_obs``-dependent ``stall_flag`` need the actual
      nonlinear residual *vector* $\\mathbf{R}^{(k+1)}$ (not just its
      norm) for $\\chi_k$'s vector-difference term -- only available if
      the caller supplies ``compute_residual_vector_fn``.
    - Test-error columns (``eps_u`` etc.) and GPU-path columns
      (``solver_path`` beyond the ``"cpu_gelsy"`` default,
      ``gpu_mem_peak_bytes``) are out of scope for this generic tracker
      entirely -- problem-specific evaluation and the GPU solve path are
      later pieces of this work.
    """

    def __init__(
        self,
        conditioning_svd_threshold: int = DEFAULT_SVD_CONDITIONING_THRESHOLD,
        n_interior_rows: Optional[int] = None,
        interior_weight: Optional[float] = None,
    ) -> None:
        self._norm_R_h_km1: Optional[float] = None
        self._norm_Rlin_h_km1: Optional[float] = None
        self._conditioning_svd_threshold = conditioning_svd_threshold
        self._t_cum_s: float = 0.0
        self._n_interior_rows = n_interior_rows
        self._interior_weight = interior_weight
        self._final_norm_R_h: Optional[float] = None
        self._final_R_vector: Optional[np.ndarray] = None
        self._b2_rel_err_by_k: Dict[int, float] = {}
        self.kappa_qr_raw_ratio: Optional[float] = None

    @property
    def b2_check(self) -> Optional[Dict[str, Any]]:
        """``{"k", "rel_err", "max_rel_err_over_run"}`` -- ``rel_err`` at
        ``k = 1`` (``k = 0`` for a single-solve run), or ``None`` without a
        residual callback."""
        if not self._b2_rel_err_by_k:
            return None
        k = 1 if 1 in self._b2_rel_err_by_k else min(self._b2_rel_err_by_k)
        return {"k": k, "rel_err": self._b2_rel_err_by_k[k],
                "max_rel_err_over_run": max(self._b2_rel_err_by_k.values())}

    def _interior_norm(self, vector: Optional[np.ndarray]) -> float:
        if vector is None or self._n_interior_rows is None or self._interior_weight is None:
            return float("nan")
        return float(np.linalg.norm(vector[:self._n_interior_rows]) / self._interior_weight)

    def step(
        self,
        k: int,
        A_stacked: np.ndarray,
        b_stacked: np.ndarray,
        beta_prev: np.ndarray,
        beta_new: np.ndarray,
        total_loss: float,
        rank_gelsy: Optional[int],
        t_assemble_s: float,
        t_solve_s: float,
        is_final_iterate: bool,
        compute_residual_vector_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        solver_path: str = "cpu_gelsy",
        gpu_mem_peak_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute row ``k`` (0-based outer iteration). ``A_stacked``/
        ``b_stacked`` are $\\mathbf{A}^{(k)}$, $\\mathbf{f}^{(k)}$ --
        already row-weighted, matching every solver in this codebase.
        ``beta_prev`` is $\\boldsymbol\\beta^{(k)}$ (the point the system
        was assembled at), ``beta_new`` is $\\boldsymbol\\beta^{(k+1)}$
        (its solution). ``total_loss`` is $\\|\\mathbf{R}^{(k+1)}\\|_h^2$
        by this codebase's own definition of the weighted total loss.

        ``rank_gelsy`` may be ``None`` -- the Section 3.2 GPU path
        (``torch.linalg.qr`` + ``solve_triangular``, a full-rank solve
        with no rank-revealing step) genuinely doesn't produce one;
        logged as empty rather than a fabricated value. ``solver_path``/
        ``gpu_mem_peak_bytes`` default to the CPU-only values every
        caller before Section 3.2's GPU path used implicitly; a caller
        running on GPU passes ``solver_path="gpu_qr"`` and the measured
        ``torch.cuda.max_memory_allocated()``.
        """
        self._t_cum_s += t_assemble_s + t_solve_s

        R_k = A_stacked @ beta_prev - b_stacked
        norm_R_h = float(np.linalg.norm(R_k))
        R_lin_k = A_stacked @ beta_new - b_stacked
        norm_Rlin_h = float(np.linalg.norm(R_lin_k))
        norm_f_h = float(np.linalg.norm(b_stacked))

        norm_R_h_next = float(np.sqrt(total_loss))
        raw_dbeta = float(np.linalg.norm(beta_new - beta_prev))
        rel_dbeta = raw_dbeta / (float(np.linalg.norm(beta_new)) + 1e-30)

        R_next = None
        chi = float("nan")
        if compute_residual_vector_fn is not None:
            R_k_direct = compute_residual_vector_fn(beta_prev)
            self._b2_rel_err_by_k[k] = float(
                np.linalg.norm(R_k - R_k_direct) / (np.linalg.norm(R_k_direct) + 1e-300))
            R_next = compute_residual_vector_fn(beta_new)
            chi = phase_indicator(R_next, R_lin_k)

        norm_R_interior = self._interior_norm(R_k)
        norm_Rlin_interior = self._interior_norm(R_lin_k)

        order_obs = (
            observed_order(norm_R_h_next, norm_R_h, self._norm_R_h_km1)
            if self._norm_R_h_km1 is not None
            else float("nan")
        )

        stall = (
            _stall_flag(chi, norm_Rlin_h, self._norm_Rlin_h_km1)
            if self._norm_Rlin_h_km1 is not None
            else False
        )

        roundoff_ratio, kappa_eps = roundoff_comparison(
            norm_Rlin_h, norm_f_h, kappa=float("nan"),
        )

        P = A_stacked.shape[1]
        if P <= self._conditioning_svd_threshold:
            cond_result = conditioning_via_svd(A_stacked)
        elif is_final_iterate:
            cond_result = conditioning_via_pivoted_qr(A_stacked)
            self.kappa_qr_raw_ratio = cond_result["kappa_raw_ratio"]
        else:
            cond_result = {"kappa": float("nan"), "kappa_method": None, "num_rank_svd": None}

        # roundoff_ratio/kappa_eps needed kappa, computed just after --
        # recompute kappa_eps now that the real kappa is known (the NaN
        # kappa above was only a placeholder to get roundoff_ratio itself,
        # which doesn't depend on kappa).
        _, kappa_eps = roundoff_comparison(norm_Rlin_h, norm_f_h, kappa=cond_result["kappa"])

        row = dict(
            k=k,
            t_assemble_s=t_assemble_s, t_solve_s=t_solve_s, t_cum_s=self._t_cum_s,
            norm_R_h=norm_R_h, norm_R_interior=norm_R_interior,
            norm_Rlin_h=norm_Rlin_h, norm_Rlin_interior=norm_Rlin_interior,
            norm_f_h=norm_f_h,
            norm_dbeta=raw_dbeta, rel_dbeta=rel_dbeta,
            chi=chi, order_obs=order_obs, stall_flag=stall,
            roundoff_ratio=roundoff_ratio, kappa_eps=kappa_eps,
            kappa=cond_result["kappa"], kappa_method=cond_result["kappa_method"],
            num_rank_svd=cond_result["num_rank_svd"],
            num_rank_gelsy=int(rank_gelsy) if rank_gelsy is not None else None,
            rcond=EPS_MACH,
            solver_path=solver_path, gpu_mem_peak_bytes=gpu_mem_peak_bytes,
        )

        self._norm_R_h_km1 = norm_R_h
        self._norm_Rlin_h_km1 = norm_Rlin_h
        self._final_norm_R_h = norm_R_h_next
        self._final_R_vector = R_next

        return row

    def finish(self, k: int) -> Dict[str, Any]:
        """Terminal row ``k = K`` (the number of solves performed):
        $\\|\\mathbf{R}^{(K)}\\|_h$ at the returned coefficients, plus its
        interior part when the residual vector is available. No system is
        assembled at $\\boldsymbol\\beta^{(K)}$, so every other column is
        empty."""
        return dict(
            k=k, t_cum_s=self._t_cum_s,
            norm_R_h=self._final_norm_R_h,
            norm_R_interior=self._interior_norm(self._final_R_vector),
        )


def _stringify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """NaN and None both write as an empty CSV field (the spec's "empty
    where undefined"); everything else is passed through as-is for
    csv.DictWriter to format.
    """
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, float) and v != v:  # NaN check without importing math/np here
            out[k] = ""
        else:
            out[k] = v
    return out
