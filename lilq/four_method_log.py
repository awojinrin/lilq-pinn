"""
``four_method_tables.csv`` schema and writer (Component B Section 3.4)
======================================================================

Implements the "four-method tables" rerun log: NiL-N, NiL-Q, and LiL-N
reruns of Bratu/Burgers/Buckley-Leverett (viscous and gravity), one row
per (benchmark, P, method, seed, device), with the exact fields Section
3.4 asks for:

    Log per run: iterations, function evaluations, wall-clock time,
    final weighted loss, whether the target was reached, and the
    stopping reason (target, iteration_cap, line_search_cap, failure),
    plus the loss history every 10 iterations.

Addendum v2.2 adds ``optimizer_stall`` and its iteration, evaluation count
and time (Section 2.2), and fixes NiL-Q's caps (Section 2.3).

LiL-Q is deliberately absent from this schema -- it already has its own
full per-iteration log (``lilq.iteration_log``, Section 3.1) from
sub-batches 3-6; this module is for the three L-BFGS-trained methods
(NiL-N, NiL-Q, LiL-N) that share ``lilq.solvers``' ``MetricsTracker``-
based summary shape instead.
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ``seed`` is the network-initialization seed (empty for deterministic
# LiL-N); ``collocation_seed`` is the fixed collocation-set seed.
# ``training_time_s`` is the optimizer loop only (the solver's own clock);
# ``wall_total_s`` also covers setup and pretraining. ``error`` holds the
# traceback of a ``failure`` row.
FOUR_METHOD_CSV_COLUMNS = (
    "benchmark", "P", "method", "seed", "collocation_seed", "device",
    "total_iterations", "total_line_searches", "training_time_s", "wall_total_s",
    "final_loss", "converged", "stopping_reason",
    "iterations_cap", "line_searches_cap",
    "stall_iteration", "stall_evaluations", "stall_time_s",
    "loss_history_every_10", "error",
)

RowKey = Tuple[str, int, str, str, str]


def row_key(row: Dict[str, Any]) -> RowKey:
    """Identity of one run -- (benchmark, P, method, seed, device) -- used to
    skip already-completed runs on resume. Normalized so a row read back
    from CSV (all strings, ``""`` for no seed) matches a freshly built one."""
    seed = row.get("seed")
    return (str(row["benchmark"]), int(row["P"]), str(row["method"]),
            "" if seed in (None, "") else str(int(seed)), str(row["device"]))


def classify_stopping_reason(
    converged: bool,
    iterations_used: int,
    iterations_cap: int,
    line_searches_used: int,
    line_searches_cap: int,
    optimizer_stall: bool = False,
    final_loss: Optional[float] = None,
    budget_exhausted: bool = False,
) -> str:
    """``target`` / ``optimizer_stall`` / ``line_search_cap`` /
    ``iteration_cap`` / ``failure`` (Addendum v2.2 Sections 2.2-2.3).

    - A non-finite final loss is a ``failure`` (so is a run that raised,
      which the call site records without calling this).
    - ``optimizer_stall``: an L-BFGS step (NiL-N, LiL-N) or a whole outer
      iteration (NiL-Q) left the parameters bitwise unchanged.
    - The evaluation cap is tested before the iteration cap when the
      iterations are below theirs; with the cap of ``lilq.solvers``
      (16x) it cannot bind, and a ``line_search_cap`` row is an anomaly.
    - ``iteration_cap`` when the iterations reached their cap, or when
      ``budget_exhausted`` (NiL-Q: every outer iteration used, some inner
      loops having ended early on a stall).

    Use :func:`stopping_fields` rather than calling this directly: it
    applies the same caps in every driver.
    """
    if final_loss is not None and not math.isfinite(final_loss):
        return "failure"
    if converged:
        return "target"
    if optimizer_stall:
        return "optimizer_stall"
    if line_searches_used >= line_searches_cap and iterations_used < iterations_cap:
        return "line_search_cap"
    if iterations_used >= iterations_cap or budget_exhausted:
        return "iteration_cap"
    if line_searches_used >= line_searches_cap:
        return "line_search_cap"
    return "failure"


def stopping_fields(method: str, summary: Dict[str, Any], opt) -> Dict[str, Any]:
    """The iteration and evaluation counts, their caps, the stopping reason
    and the stall fields of one L-BFGS run (NiL-N, NiL-Q, LiL-N), the same
    in the four-method CSV and the B8 CSV (Addendum v2.2 Section 2.3).

    Iterations are L-BFGS steps; for NiL-Q the inner steps summed over the
    outer iterations, with cap ``max_quasi_iters_nn * max_inner_iters_nn``.
    Evaluations are real loss evaluations (``lilq.solvers.LBFGSObjective``).
    """
    from lilq.solvers import line_search_cap
    if method == 'NiL-Q':
        iterations_cap = opt.max_quasi_iters_nn * opt.max_inner_iters_nn
        budget_exhausted = (summary['n_quasi_iters'] >= opt.max_quasi_iters_nn
                            and not summary['converged'] and not summary.get('optimizer_stall'))
    else:
        iterations_cap = opt.max_iterations
        budget_exhausted = False
    evaluations_cap = line_search_cap(iterations_cap, opt.max_line_searches)
    reason = classify_stopping_reason(
        converged=summary['converged'],
        iterations_used=summary['total_iterations'], iterations_cap=iterations_cap,
        line_searches_used=summary['total_line_searches'], line_searches_cap=evaluations_cap,
        optimizer_stall=bool(summary.get('optimizer_stall')),
        final_loss=summary.get('final_loss'), budget_exhausted=budget_exhausted,
    )
    return {
        'iterations': summary['total_iterations'], 'iterations_cap': iterations_cap,
        'evaluations': summary['total_line_searches'], 'evaluations_cap': evaluations_cap,
        'stopping_reason': reason,
        'stall_iteration': summary.get('stall_iteration'),
        'stall_evaluations': summary.get('stall_evaluations'),
        'stall_time_s': summary.get('stall_time_s'),
    }


def subsample_loss_history(metrics_dict: Dict[str, list], every: int = 10) -> List[list]:
    """``[[iteration, loss], ...]`` from a ``MetricsTracker.to_dict()``
    dict, keeping every ``every``-th recorded row (iteration 0, 10, 20,
    ... since ``MetricsTracker.record`` is called once per optimizer
    step with a contiguous, 1-per-call iteration counter -- so index
    stride ``every`` is exactly "every N iterations", not an
    approximation). Always keeps the last row too, so the final loss is
    never missing from the history purely because it fell between the
    stride.
    """
    iterations = metrics_dict.get("iteration", [])
    losses = metrics_dict.get("loss", [])
    n = len(iterations)
    if n == 0:
        return []
    history = [[iterations[i], losses[i]] for i in range(0, n, every)]
    if history[-1][0] != iterations[-1]:
        history.append([iterations[-1], losses[-1]])
    return history


class FourMethodLogger:
    """Accumulates one row per (benchmark, P, method, seed, device),
    matching :data:`FOUR_METHOD_CSV_COLUMNS` exactly -- same contract as
    ``lilq.iteration_log.IterationLogger``.
    """

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def record(self, **fields: Any) -> None:
        unknown = set(fields) - set(FOUR_METHOD_CSV_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown four_method_tables.csv column(s): {sorted(unknown)}")
        row = {col: fields.get(col) for col in FOUR_METHOD_CSV_COLUMNS}
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def to_csv(self, path: Union[str, Path]) -> None:
        """Written to a temporary file and moved into place, so a job
        killed mid-write (walltime, preemption) leaves the previous
        complete file rather than a truncated one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FOUR_METHOD_CSV_COLUMNS)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(_stringify_row(row))
        os.replace(tmp, path)

    @classmethod
    def from_csv(cls, path: Union[str, Path], drop_failures: bool = True) -> "FourMethodLogger":
        """Reload a previously written CSV (values stay strings, except the
        loss history, which is parsed back so a rewrite doesn't
        double-encode it). ``drop_failures`` omits ``failure`` rows so a
        resumed sweep reruns them."""
        logger = cls()
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if drop_failures and row.get("stopping_reason") == "failure":
                    continue
                history = row.get("loss_history_every_10")
                row["loss_history_every_10"] = json.loads(history) if history else None
                logger._rows.append({col: row.get(col) for col in FOUR_METHOD_CSV_COLUMNS})
        return logger

    def completed_keys(self) -> set:
        return {row_key(r) for r in self._rows}


def _stringify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in row.items():
        if k == "loss_history_every_10" and v is not None:
            out[k] = json.dumps(v)
        elif v is None:
            out[k] = ""
        elif isinstance(v, float) and v != v:  # NaN
            out[k] = ""
        else:
            out[k] = v
    return out
