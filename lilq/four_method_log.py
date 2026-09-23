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

LiL-Q is deliberately absent from this schema -- it already has its own
full per-iteration log (``lilq.iteration_log``, Section 3.1) from
sub-batches 3-6; this module is for the three L-BFGS-trained methods
(NiL-N, NiL-Q, LiL-N) that share ``lilq.solvers``' ``MetricsTracker``-
based summary shape instead.
"""

import csv
import json
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
) -> str:
    """``target`` / ``iteration_cap`` / ``line_search_cap`` -- matches
    Section 3.4's own vocabulary (``failure`` is not returned here: it
    covers a run that raised an exception, which happens above this
    function, at the call site, not as a property of a completed
    summary dict -- see ``experiments/four_method_tables.py``).

    Every ``solve_nil_n``/``solve_lil_n`` run stops (``lilq/solvers.py``)
    via ``while iterations < iterations_cap and line_searches <
    line_searches_cap`` -- at exit, at least one of the two caps has been
    reached (unless it converged first). Checked in this order --
    ``iteration_cap`` before ``line_search_cap`` -- because that is the
    order DECISIONS.md's own empirical findings for this codebase's
    tuned budgets say actually binds first for Bratu/Burgers/BL (the
    line-search cap has never bound in stored results); a tie (both caps
    reached in the same final step) is the one case this ordering is a
    genuine judgment call rather than a re-derivation of history the
    summary dict doesn't preserve.

    For NiL-Q, pass ``iterations_used=summary['n_quasi_iters']``,
    ``iterations_cap=max_quasi_iters`` (its outer-loop budget, not
    ``total_iterations``/``max_iterations`` -- NiL-Q's inner L-BFGS loop
    has no single named cap of its own; only the outer quasi-iteration
    count and the global line-search count are capped) and
    ``line_searches_used=summary['total_line_searches']``,
    ``line_searches_cap=max_line_searches``.
    """
    if converged:
        return "target"
    if iterations_used >= iterations_cap:
        return "iteration_cap"
    if line_searches_used >= line_searches_cap:
        return "line_search_cap"
    return "failure"


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
