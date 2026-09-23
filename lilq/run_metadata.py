"""
``run.json`` schema and writer (Component B "once per run" metadata)
======================================================================

Complements ``iterations.csv`` (``lilq.iteration_log``): where that file
is one row per outer iteration, ``run.json`` is the run-level metadata
Computational_Package_1_v2.md Section 3.1 asks for once per solve --
composition of $N$ and $P$, the row weights, the collocation-set
construction, the basis description, the stopping rule actually used,
and so on (the spec's own list, verbatim):

    N and its composition by row type; P and its composition per field;
    the row weights; the collocation-set construction (random-tensor
    with seed, equispaced, cell centers) and the exact counts; basis
    family and modes per direction per field; the initial coefficient
    vector (zero, or the fitted initial profile for viscous
    Buckley-Leverett); solver driver and rcond; the stopping rule
    actually used (loss target with its value, or relative coefficient
    change with its tolerance) and K_max; stopping reason; first
    iteration at which the stall flag was true; device and thread count.

Every problem's row/field/basis composition is different (Bratu's single
scalar field vs. Kovasznay's three vs. Beltrami's four, for instance), so
unlike ``iterations.csv`` (one fixed column schema every problem fills in
identically) this module fixes the *field names*, not their internal
shape -- ``N_composition``/``P_composition``/``basis_description`` are
free-form dicts whose keys are whatever row types/fields a given problem
actually has.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Section 3.1's "once per run" list, verbatim field-for-field. A field
# not applicable to a given problem (e.g. no GPU, no random collocation)
# is still present in the output, written as null, exactly like
# iterations.csv's own "empty where undefined" convention.
RUN_METADATA_FIELDS = (
    "N_total", "N_composition",
    "P_total", "P_composition",
    "row_weights",
    "collocation_construction",
    "basis_description",
    "initial_coefficients",
    "solver_driver", "rcond",
    "stopping_rule", "K_max",
    "stopping_reason",
    "first_stall_iteration",
    "device", "thread_count",
    # Beyond the spec's list: the evidence for check B2 (relative
    # difference between A^(k) beta^(k) - f^(k) and the nonlinear operator
    # evaluated directly, at k=1, plus its maximum over the run -- see
    # LilQDiagnosticsTracker.b2_check), and Section 3.1 item 8's
    # |R_11|/|R_PP| over *all* pivoted-QR diagonal entries (``kappa`` is
    # the retained-part ratio; null when the SVD path was used).
    "b2_check",
    "kappa_qr_raw_ratio",
    # Section 3.2 (Kovasznay GPU path): the 3*8NP memory estimate, the
    # smallest min|R_pp|/max|R_pp| seen, and the iterations flagged
    # below 1e-13 (each also solved on the CPU with gelsy).
    "gpu_qr",
)


def first_stall_iteration(rows: List[Dict[str, Any]]) -> Optional[int]:
    """The first ``k`` at which ``stall_flag`` was true, or ``None`` if
    the run never stalled (including a run with no logged rows at all).

    ``rows`` is an ``IterationLogger.rows`` list (or any list of dicts
    with ``k``/``stall_flag`` keys, in ``k`` order -- the natural order
    ``IterationLogger`` already produces them in).
    """
    for row in rows:
        if row.get("stall_flag"):
            return row["k"]
    return None


def build_run_metadata(**fields: Any) -> Dict[str, Any]:
    """Build a ``run.json``-compatible dict from any subset of
    :data:`RUN_METADATA_FIELDS`. A field the caller omits (not
    applicable to that problem, or not computed) is written as ``None``.
    Raises on an unrecognized keyword -- a typo'd field name should fail
    loudly, not silently produce an extra or missing key.
    """
    unknown = set(fields) - set(RUN_METADATA_FIELDS)
    if unknown:
        raise ValueError(f"Unknown run.json field(s): {sorted(unknown)}")
    return {name: fields.get(name) for name in RUN_METADATA_FIELDS}


def write_run_json(path: Union[str, Path], metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
