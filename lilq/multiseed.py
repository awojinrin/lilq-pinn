"""
Multi-seed run harness
=========================

Implements Computational_Package_1_v2.md Section 2's seed-aggregation
requirement: "Every method with random initialization (NiL-N, NiL-Q, the
Component A baselines, the ELM rows) is run with seeds 0, 1, 2 in
Component B and 0-4 in Component A; report median, min, max. Never
report a single run of a stochastic method."

LiL-N and LiL-Q are explicitly exempt (Section 3.4: "LiL-N from zero is
deterministic; one run") -- they have no random initialization of their
own (coefficients come from a least-squares pretrain fit, not a random
draw), so sweeping seeds for them would only vary the collocation set,
which the spec asks to keep fixed at seed 42 instead. This module is for
NiL-N/NiL-Q (and, later, the Component A F1/F2 baselines), where the
neural network's random weight initialization is the thing multiple
seeds are meant to average over.
"""

import dataclasses
import statistics
from typing import Any, Callable, Dict, List, Sequence


# Summary fields aggregated by default -- matches the numeric fields every
# run_nil_n/run_nil_q summary dict in problems/*.py actually populates.
# `converged` is handled separately (see _aggregate_field) since a
# median/min/max of booleans isn't meaningful the way a convergence *rate*
# across seeds is.
DEFAULT_AGGREGATE_FIELDS = (
    "total_iterations", "total_line_searches",
    "final_loss", "final_pde_loss", "training_time",
)


def run_multiseed(
    runner: Callable[..., tuple],
    config: Any,
    seeds: Sequence[int],
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """Run ``runner(config_with_seed, *args, **kwargs)`` once per seed.

    Parameters
    ----------
    runner : callable
        A ``run_nil_n``/``run_nil_q``-style function from ``problems/*.py``.
        Must accept a config object as its first positional argument and
        return a tuple whose *last* element is the run's summary dict --
        true of every ``run_*`` function in this codebase (NiL methods
        return ``(model, metrics, summary)``, LiL methods return
        ``(basis, coefficients, metrics, summary)``).
    config : dataclass instance
        Must have a ``seed`` field. A per-seed copy is made via
        ``dataclasses.replace`` -- the object passed in is never mutated,
        so the same ``config`` can be reused across multiple
        ``run_multiseed`` calls (e.g. one per method) safely.
    seeds : sequence of int
    *args, **kwargs
        Forwarded to ``runner`` after the per-seed config.

    Returns
    -------
    dict with:
        ``per_seed``: ``{seed: summary_dict}``
        ``raw_results``: ``{seed: full_return_tuple}`` -- keeps the model/
            basis/coefficients/metrics objects too, for checkpointing or
            further inspection beyond the summary.
        ``aggregate``: ``{field: {'median', 'min', 'max', 'values'}}`` for
            every numeric field in ``DEFAULT_AGGREGATE_FIELDS`` that's
            actually present in the summaries, plus a ``converged`` entry
            reporting the fraction of seeds that converged.
    """
    if not seeds:
        raise ValueError("run_multiseed requires at least one seed")

    per_seed_summaries: Dict[int, dict] = {}
    raw_results: Dict[int, tuple] = {}

    for seed in seeds:
        seed_config = dataclasses.replace(config, seed=seed)
        result = runner(seed_config, *args, **kwargs)
        summary = result[-1]
        per_seed_summaries[seed] = summary
        raw_results[seed] = result

    return {
        "per_seed": per_seed_summaries,
        "raw_results": raw_results,
        "aggregate": aggregate_summaries(per_seed_summaries),
    }


def aggregate_summaries(
    per_seed_summaries: Dict[int, dict],
    fields: Sequence[str] = DEFAULT_AGGREGATE_FIELDS,
) -> Dict[str, dict]:
    """Compute median/min/max across seeds for each present numeric field,
    plus a convergence-rate summary for the boolean ``converged`` field.

    A field missing from every summary (e.g. ``final_pde_loss`` for a
    problem whose summary doesn't report it separately) is silently
    skipped rather than producing an empty/misleading entry.
    """
    summaries = list(per_seed_summaries.values())
    aggregate: Dict[str, dict] = {}

    for field in fields:
        values = [
            s[field] for s in summaries
            if field in s and isinstance(s[field], (int, float)) and not isinstance(s[field], bool)
        ]
        if not values:
            continue
        aggregate[field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    converged_flags = [s["converged"] for s in summaries if "converged" in s]
    if converged_flags:
        n_converged = sum(1 for c in converged_flags if c)
        aggregate["converged"] = {
            "n_converged": n_converged,
            "n_total": len(converged_flags),
            "rate": n_converged / len(converged_flags),
        }

    return aggregate
