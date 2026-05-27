"""
Shared utilities for numerical validation of Theorem 2 (residual bounds).

Theorem 2 (same iterate u^(k)):
    N(u^(k)) = r^(k) - N'(u^(k))(delta u^(k))
    eps_L - Gamma ||delta u^(k)||_X <= ||N(u^(k))||_Y <= eps_U + Gamma ||delta u^(k)||_X

Indexing aligned with LiL-Q loop:
    After completing solve step (k-1), iterate u^(k) is available.
    At the start of step k (k >= 1), validate using r^(k-1), J_(k-1), delta from step (k-1).

Discrete norms: weighted Euclidean on stacked residual vectors (Y),
    ||delta u||_X = L2 norm of function increment at collocation (from Au @ delta_beta).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Theorem2Tracker:
    """Quantities for Theorem 2 validation (iterate-aligned)."""

    # Iterate n = 0, 1, ..., K
    nonlinear_res: List[float] = field(default_factory=list)

    # Recorded at iterate n >= 1 (from data of solve step n-1)
    linear_res: List[float] = field(default_factory=list)          # ||r^(n-1)||
    delta_u_norm: List[float] = field(default_factory=list)        # ||delta u^(n-1)||
    nprime_du_norm: List[float] = field(default_factory=list)      # ||N'(u^(n-1))[delta u]||
    identity_rel_err: List[float] = field(default_factory=list)    # ||N - (r - N'du)|| / ||N||

    # Theorem 2 band at iterate n
    thm2_lower: List[float] = field(default_factory=list)          # eps_L - Gamma*||du||
    thm2_upper: List[float] = field(default_factory=list)          # eps_U + Gamma*||du||

    # Optional comparison to tight predictive correction at previous step
    predictive_tight_upper: List[float] = field(default_factory=list)  # ||r|| + ||w|| at n-1

    # Per solve step (k = 0, ..., K-1) — stored for cross-checks
    sigma_max_J: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: list(v) for k, v in self.__dict__.items()}


def check_theorem2_violations(
    tracker: Theorem2Tracker,
    tol: float = 1e-10,
) -> Dict[str, List[int]]:
    """Check eps_L - Gamma||du|| <= ||N|| <= eps_U + Gamma||du|| at iterates n >= 1."""
    violations: Dict[str, List[int]] = {"lower": [], "upper": []}
    if len(tracker.thm2_lower) == 0:
        return violations

    eps_L = min(tracker.linear_res)
    eps_U = max(tracker.linear_res)

    for i, n in enumerate(range(1, 1 + len(tracker.thm2_lower))):
        actual = tracker.nonlinear_res[n]
        lb = tracker.thm2_lower[i]
        ub = tracker.thm2_upper[i]
        if actual < lb - tol:
            violations["lower"].append(n)
        if actual > ub + tol:
            violations["upper"].append(n)

    return violations


def summarize_gamma(
    nprime_du: np.ndarray,
    delta_u: np.ndarray,
    sigma_max: np.ndarray,
) -> Dict[str, float]:
    """Empirical and Jacobian-based Gamma estimates from a run."""
    du = np.maximum(delta_u, 1e-30)
    ratios = nprime_du / du
    return {
        "gamma_empirical": float(np.max(ratios)) if len(ratios) else 0.0,
        "gamma_empirical_median": float(np.median(ratios)) if len(ratios) else 0.0,
        "gamma_jacobian_max": float(np.max(sigma_max)) if len(sigma_max) else 0.0,
    }


def save_theorem2_json(
    path: Path,
    tracker: Theorem2Tracker,
    summary: Dict,
    violations: Dict,
    gamma_info: Dict,
) -> None:
    path = Path(path)
    data = {
        "summary": summary,
        "gamma": gamma_info,
        "tracker": tracker.to_dict(),
        "violations": violations,
    }
    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else None,
        )


def apply_theorem2_at_iterate(
    tracker: Theorem2Tracker,
    N_vec: np.ndarray,
    r_vec: np.ndarray,
    J_mat: np.ndarray,
    delta_beta: np.ndarray,
    delta_u_norm: float,
    eps_L: float,
    eps_U: float,
    gamma: float,
    predictive_tight_upper: Optional[float] = None,
) -> None:
    """
    Validate Theorem 2 at current iterate u^(k) using data from the previous solve step.

    Parameters
    ----------
    N_vec : nonlinear residual vector at u^(k)
    r_vec : r^(k-1) from previous linearized solve
    J_mat : J_(k-1) from previous step
    delta_beta : beta^(k) - beta^(k-1)
    """
    N_norm = float(np.linalg.norm(N_vec))
    r_norm = float(np.linalg.norm(r_vec))
    nprime_vec = J_mat @ delta_beta
    nprime_norm = float(np.linalg.norm(nprime_vec))

    rhs = r_vec - nprime_vec
    id_err = float(np.linalg.norm(N_vec - rhs))
    id_rel = id_err / max(N_norm, 1e-30)

    tracker.nonlinear_res.append(N_norm)
    tracker.linear_res.append(r_norm)
    tracker.delta_u_norm.append(delta_u_norm)
    tracker.nprime_du_norm.append(nprime_norm)
    tracker.identity_rel_err.append(id_rel)
    tracker.thm2_lower.append(eps_L - gamma * delta_u_norm)
    tracker.thm2_upper.append(eps_U + gamma * delta_u_norm)
    if predictive_tight_upper is not None:
        tracker.predictive_tight_upper.append(predictive_tight_upper)
    else:
        tracker.predictive_tight_upper.append(np.nan)
