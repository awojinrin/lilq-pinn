"""
Theorem 2 Residual Bounds — Validation & Visualization
========================================================

Numerical validation of the non-tautological residual bounds from
Theorem 2 of the LiL-Q manuscript:

    eps_L - Gamma ||delta u||_X  <=  ||N[u^(n)]||_Y  <=  eps_U + Gamma ||delta u||_X

where N[u^(n)] is the nonlinear residual at iterate n, r^(n-1) is the
linear residual from the previous solve step, and Gamma bounds the
Lipschitz-type majorization ||N'[delta u]||_Y <= Gamma ||delta u||_X.

Indexing convention (aligned with LiL-Q solver loop):
    After completing solve step (k-1), iterate u^(k) is available.
    At the start of step k (k >= 1), validate using r^(k-1), J_(k-1),
    delta from step (k-1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Theorem2Tracker:
    """Quantities tracked for Theorem 2 validation (iterate-aligned)."""

    nonlinear_res: List[float] = field(default_factory=list)

    # Recorded at iterate n >= 1 (from data of solve step n-1)
    linear_res: List[float] = field(default_factory=list)
    delta_u_norm: List[float] = field(default_factory=list)
    nprime_du_norm: List[float] = field(default_factory=list)
    identity_rel_err: List[float] = field(default_factory=list)

    # Theorem 2 band at iterate n
    thm2_lower: List[float] = field(default_factory=list)
    thm2_upper: List[float] = field(default_factory=list)

    # Optional tight predictive correction
    predictive_tight_upper: List[float] = field(default_factory=list)

    # Jacobian singular values per solve step
    sigma_max_J: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: list(v) for k, v in self.__dict__.items()}


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
    """Validate Theorem 2 at current iterate u^(k) using previous step data.

    Parameters
    ----------
    N_vec : nonlinear residual vector at u^(k)
    r_vec : r^(k-1) from previous linearised solve
    J_mat : Jacobian from step (k-1)
    delta_beta : coefficient change beta^(k) - beta^(k-1)
    delta_u_norm : ||delta u^(k-1)||_X at collocation points
    eps_L, eps_U : running min/max of ||r||
    gamma : Lipschitz-type bound
    predictive_tight_upper : optional ||r|| + ||w|| tight bound
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
    tracker.predictive_tight_upper.append(
        predictive_tight_upper if predictive_tight_upper is not None else np.nan
    )


def check_theorem2_violations(
    tracker: Theorem2Tracker,
    tol: float = 1e-10,
) -> Dict[str, List[int]]:
    """Check bound violations at iterates n >= 1.

    Returns dict with 'lower' and 'upper' lists of violating iterate indices.
    """
    violations: Dict[str, List[int]] = {"lower": [], "upper": []}
    if len(tracker.thm2_lower) == 0:
        return violations

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
    """Empirical and Jacobian-based Gamma estimates."""
    du = np.maximum(delta_u, 1e-30)
    ratios = nprime_du / du
    return {
        "gamma_empirical": float(np.max(ratios)) if len(ratios) else 0.0,
        "gamma_empirical_median": float(np.median(ratios)) if len(ratios) else 0.0,
        "gamma_jacobian_max": float(np.max(sigma_max)) if len(sigma_max) else 0.0,
    }


def plot_theorem2_validation(
    tracker: Theorem2Tracker,
    output_dir: Path,
    N: int,
    problem_label: str,
    gamma_used: float,
) -> None:
    """Two-panel banded plot: Theorem 2 sandwich + majorisation check."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import lilq.style  # noqa: F401 — applies rcParams on import
    except ImportError:
        pass

    nonlin = np.array(tracker.nonlinear_res)
    lin = np.array(tracker.linear_res)
    n_thm = len(tracker.thm2_lower)

    eps_L = np.min(lin) if len(lin) else 0.0
    eps_U = np.max(lin) if len(lin) else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=150)

    # Top: Theorem 2 band vs actual ||N||
    ax = axes[0]
    iterates = np.arange(len(nonlin))
    band_x = np.arange(1, 1 + n_thm)
    lower = np.array(tracker.thm2_lower)
    upper = np.array(tracker.thm2_upper)
    plot_floor = max(0.1 * eps_L, 1e-30)
    lower_clip = np.maximum(lower, plot_floor)

    ax.fill_between(band_x, lower_clip, upper, alpha=0.25, color="#457B9D",
                    label=rf"Theorem 2 band ($\Gamma={gamma_used:.2e}$)")
    ax.semilogy(iterates, nonlin, "o-", color="#E63946", ms=5, lw=2,
                label=r"$\|\mathcal{N}[u_n]\|_2$ (actual)")
    if len(lin) == n_thm:
        ax.semilogy(band_x, lin, "s--", color="#6A0572", ms=4, lw=1.5,
                    label=r"$\|r_{n-1}\|_2$")
    ax.axhline(y=eps_L, color="#2A9D8F", ls=":", lw=1.5,
               label=rf"$\varepsilon_L={eps_L:.2e}$")
    ax.axhline(y=eps_U, color="#F4A261", ls=":", lw=1.5,
               label=rf"$\varepsilon_U={eps_U:.2e}$")
    ax.set_xlabel("Iterate $n$")
    ax.set_ylabel("Residual norm (weighted $L_2$)")
    ax.set_title(
        f"{problem_label}: Theorem 2 bounds on "
        r"$\|\mathcal{N}[u^{(n)}]\|$"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Bottom: ||N'du|| vs Gamma||du||
    ax = axes[1]
    du = np.array(tracker.delta_u_norm)
    nprime = np.array(tracker.nprime_du_norm)
    ax.semilogy(band_x, nprime, "o-", color="#E63946", ms=4,
                label=r"$\|\mathcal{N}'(u^{(n-1)})[\delta u^{(n-1)}]\|_Y$")
    ax.semilogy(band_x, gamma_used * du, "s--", color="#264653", ms=4,
                label=rf"$\Gamma\|\delta u^{{(n-1)}}\|_X$ ($\Gamma={gamma_used:.2e}$)")
    ax.set_xlabel("Iterate $n$")
    ax.set_ylabel("Norm")
    ax.set_title(
        r"Majorisation: $\|\mathcal{N}'[\delta u]\|_Y "
        r"\leq \Gamma\|\delta u\|_X$"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    stem = f"{problem_label}_theorem2_N{N}"
    plt.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {stem}.pdf/png")


def save_theorem2_json(
    path: Path,
    tracker: Theorem2Tracker,
    summary: Dict,
    violations: Dict,
    gamma_info: Dict,
) -> None:
    """Serialise Theorem 2 validation results to JSON."""
    path = Path(path)
    data = {
        "summary": summary,
        "gamma": gamma_info,
        "tracker": tracker.to_dict(),
        "violations": violations,
    }
    with open(path, "w") as f:
        json.dump(
            data, f, indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else None,
        )


def load_theorem2_json(path: Path) -> Dict:
    """Load Theorem 2 validation results from JSON."""
    with open(Path(path)) as f:
        return json.load(f)


def make_theorem2_callback(
    tracker: Theorem2Tracker,
    gamma: Optional[float] = None,
) -> Callable:
    """Factory: return a diagnostics callback for ``solve_lil_q``.

    The callback receives ``(iteration, A_stacked, b_stacked, coeffs,
    coeffs_prev, residual_vec)`` at each quasilinear iteration.

    If ``gamma`` is None, an empirical estimate is computed from
    accumulated data at each step.
    """
    state = {"prev_A": None, "prev_r": None, "prev_beta": None,
             "eps_L": np.inf, "eps_U": 0.0}

    def callback(iteration, A_stacked, b_stacked, coeffs, coeffs_prev,
                 residual_vec, **kwargs):
        r_vec = residual_vec
        r_norm = float(np.linalg.norm(r_vec))
        state["eps_L"] = min(state["eps_L"], r_norm)
        state["eps_U"] = max(state["eps_U"], r_norm)

        if iteration == 0:
            N_norm = float(np.linalg.norm(b_stacked - A_stacked @ coeffs))
            tracker.nonlinear_res.append(N_norm)
        else:
            delta_beta = coeffs - coeffs_prev
            Au = kwargs.get("Au", A_stacked)
            delta_u_norm = float(np.linalg.norm(Au @ delta_beta))

            N_vec = b_stacked - A_stacked @ coeffs
            g = gamma
            if g is None:
                nprime_vec = state["prev_A"] @ delta_beta
                nprime_norm = float(np.linalg.norm(nprime_vec))
                g = nprime_norm / max(delta_u_norm, 1e-30)

            apply_theorem2_at_iterate(
                tracker, N_vec, state["prev_r"],
                state["prev_A"], delta_beta, delta_u_norm,
                state["eps_L"], state["eps_U"], g,
            )

        state["prev_A"] = A_stacked.copy()
        state["prev_r"] = r_vec.copy()
        state["prev_beta"] = coeffs.copy()

        sigma_max = float(np.linalg.norm(A_stacked, ord=2))
        tracker.sigma_max_J.append(sigma_max)

    return callback
