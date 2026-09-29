"""
Generic Solver Templates for LiL-Q
====================================

Reusable solver implementations for the four methods:
    - NiL-N: Standard PINN (Nonlinear-in-Learnables, Nonlinear PDE)
    - NiL-Q: Quasilinear PINN (Nonlinear-in-Learnables, Quasilinearized PDE)
    - LiL-N: Nonlinear LiL (Linear-in-Learnables, Nonlinear PDE)
    - LiL-Q: Quasilinear LiL (Linear-in-Learnables, Quasilinearized PDE)

Each solver accepts callback functions for the problem-specific parts
(PDE residual, boundary conditions, quasilinearization formula).
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
from typing import Tuple, Callable, Dict, Optional

from .nn import MLP, calculate_hidden_dim
from .metrics import MetricsTracker, QuasilinearMetrics
from .utils import set_seed, clear_gpu_memory
from .instrumentation import EPS_MACH
from .iteration_log import IterationLogger, LilQDiagnosticsTracker

# float64 precision is set explicitly at each parameter/tensor construction
# site (MLP's own dtype=torch.float64 default; explicit dtype= on every
# bare torch.tensor/linspace/zeros call across problems/*.py) rather than
# via a process-wide torch.set_default_dtype() call here. That call used
# to live at this module's import time, silently changing every later
# nn.Module construction anywhere in the process regardless of which
# problem or file did the constructing -- fragile and import-order
# dependent. See DECISIONS.md.


# Evaluation cap, uniform across problems and methods (Addendum v2.2
# Section 2.2): 16 evaluations per allowed iteration of the method's own
# budget. With PyTorch >= 2.10 an L-BFGS step with max_eval=15 makes at most
# 16 evaluations, so this cap cannot bind before the iteration cap; the
# evaluation count stays a logged quantity. NiL-N and LiL-N budget
# ``max_iterations``; NiL-Q budgets ``max_quasi_iters * max_inner_iters``.
# An explicit cap overrides it (smoke tests, probes). DECISIONS.md, 2026-09-29.
LINE_SEARCH_CAP_FACTOR = 16


def line_search_cap(iteration_budget: int, override: Optional[int] = None) -> int:
    """The evaluation cap for a method whose iteration budget is ``iteration_budget``."""
    return int(override) if override is not None else LINE_SEARCH_CAP_FACTOR * int(iteration_budget)


class LBFGSObjective:
    """The loss as the L-BFGS loops see it, with every point evaluated once
    (Addendum v2.2 Section 2.1).

    ``evaluate()`` returns ``(total, parts)``: the loss tensor (with graph)
    and a tuple of floats (its components). :meth:`closure` is what
    ``optimizer.step`` calls; :meth:`value` is the loops' stopping test.

    PyTorch's L-BFGS starts every ``step`` with a closure call at the point
    the previous line search already evaluated, and the loops' stopping test
    then evaluated the accepted point once more. With ``memoize=True`` both
    reuse the stored loss (and gradient, restored into ``.grad``) of any
    point evaluated at the same parameter values, bit for bit, and
    ``n_evals`` counts only real evaluations. The optimizer sees exactly the
    values it would have computed, so on a deterministic device the
    trajectory is bitwise identical to recomputing (tests).

    ``memoize=False`` reproduces the earlier loops exactly -- every closure
    call evaluates and is counted, the stopping test evaluates and is not
    counted -- for the tests' reference runs.
    """

    def __init__(self, params, evaluate: Callable, memoize: bool = True, keep: int = 32):
        self.params = list(params)
        self.evaluate = evaluate
        self.memoize = memoize
        self.keep = keep
        self.n_evals = 0
        self._entries = []

    def point(self) -> torch.Tensor:
        """The current parameter vector (a copy)."""
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def clear(self) -> None:
        """Forget stored values (NiL-Q: the objective changes with each linearization)."""
        self._entries = []

    def _lookup(self, x, need_grad):
        if not self.memoize:
            return None
        for e in reversed(self._entries):
            if (e['grads'] is not None or not need_grad) and torch.equal(e['x'], x):
                return e
        return None

    def _store(self, entry):
        if self.memoize:
            self._entries.append(entry)
            del self._entries[:-self.keep]

    def closure(self):
        x = self.point()
        e = self._lookup(x, need_grad=True)
        if e is not None:
            for p, g in zip(self.params, e['grads']):
                p.grad = None if g is None else g.clone()
            return e['loss']
        for p in self.params:
            p.grad = None
        total, parts = self.evaluate()
        total.backward()
        self.n_evals += 1
        self._store({'x': x, 'loss': total.detach(), 'parts': parts,
                     'grads': [None if p.grad is None else p.grad.detach().clone() for p in self.params]})
        return total

    def value(self, with_grad: bool = False):
        """``(loss, parts)`` at the current point: stored if the point was
        evaluated, else one real evaluation (with its gradient when
        ``with_grad``, so a following step can reuse it)."""
        x = self.point()
        e = self._lookup(x, need_grad=False)
        if e is not None:
            return float(e['loss']), e['parts']
        if not self.memoize:                     # the earlier loops: evaluate, do not count
            total, parts = self.evaluate()
            return total.item(), parts
        if with_grad:
            self.closure()
            e = self._entries[-1]
            return float(e['loss']), e['parts']
        total, parts = self.evaluate()
        self.n_evals += 1
        self._store({'x': x, 'loss': total.detach(), 'parts': parts, 'grads': None})
        return total.item(), parts


# How an L-BFGS run of NiL-N, NiL-Q or LiL-N may end short of its target and
# caps. 'pytorch' (the published tables): PyTorch's absolute tolerances
# (tolerance_grad 1e-8, tolerance_change 1e-9) turn a step into a no-op, and a
# step that leaves the parameters bitwise unchanged is an optimizer_stall.
# 'f1' (the control of the advisor's reply to wave 1, item 2.5): tolerances 0,
# and F1's rule -- a step that does not lower the loss is followed by a step
# with a fresh optimizer, and the run ends (optimizer_stall) only if that step
# does not lower it either. A step that lowers the loss clears the restart.
STALL_RULES = ('pytorch', 'f1')


def _check_stall_rule(stall_rule):
    if stall_rule not in STALL_RULES:
        raise ValueError(f"stall_rule must be one of {STALL_RULES}, not {stall_rule!r}")


def _lbfgs(params, stall_rule='pytorch'):
    tolerance = 0.0 if stall_rule == 'f1' else None
    return optim.LBFGS(
        params, lr=1.0, max_iter=1, max_eval=15,
        tolerance_grad=1e-8 if tolerance is None else tolerance,
        tolerance_change=1e-9 if tolerance is None else tolerance,
        history_size=100, line_search_fn='strong_wolfe',
    )


def _stall_fields(stall, metrics):
    """Summary fields of Addendum v2.2 Section 2.2's ``optimizer_stall``."""
    return {'optimizer_stall': stall is not None,
            'stall_iteration': stall['iteration'] if stall else None,
            'stall_evaluations': stall['evaluations'] if stall else None,
            'stall_time_s': stall['time_s'] if stall else None}


def _lbfgs_loop(objective, optimizer, metrics, max_iterations, max_line_searches, R_tol, verbose,
                new_optimizer=None):
    """The NiL-N / LiL-N loop: L-BFGS steps until the target, the iteration
    cap, the evaluation cap, or an ``optimizer_stall``. Without
    ``new_optimizer`` (``stall_rule='pytorch'``), a stall is a step that
    returns with the parameters bitwise unchanged (the optimizer state is
    then unchanged too, so every later step would repeat the same no-op).
    With it (``'f1'``), a step that does not lower the loss is followed by a
    step with the fresh optimizer ``new_optimizer()``, and a stall is two
    such steps in a row. Returns ``(iterations, converged, stall, restarts)``."""
    iteration, converged, stall = 0, False, None
    restarts, restarted = 0, False
    last = objective.value()[0] if new_optimizer is not None else None   # stored: no evaluation
    while iteration < max_iterations and objective.n_evals < max_line_searches:
        x_before = objective.point()
        optimizer.step(objective.closure)
        iteration += 1
        loss, (pde_val, ic_val, bc_val) = objective.value()
        metrics.record(iteration, objective.n_evals, loss, pde_val, ic_val, bc_val)

        if verbose and iteration % 500 == 0:
            print(f"  Iter {iteration} (evals: {objective.n_evals}): loss={loss:.6e}")
        if not math.isfinite(loss):
            # Diverged: NaN parameters never compare equal, so the stall test
            # cannot fire and the run would spend its whole budget. Stop; the
            # non-finite final loss is classified 'failure' (the advisor's
            # reply to Addendum v2.2, item 2.5).
            if verbose:
                print(f"  Non-finite loss at iter {iteration}: stopping")
            break
        if loss < R_tol:
            if verbose:
                print(f"  Converged at iter {iteration} ({objective.n_evals} evals)")
            converged = True
            break
        if new_optimizer is None:
            no_progress = torch.equal(objective.point(), x_before)
        else:
            no_progress = not loss < last
            if no_progress and not restarted:
                optimizer, restarted = new_optimizer(), True
                restarts += 1
                continue
            if not no_progress:
                last, restarted = loss, False
        if no_progress:
            stall = {'iteration': iteration, 'evaluations': objective.n_evals,
                     'time_s': metrics.data['wall_time'][-1]}
            if verbose:
                print(f"  Optimizer stalled at iter {iteration} ({objective.n_evals} evals)")
            break
    return iteration, converged, stall, restarts


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: NiL-N — Standard PINN (Nonlinear-in-Learnables, Nonlinear PDE)
# ─────────────────────────────────────────────────────────────────────────────

def solve_nil_n(
    compute_pde_residual: Callable,
    compute_bc_residual: Callable,
    model: MLP,
    x_pde: torch.Tensor,
    y_pde: torch.Tensor,
    bc_data: dict,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_ic: float = 10.0,
    max_iterations: int = 10000,
    max_line_searches: int = 100000,
    R_tol: float = 1e-4,
    verbose: bool = True,
    memoize: bool = True,
    stall_rule: str = 'pytorch',
) -> Tuple[MLP, MetricsTracker, Dict]:
    """Standard PINN solver (NiL-N method).

    Parameters
    ----------
    compute_pde_residual : callable
        ``(model, x_pde, y_pde) -> residual_tensor``
    compute_bc_residual : callable
        ``(model, bc_data) -> (bc_loss_tensor, ic_loss_tensor_or_None)``
    model : MLP
        Pre-initialized neural network.
    x_pde, y_pde : torch.Tensor
        Interior collocation point coordinates (requires_grad=True).
    bc_data : dict
        Boundary/IC point data (problem-specific structure).
    lambda_pde, lambda_bc, lambda_ic : float
        Loss weights.
    max_iterations, max_line_searches : int
        Iteration cap and evaluation cap.
    R_tol : float
        Convergence tolerance on total loss.
    verbose : bool
        Print progress.
    memoize : bool
        Evaluate each point once (:class:`LBFGSObjective`); ``False`` only
        for the tests' reference runs.
    stall_rule : str
        ``'pytorch'`` (the published tables) or ``'f1'`` (the stall
        control); see :data:`STALL_RULES`.

    Returns
    -------
    model, metrics, summary
    """
    _check_stall_rule(stall_rule)
    loss_fn = nn.MSELoss()
    metrics = MetricsTracker()
    metrics.start()

    def evaluate():
        residual = compute_pde_residual(model, x_pde, y_pde)
        pde_loss = loss_fn(residual, torch.zeros_like(residual))
        bc_loss, ic_loss = compute_bc_residual(model, bc_data)
        total = lambda_pde * pde_loss + lambda_bc * bc_loss
        ic_val = 0.0
        if ic_loss is not None:
            total = total + lambda_ic * ic_loss
            ic_val = ic_loss.item()
        return total, (pde_loss.item(), ic_val, bc_loss.item())

    model.train()
    objective = LBFGSObjective(model.parameters(), evaluate, memoize=memoize)
    loss, (pde_val, ic_val, bc_val) = objective.value(with_grad=True)   # initial state
    metrics.record(0, objective.n_evals, loss, pde_val, ic_val, bc_val)

    optimizer = _lbfgs(model.parameters(), stall_rule)
    iterations, converged, stall, restarts = _lbfgs_loop(
        objective, optimizer, metrics, max_iterations, max_line_searches, R_tol, verbose,
        new_optimizer=(lambda: _lbfgs(model.parameters(), stall_rule)) if stall_rule == 'f1' else None)

    summary = {
        'method': 'NiL-N',
        'stall_rule': stall_rule,
        'lbfgs_restarts': int(restarts),
        'total_iterations': int(iterations),
        'total_line_searches': int(objective.n_evals),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(metrics.data['wall_time'][-1]),
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'converged': bool(converged),
        **_stall_fields(stall, metrics),
    }
    return model, metrics, summary


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: NiL-Q — Quasilinear PINN (Nonlinear, Quasilinearized PDE)
# ─────────────────────────────────────────────────────────────────────────────

def solve_nil_q(
    compute_pde_residual: Callable,
    compute_linearized_residual_fn: Callable,
    compute_bc_residual: Callable,
    model: MLP,
    x_pde: torch.Tensor,
    y_pde: torch.Tensor,
    bc_data: dict,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_ic: float = 10.0,
    max_quasi_iters: int = 25,
    max_inner_iters: int = 300,
    max_line_searches: int = 100000,
    R_tol: float = 1e-4,
    verbose: bool = True,
    memoize: bool = True,
    stall_rule: str = 'pytorch',
) -> Tuple[MLP, MetricsTracker, Dict]:
    """Quasilinear PINN solver (NiL-Q method).

    One L-BFGS object serves every outer iteration, so curvature pairs from
    earlier linearizations carry over (the method behind the published
    tables; Addendum v2.2 Section 1). Evaluations are memoized within each
    linearization (:class:`LBFGSObjective`); the full nonlinear loss, the
    stopping test after every inner step, is a different function and is
    evaluated each time (``monitor_evaluations``, not in the evaluation
    count). An inner step that leaves the parameters bitwise unchanged ends
    that outer iteration; an outer iteration that leaves them unchanged
    ends the run with ``optimizer_stall`` (Section 2.2).

    With ``stall_rule='f1'`` (the stall control of the advisor's reply to
    wave 1, item 2.5) the tolerances are 0 and an inner step that does not
    lower the linearized loss -- the function L-BFGS minimizes -- is followed
    by one inner step with a fresh optimizer (which then serves the rest of
    the run); a second such step in a row ends that outer iteration. Each
    linearization starts with no restart used.

    Parameters
    ----------
    compute_pde_residual : callable
        Full nonlinear: ``(model, x, y) -> residual``
    compute_linearized_residual_fn : callable
        ``(model, x, y, frozen_data) -> linearized_residual``
        where ``frozen_data`` is computed from the previous iterate.
    compute_bc_residual : callable
        ``(model, bc_data) -> (bc_loss, ic_loss_or_None)``
    model : MLP
        Pre-initialized neural network.
    """
    _check_stall_rule(stall_rule)
    loss_fn = nn.MSELoss()
    metrics = MetricsTracker()
    metrics.start()

    def compute_full_loss():
        residual = compute_pde_residual(model, x_pde, y_pde)
        pde_loss = loss_fn(residual, torch.zeros_like(residual))
        bc_loss, ic_loss = compute_bc_residual(model, bc_data)
        total = lambda_pde * pde_loss + lambda_bc * bc_loss
        ic_val = 0.0
        if ic_loss is not None:
            total = total + lambda_ic * ic_loss
            ic_val = ic_loss.item()
        return total, pde_loss.item(), ic_val, bc_loss.item()

    linearization = {'frozen': None}

    def evaluate_linearized():
        lin_res = compute_linearized_residual_fn(model, x_pde, y_pde, linearization['frozen'])
        lin_loss = loss_fn(lin_res, torch.zeros_like(lin_res))
        bc_loss, ic_loss = compute_bc_residual(model, bc_data)
        total = lambda_pde * lin_loss + lambda_bc * bc_loss
        if ic_loss is not None:
            total = total + lambda_ic * ic_loss
        return total, ()

    # Record initial state
    model.train()
    total_loss, pde_val, ic_val, bc_val = compute_full_loss()
    metrics.record(0, 0, total_loss.item(), pde_val, ic_val, bc_val)

    objective = LBFGSObjective(model.parameters(), evaluate_linearized, memoize=memoize)
    optimizer = _lbfgs(model.parameters(), stall_rule)
    restarts = 0

    total_iterations = 0
    monitor_evals = 0
    n_inner_stalls = 0
    converged = False
    stall = None
    n_quasi_iters = 0

    for quasi_iter in range(max_quasi_iters):
        n_quasi_iters = quasi_iter + 1
        if verbose:
            print(f"\n  Quasilinear iteration {quasi_iter + 1}")

        # Freeze current iterate
        model.eval()
        linearization['frozen'] = compute_linearized_residual_fn(model, x_pde, y_pde, None)
        model.train()
        objective.clear()
        x_outer = objective.point()
        if stall_rule == 'f1':
            # The linearized loss here; with its gradient, so the first
            # step's own evaluation of this point is reused (not counted twice).
            last_inner, restarted = objective.value(with_grad=True)[0], False

        # Inner L-BFGS loop on linearized problem
        for inner_iter in range(max_inner_iters):
            x_before = objective.point()
            optimizer.step(objective.closure)
            total_iterations += 1

            # Evaluate full nonlinear loss
            total_loss, pde_val, ic_val, bc_val = compute_full_loss()
            monitor_evals += 1
            metrics.record(total_iterations, objective.n_evals, total_loss.item(), pde_val, ic_val, bc_val)

            if not math.isfinite(total_loss.item()):       # diverged: stop (item 2.5 of the advisor's reply)
                break
            if total_loss.item() < R_tol or objective.n_evals >= max_line_searches:
                break
            if stall_rule == 'f1':
                inner_loss = objective.value()[0]          # the accepted point: stored
                if inner_loss < last_inner:
                    last_inner, restarted = inner_loss, False
                    continue
                if not restarted:
                    optimizer, restarted = _lbfgs(model.parameters(), stall_rule), True
                    restarts += 1
                    continue
                n_inner_stalls += 1
                break
            if torch.equal(objective.point(), x_before):
                n_inner_stalls += 1
                break

        if verbose:
            print(f"    Loss: {metrics.data['loss'][-1]:.6e} "
                  f"(iter: {total_iterations}, evals: {objective.n_evals})")

        if not math.isfinite(metrics.data['loss'][-1]):
            if verbose:
                print(f"  Non-finite loss at quasi-iteration {quasi_iter + 1}: stopping")
            break
        if metrics.data['loss'][-1] < R_tol:
            if verbose:
                print(f"  Converged at quasi-iteration {quasi_iter + 1}")
            converged = True
            break

        if objective.n_evals >= max_line_searches:
            if verbose:
                print("  Max line searches reached")
            break

        if torch.equal(objective.point(), x_outer):
            stall = {'iteration': total_iterations, 'evaluations': objective.n_evals,
                     'time_s': metrics.data['wall_time'][-1]}
            if verbose:
                print(f"  Optimizer stalled for a whole outer iteration ({quasi_iter + 1})")
            break

    summary = {
        'method': 'NiL-Q',
        'stall_rule': stall_rule,
        'lbfgs_restarts': int(restarts),
        'total_iterations': int(total_iterations),
        'total_line_searches': int(objective.n_evals),
        'monitor_evaluations': int(monitor_evals),
        'n_quasi_iters': int(n_quasi_iters),
        'n_inner_stalls': int(n_inner_stalls),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(metrics.data['wall_time'][-1]),
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'converged': bool(converged),
        **_stall_fields(stall, metrics),
    }

    return model, metrics, summary


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: LiL-N — Nonlinear LiL (Linear-in-Learnables, Nonlinear PDE)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lil_n(
    compute_loss_fn: Callable,
    init_coeffs: np.ndarray,
    device: torch.device,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_ic: float = 10.0,
    max_iterations: int = 10000,
    max_line_searches: int = 100000,
    R_tol: float = 1e-4,
    verbose: bool = True,
    memoize: bool = True,
    stall_rule: str = 'pytorch',
) -> Tuple[np.ndarray, MetricsTracker, Dict]:
    """Nonlinear LiL solver (LiL-N method).

    Optimizes basis coefficients using L-BFGS on the nonlinear PDE residual,
    each point evaluated once (:class:`LBFGSObjective`).

    Parameters
    ----------
    compute_loss_fn : callable
        ``(beta) -> (total_loss, pde_loss, ic_loss, bc_loss)``
        where ``beta`` is a torch.Tensor of coefficients.
        All return values should be torch.Tensor scalars.
    init_coeffs : np.ndarray
        Initial coefficient vector from pre-training.
    device : torch.device
        Computation device.

    Returns
    -------
    coefficients, metrics, summary
    """
    _check_stall_rule(stall_rule)
    beta = torch.from_numpy(init_coeffs).to(device).requires_grad_(True)
    n_coefs = len(init_coeffs)

    metrics = MetricsTracker()
    metrics.start()

    def evaluate():
        # No torch.no_grad -- some loss fns use autograd internally (e.g. BL flux_derivative)
        total, pde_loss, ic_loss, bc_loss = compute_loss_fn(beta)
        return total, (pde_loss.item(), ic_loss.item() if isinstance(ic_loss, torch.Tensor) else ic_loss,
                       bc_loss.item())

    objective = LBFGSObjective([beta], evaluate, memoize=memoize)
    loss, (pde_val, ic_val, bc_val) = objective.value(with_grad=True)   # initial state
    metrics.record(0, objective.n_evals, loss, pde_val, ic_val, bc_val)

    optimizer = _lbfgs([beta], stall_rule)
    iterations, converged, stall, restarts = _lbfgs_loop(
        objective, optimizer, metrics, max_iterations, max_line_searches, R_tol, verbose,
        new_optimizer=(lambda: _lbfgs([beta], stall_rule)) if stall_rule == 'f1' else None)

    summary = {
        'method': 'LiL-N',
        'stall_rule': stall_rule,
        'lbfgs_restarts': int(restarts),
        'total_iterations': int(iterations),
        'total_line_searches': int(objective.n_evals),
        'final_loss': float(metrics.data['loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(metrics.data['wall_time'][-1]),
        'n_params': int(n_coefs),
        'converged': bool(converged),
        **_stall_fields(stall, metrics),
    }

    return beta.detach().cpu().numpy(), metrics, summary


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4: LiL-Q — Quasilinear LiL (Linear-in-Learnables, Quasilinearized)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lil_q(
    assemble_system_fn: Callable,
    compute_nonlinear_loss_fn: Callable,
    init_coeffs: np.ndarray,
    max_quasi_iters: int = 100,
    R_tol: float = 1e-4,
    verbose: bool = True,
    diagnostics_callback: Optional[Callable] = None,
    iteration_logger: Optional[IterationLogger] = None,
    compute_residual_vector_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    conditioning_svd_threshold: Optional[int] = None,
    n_interior_rows: Optional[int] = None,
    interior_weight: Optional[float] = None,
    test_error_fn: Optional[Callable[[np.ndarray], Dict[str, float]]] = None,
) -> Tuple[np.ndarray, QuasilinearMetrics, Dict]:
    """Quasilinear LiL solver (LiL-Q method).

    At each outer iteration:
        1. Evaluate the current solution and derivatives.
        2. Assemble the linearized weighted least-squares system.
        3. Solve via QR (``scipy.linalg.lstsq``).
        4. Check convergence on the full nonlinear loss.

    SVD/condition number analysis is **not** embedded in this solver by
    default. Use ``diagnostics_callback`` for ad hoc opt-in analysis, or
    ``iteration_logger`` for full Computational_Package_1_v2.md Section
    3.1 instrumentation (see ``lilq.iteration_log``) -- the latter is
    what Component B's reruns use.

    Parameters
    ----------
    assemble_system_fn : callable
        ``(beta) -> (A_stacked, b_stacked)``
        where ``A_stacked`` is the weighted system matrix and ``b_stacked``
        is the weighted RHS vector. The system is solved via lstsq.
    compute_nonlinear_loss_fn : callable
        ``(beta) -> (total_loss, pde_loss, ic_loss, bc_loss)``
        Evaluates the full nonlinear loss for convergence checking.
        All return values are floats. ``total_loss`` is treated as
        $\\|\\mathbf{R}\\|_h^2$ (this codebase's own weighted-total-loss
        convention) when ``iteration_logger`` is given.
    init_coeffs : np.ndarray
        Initial coefficient vector from pre-training.
    max_quasi_iters : int
        Maximum outer iterations.
    R_tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    diagnostics_callback : callable, optional
        ``(A_stacked, beta, quasi_iter)`` called at each iteration for
        ad hoc opt-in analysis. Independent of ``iteration_logger``;
        both may be given together. Does NOT affect runtime when ``None``.
    iteration_logger : IterationLogger, optional
        When given, a full Section 3.1 row is recorded every iteration
        (see ``lilq.iteration_log.LilQDiagnosticsTracker`` for exactly
        which columns are populated from generic solver state alone vs.
        left for problem-specific wiring -- interior-only norms and test
        -error columns are NOT populated here). Adds one conditioning
        computation per iteration (SVD, or pivoted QR at the final
        iterate only, per ``conditioning_svd_threshold``) -- real cost,
        opt-in only for this reason.
    compute_residual_vector_fn : callable, optional
        ``(beta) -> weighted_residual_vector``, the full nonlinear
        residual as a vector (not the scalar loss). Only used when
        ``iteration_logger`` is given, to compute the phase indicator
        $\\chi_k$ (Section 3.1 item 4), which needs
        $\\|\\mathbf{R}^{(k+1)} - \\mathbf{R}_{\\mathrm{lin}}^{(k)}\\|_h$ -- a
        vector-difference norm, not derivable from the two residuals'
        norms alone. Without it, ``chi``/``stall_flag`` log as NaN/False.
    conditioning_svd_threshold : int, optional
        Forwarded to ``LilQDiagnosticsTracker``; defaults to
        ``lilq.instrumentation.DEFAULT_SVD_CONDITIONING_THRESHOLD`` (3200)
        when ``None``.
    n_interior_rows, interior_weight : int, float, optional
        Populates ``norm_R_interior``/``norm_Rlin_interior`` when both
        are given: the caller's interior (PDE) rows are the first
        ``n_interior_rows`` rows of ``A_stacked``/``b_stacked``, weighted
        by the single scalar ``interior_weight`` (this codebase's
        universal $\\sqrt{\\lambda_{\\mathrm{block}}/n_{\\mathrm{block}}}$
        convention -- confirmed directly in every problem's
        ``assemble_system_fn``, not assumed). Left NaN if either is
        omitted.
    test_error_fn : callable, optional
        ``(beta) -> {eps_*/maxerr_* column: value}``, forwarded to the
        tracker (Section 3.1 item 10); only used with ``iteration_logger``.

    Returns
    -------
    coefficients, metrics, summary
    """
    beta = init_coeffs.copy().astype(np.float64)
    n_coefs = len(beta)

    metrics = QuasilinearMetrics()

    # Record initial state
    total_loss, pde_loss, ic_loss, bc_loss = compute_nonlinear_loss_fn(beta)
    metrics.record(0, 0, nl_res=total_loss, update_norm=0.0,
                   pde_loss=pde_loss, ic_loss=ic_loss, bc_loss=bc_loss,
                   total_loss=total_loss)

    if verbose:
        print(f"  Initial loss: {total_loss:.6e}")

    tracker = None
    if iteration_logger is not None:
        tracker_kwargs = {}
        if conditioning_svd_threshold is not None:
            tracker_kwargs["conditioning_svd_threshold"] = conditioning_svd_threshold
        if n_interior_rows is not None:
            tracker_kwargs["n_interior_rows"] = n_interior_rows
        if interior_weight is not None:
            tracker_kwargs["interior_weight"] = interior_weight
        tracker = LilQDiagnosticsTracker(test_error_fn=test_error_fn, **tracker_kwargs)

    converged = False
    n_quasi_iters = 0

    for quasi_iter in range(max_quasi_iters):
        n_quasi_iters = quasi_iter + 1
        beta_prev = beta

        # Assemble and solve linearized system
        t0 = time.perf_counter()
        A_stacked, b_stacked = assemble_system_fn(beta)
        t_assemble_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        beta_new, _residues, rank_gelsy, _s = scipy.linalg.lstsq(
            A_stacked, b_stacked, cond=EPS_MACH, lapack_driver='gelsy',
        )
        t_solve_s = time.perf_counter() - t0

        # Opt-in diagnostics (SVD, condition number)
        if diagnostics_callback is not None:
            diagnostics_callback(A_stacked, beta_new, quasi_iter)

        # Convergence metrics
        update_norm = float(np.linalg.norm(beta_new - beta) /
                           (np.linalg.norm(beta_new) + 1e-30))
        beta = beta_new

        total_loss, pde_loss, ic_loss, bc_loss = compute_nonlinear_loss_fn(beta)

        metrics.record(
            quasi_iter + 1, quasi_iter + 1,
            nl_res=total_loss, update_norm=update_norm,
            pde_loss=pde_loss, ic_loss=ic_loss, bc_loss=bc_loss,
            total_loss=total_loss,
        )

        if verbose and (quasi_iter + 1) % 5 == 0:
            print(f"  Iter {quasi_iter + 1}: loss={total_loss:.6e}, "
                  f"d_beta={update_norm:.3e}")

        just_converged = total_loss < R_tol
        is_final_iterate = just_converged or (quasi_iter == max_quasi_iters - 1)

        if tracker is not None:
            t_diag = time.perf_counter()
            row = tracker.step(
                k=quasi_iter,
                A_stacked=A_stacked, b_stacked=b_stacked,
                beta_prev=beta_prev, beta_new=beta_new,
                total_loss=total_loss, rank_gelsy=rank_gelsy,
                t_assemble_s=t_assemble_s, t_solve_s=t_solve_s,
                is_final_iterate=is_final_iterate,
                compute_residual_vector_fn=compute_residual_vector_fn,
            )
            iteration_logger.record(**row)
            # The Section 3.1 diagnostics are passive: off the method's clock.
            metrics.exclude_time(time.perf_counter() - t_diag)

        if just_converged:
            if verbose:
                print(f"  Converged at iteration {quasi_iter + 1}")
            converged = True
            break

    if tracker is not None:
        iteration_logger.record(**tracker.finish(k=n_quasi_iters))

    coefficients = beta.astype(np.float64)
    total_time = metrics.data['wall_time'][-1]

    summary = {
        'method': 'LiL-Q',
        'total_iterations': int(n_quasi_iters),
        'total_line_searches': int(n_quasi_iters),
        'final_loss': float(metrics.data['total_loss'][-1]),
        'final_pde_loss': float(metrics.data['pde_loss'][-1]),
        'training_time': float(total_time),
        'n_params': int(n_coefs),
        'converged': bool(converged),
    }
    if tracker is not None:
        summary['b2_check'] = tracker.b2_check
        summary['kappa_qr_raw_ratio'] = tracker.kappa_qr_raw_ratio

    return coefficients, metrics, summary
