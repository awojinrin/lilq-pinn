"""
Experiment Metrics Trackers for LiL-Q
======================================

Unified metrics tracking for all four solver methods, supporting
dual iteration/function-evaluation counting and wall-clock timing.
"""

import math
import time
import pandas as pd


def _synchronize():
    """Wait for queued GPU work (no-op without CUDA)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class RunProbe:
    """Package 5, item 3: a wall-clock budget and the error along a run.

    Passed to the L-BFGS loops of :mod:`lilq.solvers` (NiL-N, NiL-Q, LiL-N)
    as ``probe``. After every iteration the loop calls :meth:`after`:
    - every ``every`` iterations (and at iteration 0) it evaluates
      ``error_fn()`` off the clock (:meth:`MetricsTracker.off_clock`) and keeps
      a row ``(iteration, evaluations, time_s, loss, error)``, ``time_s`` the
      ``wall_time`` the run recorded for that iterate (the error evaluations
      excluded);
    - it returns True once the clock reaches ``budget_s`` (the loop then
      stops with ``budget_reached`` in its summary). The iteration that crosses
      the budget completes and is logged.

    The final iterate is always logged (:meth:`finish`). ``error_fn`` must not
    change the parameters or their gradients (check C2).
    """

    def __init__(self, error_fn=None, every: int = 10, budget_s: float = None):
        self.error_fn = error_fn
        self.every = int(every)
        self.budget_s = budget_s
        self.rows = []
        self.metrics = None
        self.budget_reached = False

    def _log(self, iteration, n_evals, loss):
        if self.rows and self.rows[-1]['iteration'] == iteration:
            return
        # the run's recorded time of this iterate (the loop records it just before calling the probe)
        times = self.metrics.data['wall_time']
        t = times[-1] if times else self.metrics.elapsed()
        error = float(self.metrics.off_clock(self.error_fn)) if self.error_fn is not None else math.nan
        self.rows.append({'iteration': int(iteration), 'evaluations': int(n_evals), 'time_s': t,
                          'loss': float(loss), 'error': error})

    def begin(self, metrics, n_evals, loss):
        """Bind the run's clock and log iteration 0."""
        self.metrics = metrics
        self._log(0, n_evals, loss)

    def after(self, iteration, n_evals, loss) -> bool:
        if iteration % self.every == 0:
            self._log(iteration, n_evals, loss)
        if self.budget_s is not None and self.metrics.elapsed() >= self.budget_s:
            self.budget_reached = True
        return self.budget_reached

    def finish(self, iteration, n_evals, loss):
        self._log(iteration, n_evals, loss)


class MetricsTracker:
    """Track metrics during L-BFGS-based optimization (NiL-N, NiL-Q, LiL-N).

    Records both optimizer iterations and function evaluations (line
    searches), enabling convergence plots against either metric.

    Tracked Fields
    --------------
    iteration : int
        Optimizer step count.
    n_func_evals : int
        Cumulative function evaluation count (line searches for L-BFGS).
    loss : float
        Total weighted loss.
    pde_loss : float
        PDE residual loss component.
    ic_loss : float
        Initial condition loss component (0.0 if not applicable).
    bc_loss : float
        Boundary condition loss component.
    wall_time : float
        Elapsed wall-clock time since ``start()`` was called.
    """

    def __init__(self):
        self.data = {
            'iteration': [],
            'n_func_evals': [],
            'loss': [],
            'pde_loss': [],
            'ic_loss': [],
            'bc_loss': [],
            'wall_time': [],
        }
        self.start_time = None
        self.off_clock_s = 0.0

    def start(self) -> None:
        """Begin the wall-clock timer."""
        self.start_time = time.perf_counter()
        self.off_clock_s = 0.0

    def reset(self) -> None:
        """Clear all recorded data and restart the timer."""
        for key in self.data:
            self.data[key] = []
        self.start_time = time.perf_counter()
        self.off_clock_s = 0.0

    def elapsed(self) -> float:
        """Seconds since ``start()``, less the time spent in :meth:`off_clock`."""
        if not self.start_time:
            return 0.0
        return time.perf_counter() - self.start_time - self.off_clock_s

    def off_clock(self, fn, *args, **kwargs):
        """``fn(*args, **kwargs)``, its time excluded from ``wall_time`` and
        :meth:`elapsed` (Package 5, item 3: the error evaluated along a run).
        On a GPU the queued work is finished first, so the solver's own work
        stays on the clock and the evaluation's does not."""
        _synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _synchronize()
            self.off_clock_s += time.perf_counter() - t0

    def record(
        self,
        iteration: int,
        n_func_evals: int,
        loss: float,
        pde_loss: float = 0.0,
        ic_loss: float = 0.0,
        bc_loss: float = 0.0,
    ) -> None:
        """Record a single data point."""
        self.data['iteration'].append(int(iteration))
        self.data['n_func_evals'].append(int(n_func_evals))
        self.data['loss'].append(float(loss))
        self.data['pde_loss'].append(float(pde_loss))
        self.data['ic_loss'].append(float(ic_loss))
        self.data['bc_loss'].append(float(bc_loss))
        self.data['wall_time'].append(self.elapsed())

    def to_dict(self) -> dict:
        """Convert to plain dictionary for JSON serialization."""
        return {k: list(v) for k, v in self.data.items()}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        return pd.DataFrame(self.data)


class QuasilinearMetrics:
    """Track metrics during quasilinear (Bellman–Kalaba) outer iterations.

    Used by the LiL-Q solver where each outer iteration solves a linear
    least-squares sub-problem via QR factorization.

    Tracked Fields
    --------------
    quasi_iter : int
        Outer iteration index.
    n_func_evals : int
        Cumulative function evaluation count (for NiL-Q inner loops).
    nonlinear_residual : float
        Full nonlinear PDE residual evaluated at the current coefficients.
    update_norm : float
        Relative norm of coefficient change between iterations.
    linear_residual : float
        Residual of the linearized least-squares sub-problem.
    pde_loss, ic_loss, bc_loss, total_loss : float
        Decomposed loss components.
    wall_time : float
        Elapsed wall-clock time.
    """

    def __init__(self):
        self.data = {
            'quasi_iter': [],
            'n_func_evals': [],
            'nonlinear_residual': [],
            'update_norm': [],
            'linear_residual': [],
            'pde_loss': [],
            'ic_loss': [],
            'bc_loss': [],
            'total_loss': [],
            'wall_time': [],
        }
        self.converged = False
        self.reason = None
        self.start_time = time.perf_counter()

    def exclude_time(self, seconds: float) -> None:
        """Keep ``seconds`` (the passive diagnostics' cost) out of ``wall_time``
        from now on, so a method's recorded time is its own."""
        self.start_time += seconds

    def reset(self) -> None:
        """Clear all recorded data and restart the timer."""
        for key in self.data:
            self.data[key] = []
        self.converged = False
        self.reason = None
        self.start_time = time.perf_counter()

    def record(
        self,
        quasi_iter: int,
        n_func_evals: int = 0,
        nl_res: float = 0.0,
        update_norm: float = 0.0,
        lin_res: float = 0.0,
        pde_loss: float = 0.0,
        ic_loss: float = 0.0,
        bc_loss: float = 0.0,
        total_loss: float = 0.0,
    ) -> None:
        """Record a single outer-iteration data point."""
        self.data['quasi_iter'].append(int(quasi_iter))
        self.data['n_func_evals'].append(int(n_func_evals))
        self.data['nonlinear_residual'].append(float(nl_res))
        self.data['update_norm'].append(float(update_norm))
        self.data['linear_residual'].append(float(lin_res))
        self.data['pde_loss'].append(float(pde_loss))
        self.data['ic_loss'].append(float(ic_loss))
        self.data['bc_loss'].append(float(bc_loss))
        self.data['total_loss'].append(float(total_loss))
        self.data['wall_time'].append(time.perf_counter() - self.start_time)

    def to_dict(self) -> dict:
        """Convert to plain dictionary for JSON serialization."""
        return {k: list(v) for k, v in self.data.items()}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        return pd.DataFrame(self.data)
