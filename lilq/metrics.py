"""
Experiment Metrics Trackers for LiL-Q
======================================

Unified metrics tracking for all four solver methods, supporting
dual iteration/function-evaluation counting and wall-clock timing.
"""

import time
import pandas as pd


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

    def start(self) -> None:
        """Begin the wall-clock timer."""
        self.start_time = time.time()

    def reset(self) -> None:
        """Clear all recorded data and restart the timer."""
        for key in self.data:
            self.data[key] = []
        self.start_time = time.time()

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
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        self.data['wall_time'].append(elapsed)

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
        self.start_time = time.time()

    def reset(self) -> None:
        """Clear all recorded data and restart the timer."""
        for key in self.data:
            self.data[key] = []
        self.converged = False
        self.reason = None
        self.start_time = time.time()

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
        self.data['wall_time'].append(time.time() - self.start_time)

    def to_dict(self) -> dict:
        """Convert to plain dictionary for JSON serialization."""
        return {k: list(v) for k, v in self.data.items()}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        return pd.DataFrame(self.data)
