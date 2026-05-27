import numpy as np
import torch
import torch.nn as nn
import time
import math
import random

def set_seed(seed=42):
    """Set random seed across numpy, random, and PyTorch for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_gpu_memory():
    """Flush the CUDA cache to prevent GPU memory bloat."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_hidden_dim(n_hidden_layers: int, N_x: int, N_t: int) -> int:
    """Calculate the neural network hidden layer dimension to match the DOFs of basis coefficients."""
    n_target = N_x * N_t
    nh = n_hidden_layers
    a = nh - 1
    b = 3 + nh
    c = -n_target

    discriminant = b**2 - 4*a*c
    nl1 = (-b + np.sqrt(discriminant)) / (2*a)
    nl2 = (-b - np.sqrt(discriminant)) / (2*a)

    return math.ceil(abs(max(nl1, nl2))) if nh > 1 else math.ceil(n_target / 4)

class MLP(nn.Module):
    """Standard multi-layer perceptron for PINN and NiL solvers."""

    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1,
                 num_layers=4, activation=nn.Tanh(), output_activation=None):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.network = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.network(x)

class QuasilinearMetrics:
    """Track metrics during quasilinear iteration.
    
    Tracks BOTH iterations AND function evaluations, plus wall-clock time.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data = {
            'quasi_iter': [], 'n_func_evals': [], 'nonlinear_residual': [], 'update_norm': [],
            'linear_residual': [], 'wall_time': [],
            'pde_loss': [], 'ic_loss': [], 'bc_loss': [], 'total_loss': [], 'cond_number': []
        }
        self.converged = False
        self.reason = None
        self.start_time = time.time()
    
    def record(self, quasi_iter, n_func_evals, nl_res, update_norm, lin_res=0, 
               pde_loss=0, ic_loss=0, bc_loss=0, total_loss=0, cond_number=0):
        self.data['quasi_iter'].append(int(quasi_iter))
        self.data['n_func_evals'].append(int(n_func_evals))
        self.data['nonlinear_residual'].append(float(nl_res))
        self.data['update_norm'].append(float(update_norm))
        self.data['linear_residual'].append(float(lin_res))
        self.data['wall_time'].append(time.time() - self.start_time)
        self.data['pde_loss'].append(float(pde_loss))
        self.data['ic_loss'].append(float(ic_loss))
        self.data['bc_loss'].append(float(bc_loss))
        self.data['total_loss'].append(float(total_loss))
        self.data['cond_number'].append(float(cond_number))
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {k: list(v) for k, v in self.data.items()}
    
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.data)

class MetricsTracker:
    """Standardized metrics tracker with dual iteration and line search evaluation tracking."""

    def __init__(self):
        self.data = {
            'iteration': [],
            'n_func_evals': [],
            'loss': [],
            'residual_norm': [],
            'pde_loss': [],
            'ic_loss': [],
            'bc_loss': [],
            'wall_time': [],
            'linear_mse': [],
            'cond_number': []
        }
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def record(self, iteration, n_func_evals, loss, pde_loss=None, ic_loss=None, bc_loss=None, 
               linear_mse=None, cond_number=None, residual_norm=None):
        self.data['iteration'].append(int(iteration))
        self.data['n_func_evals'].append(int(n_func_evals))
        self.data['loss'].append(float(loss))
        self.data['residual_norm'].append(float(residual_norm) if residual_norm is not None else float(loss))
        self.data['pde_loss'].append(float(pde_loss) if pde_loss is not None else float(loss))
        self.data['ic_loss'].append(float(ic_loss) if ic_loss is not None else 0.0)
        self.data['bc_loss'].append(float(bc_loss) if bc_loss is not None else 0.0)
        self.data['wall_time'].append(time.time() - self.start_time if self.start_time else 0.0)
        self.data['linear_mse'].append(float(linear_mse) if linear_mse is not None else 0.0)
        self.data['cond_number'].append(float(cond_number) if cond_number is not None else 0.0)

    def to_dict(self):
        return {k: list(v) for k, v in self.data.items()}

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.data)
