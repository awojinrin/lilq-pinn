"""
Utility Functions for LiL-Q
============================

Reproducibility, GPU management, and checkpointing utilities
shared across all experiments.
"""

import numpy as np
import torch
import random
import gc


# ─────────────────────────────────────────────────────────────────────────────
# Device Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds across all backends for reproducibility.

    Sets seeds for: numpy, Python random, PyTorch CPU, PyTorch CUDA.
    Also disables cuDNN non-deterministic algorithms.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# GPU Memory Management
# ─────────────────────────────────────────────────────────────────────────────

def clear_gpu_memory() -> None:
    """Flush CUDA cache and run garbage collection to free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory() -> None:
    """Print current GPU memory usage (allocated and reserved)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("CUDA not available")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O for LiL Methods
# ─────────────────────────────────────────────────────────────────────────────

def save_lil_checkpoint(basis, coefficients, metrics, summary, filepath) -> None:
    """Save LiL method checkpoint (basis info, coefficients, metrics).

    Parameters
    ----------
    basis : TensorProductBasis2D or ELMBasis2D
        The basis object used.
    coefficients : np.ndarray
        Final basis coefficients.
    metrics : MetricsTracker or QuasilinearMetrics
        Training metrics.
    summary : dict
        Experiment summary dictionary.
    filepath : str or Path
        Output file path (.npz).
    """
    save_dict = {
        'coefficients': np.asarray(coefficients),
        'n_basis': np.array([basis.n_basis]),
    }

    # Store metrics
    if hasattr(metrics, 'to_dict'):
        metrics_dict = metrics.to_dict()
    elif hasattr(metrics, 'data'):
        metrics_dict = {k: list(v) for k, v in metrics.data.items()}
    else:
        metrics_dict = metrics

    for key, values in metrics_dict.items():
        save_dict[f'metrics_{key}'] = np.array(values)

    # Store summary scalars
    for key, value in summary.items():
        if isinstance(value, (int, float, bool)):
            save_dict[f'summary_{key}'] = np.array([value])

    np.savez(filepath, **save_dict)


def load_lil_checkpoint(filepath) -> dict:
    """Load LiL method checkpoint.

    Returns
    -------
    dict with keys: 'coefficients', 'n_basis', 'metrics', 'summary'
    """
    data = np.load(filepath, allow_pickle=True)

    result = {
        'coefficients': data['coefficients'],
        'n_basis': int(data['n_basis'][0]),
    }

    # Reconstruct metrics
    metrics = {}
    summary = {}
    for key in data.files:
        if key.startswith('metrics_'):
            metrics[key[8:]] = data[key].tolist()
        elif key.startswith('summary_'):
            val = data[key]
            summary[key[8:]] = val[0] if len(val) == 1 else val.tolist()

    result['metrics'] = metrics
    result['summary'] = summary
    return result
