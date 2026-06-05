"""
Synthetic Permeability Field Generator
========================================

Generates synthetic log-normal permeability fields for testing the
Darcy/SPE10 solver without requiring the full SPE10 dataset.

The actual SPE10 permeability data in ``data/spe10/`` was extracted from
the SPE10 Model 2 benchmark (Christie & Blunt, 2001).

Usage::

    python scripts/generate_permeability.py
    python scripts/generate_permeability.py --nx 60 --ny 220 --n-fields 3
"""

import argparse
from pathlib import Path

import numpy as np


def generate_lognormal_field(
    nx: int, ny: int,
    log_mean: float = 2.0,
    log_std: float = 3.0,
    correlation_length: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a spatially correlated log-normal permeability field.

    Uses a simple Gaussian kernel smoothing approach to introduce
    spatial correlation.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((nx, ny))

    # Smooth with a Gaussian kernel in Fourier space
    kx = np.fft.fftfreq(nx, d=1.0)
    ky = np.fft.fftfreq(ny, d=1.0)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    kernel = np.exp(-2 * np.pi**2 * correlation_length**2 * (KX**2 + KY**2))

    smoothed = np.real(np.fft.ifft2(np.fft.fft2(white) * kernel))
    smoothed = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-12)

    log_K = log_mean + log_std * smoothed
    return np.exp(log_K)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic permeability fields")
    parser.add_argument('--nx', type=int, default=60)
    parser.add_argument('--ny', type=int, default=220)
    parser.add_argument('--n-fields', type=int, default=3)
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'data' / 'spe10'))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seeds = [100, 200, 300]
    corr_lengths = [3.0, 5.0, 8.0]

    for i in range(args.n_fields):
        K = generate_lognormal_field(
            args.nx, args.ny,
            seed=seeds[i % len(seeds)],
            correlation_length=corr_lengths[i % len(corr_lengths)],
        )
        fname = f'perm_field_S{i + 1}.txt'
        np.savetxt(out / fname, K.ravel(), fmt='%.6e')
        print(f"Generated {fname}: shape=({args.nx}, {args.ny}), "
              f"K range=[{K.min():.4f}, {K.max():.2f}] mD")


if __name__ == "__main__":
    main()
