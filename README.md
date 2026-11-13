# lilq-pinn

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23167176-blue.svg)](https://doi.org/10.5281/zenodo.23167176)

Quasilinearized least-squares collocation for nonlinear partial differential equations in trial spaces that are linear in their parameters.

## Overview

LiL-Q solves a nonlinear PDE by quasilinearization (Newton–Kantorovich).
- Each outer iteration linearizes the operator about the current iterate.
- It collocates the resulting linear problem at more points than unknowns.
- It solves the weighted linear least-squares system by a column-pivoted QR factorization.

The trial space is a fixed expansion whose coefficients enter linearly: tensor Chebyshev or Fourier bases, or random features. No step involves gradient-based training.

The iteration stops by an a posteriori rule built from quantities it already computes:
- the phase indicator χ;
- the stationarity of the linearized residual;
- a classification of the stall as limited by the approximation or by conditioning.

The repository also implements three comparison formulations that keep one or both sources of nonlinearity:

| Formulation | Trial function | Nonlinearity | Solver |
|---|---|---|---|
| **NiL-N** | neural network | minimizes the nonlinear residual | L-BFGS or Levenberg–Marquardt |
| **NiL-Q** | neural network | quasilinearized, one training problem per iteration | L-BFGS |
| **LiL-N** | linear expansion | minimizes the nonlinear residual | L-BFGS |
| **LiL-Q** | linear expansion | quasilinearized, one least-squares problem per iteration | column-pivoted QR |

## Features

- **Trial spaces:**
  - Chebyshev, Fourier (sine, cosine, mixed) and augmented one-dimensional families;
  - tensor products in two to four dimensions;
  - random-feature (ELM) bases.
- **Collocation sets:**
  - random tensor and scattered points;
  - equispaced grids;
  - Chebyshev–Gauss–Lobatto grids with Clenshaw–Curtis weights;
  - cell-centred points for gridded coefficients.
- **Monitoring:**
  - per-iteration logs (`iterations.csv`) of the nonlinear and linearized residuals, the phase indicator and the observed order;
  - the numerical rank and the condition number;
  - the termination rule (`lilq/instrumentation.py`).
- **Stability constants:** a posteriori norm-equivalence constants of a collocation set, and the realized quasi-optimality ratio (`lilq/certified.py`).
- **Baselines:**
  - physics-informed networks trained by L-BFGS and Levenberg–Marquardt;
  - square Chebyshev collocation with Newton's method;
  - a two-point-flux finite-volume solver for Darcy flow.
- **Provenance:** every run records its commit, hardware, BLAS threading and settings (`run.json`, `hardware.json`). Trained models can be saved and reloaded (`lilq/saved_models.py`).

## Benchmark problems

| Problem | Type | Module |
|---|---|---|
| Bratu equation | 2D nonlinear elliptic | `problems/bratu.py` |
| Viscous Burgers equation | 1D + time, nonlinear parabolic | `problems/burgers.py` |
| Buckley–Leverett equation (viscous and gravity-influenced) | 1D + time, regularized conservation law | `problems/buckley_leverett.py` |
| Plane-strain linear elasticity | 2D linear system | `problems/elasticity.py` |
| Kovasznay flow | 2D steady incompressible Navier–Stokes | `problems/kovasznay.py` |
| Beltrami flow | 3D + time incompressible Navier–Stokes | `problems/beltrami.py` |
| Single-phase Darcy flow (synthetic and SPE10 permeability) | 2D heterogeneous elliptic, mixed form | `problems/darcy.py` |

## Installation

```bash
git clone https://github.com/awojinrin/lilq-pinn.git
cd lilq-pinn
pip install -r requirements.txt
```

Requirements:
- Python 3.10 or later.
- The pinned versions in `requirements.txt`: NumPy 2.4.2, SciPy 1.17.0, PyTorch 2.10.0, Matplotlib 3.10.8, pandas 2.3.0 and threadpoolctl 3.6.0.
- PyTorch is needed only for the network formulations and the GPU solve path.

## Quick start

```python
from problems.bratu import BratuConfig, BratuOptConfig, run_lil_q

config = BratuConfig(N_x=15, N_y=15, basis_type='chebyshev')
opt = BratuOptConfig(R_tol=1e-7)

basis, coefficients, metrics, summary = run_lil_q(config, opt)
print(f"Final loss: {summary['final_loss']:.2e}")
```

Standalone scripts for several problems are in [`examples/`](examples/), along with an [interactive notebook](examples/lilq_demo.ipynb).

## Running experiments

Each problem has a command-line runner in `experiments/`:

```bash
python experiments/run_all_dry.py                        # every problem at minimal size (< 2 min)

python experiments/run_bratu.py --basis fourier --N 5 10 15
python experiments/run_burgers.py --N 5 10 15 20 25
python experiments/run_bl.py --gravity --N 8 16 24 32
python experiments/run_kovasznay.py --N 5 10 15 20 25
python experiments/run_beltrami.py --N_vel 6 --N_p 8
python experiments/run_elasticity.py --N 5 10 15 20 25
python experiments/run_darcy.py --fields S1 S2 S3 SPE10 --order 32
python experiments/run_burgers_basis_comparison.py --bases sin_cheb cheb_cheb elm
```

- **Bratu, Burgers and Buckley–Leverett.** The runners compare all four formulations. `--lil-q-only` restricts them to LiL-Q.
- **Sizes.** `--N` gives the number of modes per direction (P = N² coefficients for the scalar problems).

**Reloading saved models.** Instrumented runs save their models next to their logs:

```python
from lilq.saved_models import load_solution, evaluate_field
sol = load_solution('<run directory>')        # basis, coefficients and configuration
u = evaluate_field(sol, 'u', x, t)
```

## Repository layout

```
lilq/          core library: bases, collocation, solvers, monitors and the termination rule,
               stability constants, iteration logs, provenance, saved models
problems/      one module per benchmark problem
baselines/     calibrated network baselines and square-collocation baselines
experiments/   runners, studies and the scripts that regenerate published tables and figures
scripts/       cluster job scripts (SLURM) and data utilities
data/spe10/    permeability fields
examples/      quick-start scripts and notebook
tests/         pytest suite
```

## Testing

```bash
python -m pytest -q                  # unit and regression tests
python experiments/run_all_dry.py    # end-to-end validation (also run by CI)
```

## Results and reproducibility

[RELEASE.md](RELEASE.md) maps each published table and figure to the script that regenerates it and the result files it reads. [REPRODUCE.md](REPRODUCE.md) lists the commands for each study.

The result files are on the repository's `*-results` branches and in the Zenodo record [10.5281/zenodo.23147620](https://doi.org/10.5281/zenodo.23147620). They include:
- the per-iteration logs;
- the trained models;
- the timing records.

[DECISIONS.md](DECISIONS.md) records the design decisions and deviations, in date order.

## Citation

If you use this software, please cite it as described in [CITATION.cff](CITATION.cff). The archived code releases are at [10.5281/zenodo.23167176](https://doi.org/10.5281/zenodo.23167176).

## License

MIT; see [LICENSE](LICENSE).
