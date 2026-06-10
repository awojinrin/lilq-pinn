# A Convex Quasilinearization Method for Solving Nonlinear PDEs with Physics-Informed Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Codebase for the paper:

> **A Convex Quasilinearization Method for Solving Nonlinear PDEs with Physics-Informed Neural Networks**
> Gbenga T. Awojinrin, Abdul-Akeem Olawoyin, and Rami M. Younis
> (under review)

## Overview

LiL-Q parameterises PDE solution fields with **linear basis expansions** (linear networks or orthogonal polynomials) and applies **Bellman-Kalaba quasilinearisation** to reduce nonlinear PDEs to a sequence of weighted linear least-squares problems solved directly via QR factorisation. This eliminates neural network training epochs, bypasses non-convex optimisation landscapes, and provides reproducible, high-accuracy solutions.

The framework implements four solver variants for systematic comparison:

| Method | Representation | Optimisation |
|--------|---------------|-------------|
| **NiL-N** | Neural Network (nonlinear) | L-BFGS (nonlinear) |
| **NiL-Q** | Neural Network (nonlinear) | L-BFGS (nonlinear) |
| **LiL-N** | Linear basis expansion | L-BFGS (nonlinear) |
| **LiL-Q** | Linear basis expansion | QR factorisation (linear) |

## Problems

### Four-Method Benchmark Problems
- **Bratu equation** — 2D nonlinear elliptic PDE
- **Burgers equation** — 1D+time nonlinear parabolic PDE
- **Buckley-Leverett** — 1D+time nonlinear conservation law (with optional gravity)

### Application Problems (LiL-Q Only)
- **Kovasznay flow** — 2D steady incompressible Navier-Stokes
- **Beltrami flow** — 3D+time unsteady incompressible Navier-Stokes
- **Linear elasticity** — 2D plane-strain (Haghighat et al., CMAME 2021)

### Porous Media (LiL-Q + FVM + NiL-N)
- **SPE10/Darcy flow** — Heterogeneous permeability fields

## Installation

```bash
git clone https://github.com/awojinrin/lilq-pinn.git
cd lilq-pinn
pip install -r requirements.txt
```

**Dependencies** (pinned for reproducibility):
- Python >= 3.10
- NumPy 2.4.2, SciPy 1.17.0, Matplotlib 3.10.8, Pandas 2.3.0
- PyTorch 2.10.0 (for NiL-N/NiL-Q baseline comparisons)

## Quick Start

```python
from problems.bratu import BratuConfig, BratuOptConfig, run_lil_q

config = BratuConfig(N_x=15, N_y=15, basis_type='chebyshev')
opt = BratuOptConfig(R_tol=1e-7)

basis, coefficients, metrics, summary = run_lil_q(config, opt)
print(f"Final loss: {summary['final_loss']:.2e}")
```

See [`examples/`](examples/) for standalone scripts and an
[interactive notebook](examples/lilq_demo.ipynb).

## Running Experiments

```bash
# End-to-end validation (all problems, minimal settings, < 2 min)
python experiments/run_all_dry.py

# Full experiments per problem
python experiments/run_bratu.py --basis fourier --N 5 10 15
python experiments/run_burgers.py --N 5 10 15
python experiments/run_bl.py --gravity --N 8 16 24 32
python experiments/run_kovasznay.py --N 5 10 15 20 25
python experiments/run_beltrami.py --N 3 4 5 6 8
python experiments/run_elasticity.py --N 5 10 15 20 25
python experiments/run_darcy.py --fields S3 --order 32

# Basis comparison study
python experiments/run_burgers_basis_comparison.py
```

## Project Structure

```
LiL-Q/
├── .github/workflows/       # Continuous Integration workflows
│   └── test.yml             # Dry-run validation suite runner
├── lilq/                    # Core library
│   ├── basis.py             # Chebyshev, Fourier, ELM, tensor products (1D/2D/ND)
│   ├── solvers.py           # Generic NiL-N, NiL-Q, LiL-N, LiL-Q templates
│   ├── collocation.py       # Collocation point generation
│   ├── pretraining.py       # NN and LiL pretraining
│   ├── nn.py                # MLP architecture (tanh/SiLU)
│   ├── metrics.py           # Experiment tracking
│   ├── properties.py        # Theorem 2 residual bounds validation
│   ├── analysis.py          # SVD, condition number studies
│   ├── plotting.py          # Publication-quality plots
│   ├── style.py             # JCP/Elsevier styling
│   └── utils.py             # Seeds, GPU management, checkpointing
├── problems/                # Problem-specific physics
│   ├── bratu.py
│   ├── burgers.py
│   ├── buckley_leverett.py
│   ├── kovasznay.py
│   ├── beltrami.py
│   ├── elasticity.py
│   └── darcy.py
├── experiments/             # Experiment runners
│   ├── exp_utils.py         # Shared saving/plotting utilities
│   ├── run_all_dry.py       # End-to-end validation
│   ├── run_bratu.py
│   ├── run_burgers.py
│   ├── run_bl.py
│   ├── run_kovasznay.py
│   ├── run_beltrami.py
│   ├── run_elasticity.py
│   ├── run_darcy.py
│   └── run_burgers_basis_comparison.py
├── reference_results/       # Pre-computed paper results (committed)
│   └── <problem>_experiments_<basis>/
│       ├── N_<n>/           # Per-N checkpoints + metrics
│       ├── figures/         # Convergence + solution plots (PNG only)
│       └── master_results.json
├── examples/                # Quick-start scripts + notebook
│   ├── run_bratu_lilq.py
│   ├── run_burgers_lilq.py
│   ├── run_kovasznay_lilq.py
│   ├── run_elasticity_lilq.py
│   ├── run_bl_lilq.py
│   └── lilq_demo.ipynb
├── data/spe10/              # SPE10 permeability field data
├── scripts/
│   └── generate_permeability.py
├── pyproject.toml           # Package metadata and dynamic install setup
├── CITATION.cff             # Author citation metadata (full author list)
├── REPRODUCE.md             # Detailed replication guide
├── requirements.txt
├── LICENSE
└── README.md

```

## Repository Status

**Fully implemented:**
- All 7 benchmark problems with complete solver implementations
- 4-method comparison (NiL-N, NiL-Q, LiL-N, LiL-Q) for Bratu, Burgers, Buckley-Leverett
- LiL-Q for Kovasznay, Beltrami, elasticity
- FVM + LiL-Q + NiL-N for Darcy/SPE10
- Theorem 2 residual bounds validation
- Interactive Jupyter notebook

## Citation

Manuscript under review. The BibTeX citation entry will be updated here upon acceptance.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
