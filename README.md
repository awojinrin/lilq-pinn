# Quasilinearized Physics-Informed Least-Squares Collocation in Linear-in-Learnables Trial Spaces

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Codebase for the paper:

> **Quasilinearized Physics-Informed Least-Squares Collocation in Linear-in-Learnables Trial Spaces: Convergence Theory and Practical Stopping Criteria**
> Gbenga T. Awojinrin, Abdul-Akeem Olawoyin, and Rami M. Younis
> (under review)

An earlier version of the paper appeared on arXiv as *A Convex Quasilinearization Method for Solving Nonlinear PDEs with Physics-Informed Neural Networks*; its code is the tag `v1.0-manuscript`.

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

### Reloading trained models

The Package 1 drivers save every trained model next to its logs
(`lilq/saved_models.py`): `solution.pt` for LiL runs, `network.pt` for NiL
runs, F1's `model.pt` and F2's `theta.pt` for the Component A baselines, and
the Darcy PINN's `network.pt`. Load them with the code at the commit in the
run's `hardware.json`:

```python
from lilq.saved_models import load_solution, evaluate_field, load_network, load_f1, load_f2
sol = load_solution('B_instrumentation/bl_gravity_P1024_cpu_paper')   # basis + coefficients + config
S = evaluate_field(sol, 'u', x, t)
net = load_network('B_instrumentation/four_method_jobs/bl_gravity_gpu/models/bl_gravity_P1024_NiL-N_s0_cuda')['model']
from problems.darcy import load_darcy_pinn
pinn, saved = load_darcy_pinn('B_instrumentation/darcy_fv/S1_s0/models/NiL_S1_s0')
```

## Project Structure

```
lilq-pinn/
├── .github/workflows/test.yml  # CI: the dry-run validation suite
├── lilq/                       # Core library
│   ├── basis.py                # Chebyshev, Fourier, ELM, tensor products (1D/2D/ND)
│   ├── solvers.py              # Generic NiL-N, NiL-Q, LiL-N, LiL-Q templates
│   ├── collocation.py          # Collocation point generation
│   ├── pretraining.py, nn.py   # NN and LiL pretraining; MLP architecture (tanh/SiLU)
│   ├── iteration_log.py        # Per-iteration log (iterations.csv)
│   ├── instrumentation.py      # Monitors (chi, round-off, rank) and the stall-based termination rule
│   ├── test_errors.py, references.py  # Errors against exact and reference solutions
│   ├── saved_models.py         # Save and reload every trained model
│   ├── provenance.py, source_lock.py, run_metadata.py  # Commit, hardware and run records
│   ├── properties.py, analysis.py  # Residual-bound checks; SVD and condition numbers
│   └── plotting.py, style.py, utils.py, ...
├── problems/                   # Problem-specific physics (Bratu, Burgers, Buckley-Leverett,
│                               #   Kovasznay, Beltrami, elasticity, Darcy)
├── baselines/                  # Calibrated network baselines (F1, F2) and the classical
│                               #   square Chebyshev collocation baselines
├── experiments/                # Experiment runners
│   ├── run_<problem>.py        # One runner per problem; run_all_dry.py validates them all
│   ├── component_a.py, component_b.py, component_c.py  # Network baselines, instrumented runs,
│   │                           #   oversampling study
│   ├── four_method_tables.py, clean_timing.py, stopping_rule_table.py, ...
│   ├── release_tables.py       # Regenerates the manuscript's tables and checks them (RELEASE.md)
│   └── manuscript_scripts.py   # Runs the manuscript's figure scripts on the results (RELEASE.md)
├── scripts/
│   ├── cluster/                # SLURM jobs and submission scripts (TAMU HPRC Grace and FASTER)
│   ├── make_hprc_bundle.py     # The upload bundle, locked to a commit
│   └── generate_permeability.py, plot_permeability_grid.py
├── reference_results/          # Results of the arXiv version (committed)
├── examples/                   # Quick-start scripts and notebook
├── data/spe10/                 # SPE10 permeability field data
├── tests/                      # pytest suite
├── RELEASE.md                  # Every table and figure of the manuscript -> script and result files
├── REPRODUCE.md                # Replication guide
├── DECISIONS.md                # Dated log of every design decision and deviation
├── pyproject.toml, requirements.txt, CITATION.cff, LICENSE
└── README.md
```

## Repository Status

**Fully implemented:**
- All 7 benchmark problems with complete solver implementations
- 4-method comparison (NiL-N, NiL-Q, LiL-N, LiL-Q) for Bratu, Burgers, Buckley-Leverett
- LiL-Q for Kovasznay, Beltrami, elasticity
- FVM + LiL-Q + NiL-N for Darcy/SPE10
- Residual-bound checks of the convergence theory
- Interactive Jupyter notebook

## Releases

| Tag | Paper version |
|---|---|
| `v2.0.1` | the code release the revised manuscript cites (DOI [10.5281/zenodo.23167177](https://doi.org/10.5281/zenodo.23167177)): `v2.0.0` with corrected citation metadata (`CITATION.cff`) only |
| `v2.0.0` | the revised manuscript (4 October 2026). [RELEASE.md](RELEASE.md) maps every table and figure of its Sections 6 and Appendix B to the script that regenerates it and the result files it reads. |
| `v1.0-manuscript` | the arXiv version |

## Citation

G. T. Awojinrin, A.-A. Olawoyin, R. M. Younis, *Quasilinearized Physics-Informed Least-Squares Collocation in Linear-in-Learnables Trial Spaces: Convergence Theory and Practical Stopping Criteria*, arXiv:2606.18175 (2026); revised version in preparation. [CITATION.cff](CITATION.cff) has the software and paper citations.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
