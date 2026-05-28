# Linear-in-Learnables Quasilinearized Solvers for Nonlinear PDEs (LiL-Q)

This repository contains the official, publication-grade implementation of the **Linear-in-Learnables Quasilinearized Solver (LiL-Q)** framework. 

LiL-Q is an accelerated, direct solver architecture for solving non-convex, highly nonlinear boundary value problems and Navier-Stokes formulations. By parameterizing solution fields with linear basis expansions (such as Chebyshev polynomials, Fourier series, or Extreme Learning Machines) and applying Bellman-Kalaba quasilinearization, the nonlinear PDEs are reduced to a sequence of weighted linear least-squares problems that are solved directly via QR factorization. This framework accelerates convergence, bypasses non-convex optimization landscapes, eliminates expensive neural network training epochs, and provides extreme numerical accuracy.

---

## Citation

Manuscript under review at the *Journal of Computational Physics (JCP)*. The BibTeX citation entry will be updated here upon acceptance.

---

## Problems Investigated

This repository contains the implementations of the benchmark problems discussed in the manuscript:

1.  **Linear Elasticity** — 2D plane-strain displacement-field solid mechanics solver.
2.  **Kovasznay Flow** — steady 2D incompressible Navier-Stokes equations.
3.  **Bratu Equation** — a classical highly nonlinear elliptic boundary value problem modeling thermal combustion.

*Note: Other physics modules from the paper (including Burgers' equation, Buckley-Leverett flow, 3D Beltrami flow, and heterogeneous Darcy SPE10 flow) are being uploaded to this repository progressively.*

---

## Installation

### Prerequisites
- **Python Version**: `python>=3.8` (tested and verified on Python 3.10)
- **Dependencies** (Pinned for exact reproducibility):
  - `numpy==2.4.2`
  - `scipy==1.17.0`
  - `matplotlib==3.10.8`
  - `torch==2.10.0` (used solely for baseline standard PINN/NiL comparison models)

Install dependencies:
```bash
pip install -r requirements.txt
```

To install the core framework as a local package in editable mode:
```bash
pip install -e .
```

---

## Repository Status

This repository currently includes the fully functional **Linear Elasticity**, **Kovasznay Flow**, and **Bratu Equation** solvers. Additional physics modules discussed in the manuscript will be uploaded to the repository incrementally.

---

## Usage

To run the comparative sweeps across all four solver formulations (NiL-N, NiL-Q, LiL-N, LiL-Q) from the paper, navigate to the respective directory and execute the runner:

*   **Linear Elasticity**:
    ```bash
    cd LinearElasticity
    python run_elasticity_experiments.py
    ```
*   **Kovasznay Flow**:
    ```bash
    cd Kovasznay
    python run_kovasznay_experiments.py
    ```
*   **Bratu Equation**:
    ```bash
    cd Bratu
    python run_bratu_experiments.py
    ```
