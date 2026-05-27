# Linear-in-Learnables Quasilinearized Solvers for Nonlinear PDEs (LiL-Q)

This repository contains the official, publication-grade implementation of the **Linear-in-Learnables Quasilinearized Solver (LiL-Q)** framework. 

LiL-Q is an accelerated, direct solver architecture for solving non-convex, highly nonlinear boundary value problems and Navier-Stokes formulations. By parameterizing solution fields with linear basis expansions (such as Chebyshev polynomials, Fourier series, or Extreme Learning Machines) and applying Bellman-Kalaba quasilinearization, the nonlinear PDEs are reduced to a sequence of weighted linear least-squares problems that are solved directly via QR factorization. This framework accelerates convergence, bypasses non-convex optimization landscapes, eliminates expensive neural network training epochs, and provides extreme numerical accuracy.

---

## Citation

Manuscript under review at the *Journal of Computational Physics (JCP)*. The BibTeX citation entry will be updated here upon acceptance.

---

## Supported Problems

This repository contains the complete implementation of the benchmarks and applications discussed in the paper:

1.  **Bratu Equation** — 2D nonlinear elliptic PDE (thermal ignition benchmark).
2.  **Burgers Equation** — 1D+time advection-diffusion equation.
3.  **Buckley-Leverett Flow** — 1D+time highly non-convex conservation law (with optional gravity).
4.  **Kovasznay Flow** — steady 2D incompressible Navier-Stokes equations.
5.  **3D Beltrami Flow** — unsteady 3D incompressible Navier-Stokes equations.
6.  **Linear Elasticity** — 2D plane-strain displacement-field solid mechanics solver.
7.  *Heterogeneous SPE10 Darcy Flow* (Planned Future Release)

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

This repository currently includes the fully functional **Bratu, Burgers, Buckley-Leverett (with/without gravity), Kovasznay flow, 3D Beltrami flow, and Linear Elasticity** solvers. 

The remaining high-contrast heterogeneous **SPE10/Darcy flow** experiment from the paper will be added directly to this repository over the coming weeks.

---

## Usage

### 1. High-Speed LiL-Q Direct Solver Examples (Recommended)
While the benchmark comparison experiments compare multiple formulations (including neural network baseline models which take considerable time to train), the top-level **`examples/`** folder runs **strictly the pure LiL-Q direct solver**. 

These scripts run directly on CPU and solve in seconds—sometimes milliseconds—reproducing the extreme accuracy and speeds of the quasilinear QR method. By default, they are configured to run at the **largest basis sizes ($N$) used in the paper**, but you can specify custom basis sizes dynamically using the `--N` flag:

*   **Bratu Equation (Chebyshev basis, Default: $N=15$, $P=225$)**:
    ```bash
    python examples/run_bratu_lilq.py
    # Test a smaller, faster basis dimension:
    python examples/run_bratu_lilq.py --N 10
    ```
*   **Burgers Equation (Chebyshev basis, Default: $N=25$, $P=625$)**:
    ```bash
    python examples/run_burgers_lilq.py
    python examples/run_burgers_lilq.py --N 15
    ```
*   **Kovasznay Flow (Chebyshev basis, Default: $N=25$, $P=1875$ total)**:
    ```bash
    python examples/run_kovasznay_lilq.py
    python examples/run_kovasznay_lilq.py --N 15
    ```
*   **Linear Elasticity (Mixed Fourier/Chebyshev basis, Default: $N=15$, $P=450$ total)**:
    *Elasticity is a linear PDE, meaning LiL-Q converges perfectly in a single direct QR step (~0.3s).*
    ```bash
    python examples/run_elasticity_lilq.py
    python examples/run_elasticity_lilq.py --N 10
    ```

### 2. Benchmark Comparison Sweeps
To run the full comparative sweeps across all four solver formulations (NiL-N, NiL-Q, LiL-N, LiL-Q) from the paper, navigate to the respective directory and execute the runner:

*   **Bratu Equation**:
    ```bash
    cd Bratu
    python run_bratu_experiments.py
    ```
*   **Burgers Equation**:
    ```bash
    cd Burgers
    python run_burgers_experiments.py
    ```
*   **Buckley-Leverett**:
    ```bash
    cd BL
    python run_bl_experiments.py          # Standard Buckley-Leverett
    python run_bl_gravity_experiments.py  # Buckley-Leverett with gravity
    ```
*   **Kovasznay Flow**:
    ```bash
    cd Kovasznay
    python run_kovasznay_experiments.py
    ```
*   **3D Beltrami Flow**:
    ```bash
    cd Beltrami_3D
    python run_beltrami_experiments.py
    ```
*   **Linear Elasticity**:
    ```bash
    cd LinearElasticity
    python run_elasticity_experiments.py
    ```
