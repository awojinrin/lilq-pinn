# Replication and Reproduction Guide

This guide maps each manuscript table and figure to its corresponding command-line interface (CLI) execution command in the repository.

---

## 1. Quick Validation (Dry Run)
Before running full-scale sweeps, you can run a complete end-to-end dry run to verify that all solvers compile and run successfully on CPU in under 2 minutes:
```bash
python experiments/run_all_dry.py
```

---

## 2. Parameter Mappings: Basis Modes ($N$) to Total DOFs ($P$)
The experiment scripts accept a basis mode parameter `--N` (modes per dimension), whereas the manuscript tables and figures report the total degrees of freedom ($P$). The translations are:

* **Bratu Equation** (2D scalar):
  $P = N_x \cdot N_y = N^2$
  * $N = 5 \rightarrow P = 25$
  * $N = 10 \rightarrow P = 100$
  * $N = 15 \rightarrow P = 225$

* **Burgers Equation** (1D+time scalar):
  $P = N_x \cdot N_t = N^2$
  * $N = 5 \rightarrow P = 25$
  * $N = 10 \rightarrow P = 100$
  * $N = 15 \rightarrow P = 225$
  * $N = 20 \rightarrow P = 400$
  * $N = 25 \rightarrow P = 625$

* **Buckley-Leverett Equation** (1D+time scalar):
  $P = N_x \cdot N_t = N^2$
  * $N = 8 \rightarrow P = 64$
  * $N = 16 \rightarrow P = 256$
  * $N = 24 \rightarrow P = 576$
  * $N = 32 \rightarrow P = 1024$

* **Kovasznay Flow** (2D steady Navier-Stokes, 3 variables: $u, v, p$):
  $P = 3 \cdot N_x \cdot N_y = 3 N^2$
  * $N = 5 \rightarrow P = 75$
  * $N = 10 \rightarrow P = 300$
  * $N = 15 \rightarrow P = 675$
  * $N = 20 \rightarrow P = 1200$
  * $N = 25 \rightarrow P = 1875$

* **Linear Elasticity** (2D plane-strain, 2 displacement variables: $u, v$):
  $P = 2 \cdot N_x \cdot N_y = 2 N^2$
  * $N = 5 \rightarrow P = 50$
  * $N = 10 \rightarrow P = 200$
  * $N = 15 \rightarrow P = 450$
  * $N = 20 \rightarrow P = 800$
  * $N = 25 \rightarrow P = 1250$

* **Beltrami Flow** (3D+time unsteady Navier-Stokes, 4 variables: $u, v, w, p$):
  For custom Chebyshev-Fourier tensor configurations, see script mapping:
  * $N_x=N_y=N_z=3, N_t=4 \rightarrow P = 243$
  * $N_x=N_y=N_z=4, N_t=4 \rightarrow P = 576$
  * $N_x=N_y=N_z=5, N_t=4 \rightarrow P = 1125$
  * $N_x=N_y=N_z=6, N_t=4 \rightarrow P = 1944$
  * $N_x=N_y=N_z=8, N_t=4 \rightarrow P = 4608$

---

## 3. Experiment Runner Commands

### Table 1 & Figure (Bratu Convergence Sweep)
Runs NiL-N, NiL-Q, LiL-N, and LiL-Q solvers across basis sizes:
```bash
python experiments/run_bratu.py --basis fourier --N 5 10 15
```
*Expected CPU Runtime*: ~2–5 minutes.

### Table 2 & Figure (Burgers Convergence Sweep)
Runs all four solvers across basis sizes:
```bash
python experiments/run_burgers.py --N 5 10 15 20 25
```
*Expected CPU Runtime*: ~5–15 minutes.

### Table 4 & Table 5 (Buckley-Leverett Sweeps)
* **Viscous Case (No Gravity)**:
  ```bash
  python experiments/run_bl.py --N 8 16 24 32
  ```
* **Gravity Case**:
  ```bash
  python experiments/run_bl.py --gravity --N 8 16 24 32
  ```
*Expected CPU Runtime*: ~10–25 minutes per case.

### Table 8 (Kovasznay Navier-Stokes)
Runs the steady 2D Navier-Stokes solver:
```bash
python experiments/run_kovasznay.py --N 5 10 15 20 25
```
*Expected CPU Runtime*: ~3–5 minutes.

### Table 10 (Beltrami 3D Flow)
Runs the unsteady 3D Navier-Stokes solver:
```bash
python experiments/run_beltrami.py --N 3 4 5 6 8
```
*Expected CPU Runtime*: ~15–30 minutes.

### Table 11 (Linear Elasticity)
Runs plane-strain linear elasticity solver:
```bash
python experiments/run_elasticity.py --N 5 10 15 20 25
```
*Expected CPU Runtime*: ~2–3 minutes.

### Table 13 (SPE10 Darcy Flow)
Runs Darcy flow solver for heterogeneous porous media (layers S1, S2, S3, and the full SPE10 field):
```bash
python experiments/run_darcy.py --fields S1 S2 S3 SPE10 --order 32
```
*Expected CPU Runtime*: ~8–12 minutes.

### Figure (Burgers Basis Comparison Study)
Runs Burgers equation comparing Chebyshev, Fourier, Legendre, Jacobi, and Hermite bases:
```bash
python experiments/run_burgers_basis_comparison.py
```
*Expected CPU Runtime*: ~5–10 minutes.
