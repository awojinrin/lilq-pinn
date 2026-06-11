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
  Each velocity component uses a 4D tensor product basis with $N_\text{vel}$ modes per dimension;
  pressure uses $N_p$ modes per dimension. Total DOFs: $P = 3 N_\text{vel}^4 + N_p^4$.
  * $N_\text{vel}=6, N_p=8 \rightarrow P = 3 \times 1296 + 4096 = 7{,}984$ (manuscript configuration)

---

## 3. Experiment Runner Commands

> **Hardware note.** All runtimes below were recorded on an Intel Core Ultra 9 275HX CPU.
> The dominant cost in each 4-method benchmark is NiL-N (L-BFGS) training; LiL-Q itself
> completes in seconds. For LiL-Q-only problems (Kovasznay, Beltrami, Elasticity), the
> runtime is the LiL-Q solver time alone.

### Table 1 & Figure (Bratu Convergence Sweep)
Runs NiL-N, NiL-Q, LiL-N, and LiL-Q solvers across basis sizes:
```bash
python experiments/run_bratu.py --basis fourier --N 5 10 15
```
*Expected CPU Runtime*: ~15–20 minutes. NiL-N and NiL-Q each take ~5–8 min across all 3 sizes (dominated by $P=225$); LiL-N ~3 min; LiL-Q < 1 s total.

### Table 2 & Figure (Burgers Convergence Sweep)
Runs all four solvers across basis sizes:
```bash
python experiments/run_burgers.py --N 5 10 15 20 25
```
*Expected CPU Runtime*: ~55–60 minutes. NiL-N and NiL-Q each take ~22 min total across all 5 sizes (dominated by $N \geq 15$); LiL-N ~13 min; LiL-Q < 2 s total.

### Table 4 & Table 5 (Buckley-Leverett Sweeps)
* **Viscous Case (No Gravity)**:
  ```bash
  python experiments/run_bl.py --N 8 16 24 32
  ```
* **Gravity Case**:
  ```bash
  python experiments/run_bl.py --gravity --N 8 16 24 32
  ```
*Expected CPU Runtime*: ~40–60 minutes per case. At $N=32$ alone, NiL-N takes ~16 min, NiL-Q ~15 min, LiL-N ~5 min, and LiL-Q ~3.5 s. Smaller $N$ values add another ~5–10 minutes total.

### Table 8 (Kovasznay Navier-Stokes)
Runs the LiL-Q steady 2D Navier-Stokes solver:
```bash
python experiments/run_kovasznay.py --N 5 10 15 20 25
```
*Expected CPU Runtime*: < 1 minute (LiL-Q only; $N=25$ takes ~13 s, total ~20 s for all sizes).

### Table 10 (Beltrami 3D Flow)
Runs the LiL-Q unsteady 3D Navier-Stokes solver with the manuscript configuration ($N_\text{vel}=6$, $N_p=8$, $P = 7{,}984$):
```bash
python experiments/run_beltrami.py --N_vel 6 --N_p 8
```
*Expected CPU Runtime*: ~9 minutes for the single manuscript configuration.

### Table 11 (Linear Elasticity)
Runs the LiL-Q plane-strain linear elasticity solver:
```bash
python experiments/run_elasticity.py --N 5 10 15 20 25
```
*Expected CPU Runtime*: < 1 minute (LiL-Q only; total ~3 s for all sizes).

### Table 13 (SPE10 Darcy Flow)
Runs FVM reference + LiL-Q solver for heterogeneous porous media. Use `--skip-pinn` to omit the NiL-N baseline (which trains for 150 K epochs per field):
```bash
# LiL-Q + FVM only (recommended for quick reproduction)
python experiments/run_darcy.py --fields S1 S2 S3 SPE10 --order 32 --skip-pinn

# Full run including NiL-N PINN baseline
python experiments/run_darcy.py --fields S1 S2 S3 SPE10 --order 32
```
*Expected Runtime*: LiL-Q + FVM only: ~2 minutes for all 4 fields (LiL-Q ~24 s per field, FVM ~0.1 s per field). With NiL-N (default): ~75 minutes on GPU (NiL-N takes 960–1,580 s per field); significantly longer on CPU.

### Figure (Burgers Basis Comparison Study)
Runs Burgers equation comparing Chebyshev, Fourier, and ELM bases across basis sizes:
```bash
python experiments/run_burgers_basis_comparison.py
```
*Expected CPU Runtime*: < 1 minute (LiL-Q only across all basis types).
