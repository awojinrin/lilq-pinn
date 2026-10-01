# Wave 2 results (Computational Package 1; the advisor's reply to wave 1)

Wave 2 ran on TAMU Grace on 29–30 September 2026 at commit `905e58c`. `COMMIT` holds that commit and the source-tree hash every job checked. This branch starts from `bc241bf`, which adds only job 12's script (`scripts/cluster/12_timing_reruns_beltrami_darcy.slurm`) to `905e58c`. Job 12 was submitted by hand from a copy of that file, so the code it ran is `905e58c`'s. The branch's own commit adds `results/wave2/` and nothing else.

The folder holds every file the wave wrote except the trained models (`*.pt`, 294 files), which are kept on Grace and offline. It contains all 490 files of `wave2_report.tar.gz`, byte for byte, plus each run's own logs and the per-iteration histories. `analysis/` holds a post-hoc computation made on the laptop after the wave (Section 2).

Wave 1's results are on branch `wave1-results`.

## 1. Jobs and SUs

All jobs completed with exit code 0 (`sacct.txt`; the report job lists itself as RUNNING because it ran `sacct` before finishing, at 3:38). Each job's output is in `slurm_logs/`. One more preflight, 19905468, was submitted by a timed-out `sbatch` and cancelled after under a minute; it charged 8 SU.

`myproject` shows **8,498 SU** charged for wave 2: 8,860.13 used, against 361.78 before it.

**Timed GPU jobs were charged 192 SU/h, not 120.** Each of them held one A100 (`gres/gpu=1`, 48 cores, 360G, without `--exclusive`; `sacct.txt`), as the advisor's item 2.3 asked. The other jobs were charged at the rates expected:

| Charging rate | Applies to | Wave 2 total |
|---|---|---|
| 192 SU/h | 40.4 timed GPU hours | 8,402 SU |
| 120 SU/h, as intended | the same 40.4 hours | 5,492 SU |

The 8,402 SU total is within about 1% of the 8,498 charged. The likely reason, not yet confirmed with HPRC, is that Grace charges a GPU-node job by the largest fraction of the node it holds: all 48 cores counts as the whole node, both GPUs included. Shared GPU jobs (8 cores, 1 A100) are charged 80 SU/h, as expected; wave 3's GPU jobs are of that class.

SU column computed as rate × elapsed hours; rates 80 shared GPU, 192 timed GPU, 48 timed CPU, 24 CPU:

| Job | What | Allocation | Elapsed | SU/h | SU |
|---|---|---|---|---|---|
| 19905478 | 00 preflight | 8 cores + 1 A100 | 00:10:52 | 80 | 14.5 |
| 19905479 | 29 A1 gate (search, A1, A2, F2 Jacobian) | 8 cores + 1 A100 | 01:08:41 | 80 | 91.6 |
| 19905481_0 | 30 A screening, F1 | 48 cores + 1 A100 | 04:06:17 | 192 | 788.1 |
| 19905481_1 | 30 A screening, F2 | 48 cores + 1 A100 | 04:08:57 | 192 | 796.6 |
| 19905482_0 | 31 A full, F1 | 48 cores + 1 A100 | 15:24:21 | 192 | 2,957.9 |
| 19905482_1 | 31 A full, F2 | 48 cores + 1 A100 | 11:31:07 | 192 | 2,211.6 |
| 19905483_0 | 32 A CPU, F1 | 48 cores | 05:11:08 | 48 | 248.9 |
| 19905483_1 | 32 A CPU, F2 | 48 cores | 01:38:33 | 48 | 78.8 |
| 19905484 | 33 A float32 | 48 cores + 1 A100 | 01:08:41 | 192 | 219.8 |
| 19905486_0 | 20 B4 GPU: Bratu (wave 1's controls) | 48 cores + 1 A100 | 00:32:14 | 192 | 103.1 |
| 19905486_1 | 20 B4 GPU: Burgers | 48 cores + 1 A100 | 01:35:41 | 192 | 306.2 |
| 19905486_2 | 20 B4 GPU: viscous BL | 48 cores + 1 A100 | 00:59:35 | 192 | 190.7 |
| 19905486_3 | 20 B4 GPU: gravity BL | 48 cores + 1 A100 | 00:53:38 | 192 | 171.6 |
| 19905487_0 | 21 B4 CPU: Bratu (wave 1's controls) | 48 cores | 00:14:09 | 48 | 11.3 |
| 19905487_1 | 21 B4 CPU: Burgers | 48 cores | 00:53:18 | 48 | 42.6 |
| 19905487_2 | 21 B4 CPU: viscous BL | 48 cores | 01:33:42 | 48 | 75.0 |
| 19905487_3 | 21 B4 CPU: gravity BL | 48 cores | 01:15:30 | 48 | 60.4 |
| 19905488 | 11a timing reruns, CPU | 48 cores | 00:04:42 | 48 | 3.8 |
| 19905489 | 11b timing reruns, GPU, and check B3 | 48 cores + 1 A100 | 00:04:17 | 192 | 13.7 |
| 19905751 | 12 timing reruns, Beltrami and Darcy | 48 cores | 00:17:46 | 48 | 14.2 |
| 19905490 | 91 wave report | 24 cores | 00:03:38 | 24 | 1.5 |
| | **Total** | | | | **8,401.9** |

## 2. Component A: the selection picked the configuration that generalizes worst, in both families

Screening, full and CPU runs, the float32 run and the checks all completed (`A_calibration/`). Each family's top three and representative were chosen by final training loss, as specified (F1: the unweighted loss). In both families that rule picked a configuration that drives the training loss down without learning the solution:

| Family | Representative chosen | Median training loss | Median ε_u (5 seeds) | The other two finalists |
|---|---|---|---|---|
| F1 | F1_04 | 4.8e-8 | 7.4e-2 (pressure error ~100%) | F1_15: 8.7e-5; F1_18: 4.2e-5 |
| F2 | F2_16 | 8.4e-31 | 7.6e-3 | F2_03: 2.8e-7; F2_17: 1.7e-8 |

**F2.** The six configurations with the lowest training loss all have fewer residual rows than parameters (6,001 rows; 7,395–16,835 parameters). They can fit the collocation points almost exactly. F2_16 reaches a loss of about 1e-30, after which no Levenberg–Marquardt step lowers it and μ overflows: `mu_overflow` after 37–56 s in four of its five full runs, and in its screening run. The configurations with more rows than parameters (8,000 interior points, 24,001 rows) generalize best; F2_05 (4,291 parameters) reaches ε_u = 2.3e-10 in screening.

**F1.** F1_04 combines σ_FF = 5 with 2,000 interior points. F1_09, the only other σ_FF = 5 configuration with 2,000 points, also generalizes badly. The hard boundary conditions are not the cause: hard-BC configurations with σ_FF ≤ 2 are accurate, e.g. F1_10 and F1_12 at about 9e-5.

**Validation residual** (`analysis/validation_residuals.csv`, `analysis/validation_residual.py`). For every Component A model (89 runs), F1's unweighted loss was evaluated on points no run trained on: the momentum and continuity mean squares on 20,000 fresh uniform interior points, plus the boundary mean square on 400 points per face for soft-BC F1 runs. No exact solution is used except the boundary data. Spearman rank correlation with the test error ε_u across the 24 screening runs of each family:

| Criterion | F1 | F2 |
|---|---|---|
| Final training loss (the rule used) | 0.70 | 0.11 |
| Validation residual | 0.93 | 0.94 |

Between the two criteria, among the finalists:
- **F1_04:** validation residual 3.8, against a training loss of 5e-8.
- **F2_16:** 2.9e-2, against 8e-31.
- **Re-picked among each family's three finalists by median validation residual:** F1_18 (median ε_u 4.2e-5) and F2_17 (1.7e-8).
- **Re-ranked from all 24 screening runs by validation residual:** the top three would be F1_19, F1_01, F1_08 (screening ε_u 4.0e-5 to 1.0e-4) and F2_10, F2_05, F2_14 (screening ε_u 2.3e-10 to 1.6e-9).

The full stage's results are kept as they are. The CPU reruns and the float32 run were made on F1_04 and F2_16.

`A_calibration/run_endings.csv` lists how every run ended. The 12 runs that ended before their budget are the two A2 checks (`max_steps`, by design) and ten F2_16 runs (`mu_overflow`).

**Checks** (gate job): A1 passed (ε_u = 5.8e-5), A2 was bit-identical, and the F2 Jacobian check passed (3.8e-9). The F2 search list (`A_calibration/search/F2_configs.json`) has 24 distinct configurations: 32 draws were redrawn for the n_θ cap and 11 for repeats.

## 3. Four-method tables and stall controls (B4; the reply's item 2.5)

`B_instrumentation/four_method_tables.csv` has 112 rows, for Burgers and both BL problems on GPU and CPU: 59 `target`, 40 `optimizer_stall`, 13 `iteration_cap`, and no failures. Wave 1's 28 Bratu rows are on branch `wave1-results`.

`B_instrumentation/four_method_controls.csv` has **54 controls**: the 40 stalled runs of wave 2 and wave 1's 14 stalled Bratu runs. Each ran on its original's device, with tolerances 0 and F1's restart-once rule.

- All 54 have `same_start = True`: iteration 0's loss equals the original's bit for bit, on GPU and CPU, including wave 1's runs (another commit and node).
- 48 retrace the original's logged history exactly up to its stall. The other 6 depart from it at iterations 5,560–8,140. With `tolerance_change = 0`, the line search's own bracketing test also changes, which is the likely cause.
- No control needed the restart (`lbfgs_restarts = 0` everywhere): with tolerances 0, every step lowered the loss.
- 46 end on `iteration_cap` and 8 reach the target.

| Benchmark | Device | Controls | Reached target | Median control/original final loss | Range |
|---|---|---|---|---|---|
| Bratu | GPU | 9 | 0 | 0.72 | 0.37–0.95 |
| Bratu | CPU | 5 | 0 | 0.79 | 0.69–0.90 |
| Burgers | GPU | 20 | 4 | 0.38 | 0.16–0.94 |
| Burgers | CPU | 7 | 0 | 0.30 | 0.068–0.50 |
| Viscous BL | GPU | 5 | 2 | 0.81 | 0.53–1.0 |
| Viscous BL | CPU | 5 | 2 | 0.63 | 0.34–0.73 |
| Gravity BL | GPU | 1 | 0 | 1.0 | — |
| Gravity BL | CPU | 2 | 0 | 0.80 | 0.61–1.0 |

## 4. Timing reruns (the reply's item 2.2 and its reply of 30 September)

Every paper pass of Bratu, Burgers, both BL problems, elasticity and Kovasznay (CPU and GPU), plus Beltrami and Darcy, was rerun with one complete untimed warm-up run of the same configuration on the same device first. All 36 runs are `ok` (`B_instrumentation/runs_index.csv`). Check B3 was rerun as well.

Selected reported times in seconds: wave 1, wave 2, and wave 2's untimed warm-up run, i.e. the cold time:

| Run | Quantity | Wave 1 | Wave 2 | Wave 2 warm-up (cold) |
|---|---|---|---|---|
| `kovasznay_P75_cuda_paper` | solve_time_total | 0.0474 | 0.0477 | 0.170 |
| `kovasznay_P300_cuda_paper` | solve_time_total | 1.416 | 0.0606 | 1.411 |
| `kovasznay_P675_cuda_paper` | solve_time_total | 0.153 | 0.156 | 0.159 |
| `kovasznay_P1200_cuda_paper` | solve_time_total | 1.095 | 0.421 | 0.970 |
| `kovasznay_P1875_cuda_paper` | solve_time_total | 1.137 | 1.142 | 0.955 |
| `kovasznay_P1875_cpu_paper` | solve_time_total | 4.94 | 4.32 | 4.48 |
| `elasticity_P50_cpu_paper` | solve_time_qr | 0.00192 | 0.00201 | — |
| `elasticity_P1250_cpu_paper` | solve_time_qr | 0.800 | 0.830 | — |
| `beltrami_P7984_cpu_paper` | solve_time_total | 286.6 | 302.7 | 226.8 |
| `darcy_SPE10_cpu_paper` | total_time | 29.8 | 21.1 | 25.2 |

For elasticity the warm-up column is not comparable: the warm-up run records `time_lil_s`, assembly plus solve, not the QR time. Elasticity's summaries now also give `time_lil_s`, the median of five repeats of the phase of `t_cum_s`.

**Beltrami.** The untimed warm-up run, which has no per-iteration logger, took 227 s. The timed, logged run took 303 s; wave 1's took 287 s. The passive diagnostics are off the clock, but the logged run may still be slowed by them, e.g. through memory. This is reported, not changed.

**Checks B1 and B3.** B1's 15 violations are identical to wave 1's (`B_instrumentation/reproduction_check.csv`). B3 is equivalent at every size, at P = 1,875 under the amended round-off rule only, as in wave 1; its same-system timings are unchanged (`B_instrumentation/gpu_cpu_equivalence.csv`).

These runs replace wave 1's in `package1`: the assembly lists each replaced file in `WAVES.json`.
