# Running the studies

This guide covers:
- the commands that run each benchmark;
- the instrumented drivers behind the released result files;
- the cluster job scripts.

[RELEASE.md](RELEASE.md) says where the released results are and how the published tables and figures are regenerated from them.

## 1. Validation

```bash
python -m pytest -q                  # unit and regression tests
python experiments/run_all_dry.py    # every problem at minimal size, on the CPU, in under 2 minutes
```

## 2. Sizes

The runners take the number of modes per direction. The number of coefficients P follows from it:

| Problem | Unknown fields | P | Sizes used |
|---|---|---|---|
| Bratu | u | N² | N = 5, 10, 15 (P = 25, 100, 225) |
| Burgers | u | N² | N = 5, 10, 15, 20, 25 (P = 25 to 625) |
| Buckley–Leverett | S | N² | N = 8, 16, 24, 32 (P = 64 to 1,024) |
| Kovasznay flow | u, v, p | 3N² | N = 5, 10, 15, 20, 25 (P = 75 to 1,875) |
| Linear elasticity | u_x, u_y | 2N² | N = 5, 10, 15, 20, 25 (P = 50 to 1,250) |
| Beltrami flow | u, v, w, p | 3N_vel⁴ + N_p⁴ | N_vel = 6, N_p = 8 (P = 7,984) |
| Darcy flow | h, u, v | about 3N² | order 32 (P = 3,169) |

## 3. Benchmark runners

Each runner writes its results under `results/`. The runtimes are approximate, for one laptop CPU (Intel Core Ultra 9 275HX).

| Problem | Command | Runtime |
|---|---|---|
| Bratu, all four formulations | `python experiments/run_bratu.py --basis fourier --N 5 10 15` | 15–20 min |
| Burgers, all four formulations | `python experiments/run_burgers.py --N 5 10 15 20 25` | about 1 h |
| Buckley–Leverett, viscous | `python experiments/run_bl.py --N 8 16 24 32` | 40–60 min |
| Buckley–Leverett, gravity-influenced | `python experiments/run_bl.py --gravity --N 8 16 24 32` | 40–60 min |
| Kovasznay flow | `python experiments/run_kovasznay.py --N 5 10 15 20 25` | under 1 min |
| Linear elasticity | `python experiments/run_elasticity.py --N 5 10 15 20 25` | seconds |
| Beltrami flow | `python experiments/run_beltrami.py --N_vel 6 --N_p 8` | 5–10 min |
| Darcy flow, LiL-Q and finite volumes only | `python experiments/run_darcy.py --fields S1 S2 S3 SPE10 --order 32 --skip-pinn` | about 2 min |
| Darcy flow, with the network baseline | `python experiments/run_darcy.py --fields S1 S2 S3 SPE10 --order 32` | about 75 min on a GPU |
| Burgers, basis comparison | `python experiments/run_burgers_basis_comparison.py` | about 1 min |

**Where the runtime goes.**
- **The four-formulation benchmarks:** most of the time is network training. LiL-Q itself takes seconds.
- **The Darcy network baseline:** it trains for 150,000 iterations per field.

## 4. Instrumented runs

The released results were produced by these drivers. Each writes:
- `iterations.csv`: every iterate's residuals, monitors, rank, condition number and errors;
- `run.json`: settings, commit and checks;
- `hardware.json`;
- the saved model.

| Driver | What it runs |
|---|---|
| `experiments/component_b.py` | The logged LiL-Q runs of every benchmark: the reported pass and the pass with termination disabled to 60 iterations (`--passes paper kmax`) |
| `experiments/four_method_tables.py` | The four-formulation comparison over seeds, with the stall controls (`--controls`) |
| `experiments/clean_timing.py` | Timings with the diagnostics off: one warm-up, then three timed runs, reported as their median |
| `experiments/component_a.py`, `experiments/kovasznay_comparison.py` | The calibrated network baselines on the Kovasznay flow and their comparison with LiL-Q |
| `experiments/component_c.py` | The oversampling and collocation-set study |
| `experiments/stopping_rule_table.py` | The termination rule applied to logged histories, with its variants |
| `experiments/p2_*.py`, `experiments/p3_*.py` | The supplementary studies: classical baselines, Levenberg–Marquardt networks, scaling, bases, refinement, certified grids, a posteriori constants |
| `experiments/figure_data.py` | The solution fields of the field figures, from the saved models |

Every driver has a `--help`. Most have a `--smoke` or `--quick` mode for a short local check.

## 5. Cluster runs

`scripts/cluster/` holds the SLURM job scripts and the submission scripts used on the Texas A&M HPRC Grace and FASTER clusters. Its README explains the environment and the submission order.

To build the upload bundle, locked to a commit:

```bash
python scripts/make_hprc_bundle.py
```

## 6. Reproducibility notes

- **Determinism:** LiL-Q runs are deterministic for a given machine, BLAS library and thread count.
- **Thread count:** `lilq/blas_threads.py` sets it before NumPy is imported.
- **Records:** `hardware.json` records the CPU, the BLAS backend and its threading.
- **Network baselines:** they use seeds 0, 1 and 2. Their collocation sets are fixed by a separate seed (42).
- **Across machines and library versions:** results agree to round-off, except in runs whose iteration diverges or whose systems are numerically rank deficient. There the individual iterates can differ.
