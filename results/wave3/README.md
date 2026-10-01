# Wave 3 results (Computational Package 1: tasks B8 and B9)

Wave 3 ran on TAMU Grace on 1 October 2026 at commit `905e58c`, the commit
this branch starts from; the branch's own commit adds only `results/wave3/`.
`COMMIT` holds the commit and the source-tree hash every job checked.

The folder holds every file the wave wrote, including the trained models
(164 `*.pt`, 6.7 MB), and the jobs' Slurm logs (`slurm_logs/`). It contains
all 196 files of `wave3_report.tar.gz`, byte for byte. `analysis/` holds the
B8 and B9 tables merged across jobs. The package assembly (`90_finalize`)
normally merges them; it was cancelled for this wave and runs after wave 4,
which the advisor added on 1 October.

Waves 1 and 2 are on branches `wave1-results` and `wave2-results`.

## 1. Jobs and SUs

Every task completed with exit code 0 (`sacct.txt`; the report job lists
itself as RUNNING because it ran `sacct` before finishing, at 2:02). All B8
and B9 tasks ran on the shared GPU allocation, 8 cores and one A100 (40 GB)
each, and were charged 80 SU/h. `myproject` shows **1,019 SU** charged for
wave 3: 9,878.76 used, against 8,860.13 before. The computed total is
1,018.6.

| Job | What | Allocation | Elapsed | SU/h | SU |
|---|---|---|---|---|---|
| 19925482_0 | 40 B9 NiL Darcy: S1, seed 0 | 8 cores + 1 A100 | 00:58:11 | 80 | 77.6 |
| 19925482_1 | 40 B9: S1, seed 1 | 8 cores + 1 A100 | 00:50:14 | 80 | 67.0 |
| 19925482_2 | 40 B9: S1, seed 2 | 8 cores + 1 A100 | 00:51:34 | 80 | 68.8 |
| 19925482_3 | 40 B9: S2, seed 0 | 8 cores + 1 A100 | 00:49:41 | 80 | 66.2 |
| 19925482_4 | 40 B9: S2, seed 1 | 8 cores + 1 A100 | 00:52:46 | 80 | 70.4 |
| 19925482_5 | 40 B9: S2, seed 2 | 8 cores + 1 A100 | 00:48:57 | 80 | 65.3 |
| 19925482_6 | 40 B9: S3, seed 0 | 8 cores + 1 A100 | 01:01:03 | 80 | 81.4 |
| 19925482_7 | 40 B9: S3, seed 1 | 8 cores + 1 A100 | 00:48:57 | 80 | 65.3 |
| 19925482_8 | 40 B9: S3, seed 2 | 8 cores + 1 A100 | 00:49:45 | 80 | 66.3 |
| 19925482_9 | 40 B9: SPE10, seed 0 | 8 cores + 1 A100 | 00:46:39 | 80 | 62.2 |
| 19925482_10 | 40 B9: SPE10, seed 1 | 8 cores + 1 A100 | 00:49:39 | 80 | 66.2 |
| 19925482_11 | 40 B9: SPE10, seed 2 | 8 cores + 1 A100 | 00:54:08 | 80 | 72.2 |
| 19925495_0 | 41 B8: viscous BL, zero guess | 8 cores + 1 A100 | 00:29:19 | 80 | 39.1 |
| 19925495_1 | 41 B8: viscous BL, initial-condition guess | 8 cores + 1 A100 | 00:37:18 | 80 | 49.7 |
| 19925495_2 | 41 B8: gravity BL, zero guess | 8 cores + 1 A100 | 00:43:16 | 80 | 57.7 |
| 19925495_3 | 41 B8: gravity BL, initial-condition guess | 8 cores + 1 A100 | 00:31:55 | 80 | 42.6 |
| 19925497 | 91 wave report | 24 cores | 00:02:02 | 24 | 0.8 |
| | **Total** | | | | **1,018.6** |

## 2. B9: Darcy pressures against the finite-volume solution

`B_instrumentation/darcy_fv/<field>_s<seed>/darcy_fv_comparison.csv`, merged
in `analysis/b9_darcy_fv_merged.csv` (28 rows: LiL once per field, NiL
4 fields x 3 seeds x float64 and float32). NiL is the manuscript's network,
2 x 32 (3,555 parameters), trained 150,000 epochs.

Median delta_FV over the three seeds, with the range:

| Field | LiL | NiL float64 | NiL float32 | float32 / float64 |
|---|---|---|---|---|
| S1 | 1.37e-4 | 2.26e-2 (2.14-2.36e-2) | 1.93e-2 (1.92-2.17e-2) | 0.85 |
| S2 | 2.33e-4 | 4.68e-2 (4.09-5.38e-2) | 5.21e-2 (5.12-6.33e-2) | 1.11 |
| S3 | 6.74e-4 | 9.78e-2 (5.12e-2-1.47e-1) | 1.03e-1 (5.46e-2-1.08e-1) | 1.05 |
| SPE10 | 3.42e-2 | 8.99e-2 (8.74e-2-1.07e-1) | 9.05e-2 (8.91e-2-1.09e-1) | 1.01 |

- LiL's delta_FV is about 100x smaller than NiL's on the synthetic fields S1-S3, and 2.6x smaller on SPE10.
- The float32 and float64 medians agree within the 20% rule (Addendum v2.2 reply, Section 3) in every field.
- NiL training times are 1,340-1,710 s per run (`time_s`).
- LiL's `time_s` here (27-38 s) is the whole comparison call, including the finite-volume reference. The clean LiL Darcy times come from wave 4's clean-timing job.

**Records for the B9 times (the advisor's follow-up of 1 October, item 4):**
- **Allocation:** every NiL run ran on the shared GPU allocation, 8 cores and one A100-PCIE-40GB, not exclusive (`hardware.json` per job, `scheduler`).
- **Precision:** the `dtype` column.
- **Checkpoint resumes:** none of the 24 NiL networks resumed from a checkpoint; `resumed_at` is empty in every `models/NiL_*/network.pt`.

## 3. B8: initial guesses for Buckley-Leverett

`B_instrumentation/b8_jobs/<case>_<guess>/b8_initial_guess.csv`, merged in
`analysis/b8_initial_guess_merged.csv`. 128 rows: 2 cases x 2 guesses x 4
sizes x (NiL-N and NiL-Q at 3 seeds, LiL-N, LiL-Q). There are no failures.

Stopping reasons:

| Case, guess | LiL-Q | LiL-N | NiL-N | NiL-Q |
|---|---|---|---|---|
| viscous, zero | target 4 | target 2, cap 1, stall 1 | target 9, stall 3 | target 9, stall 3 |
| viscous, IC | target 4 | target 2, cap 2 | target 10, stall 2 | target 9, stall 3 |
| gravity, zero | target 3, **cap 1** | target 3, stall 1 | target 10, cap 2 | target 8, cap 4 |
| gravity, IC | target 2, **cap 2** | target 3, stall 1 | target 12 | target 11, cap 1 |

**Three LiL-Q rows ended at K_max = 20 without their target**, all gravity BL:

| Guess | P | Final loss | Target |
|---|---|---|---|
| IC | 256 | 0.099 | 0.075 |
| IC | 576 | 0.125 | 0.045 |
| zero | 64 | 0.951 | 0.24 |

Per the advisor's follow-up of 1 October (item 1), these are rerun on the
CPU in wave 4 with K_max = 60, and reported beside these rows with a `K_max`
column.
