# Computational Package 5, item 3: report

**To:** R. Younis. **From:** G. Awojinrin. **Date:** 18 November 2026.
**Refers to:** the Package 5 instructions (13 November 2026, Section 6) and your reply to the item 3 cost check (17 November 2026).

**Code:** `awojinrin/lilq-pinn`, branch `v3-dev` at 37af5a4. These are the commits after fcb9952, the code of the cost check:

| commit | what | DECISIONS.md entry (Package 5, item 3, ...) |
|---|---|---|
| 710f49c | numpy before torch; the OpenMP library check in every job; the CPU network runs again (stage p5r) | the diagnostic's result; numpy first, the OpenMP library check |
| de4df64, decae59 | the library check of Package 1's Component A baselines (stage p5k); C4 timed three times | the advisor's reply to the cost check; C4's three timings; Component A's OpenMP library |
| 3df7b7b | line endings of one file (the bundle of the rerun) | the same entry |
| 1a88feb | the C4 diagnostic (stage p5e; the bundle of the A100 set) | the rerun's result, C4 on Buckley-Leverett LM, and the C4 diagnostic |
| a61de73 | the assembly of item 3; the table leads with the time including setup and pretraining | the assembly and the table's headline |
| 37af5a4 | C1 reads the published extract of a run's log (iterations 0 to 100) when the whole log is absent | the assembly and the table's headline |
| 03bd5ab, b6ff9f5, 6d28d09, 0e3cb8d, 3edbd66, e99e5d3, 5adea2f, 99fbd07 | DECISIONS.md only: rehearsals, scans, results | as above |

**Where each run ran:** every run under its stage's source lock. The network runs on the CPU ran at 3df7b7b. The A100 runs ran at 1a88feb. The LiL runs on the CPU ran at 0f74bdf (the CPU set of the cost check, kept as you said). From each stage's commit to 37af5a4, the code its runs used changed only by additions: the LiL driver gained the option that checks the OpenMP library, C2/C4 gained a folder argument for the diagnostic, and C1 can read the published extract of a run's log.

**Results:** `package5_results/`, on the branch `package5-results` of `awojinrin/lilq-pinn`, under `results/package5/`. Item 3 is in `P5_3_cost/`.

## Summary

Item 3 is done: the CPU network runs again, then the A100 set, as you approved.
- **Cost:** 3,785.0 SU for item 3 (all stages, the diagnostics included). Package 5 in all: 3,938.5 SU against your cap of 4,400.
- **Checks** (Section 2):
  - **C1 passes** on all 90 runs that have a Package 1 run of the same configuration, with a difference of exactly 0. These are the 18 CPU runs and also the 72 A100 runs: the A100 runs reproduce Package 1's GPU losses bit for bit.
  - **C2 and C3 pass.**
  - **C4: 4 of 6 pairs pass.** Two CPU pairs are outside 5%: Bratu NiL-N (+5.3%) and Buckley-Leverett LM (+18.1%). A diagnostic of 67 SU shows that the error log puts no time on the clock. The two values are the clock's spread on the CPU node (Section 3).
- **The table** is in Section 4. It leads with the time from the starting data, as you asked, and gives the optimizer's time beside it.
  - **Bratu:** only square collocation reaches 1e-5 (0.020 s), and it is also the fastest to 1e-4 (0.009 s). LiL-Q reaches 1e-4 in 0.23 s, NiL-N with LM in 142 s on the A100.
  - **Burgers:** the LiL methods reach 1e-6. LiL-Q gets there fastest, in 0.97 s. The fastest network method reaches 1e-5 in 12.4 s, NiL-N with LM on the A100.
  - **Buckley-Leverett:** no method reaches 1e-5. LiL-Q reaches 1e-4 in 3.4 s, and NiL-N with LM in 8.0 s on the A100 and 40 s on the CPU.
- **Package 1's Component A** (Section 6): its Kovasznay CPU baselines ran with the PyTorch wheel's OpenMP library, which is the faster one for them. That comparison is therefore conservative for LiL-Q, and nothing needs rerunning.

**For your reply:**
1. **C4 on the CPU** (Section 3). The two pairs outside 5% stay reported as failed, as the rule of Section 7 says. The diagnostic shows that the same 213-iteration run varies from 29 to 35 s on that node, with or without the log. So a single pair on that node cannot resolve 5%. Is the diagnostic enough for the manuscript? The alternative is a C4 on the median of several pairs. That would be a new check of about 50 SU, and I do not think it is needed.

## 1. What ran

| Stage | What | Commit | Jobs | SU |
|---|---|---|---|---|
| p5c | the CPU set (cost check): its LiL runs, protocol runs and C3 are kept; its network runs are superseded | 0f74bdf | 7 + report | 938.9 |
| p5o | the OpenMP library diagnostic (cost check) | fcb9952 | 1 | 14.4 |
| p5r | the CPU network runs again, numpy first; C2 and C4 | 3df7b7b | 7 + report | 875.3 |
| p5k | the library of Package 1's Component A baselines | 3df7b7b | 1 | 20.9 |
| p5e | the C4 diagnostic | 1a88feb | 2 + report | 67.0 |
| p5g | the A100 set | 1a88feb | 4 + report | 1,868.5 |
| **item 3** | | | | **3,785.0** |

- **The CPU runs** each ran on a whole node of partition medium (Intel Xeon Gold 6248R, 48 threads), one at a time.
- **The A100 runs** ran on whole two-A100 nodes, two processes side by side with 24 threads and one A100 each, as approved on 16 November.
- **Every process of the table's runs loaded the GCC module's OpenMP library**, the one Package 1 used. 46 process records were checked at the assembly, and each job also checks before its first run.
- **LiL-Q on the A100** (optional in Section 6) was left out for cost, as in the plan. `trf` and square collocation are scipy and run on the CPU only.

**What is in the table:**
- **Networks:** 108 budget runs per device (NiL-N and NiL-Q with L-BFGS, NiL-N with LM; seeds 0 to 2; 600 s each).
- **LiL-N L-BFGS:** 14 budget runs per device.
- **The usual protocol:** 14 LiL-Q runs, 14 `trf` runs and 3 square collocation runs (Bratu).

## 2. Checks

| Check | Result | Numbers |
|---|---|---|
| C1: the loss over iterations 0 to 100 equals Package 1 | pass | 90 runs (18 CPU, 72 A100), every one with difference 0 (cutoff 1e-6 relative) |
| C2: the error log does not change the iterates | pass | 6 CPU pairs bit for bit; the 11 processes of the C4 diagnostic, 2 of them on the A100 |
| C3: LiL-Q on Burgers equals Package 4 | pass | 7 sizes: the same rule step; errors within 2.5e-16 relative (cutoff 1e-8) |
| C4: the clock with and without the log within 5% | 4 of 6 | Section 3 |
| Pairing (approved 16 November): a paired A100 run within 5% of the same run alone | pass | Bratu -2.1%, Burgers +1.8%, Buckley-Leverett +2.8%; each job's gate before its budget runs: +0.5%, +2.1%, +3.1% |

**C1 also shows the cause of the cost check's failure.** The diagnostic's runs with numpy first match Package 1 exactly. The runs with torch first differ by up to 0.37 relative by iteration 100.

## 3. C4 and the diagnostic

**The six pairs** (CPU, the first pair of each, the verdict fixed before the runs):

| Problem, method | Iterations | Without the log (s) | With the log (s) | Difference | C4 |
|---|---|---|---|---|---|
| Bratu, NiL-N (L-BFGS) | 3,158 | 32.80 | 34.52 | +5.25% | fails |
| Bratu, NiL-N (LM) | 766 | 27.45 | 27.27 | -0.65% | passes |
| Burgers, NiL-N (L-BFGS) | 1,934 | 35.23 | 36.89 | +4.72% | passes |
| Burgers, NiL-N (LM) | 239 | 17.74 | 17.04 | -3.96% | passes |
| Buckley-Leverett, NiL-N (L-BFGS) | 758 | 26.87 | 27.19 | +1.16% | passes |
| Buckley-Leverett, NiL-N (LM) | 213 | 28.17 | 33.28 | +18.13% | fails |

**Buckley-Leverett NiL-N, the three timings you asked for:**
- The three differences were +1.16%, -2.58% and -6.08%, a spread of 7.2%.
- The first pair, which gives the verdict, is within 5%; the third is outside.
- In the CPU set the same pair was 5.9% faster with the log.

**Buckley-Leverett LM at +18.1% held the A100 set** until its cause was shown (my decision with the user, 17 November). The LM runs log the error at every iteration, so a leak there would be in every A100 LM time. The diagnostic (stage p5e, 67 SU) timed the same pair three times in each of 11 processes. Its criteria were fixed before the job:

| Group | Pairs | Median | Range | Verdict |
|---|---|---|---|---|
| CPU, the module's library (as the rerun) | 9 | -1.9% | -10.7% to +4.9% | clean |
| CPU, the wheel's library | 9 | -1.4% | -14.7% to +3.8% | clean |
| CPU, the module's library, threads kept awake (`OMP_WAIT_POLICY=ACTIVE`) | 9 | -6.1% | -40.3% to +6.4% | faster with the log |
| A100, two processes side by side, as the A100 set | 6 | -0.0% | -1.4% to +12.3% | clean |

What this shows:
- **The rerun's +18% was not reproduced, and the log leaks with neither library.**
  - On that node the same 213 iterations took 29.5 to 34.7 s with the module's library, with or without the log.
  - The rerun's pair, 28.17 s without and 33.28 s with, is one fast run and one ordinary one.
- **On the A100 the log is clean.** Four of six pairs are within 0.3%, and a fifth is -1.4%.
  - The one +12.3% pair is a slow stretch on GPU 1.
  - The run after it, without the log, was as slow (31.02 s against 31.01 s).
- **The wheel's library is about 65% slower on this run** (51 s against 31 s). The cost check measured +61 to +74% on Buckley-Leverett.
- **The kept-awake group came out faster with the log, which I did not foresee.**
  - In each of its three processes the first run without the log took 41 to 51 s, against about 31 s afterwards. No other group shows this.
  - Runs without the log always go first in a pair, so a one-time cost lands there.
  - The verdict stands as fixed. Its other six pairs have a median near -1.5%.
- **On the A100 set's clock:** the diagnostic gave the go-ahead, and the A100 set ran after it.

## 4. Time to reach an error

Seconds from the starting data to the iterate, as your reply asks: the run's clock plus its setup and pretraining. Each entry is the smallest over the sizes of the median over the seeds, with the size P that gives it. "opt." is the optimizer alone, the clock from the initial coefficients; its size is given when it differs. A dash means no size reaches the level within the budget. The full table, with every size, is `P5_3_cost/time_to_error.csv`; every run and level is in `time_to_error_runs.csv`.

### Bratu, CPU

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.85 (P = 225); optimizer only 5.81 | - | - | - | - |
| NiL-Q (L-BFGS) | 8.67 (P = 100); optimizer only 8.54 | 185 (P = 225); optimizer only 185 | - | - | - |
| NiL-N (LM) | 3.07 (P = 100); optimizer only 1.75 (P = 225) | 12.4 (P = 225); optimizer only 9.73 | 302 (P = 225); optimizer only 300 | - | - |
| LiL-N (L-BFGS) | 0.147 (P = 100); optimizer only 0.133 | 0.176 (P = 100); optimizer only 0.162 | 1.35 (P = 225); optimizer only 1.31 | - | - |
| LiL-N (trf) | 0.084 (P = 100); optimizer only 0.0706 | 0.084 (P = 100); optimizer only 0.0706 | 0.34 (P = 225); optimizer only 0.306 | - | - |
| LiL-Q | 0.0435 (P = 100); optimizer only 0.0299 | 0.0435 (P = 100); optimizer only 0.0299 | 0.23 (P = 225); optimizer only 0.194 | - | - |
| square collocation (SQ-CGL) | 0.0024 (P = 25); optimizer only 0.000233 | 0.0024 (P = 25); optimizer only 0.000233 | 0.00919 (P = 100); optimizer only 0.00355 | 0.0198 (P = 225); optimizer only 0.00865 | - |

### Bratu, A100

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 9.93 (P = 225); optimizer only 9.76 | - | - | - | - |
| NiL-Q (L-BFGS) | 15.7 (P = 225); optimizer only 15.7 | 238 (P = 225); optimizer only 238 | - | - | - |
| NiL-N (LM) | 0.682 (P = 225); optimizer only 0.643 | 4.41 (P = 225); optimizer only 4.38 | 142 (P = 225); optimizer only 142 | - | - |
| LiL-N (L-BFGS) | 0.379 (P = 100); optimizer only 0.364 | 0.471 (P = 100); optimizer only 0.456 | 3.91 (P = 225); optimizer only 3.87 | - | - |

### Burgers, CPU

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 3.01 (P = 225); optimizer only 2.82 | 44.1 (P = 225); optimizer only 44.1 | - | - | - |
| NiL-Q (L-BFGS) | 8.27 (P = 100); optimizer only 8 | 57.3 (P = 400); optimizer only 57.1 | - | - | - |
| NiL-N (LM) | 1.23 (P = 225); optimizer only 0.686 (P = 100) | 2.19 (P = 225); optimizer only 2.13 | 7.57 (P = 400); optimizer only 7.44 | 55.4 (P = 625); optimizer only 55.1 | - |
| LiL-N (L-BFGS) | 0.0873 (P = 100); optimizer only 0.0832 | 0.191 (P = 196); optimizer only 0.178 | 0.329 (P = 324); optimizer only 0.282 | 0.809 (P = 484); optimizer only 0.756 (P = 676) | 1.49 (P = 676); optimizer only 1.38 |
| LiL-N (trf) | 0.0512 (P = 100); optimizer only 0.0472 | 0.187 (P = 196); optimizer only 0.179 | 0.415 (P = 324); optimizer only 0.398 | 1.12 (P = 484); optimizer only 1.09 | 2.08 (P = 676); optimizer only 2.01 |
| LiL-Q | 0.0221 (P = 100); optimizer only 0.0181 | 0.132 (P = 196); optimizer only 0.113 | 0.231 (P = 324); optimizer only 0.211 | 0.55 (P = 484); optimizer only 0.506 | 0.974 (P = 676); optimizer only 0.891 |

### Burgers, A100

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.49 (P = 225); optimizer only 5.44 | 43.2 (P = 625); optimizer only 42.7 | - | - | - |
| NiL-Q (L-BFGS) | 13.1 (P = 225); optimizer only 12.7 | 68 (P = 400); optimizer only 68 | - | - | - |
| NiL-N (LM) | 0.536 (P = 225); optimizer only 0.413 (P = 400) | 0.919 (P = 400); optimizer only 0.878 | 2.42 (P = 400); optimizer only 2.37 | 12.4 (P = 625); optimizer only 9.97 | - |
| LiL-N (L-BFGS) | 0.373 (P = 100); optimizer only 0.368 | 0.572 (P = 196); optimizer only 0.551 | 1.09 (P = 324); optimizer only 1.05 | 1.87 (P = 484); optimizer only 1.8 | 6.66 (P = 676); optimizer only 6.58 |

### Buckley-Leverett (viscous), CPU

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 3.73 (P = 64); optimizer only 3.48 | 245 (P = 576); optimizer only 245 | - | - | - |
| NiL-Q (L-BFGS) | 12.8 (P = 256); optimizer only 12.4 (P = 64) | 245 (P = 576); optimizer only 244 | - | - | - |
| NiL-N (LM) | 1.24 (P = 64); optimizer only 0.922 | 14.4 (P = 1024); optimizer only 13.8 | 40.1 (P = 576); optimizer only 39.6 | - | - |
| LiL-N (L-BFGS) | 0.881 (P = 256); optimizer only 0.766 | 1.03 (P = 256); optimizer only 0.911 | 20.2 (P = 1024); optimizer only 19.8 | - | - |
| LiL-N (trf) | 1.07 (P = 256); optimizer only 0.99 | 1.2 (P = 256); optimizer only 1.11 | 14 (P = 1024); optimizer only 13.6 | - | - |
| LiL-Q | 0.327 (P = 256); optimizer only 0.244 | 0.388 (P = 256); optimizer only 0.305 | 3.44 (P = 1024); optimizer only 3.05 | - | - |

### Buckley-Leverett (viscous), A100

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.82 (P = 64); optimizer only 5.68 | 164 (P = 1024); optimizer only 164 | - | - | - |
| NiL-Q (L-BFGS) | 14.7 (P = 256); optimizer only 14.6 | 249 (P = 576); optimizer only 249 | - | - | - |
| NiL-N (LM) | 0.677 (P = 576); optimizer only 0.487 | 1.94 (P = 1024); optimizer only 1.67 | 8.04 (P = 1024); optimizer only 7.77 | - | - |
| LiL-N (L-BFGS) | 2.01 (P = 256); optimizer only 1.86 | 2.39 (P = 256); optimizer only 2.24 | 15.3 (P = 1024); optimizer only 14.8 | - | - |

**What the setup includes** (your reply, item 2):
- **Networks:** setup (the collocation points and the model, on the device) and pretraining.
- **The LiL methods:** the system (collocation set and basis matrices) and the initial coefficients.
  - On Bratu and Burgers the initial coefficients are zero.
  - On Buckley-Leverett they are the least-squares fit of the initial profile (Package 3's initial guess). That fit is timed together with the system's build in one figure (`setup_s`), not separately. The two together take 0.015 to 0.50 s, depending on the size; for LiL-Q at P = 1024 they are 0.39 s of its 3.44 s to 1e-4.
- **LiL-N L-BFGS:** also the move to the device (`device_setup_s`, given apart).
- **Square collocation:** the setup Package 2's driver times (`t_setup_s`).
- **Not included,** as your reply says: imports, device start-up, the warm-up run.

## 5. The CPU network runs again

All five CPU network jobs ran again with numpy imported first and the library checked before each run, and C1 now passes.
- **The rerun used 874 SU** for its seven compute jobs (875.3 with its report job), under your line of 1,000, and every job completed. So the A100 set went ahead without asking, as you said.
- **The CPU set's network runs are kept** in `package5_item3_cpu/`, as the record of the cost check. They are not in the table.

## 6. Package 1's Component A baselines

The same import order put Package 1's Component A processes on the wheel's library: Kovasznay, F1 (PINN) and F2 (LM), on the CPU. Their CPU times sit beside LiL-Q's in the same-hardware table. A check of 20.9 SU (stage p5k) ran both on one whole node, the two libraries alternating over three rounds:

| Measure | Module's library | Wheel's library | Wheel / module | Package 1's runs |
|---|---|---|---|---|
| F1, Adam, s per iteration | 0.458 | 0.336 | 0.73 | 0.327 |
| F1, L-BFGS, s per iteration | 0.450 | 0.360 | 0.80 | 0.353 |
| F2, s per evaluation | 10.4 | 9.27 | 0.89, the rounds overlap | not comparable (below) |

- **F1 is 20 to 27% faster with the wheel's library**, the one Package 1 used. Its torch-first values are within 3% of Package 1's.
- **F2 does not change beyond the cutoff.** Its time per evaluation is not constant over a run (rejected tries late in a run are cheap), so Package 1's whole-run figure cannot be set beside these short runs.
- So the Component A baselines had the faster library. Their comparison with LiL-Q is conservative for LiL-Q, and I do not propose a rerun. `REPRODUCE.md` will name both libraries at the release, as you asked.

## 7. Rules applied

1. **Section 7, report both values:** C4 on Bratu NiL-N and on Buckley-Leverett LM; Buckley-Leverett NiL-N's three timings.
2. **Section 7, report the status as returned:**
   - 11 of the 108 A100 network runs and 13 of the 108 CPU ones stopped before the budget on an optimizer stall. Their histories up to the stall count.
   - No run failed, and none is flagged.
3. **Your reply:** pretraining is on the clock (Section 4). The rerun stayed under 1,000 SU, and every job completed.
4. **The pairing gate** (16 November): it passed in each A100 job before its budget runs.

## 8. SU

| Part | Jobs | SU |
|---|---|---|
| Items 1, 2 and 4 | 8 | 118.7 |
| The re-time of item 2 | 1 | 3.9 |
| Addendum 2 (one job ran out of time and was run again) | 2 | 30.9 |
| Item 3: the CPU set (p5c) | 8 | 938.9 |
| Item 3: the OpenMP library diagnostic (p5o) | 1 | 14.4 |
| Item 3: the CPU network runs again (p5r) | 8 | 875.3 |
| Item 3: Component A's library (p5k) | 1 | 20.9 |
| Item 3: the C4 diagnostic (p5e) | 3 | 67.0 |
| Item 3: the A100 set (p5g) | 5 | 1,868.5 |
| **Package 5** | **37** | **3,938.5** |

Against your cap of 4,400 this leaves 461.5 SU. Every job's row is in `su.csv`, at Grace's rates (a CPU core-hour 1 SU, an A100-hour 72 SU more). Every job completed except the first Addendum 2 job, as reported then.

## Files

- **The item:** `P5_3_cost/`.
  - `networks/` and `lil/`: one folder per run, each with `history.csv` (iteration, time, loss, error) and `run.json`.
  - `time_to_error.md`, `time_to_error.csv` and `time_to_error_runs.csv`: the table.
  - `checks.json`: C1 to C4, the pairing, and the diagnostics. Also `networks/checks/` (C1, C2/C4, the solo and gate runs) and `lil/checks/` (C3).
  - **For checking C1 and Section 6 from the branch alone:** `package1_reference/` holds the Package 1 files they compare against, in Package 1's own layout: the 90 histories of C1, and the Component A runs F1_18 and F2_05 (seeds 0 to 4). Each run folder of C1 holds `metrics_iterations_0_100.csv`, the first 101 rows of its per-iteration log, byte for byte (the whole logs are left out for size). C1, the diagnostic's C1 and the Component A summary, rerun on the branch alone, give exactly the values here. The Package 4 runs of C3 are on the branch `package4-results`.
  - `openmp/`, `openmp_kovasznay/` and `c4_diagnostic/`: the three diagnostics.
- **The stages' records:** `package5_item3_cpu/`, `package5_item3_cpu_rerun/`, `package5_item3_gpu/`, `package5_item3_openmp/`, `package5_item3_openmp_kovasznay/` and `package5_item3_c4_diagnostic/`. Each holds its `COMMIT`, provenance (a `hardware.json` per process) and `report_log.txt`. The four stages with a report job also hold `sacct.txt` and `slurm_logs/`. The two single-job diagnostics (`openmp`, `openmp_kovasznay`) reported inside their job, before Slurm had their accounting: their rows in `su.csv` are from sacct afterwards.
- **Package 5's records:** `su.csv`, `hardware.json` (item 3 per job), `provenance.json` (each stage's commit) and `assembly_log.txt`.
- **Left out for size** (1.4 GB, kept with the cluster archives): each run's per-iteration logs (`log.csv`, `metrics.csv`), logged coefficients and saved networks. The table is rebuilt from `history.csv` and `run.json` alone, and gives exactly the table here.
