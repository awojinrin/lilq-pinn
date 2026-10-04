# package1, final: the derived tables (3 October 2026)

This is the final assembly of Package 1 from waves 1-4, after the
advisor's reply to wave 4 (3 October 2026). The branch starts from commit
`8a86bbc` on `v3-dev`, the code that assembled the package
(`WAVES.json`'s `assembled_by_commit`). The branch's own commit adds only
`results/package1_final/`.

**What is here:** what the assembly (`90_finalize`) derived:
- `WAVES.json`;
- the merged and derived tables, the stopping-rule tables, and the Section
  4.6 results;
- the figures and B9's annotated tables.

**The rest of `package1`** is the waves' own files, copied, and is on
branches `wave1-results` ... `wave4-results`. The screening runs and logs
are in `A_calibration/screening/`, unchanged (wave 2's runs, with
`validation.json` from wave 4), as the advisor asked for the appendix
table.

## How it was assembled

`scripts/cluster/90_finalize.slurm`, every step, with
`LILQ_PACKAGE_FINAL=1`.

**Where:** on a laptop, on the downloaded results of the four waves:
- waves 1-3 from `wave<N>_full.tar.gz`;
- wave 4 from `wave4_full.tar.gz` (no `*.npz`), plus
  `wave4_fixes.tar.gz` (jobs 14a and 14b).

The advisor asked for no further Grace jobs. Also, the wave folders' locks
(`env.sh`) would refuse code newer than wave 4's. Grace's own `package1`
is still the provisional one of 2 October.

**Checks:**
- **Waves unchanged:** a SHA-256 of every file of the four wave folders,
  3,777 files, was taken before and after. Nothing changed.
- **The log:** `finalize_log.txt`.

**What the assembly found:**
- **Assembly:** 3,391 files, the waves' commits `8a3f5f7`, `905e58c`,
  `905e58c` and `17b3539`.
- **Overrides:** 226, listed in `WAVES.json`.
- **Superseded runs:** 26 marked.
- **Section 3.3 and check B1:** 63 runs, 0 failed. Check B1 has the same 15
  violations as before.
- **Four-method:** 140 rows and 54 controls, 0 failures, plus the 16 LiL-Q
  rows.
- **B8:** 131 rows. **B9:** 28 rows.

## What changed from the provisional package

**1. `status`: final.** `WAVES.json` also records `assembled_by_commit`.

**2. The clean times are option B** (the advisor's reply to wave 4, item 1).
- **CPU:** every CPU clean time is the median of three independent clean
  timings: wave 4's job 13a, and job 14b's two replicates
  (`scripts/cluster/after_wave4/`). Each followed the same protocol
  (warm-up, timed run, median of five under 1 s).
- **GPU:** the times (13b, all under 1 s, medians of five) are unchanged.
- **`B_instrumentation/clean_timing/`:**
  - **`clean_timing.csv`:** the composed table. Each row has its three
    timings (`clean_time_timings_s`), their spread and the rule.
  - **`clean_timing_single.csv`:** wave 4's table as measured.
  - **`clean_vs_logged.csv`:** from the wave 4 report, against the single
    timings.
- **`WAVES.json`'s `clean_timing`** records the rule, the replicate tables
  and all 37 CPU rows' three timings.
- **What reads the composed table:** the Section 4.6 comparison
  (`A_calibration/results/`) and the four-method LiL-Q rows
  (`four_method_lilq.csv`).
- **The check:** all three tables are identical to `fixes/option_b/` on
  branch `wave4-results`.

**3. Section 4.6 reports the final errors** (item 2(a); commit `ce0afda`).
- `eps_*_median`, `_min` and `_max` are of each seed's final errors in its
  `run.json`.
- The earlier "best within the budget" values are beside them, as
  `eps_*_best_logged_*`.
- So are `n_final_past_budget` and `max_final_overrun_s`: at most 0.2 s
  (F1) and 10.3 s (F2) past the one-hour budget.

| Representative | Device | Median eps_u (final) | Best logged |
|---|---|---|---|
| F1_18 | GPU | 4.19e-5 | 4.11e-5 |
| F1_18 | CPU | 5.88e-5 | 5.66e-5 |
| F2_05 | GPU | 1.66e-10 | 1.66e-10 |
| F2_05 | CPU | 1.99e-9 | 1.99e-9 |

**4. The stall-based termination rule** (item 2(b); commit `3f75a58`):
`B_instrumentation/stopping_rule/`. The rule is tau_chi = 0.1, tau_r =
0.01, both conditions at n_s consecutive steps, recomputed from the logs
(`experiments/stopping_rule_table.py`).
- **`stopping_rule_table16.csv`:** Table 16, the 22 kmax passes x n_s =
  1, 2, 3. No early stop and no premature stop at any n_s. R max is 1.0055,
  1.0023 and 1.0013.
- **`stopping_rule_heldout.csv`:** the 192 held-out Component C runs.

  | n_s | Fire | Censored | R max |
  |---|---|---|---|
  | 1 | 145 | 0 | 1.00320 |
  | 2 | 69 | 76 | 1.00039 |
  | 3 | 59 | 86 | 1.00009 |

  For n_s = 2, the Kovasznay runs that fire (55) return eps_u at most
  1.746x their smallest, median 1.0039.
- **`stopping_rule_summary.json`:** the above, with tau and n_s.
- **The check:** the script reproduces the advisor's `ns_eval.py` row by
  row on these logs: 66 Table 16 rows and 576 held-out rows, no
  difference.
- **`stall_columns.csv`:** for every `iterations.csv` in the package (306
  logs), the first stall as logged (tau_r = 0.1, as every run was
  logged), the first at tau_r = 0.01, and the n_s = 2 rule's step,
  returned iterate and censoring.
  - The tables' own `first_stall_iteration` columns (run index, B8, B10,
    Component C, Table 3) and B8's `stall_flag_ever` are left as logged,
    at tau_r = 0.1.
  - Where the manuscript quotes a first stall, the tau_r = 0.01 value is in
    this file.

  The logged first stall differs at tau_r = 0.01 in:

  | Logs | Count | Differs |
  |---|---|---|
  | Section 3.3 runs | 64 | 20 |
  | B8's LiL-Q runs | 19 | 9 |
  | Table 3 | 10 | 2 |
  | Component C | 196 | 5 |
  | the pinned Beltrami run | 1 | 1 |
  | B10 | 16 | 0 |

## Files

| Path | What |
|---|---|
| `WAVES.json` | wave commits, overrides, superseded runs, `clean_timing`, `status`, `assembled_by_commit` |
| `A_calibration/results/kovasznay_comparison.csv`, `time_to_accuracy.csv` | Section 4.6 |
| `A_calibration/figures/eps_u_vs_time.*` | Section 4.6's figure |
| `B_instrumentation/clean_timing/` | the clean times (option B), as measured, against logged |
| `B_instrumentation/stopping_rule/` | the stall-based rule's tables |
| `B_instrumentation/four_method_tables.csv`, `four_method_controls.csv`, `four_method_lilq.csv` | Section 3.4 |
| `B_instrumentation/runs_index.csv`, `reproduction_check.csv`, `gpu_cpu_equivalence.csv` | Section 3.3, checks B1 and B3 |
| `B_instrumentation/residual_band_figures/` | task B5 |
| `B_instrumentation/b8_initial_guess.csv`, `b8_kmax60_vs_wave3.csv` | B8 (131 rows), its K_max = 60 reruns |
| `B_instrumentation/darcy_fv_comparison.csv`, `darcy_fv/*/` | B9, the allocation filled |
| `finalize_log.txt` | the assembly's output |
