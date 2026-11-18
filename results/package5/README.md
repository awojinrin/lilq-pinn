# Computational Package 5 results (items 1 to 4, and Addendum 2)

This is `package5_results/`, in the Section 8 layout of Computational Package 5 (the advisor's
instructions of 13 November 2026), for items 1 to 4, and for Addendum 2 (15 November 2026). The
branch holds `v3-dev` at `99fbd07`, the commit item 3 was assembled at, and adds only
`results/package5/`. Items 1, 2 and 4 and Addendum 2 were assembled at `4417866`; the assembly at
`99fbd07` leaves every one of their files as it was (it adds item 3 and rewrites `su.csv`,
`provenance.json`, `hardware.json`, `assembly_log.txt` and `DECISIONS_package5.md`).

**Start with `report.md`** (items 1, 2 and 4) **and `report_item3.md`** (item 3). Each ends with a
list of its files. Addendum 2 is reported in the item 3 cost-check memo.

**Commits:**
- **The runs:** every Grace job of 14 November 2026 ran at `60c2d8c` (`COMMIT`), under the source lock.
- **The re-time:** one item 2 run (Burgers p = 10, the PyTorch L-BFGS) was run again at `16b128b`,
  in its own folder under its own lock (`package5_retime/`, its own `COMMIT`). In the timed job, a
  one-time cost of about 29 s fell inside that run's only timed pass. `16b128b` adds an untimed
  warm-up of 5 iterations before every solver timed by one pass. The warm-up changes the clock only:
  the re-timed run has the same iterations and identical coefficients.
- **Addendum 2:** its 18 runs, K1 and K2 ran at `7659961` on 16 November 2026, in one job under its
  own lock (`package5_a2/COMMIT`). A first job that day produced nothing (see Cost).
- **After the runs:** `8674498` and `21ec859` add the assembly (`experiments/p5_assemble.py`) and
  the comparisons with the advisor's files; `57ba02a` adds the re-time to the assembly; `99303ee`
  adds Addendum 2 to it; `4417866` is a DECISIONS note. Item 3's commits (`13a23a5`, `b3fddf0`,
  `bbce1ac`) add PyTorch residuals beside code item 1 uses, and a path that evaluates the residual
  alone, which equals the full assembly's residual to 1e-14 (tested); the item 1 code paths the runs
  took are otherwise unchanged. Otherwise, apart from `16b128b`'s warm-up, no code a
  run uses changed between the runs and the assembly, which reruns summaries only.
- **Where they are recorded:** `provenance.json` records the runs' commit, the re-time's, Addendum 2's,
  the assembly's and the commits between. `DECISIONS_package5.md` gives the reasons.

**What is Grace's.** Every run folder, table, log and timing is as Grace wrote it. The cluster did
not have the advisor's expected files, so the assembly reran the summaries on Grace's files with
them:
- `P5_1_weights/compare_expected.csv`, `P5_2_nlls/compare_expected.csv` and
  `P5_4_pb/compare_expected_4b.csv` set the runs beside the advisor's values;
- `P3_1_beltrami_certified/beside_delivered.csv` sets the Beltrami runs beside the rows Package 3
  delivered. Since 16 November it reads Package 3 as delivered, with the pressure delta_P as
  re-evaluated after the advisor's reply of 10 October; the version of 14 November read Grace's first
  value. The p and combined ratios moved by at most 2e-4 relative; one figure printed in `report.md`
  changes (B2, weight 10^3, p (pin gauge) / delta_P: 169 there, 170 now);
- `checks.json` holds the checks C1 to C3 of the delivered packages; all three pass;
- the summaries also rewrote Grace's tables. The tables kept here are Grace's; every column agreed
  exactly, except delta_P (the error of the best approximation) and the ratios to it, which differ by
  round-off of at most 1e-12 relative, and the Burgers PDE check's residual in
  `P5_1_weights/references.json` (about 1e-12, round-off). Both are recorded in
  `table_rerun_differences.json`;
- `P5_2_nlls/retime.csv` sets the re-timed run beside the timed job's. Grace's item 2 tables keep the
  timed job's time (30.9 s); the re-timed time is 0.372 s.

**Addendum 2** (the advisor's Section 6 layout):
- `P5_A2_initial_profile/`: one folder per run (`iterations.csv`, `errors_by_iterate.csv`,
  `updates_by_iterate.csv`, `betas.npy`, `beta_last.npy`, `run.json`), `terminal.csv`,
  `stretches.csv`, `trig_delta_P.json`, `item1_references.json`, `checks.json` (K1 and K2, both
  pass), and `K1/` (item 1's two runs rerun in the job). All as Grace wrote them.
- The summary was rerun at the assembly on Grace's stored iterates: every number agrees within 2e-14
  relative but delta_P's, which differ by round-off of at most 1.8e-9
  (`package5_a2_rerun_differences.json`).
- Section 3a: `updates_by_iterate.csv` in the 96 run folders of item 1 and the 54 Poisson-Boltzmann
  run folders of items 4a and 4b, and under `P5_A2_initial_profile/updates_package4/` for Package 4's
  18 runs (so that the `package4-results` branch stays as delivered). Written at the assembly from the
  stored coefficients.
- `package5_a2/`: the stage's records (`COMMIT`, `provenance/a2`, `slurm_logs`, `sacct.txt`,
  `su_per_job.csv`, `report_log.txt`).

**Item 3** (the time to reach an error; `report_item3.md`):
- `P5_3_cost/`: one folder per run under `networks/` and `lil/` (`history.csv`, `run.json`); the
  table (`time_to_error.md`, `.csv`, `time_to_error_runs.csv`); `checks.json` (C1 to C4, the pairing,
  the diagnostics) with the check records under `networks/checks/` and `lil/checks/`; the three
  diagnostics `openmp/`, `openmp_kovasznay/` and `c4_diagnostic/`.
- `package5_item3_<stage>/`: each Grace stage's records (`COMMIT`, provenance with a `hardware.json`
  per process, `sacct.txt`, `slurm_logs/`, `report_log.txt`). The runs ran at the stage's `COMMIT`,
  under the source lock: the CPU set at `0f74bdf` (its LiL runs, protocol runs and C3 are in the
  table; its network runs, superseded, stay in `package5_item3_cpu/`), the OpenMP diagnostic at
  `fcb9952`, the CPU network runs again and Component A's library check at `3df7b7b`, the C4
  diagnostic and the A100 set at `1a88feb`.
- **What is Grace's:** every run folder, history and check record. The assembly rebuilt the table
  from the histories; every row equals the table each stage's report job wrote on Grace. It reran
  C1 against Package 1 and the diagnostics' summaries on the downloaded data (the C4 diagnostic's
  groups equal Grace's).
- **Left out here, for size** (1.4 GB, kept by the authors with the cluster tarballs): each run's
  per-iteration logs `log.csv` and `metrics.csv`, the logged coefficients `betas_logged.npy`, and
  the saved networks (`network.pt`, `model.pt`). The table needs none of them: it is rebuilt from
  `history.csv` and `run.json` alone.

**Cost:** `su.csv`, 37 jobs, 3,938.5 SU at Grace's rates (item 3: 26 jobs, 3,785.0 SU; the cap is
4,400). Items 1, 2 and 4, the re-time and Addendum 2: 11 jobs, 153.5 SU at Grace's rates (the report job's, the re-time job's and both
Addendum 2 jobs' rows from sacct after they ended). The first Addendum 2 job (20549881, 24.5 SU) ran
its full hour without producing a run: its processes sat idle after starting, most likely a stall of
the scratch filesystem during that day's outage. The second (20553982) ran the same bundle in 954 s.

**What is not here:**
- the raw cluster tarballs (`package5.tar.gz`, `package5_retime.tar.gz`, `package5_a2.tar.gz`, and
  item 3's six). Everything in them is here, assembled, except item 3's per-iteration logs,
  logged coefficients and saved networks (above). The tarballs are kept by the authors;
- the advisor's `expected_weight_bratu.json`, `expected_weight_bl.json`,
  `expected_nlls_liln.json` and `expected_rf_pb.json`. `provenance.json` records their SHA-256.

**Redactions.** Before publishing, these were replaced by placeholders in 226 text files (listed in
`REDACTED_FILES.txt`):
- the cluster user's scratch path (`$SCRATCH`);
- the HPRC allocation account (`<account>`);
- the user name (`<user>`);
- the laptop's host name, where the bundle was built (`<laptop-host>`).

Nothing else changed. Every number, every binary file (`*.npy`, `*.npz`) and every CSV is as
assembled; the only numbers removed are the account numbers. The unredacted originals are kept by
the authors.
