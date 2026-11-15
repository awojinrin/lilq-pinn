# Computational Package 5 results (items 1, 2 and 4)

This is `package5_results/`, in the Section 8 layout of Computational Package 5 (the advisor's
instructions of 13 November 2026), for items 1, 2 and 4. Item 3 follows separately. The branch holds
`v3-dev` at `57ba02a`, the commit the results were assembled at, and adds only `results/package5/`.

**Start with `report.md`.** It ends with a list of its files.

**Commits:**
- **The runs:** every Grace job of 14 November 2026 ran at `60c2d8c` (`COMMIT`), under the source lock.
- **The re-time:** one item 2 run (Burgers p = 10, the PyTorch L-BFGS) was run again at `16b128b`,
  in its own folder under its own lock (`package5_retime/`, its own `COMMIT`). In the timed job, a
  one-time cost of about 29 s fell inside that run's only timed pass. `16b128b` adds an untimed
  warm-up of 5 iterations before every solver timed by one pass. The warm-up changes the clock only:
  the re-timed run has the same iterations and identical coefficients.
- **After the runs:** `8674498` and `21ec859` add the assembly (`experiments/p5_assemble.py`) and
  the comparisons with the advisor's files; `57ba02a` adds the re-time to the assembly. Apart from
  `16b128b`'s warm-up, no code a run uses changed.
- **Where they are recorded:** `provenance.json` records the runs' commit, the re-time's, the
  assembly's and the commits between. `DECISIONS_package5.md` gives the reasons.

**What is Grace's.** Every run folder, table, log and timing is as Grace wrote it. The cluster did
not have the advisor's expected files, so the assembly reran the summaries on Grace's files with
them:
- `P5_1_weights/compare_expected.csv`, `P5_2_nlls/compare_expected.csv` and
  `P5_4_pb/compare_expected_4b.csv` set the runs beside the advisor's values;
- `P3_1_beltrami_certified/beside_delivered.csv` sets the Beltrami runs beside the rows Package 3
  delivered;
- `checks.json` holds the checks C1 to C3 of the delivered packages; all three pass;
- the summaries also rewrote Grace's tables. The tables kept here are Grace's; every column agreed
  exactly, except delta_P (the error of the best approximation) and the ratios to it, which differ by
  round-off of at most 1e-12 relative, and the Burgers PDE check's residual in
  `P5_1_weights/references.json` (about 1e-12, round-off). Both are recorded in
  `table_rerun_differences.json`;
- `P5_2_nlls/retime.csv` sets the re-timed run beside the timed job's. Grace's item 2 tables keep the
  timed job's time (30.9 s); the re-timed time is 0.372 s.

**Cost:** `su.csv`, 9 jobs, 122.6 SU at Grace's rates (the report job's and the re-time job's rows
from sacct after they ended).

**What is not here:**
- the raw cluster tarballs (`package5.tar.gz`, `package5_retime.tar.gz`). Everything in them is
  here, assembled. The tarballs are kept by the authors;
- the advisor's `expected_weight_bratu.json`, `expected_weight_bl.json`,
  `expected_nlls_liln.json` and `expected_rf_pb.json`. `provenance.json` records their SHA-256.

**Redactions.** Before publishing, these were replaced by placeholders in 10 text files (listed in
`REDACTED_FILES.txt`):
- the cluster user's scratch path (`$SCRATCH`);
- the HPRC allocation account (`<account>`);
- the user name (`<user>`);
- the laptop's host name, where the bundle was built (`<laptop-host>`).

Nothing else changed. Every number, every binary file (`*.npy`, `*.npz`) and every CSV is as
assembled; the only numbers removed are the account numbers. The unredacted originals are kept by
the authors.
