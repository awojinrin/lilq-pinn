# DECISIONS.md: the Package 5 entries

Copied from the repository's `DECISIONS.md` at the assembly's commit (57ba02a), newest first.

---

## 2026-11-14 -- Package 5, item 2: the one-pass timings get an untimed warm-up; one run re-timed on Grace (`experiments/p5_2_nlls.py`, stage `p5t`)

**Found at the assembly.** In Grace's timed job, Burgers p = 10 with the PyTorch L-BFGS took 30.9 s
(224 ms per iteration); on the laptop it takes 1.4 s (10 ms per iteration). Every other Grace run of
item 2 is 1-2x its laptop time, including the same solver at Burgers p = 14 to 30. That run was the
job's first PyTorch L-BFGS pass, and the solvers timed by one logged pass (his answer 2: the L-BFGS
solvers) had no warm-up, so a one-time cost of about 29 s fell inside its only timed pass.

**Not the import.** The first diagnosis (PyTorch imported lazily inside the first call) was wrong:
torch and `lilq.solvers` are loaded when the driver is imported, before any clock. What the 29 s is
(a first-call initialization in torch or the thread pools on that node) is not known; the fix does not
need to know it.

**Decided:**
- `prepare(name, prob)`: before the clock, each one-pass solver (the ONE_PASS names) runs the same
  solve for 5 iterations (`WARMUP_IT`), untimed; its seconds are kept as `warmup_s` in `result.json`.
  The timed pass is unchanged (same solver, same start, same cap) and its iterates do not depend on
  the warm-up (tested). The timing note says "after an untimed warm-up of 5 iterations".
- **Re-time (the user's decision):** the one run, Burgers p = 10 PyTorch L-BFGS, rerun on a whole
  node (timed-cpu, as the timed job) in its own stage `p5t` -> `results/package5_retime`, under its
  own commit lock (`scripts/cluster/package5/submit_p5_retime.sh`, `p5_retime.slurm`; 12 SU requested,
  about 5 expected). The other item 2 runs are not rerun: their times sit at 1-2x the laptop's.
- The report shows the original 30.9 s beside the re-timed value and its `warmup_s`. The assembly
  (`p5_assemble.py --retime`) copies `package5_retime/` as Grace wrote it, stops unless its run is the
  timed job's (same counts and status; errors and coefficients within 1e-8 of their scale), and
  writes `P5_2_nlls/retime.csv`; Grace's own tables keep the timed job's 30.9 s.
- **Result (job 20534654, node c057, 3.9 SU):** 0.372 s (2.7 ms per iteration) against 30.9 s
  (224 ms); the same 138 iterations and identical coefficients. The warm-up itself took 59.7 s for 5
  iterations: the one-time cost is real and was larger on this node.
- Item 3's one-pass optimizer timings will use the same warm-up.

**Affects:** item 2's `time_s` for that one run (and `warmup_s` added for the one-pass solvers in
later runs). No iterate, error or iteration count changes.

**Revisit:** if a warm-up of 5 iterations is ever comparable to the timed pass itself (it is not
here: the cap is 2,000 iterations).

---

## 2026-11-14 -- Package 5, items 1, 2 and 4: the assembly (`experiments/p5_assemble.py`)

As Package 4's: Grace's run folders copied as they are; the summaries rerun on them with the advisor's
files (which Grace had not) to write the comparisons; Grace's tables kept, and the assembly stops if a
rewritten number differs from Grace's by more than 1e-8 relative, but for delta_P's columns (recorded).
The table comparison moved from `p4_assemble.py` to `assemble_common.py` (`compare_tables`), unchanged.

- **Item 1's `references.json`** is compared the same way. One quantity in it is round-off: the Burgers
  PDE check's residual (Grace 2.46e-12, the assembly 2.37e-12; 5e-13 of the equation's terms against the
  check's 1e-8). It differs between machines in its first digits, so its two values are recorded in
  `table_rerun_differences.json`; its verdict and every other number must be Grace's (they agree to 2e-12).
- **On the Grace runs (60c2d8c):** every table outside delta_P's columns equals Grace's exactly; delta_P's
  columns differ by up to 1e-12. C1, C2 and C3 pass.
- **`su.csv`** takes the report job's own row from `--sacct-extra` (its tarball was written before it ended).

---

## 2026-11-14 -- Package 5, item 1: how the runs are set beside the advisor's values and the delivered Beltrami rows

Assembly of the Grace runs (commit 60c2d8c). Two comparisons the summaries did not yet write
(`experiments/p5_1_weights.py`: `compare_expected`, `beltrami_beside_delivered`).

- **Bratu and viscous Buckley-Leverett beside `expected_weight_*.json`** (Section 3: his runs on
  Gauss-Legendre grids, "agreement to one figure is expected"). Rows where both ran the same size and
  weight: Bratu at 10, 10^3, 10^6 (and his p^4 = 10^4 at p = 10, the same weight as our 10^4);
  Buckley-Leverett at 10 and 10^3 (his other weights are p^3 and 10^5). His p^3 = 10^3 at p = 10 repeats
  his 10^3 row: matched once. His `solves` is our returned iterate (the rule's step + 1), as in item 4.
  Ratios ours / his are given; no verdict is drawn (the margin is "one figure"). Buckley-Leverett at
  weight 10 differs by up to 2.2 times: his script splits the weight as initial lambda, sides lambda / 2,
  while Package 3's split, which item 1 keeps, is by |line| / |dOmega| (the advisor's reply, answer 3:
  report his values and the check, no rerun).
- **Beltrami beside the delivered rows** (Section 3: errors of u, v, w and p in the pin gauge, their
  ratios to the delivered delta_P of each field, the rule step and the stall level). The ratios use the
  DELIVERED delta_P (Package 5's recomputation differs by about 1e-4 in delta_P_p, round-off). p is given
  in the pin gauge, as asked, and gauge-corrected, as in the delivered `error_over_delta_P_p`. The errors
  are at the delivered convention's returned iterate (the 1e-9 criterion's, else the last), with the
  combined error at the rule's iterate beside them. The delivered table has no stall level: it is read
  from the delivered log at the rule's step, as for the new rows. The stall level is the weighted
  ||R_lin||, so it grows with the weight and is not compared across weights.

---

## 2026-11-14 -- Package 5, items 1, 2 and 4: the advisor's reply to the cost check, and the Grace jobs (`scripts/cluster/package5/`)

**The reply** (`Computational Package 5/Package5_Reply_to_cost_check.md`, 14 November 2026). The runs as implemented match the intent; submit them with three answers. About 135 SU expected, up to about 170 if Beltrami is slower on 24 cores; both approved. All other choices accepted.
1. **The fine reference: p = 88 with 80.** The viscous tensor Chebyshev P = 1,024 ratio must be reported and settled. Report both cases' L2 difference from the finite-difference reference and the H^{2,1} difference between 88 and 80 (`references.json`, `l2_vs_fd` and `h21_diff_neighbour`). **Fallback:** if the p = 88 solve fails or does not stop by the rule, use p = 64 with 56, apply the factor-of-three rule, and say which runs are left out.
2. **Item 2's L-BFGS times: one logged pass.** The least-squares solvers and LiL-Q keep the protocol.
3. **The weight split: the Package 3 rule** (no rerun); his indicative values and our check go beside our results.

**How the fallback is implemented** (`p5_1_weights`; the user approved per case and the 60-minute limit, 14 November 2026):
- **Per case.** Each case has its own reference, so a failure in one does not discard the other's p = 88. `choose` writes `bl_reference/chosen_<case>.json` with each case's size and the reason (one file per case, written whole: the two cases' jobs run side by side), and every later stage (`Context`, `summarize`, `run_all`) reads each case's size from it. On Grace, item 1's runs stop with the reason if a case has no usable reference; the summary records that case and goes on with the others.
- **"Fails or does not stop by the rule":** no `run.json` (the job stopped the solve at its time limit, or it failed), or a rule step of None (60 iterations without the rule).
- **Time limits:** p = 88 has 60 min (expected 25 to 32 min). Its neighbour p = 80 has 40 min (expected 15 to 18 min). An iteration at p = 88 costs about 4.5 min, so K_max = 60 without the rule would take about 4.5 h.
- **The neighbour gets the same treatment.** If p = 80 does not stop by the rule, p = 88 is not run and the case falls back: the pair needs both. This goes slightly beyond the reply's wording, which names p = 88 only, and the report says so.
- **The new stages:**
  - `ref-ok` (exit 0 if a reference stopped by the rule);
  - `choose` (keeps the pair, or runs and checks 64 and 56);
  - `runs` (the families without running a reference or writing the summary, so six processes can run side by side).

**Item 2** (`p5_2_nlls.ONE_PASS`): `lbfgs` and `lbfgs_scipy` take the time of their one logged pass; the others keep `p4_common.timed`.

**Item 4:** `rf --sigma` splits 4b among processes. `compare-points` records Grace's scipy against the shipped 16,000 points: a record only, as in Package 4.

**The jobs** (`_generate_jobs.py`; 330 SU requested, about 145 expected):
- **Classes.**
  - A new `cpu-mem` class (24 cores, 96 GB) for everything untimed. The p = 88 reference needs about 38 GB: measured 2.5 GB at p = 40 and 7.0 GB at p = 56, fitted as a + b p^4. Beltrami B2 needs about 18 GB.
  - Items 1 and 4 run six processes at 4 threads each.
  - 96 GB is under 24 cores' share of a node (180 GB), so the class should bill as 24 cores. The preflight (itself `cpu-mem`) reads `billing=` from `scontrol` and stops everything if it is above 24.
  - `wave_report.su_rate` now takes `billing` when it exceeds the cores, so the SU record would show such a charge.
- **Jobs and dependencies:**
  - Both references come after the preflight.
  - The item 1 sweep runs after the preflight and after both references, whatever their outcome. A failed case's families then fail, and Bratu and Burgers still run.
  - Beltrami, the timed job (4a's timings, then item 2) and item 4 each run after the preflight.
  - The report runs after all of them (afterany): summaries, sacct, `su_per_job.csv`, tarball.
- **Inputs.**
  - Item 1 reads Package 2 stage 2's references: P2-16's, and P3-2's finite-difference ones.
  - Item 2 reads stage 1's Cole-Hopf, as Package 4 did.
  - The checks C1 to C3 and the comparisons with the advisor's expected values run at the assembly.

---

## 2026-11-14 -- Package 5, items 1, 2 and 4: the checks against the delivered runs (`experiments/p5_checks.py`)

Reading 7 of the plan entry, made concrete. Each check compares a Package 5 run from Grace with a delivered Grace run of the same configuration. The checks are evaluated at the assembly on the laptop, where the delivered results are. `checks.json` holds the rows and the verdicts.

- **C1: item 1 at weight 10 against Packages 2 and 3.**
  - The pairs: Bratu and Burgers against P2-16 at N/P = 10 (Package 2, stage 2); Buckley-Leverett against P3-2a and P3-2b at N/P = 10 with the delivered guess.
  - Compared up to the delivered run's first stall: ||N|| and the L2 error at every iterate, and the linearized residual of every step before the stall step.
  - The linearized residual's difference is taken relative to its step's ||f||. Burgers' first step solves its system to round-off (||r|| about 1e-15), where a relative difference means nothing: 60% between the laptop and Grace at P = 625.
  - Tolerance 1e-8: far above round-off, far below the figures any table quotes.
  - The rule step and the stall level are recorded beside the delivered ones, with no verdict. After the stall the values are round-off, as in Package 4's M5.
- **C2: item 2's LiL-Q against Package 4** (1a: Burgers p = 10, 18, 26, 30; 2a: Poisson-Boltzmann seed 0, p = 10, 18, 22).
  - The solves must equal Package 4's returned iterate.
  - The errors must agree to 5e-3 relative, the same cutoff as C3, set by the two figures to which the ratios are quoted.
  - ||N|| is recorded.
  - I first set 1e-6. The laptop run gives 5e-5 in L2 at Poisson-Boltzmann p = 22: there the space is rank deficient and the error at the stall carries round-off in its fourth figure, as delta_P does (Package 4). So 1e-6 did not separate a code change from round-off. The cutoff now comes from the quoted figures, not from that value.
- **C3: delta_P of Package 4 by gelsy.** Item 4's delta_P for p = 10 to 30 against Package 4's, to 5e-3 relative.

**On the laptop data** (laptop runs against the delivered Grace runs):
- C1: worst 3.1e-12 (Bratu, Burgers) and 9.1e-13 (Buckley-Leverett, 16 runs); every rule step equal.
- C2: Burgers within 1e-10; Poisson-Boltzmann within 5e-5; solves equal.
- C3: within 1.4e-3 (L2, p = 30).
- All three pass.

---

## 2026-11-14 -- Package 5, items 1, 2 and 4: the cost, firmed before the Grace runs; three questions to the advisor

**This entry replaces the cost figures of the four entries below** (item 1, its fine reference, item 2 and item 4). Nothing here changes a computed result.

**Job sizes, from the Package 3 and 4 Grace logs** (median gelsy rate per iterate, about 2NP^2 flops per solve):
- one process at 24 threads: 19 to 43 GFLOP/s (P = 484 to 900, N/P about 8 to 10);
- one process at 48 threads: 19 to 40 GFLOP/s (P = 576 to 7,984; Beltrami B2 level 1: 36);
- six processes at 4 threads each in one 24-core job (Package 3, Buckley-Leverett): 10 to 16 GFLOP/s each, about 1.7 times the throughput of one 24-thread process.

So a whole node is no faster than 24 cores for these solves. The untimed runs (item 1, item 4's logged runs, delta_P) go on 24-core jobs, several runs to a job where there are many small ones. The thread count changes only the order of the arithmetic, so results change only by round-off. The timed runs (item 2, item 4a's timings) keep a whole node, as Section 2 asks.

The laptop (a hybrid CPU with performance and efficiency cores) is no guide to thread scaling: there, gelsy stops gaining at about 8 threads. Only its single-process times are used, as the Grace 24-core time.

**Expected SU:**

| Part | As planned | With the cheaper choices of questions 1 and 2 |
|---|---|---|
| Item 1, the six Beltrami runs (24 cores; Package 3's B1 and B2 level-1 times, 398 s and 1,301 s per run) | 35 | 35 |
| Item 1, Bratu, Burgers and the Buckley-Leverett weight runs (packed) | 12 | 12 |
| Item 1, the fine Buckley-Leverett references (laptop ladder to p = 64, scaled as p^6) | 35 (p = 88, 80) | 6 (p = 64, 56) |
| Item 2 (one whole node; laptop pass 913 s, of which L-BFGS 835 s) | 50 (full protocol) | 15 (one pass for the L-BFGS solvers) |
| Item 4 (packed), with the 4a timings (about 10) | 37 | 37 |
| **Total** | **about 170** | **about 105** |

**Questions to the advisor** (the memo `Package5_items124_cost_check.md`, sent with these commits), to settle before any SU are spent:
1. **The fine Buckley-Leverett reference:** p = 88 with 80, or p = 64 with 56.
   - With p = 64, Section 3's factor-of-three rule leaves out the H^{2,1} ratios of viscous tensor Chebyshev at P = 1,024 (all four weights). That ratio is still moving with the reference: 1.61, 1.47, 1.38 at weight 10^6 against references at p = 48, 56, 64. The other ratios move by less than 1%.
2. **Item 2's L-BFGS times:** may they come from one logged pass, as Section 6 allows for item 3's optimizer runs?
   - In Package 4's 16 timed Grace runs, the three timings typically differed by 0.8% (the median spread), but one differed by 23%.
3. **The Buckley-Leverett weight split:** the Package 3 rule (run as planned) or the split of `ref_weight_bl.py`. No cost either way.

Until he answers, the plan stands as decided in the entries below.

---

## 2026-11-13 -- Package 5, item 4: the irregular domain (`experiments/p5_4_pb.py`)

The user approved these readings before the code.
- **More Halton points** (forced by the sizes, not stated): 4a at p = 42 needs 8P = 14,112 interior points per seed, and 4b at P = 1,600 needs 12,800. Package 4 shipped 8,000.
  - The `points` stage drew 16,000 per seed on the laptop (`pb.halton`, scipy 1.17.0) into `data/package5`.
  - Its first 8,000 are Package 4's `data/package4` points bit for bit (checked when drawn, and by a test), so every Package 4 set is unchanged.
- **4a** is Package 4's own `run_lilq`, `run_under` and `run_timing`, at p = 34, 38, 42 with seeds 0 to 2. Their new `item` argument names the results folder; its default is unchanged. Everything else is as delivered: 2a logged with K_max = 60 and every iterate stored; the timing by the paper's protocol (Section 2); 2c with N = P/2 rows and Newton steps by gelsd.
- **4b:** `pb.RandomFeatures(P, sigma, draw)`, drawn as `ref_rf_pb.py` does (test).
  - `pb.System`, `pb.lilq_system` and `pb.Norms` accept such a space in place of p. With an int p the behaviour is unchanged (test).
  - Boundary points: 12 round(sqrt(P)), his script's reading of "about 12 sqrt(P)".
  - Logged as 2a: K_max = 60, every iterate stored, the rule read from the log. Untimed.
- **delta_P two ways:** `pb.Norms.best(space, driver)`, gelsy (the default, as delivered) or gelsd, both at rcond = eps. It is computed for p = 10..30 of Package 4, p = 34, 38, 42, and the 45 spaces of 4b. The record has both values and |gelsy - gelsd| / gelsd.
- **Every iterate (Section 2)** is stored for 4a and 4b (`betas.npy`), with `beta_returned.npy`.
- `p4_2_pb_inclusions.run_row` is now `rule_row` (the values at the rule) plus the run's identity. 4b reads `rule_row`, and Package 4's rows are unchanged.

**Laptop run** (16 threads):
- **4a seed 0 against Section 5's indicative values**, all within the stated factor of two:

  | p | rule step | stall level | H2 error | rank |
  |---|---|---|---|---|
  | 34 | 4 | 4.14e-8 | 1.67e-8 | 1,099 |
  | 38 | 4 | 6.59e-10 | 1.88e-10 | 1,347 |
  | 42 | 5 | 1.52e-11 | 5.01e-12 | 1,599 |

  His values: 4.1e-8 / 6.6e-10 / 1.5e-11; 1.7e-8 / 1.8e-10 / 5.0e-12; about 1,098 / 1,340 / 1,594. His text gives rule step 4; his own script, rerun in Batch 0, gives 4, 4, 5. H2 ratios to delta_H2 (gelsy): 3.6, 2.8, 3.2.
- **2c at p = 34 to 42:** Newton stops at the 30-step budget each time, with H2 errors of 380 to 1,000.
- **4b against `expected_rf_pb.json`** (24 runs: P up to 800, draws 0 and 1):
  - Ranks are identical. Stall levels agree to 3 figures, H2 ratios to 2, delta_H2 to 3e-3 or better.
  - The step count agrees in 22 of 24 runs. The two others are at P = 800, sigma = 2, where the stall is near round-off and chi stays within 10% of tau_chi on the plateau: draw 0 gives 5 solves (his 6), draw 1 gives 20 (his 10). In Batch 0 his own script on the laptop gave 5 and 7 there. Reported beside his, as Section 7 asks.
- **4b at P = 1,600, sigma = 2, draw 0 (one laptop run):** the rule does not fire within 60 iterations.
  - The iteration reaches its plateau at step 2: ||r|| about 4.8e-11, H2 error 2.7e-11, rank 968 of 1,600.
  - On the plateau chi stays between 0.21 and 0.25, above tau_chi = 0.1. The run is reported as logged (the last iterate), as in Package 4.
  - Its H2 ratio is 37 against delta_H2 by gelsy (7.4e-13), or 9 against gelsd (3.0e-12).
- **delta_P two ways** (|gelsy - gelsd| / gelsd):
  - **Well-conditioned spaces: round-off.** Chebyshev p <= 18: up to 2.5e-10 in H2, 3e-8 in L2. Random features P <= 200: up to 1e-8. H2 at P = 400, sigma = 4 and 8: up to 1e-10.
  - **The difference grows with the rank deficiency.**
    - Chebyshev H2: 1e-5 (p = 22) to 2.8e-3 (p = 42). L2 at p = 42: 8.5e-2.
    - Random features H2: 6e-3 and 3e-2 at P = 400, sigma = 2; 0.18 to 0.86 at P = 800 for sigma = 2 and 4; 0.76 at P = 1,600.
    - Random features L2: from 1e-2 at P = 400 to 0.93 at P = 1,600.
  - Where the two differ by more than 1e-3, gelsd's delta is the larger in every case but one, Chebyshev L2 at p = 38 (2.3e-3).
- **Cost** (laptop times, which match a 24-core Grace job for gelsy):
  - 4b: 770 s for the 36 runs up to P = 800, and 314 s per P = 1,600 run, about 1 h in all.
  - 4a logged: 691 s per seed, 35 min for three. 2c: 1.5 min.
  - delta_P: about 17 min.
  - About 2 h of 24-core jobs (about 50 SU), plus the 4a timings in one timed-cpu job of 0.5 h (24 SU requested, about 10 expected).

---

## 2026-11-13 -- Package 5, item 2: LiL-N by nonlinear least squares (`experiments/p5_2_nlls.py`)

The user approved these readings before the code. Where Section 4 is silent, the advisor's `ref_nlls_liln.py` fills the gap; none of them departs from his text.
- **Problems:** Package 4's own classes, unchanged. Burgers is `p4_1_burgers_gauss.Rows` at p = 10, 18, 26, 30. Poisson-Boltzmann is `p4_2_pb_inclusions.lilq_set`, seed 0, on the shipped Halton points, at p = 10, 18, 22.
- **The error.** Burgers: the relative L2 error on the 201 x 201 grid, against Package 4's delta_P. Poisson-Boltzmann: the H2 error against delta_H2 (his script's), with L2 beside it.
- **least_squares:** `trf`, `lm`, `dogbox`, xtol = ftol = gtol = 1e-12. The analytic Jacobian is A(beta), the linearized collocation matrix (test: equal to the centred differences of N). At most 200 evaluations, from his script; his runs need at most 51.
- **L-BFGS:** Batch 0's reading 4. The four-method runs' `solve_lil_n` runs with `stall_rule='f1'` (its tolerances are 0), no target, and 2,000 iterations, on ||N||^2 with the gradient 2 A^T N. His scipy `L-BFGS-B` (maxcor 100, ftol = gtol = 0, 2,000 iterations) runs beside it as `lbfgs_scipy`.
- **LiL-Q:** Algorithm 1 runs to the rule and returns the iterate after the rule's step, as in Package 4's timings. It is compared with the delivered Package 4 runs (Batch 5).
- **Counts and status:** `nfev` / `njev` for least_squares; iterations and evaluations for L-BFGS; solves for LiL-Q. Each solver's status is recorded as returned (Section 7).
- **Timing:** the protocol of the paper, which Section 2 sets for every timing "unless stated otherwise" (dedicated node, warm-up, median of three: `p4_common.timed`). It applies to all six solvers, in one job on one whole node at 48 threads. The user confirmed the full protocol for the L-BFGS runs too, after weighing a single timing (about 23 SU less).
- **Every iterate (Section 2):** LiL-Q's coefficient run stores beta^(0..K) in `betas.npy` (`iterate_to_rule(..., on_iterate=...)`); the timed runs do not.
  - The clock runs from zero coefficients to the solver's return. The collocation set and the basis matrices are built before it, as in his script; the errors are evaluated after it.
  - The run that gives the coefficients serves as the warm-up (`p4_common.timed(fn, warmup=...)`; the default is unchanged), so each solver runs four times.
  - **Cost, corrected after the laptop run:** one pass of all 42 runs takes 913 s on the laptop (8 threads). The PyTorch L-BFGS takes 625 s of that and the scipy L-BFGS-B 210 s: both run to the 2,000-iteration cap where they do not converge, and every evaluation assembles A. Four passes take about 61 min. That is one timed-cpu job of about 1.5 h: 72 SU requested, about 50 expected, not the 24 / 10 first estimated.

**Laptop run** (one pass each, no timing protocol):
- **The three least-squares solvers reach LiL-Q's point** at every size: ||N|| equal to 1e-14 relative. The error and the ratio match LiL-Q's to every printed figure: Burgers 2.01, 1.04, 1.00, 1.00 x delta_P at p = 10, 18, 26, 30; Poisson-Boltzmann H2 1.45, 1.81, 2.32 x delta_H2 at p = 10, 18, 22.
- **Against `expected_nlls_liln.json`**, the errors agree to 4 figures. Evaluation counts are the same, or differ by one or two (Poisson-Boltzmann `trf` at p = 10: 16 / 14 against his 21 / 13). LiL-Q's solves are identical: 5, 6, 6 on Burgers and 3, 4 on Poisson-Boltzmann.
- **scipy L-BFGS-B, Burgers:** it reaches LiL-Q's point in 78, 287, 620 iterations at p = 10, 18, 26 (his 77, 274, 594), and in 841 at p = 30. On Poisson-Boltzmann it stops at the cap at every size, as in his runs: H2 1.42, 1.67, 6.1 x delta_H2. At p = 10 and 18 that is slightly below LiL-Q's error, with ||N|| above LiL-Q's (2.706 against 2.688; 0.054 against 0.0469).
- **The four-method PyTorch L-BFGS, Burgers:** at p = 10 it stops on the f1 stall at LiL-Q's point, after 132 iterations. At p = 18 it reaches that point by the cap. At p = 26 and 30 it ends at the cap short of it: error 1.27 and 36.6 x delta_P, ||N|| 4.91e-6 against 4.15e-6 at p = 30. On Poisson-Boltzmann it ends at the cap close to the scipy values.

---

## 2026-11-13 -- Package 5, item 1, Beltrami: `--lambda-bc` / `--lambda-ic` in `p3_1_beltrami_certified.py`

- `config_for(size, lambda_bc=None, lambda_ic=None)` and `run(..., lambda_bc=None, lambda_ic=None)`. None keeps the delivered 10, so the delivered configuration is unchanged (test: same config, same row weights).
- The faces carry `lambda_bc`, the initial slab `lambda_ic`, each with the Package 3 shares (4/40, 8/40), as delivered. The interior rows are unchanged.
- **The pins**, read literally from Section 3 (`pin_scale` = 1): their squared weight stays `pin_scale x lambda_bc / N_p`, so they follow the new `lambda_bc`. They fix the pressure constant only.
- `run.json` records `lambda_bc` and `lambda_ic`. `terminal.csv` gains `lambda_bc`, `lambda_ic` and `stall_level` (||r|| at the rule's step, as in Package 4); delivered runs read as 10.
- Runs: B1 and B2 at level 1, with `lambda_bc = lambda_ic` in {1e3, 1e4, 1e6}, tagged `_lam1e3`, `_lam1e4`, `_lam1e6`, with eight iterates as delivered. Errors stay in L2 against the delivered delta_P.

---

## 2026-11-13 -- Package 5, item 1: the weight option, the norm of X, the fine Buckley-Leverett reference (`experiments/p5_1_weights.py`, `lilq/xnorm.py`, `lilq/references.cole_hopf_derivatives`)

**New options, old defaults.** `p2_16_certified.grid(..., lam_aux=LAM_AUX)` and `p3_2_bl_certified.collocation_blocks(..., lam_aux=LAM_AUX)` take the weight of the boundary and initial lines; at the default every array is bit-identical (tests). `p2_16_certified.tensor` also returns the mixed derivative (1, 1), which no existing run reads. On the laptop, item 1 at lam = 10 reproduces the delivered P2-16 runs (Bratu P = 100, Burgers P = 225, Grace stage 2) to 1e-13 in ||N||, ||r|| and eps_ref at every iterate; chi on the plateau differs by up to 18%, a ratio of round-off-sized differences, as in Package 4.

**The run** is Package 4's `logged_run`: termination disabled, K_max = 60, the coefficients of every iterate in `betas.npy`, the scaled identity of M3. The termination rule is read from the log.

**The norm of X** (`lilq.xnorm`): a tensor Gauss-Legendre rule, 96 x 96 (Buckley-Leverett: max(96, 2 p_ref) = 176), components weighted equally.
- Bratu: H2 of the p = 48 reference, from the Chebyshev coefficients stored in `bratu_ref_p48.npz` (they reproduce its field to 1e-14).
- Burgers: H^{2,1} by `cole_hopf_derivatives`, which differentiates the Cole-Hopf integrals (u_t from the heat kernel, not from the PDE). Section 3's check u_t + u u_x - nu u_xx = 0 holds to 5e-13 relative to the terms at the nodes (tolerance 1e-8); `references.json` records it.
- Buckley-Leverett: H^{2,1} of the fine LiL-Q solution below.

**The fine Buckley-Leverett reference.** The lifted sine x Chebyshev space at p_ref, weight 10^4, N/P = 10, the Package 3 guess of the case, stopped at the rule's returned iterate.
- **p_ref = 88 (P = 7,744), neighbour p = 80: the user's reading of "the largest size you can afford".** The cost is set by compute, not memory: gelsy runs at about 35 GFLOP/s whether on 16 laptop threads or a whole Grace node (Package 3, Beltrami B2), and a solve grows like p^6. Estimated, as 24-core jobs: p = 64 with 56, about 7 SU; 80 with 88, about 35 SU; 88 with 96, about 60 SU; the memory limit of a node (p about 150), about 1,000 SU or more. The neighbouring size is Section 3's: the H^{2,1} difference between two fine solutions.
- **The laptop ladder** (p = 24 to 64): the L2 difference from the finite-difference reference falls to 8.6e-6 (viscous) and 4.7e-5 (gravity) at p = 64; the H^{2,1} difference from p - 8 is 5.7e-3 (viscous) and 3.8e-2 (gravity), decreasing about like p^-2. Every item-1 run's delta_X exceeds three times the p = 64 figure except viscous tensor Chebyshev at P = 1,024 (delta_X = 0.014), so p = 88 should report every run.
- **Sensitivity:** recomputed against references at p = 48, 56 and 64, the reported H^{2,1} ratios change by less than 1% except at viscous tensor Chebyshev P = 1,024, where they still move by a decreasing amount (weight 10^6: 1.61, 1.47, 1.38). Between references at 88 and 96 the change there is estimated at about 2%.
- **The gate** is Section 3's: an H^{2,1} ratio is reported only where delta_X exceeds three times the neighbour difference (`x_ratio_reported`); the L2 ratio is always reported.

**The laptop runs** (all 96 scalar runs of item 1; Buckley-Leverett against a p = 64 reference):
- Bratu agrees with the advisor's indicative values (Gauss-Legendre grids) to about two figures, e.g. L2 / H2 at weight 10: 71.8 / 1.83, 16.1 / 2.63, 15.0 / 2.64 at p = 5, 10, 15 (his 74 / 1.8, 17 / 2.7, 15 / 2.9).
- Burgers: H^{2,1} ratios 1.01 to 1.13 at every size and weight; L2 ratios 1.3 to 5.
- Buckley-Leverett, viscous, tensor Chebyshev: at weight 10^3 the ratios agree with his to about two figures (4.87 / 1.11 against 4.81 / 1.11 at p = 8). At weight 10 they are about twice his in L2 (45 against 28 at p = 16). The cause is the split of the weight among the lines:
  - Our runs use the Package 3 rule his item 1 asks for ("unchanged except for the weight"): each line's squared weight is lambda x CC weights x |line| / |dOmega|, |dOmega| = 2.8. The initial line therefore gets lambda / 2.8 and each side line 0.4 lambda / 2.8.
  - His `ref_weight_bl.py` gives the initial line lambda and each side line lambda / 2.
  - **Laptop check** (scratch script; the same code with only the shares replaced by his): at weight 10, L2 / H^{2,1} ratios 8.97 / 1.30, 27.1 / 1.60, 32.0 / 1.33, 19.0 / 1.42 at p = 8, 16, 24, 32, against his 9.1 / 1.3, 27.6 / 1.59, 32.5 / 1.3, 19.3 / 1.34. At weight 10^3: 4.75 / 1.11, 10.8 / 1.15, 10.2 / 1.06, 5.8 / 1.15. The split accounts for the difference.
  - The report gives our runs (the Package 3 rule), his values beside them, and this check. No change to the runs.
- The laptop needed 42 minutes for the 64 Buckley-Leverett runs on 16 threads.

---

## 2026-11-13 -- Package 5, the plan: items 1, 2 and 4 before item 3; the readings of the instructions

The advisor's Package 5 instructions (13 November 2026) have four items: 1, the errors in the norm of X and the boundary weight; 2, LiL-N by nonlinear least squares; 4, the irregular domain at larger sizes and with random features; 3, the cost at a common accuracy. Items 1, 2 and 4 come first and get their own report. Item 3 waits for the advisor's reply to that report, which proposes the Bratu boundary weight item 3 uses.

**His reply to Package 4** (Section 1 of the instructions):
- M5 is judged on the stall levels;
- M3 is judged by the scaled identity, with the plain difference kept in the record;
- the manuscript quotes the ratios to delta_P to two figures;
- the manuscript no longer uses the class of a stall (A or C). The `class` columns stay in the files; no new decision is based on them.

**The advisor's scripts on the laptop** (numpy 2.4.2, scipy 1.17.0), before any code:
- `ref_weight_bratu.py` and `ref_weight_bl.py` reproduce `expected_weight_bratu.json` and `expected_weight_bl.json` to 4e-11 relative;
- `ref_nlls_liln.py`: the errors and residuals of the three least-squares solvers and of LiL-Q agree with `expected_nlls_liln.json` to three figures. The evaluation counts differ by one in places, and the L-BFGS iteration counts by up to 30% (round-off in a long line-search iteration);
- `ref_rf_pb.py` reproduces `expected_rf_pb.json` except at P = 800, sigma = 2, where the space is rank deficient (rank 693 to 695 of 800): 5 and 7 solves against 6 and 10, errors within 1.2% (H2) and 20% (L2);
- `ref_pb_large.py`, seed 0: stall levels 4.1e-8, 6.6e-10, 1.5e-11, H2 errors 1.7e-8, 1.8e-10, 5.0e-12, ranks 1,099, 1,345, 1,598 at p = 34, 38, 42, as Section 5 states; the rule fires at step 4, 4 and 5 (Section 5: 4). delta_L2 at p = 42 is 6e-15, at the level of round-off.
- Section 5's summary of item 4b ("3 to 5 solves", "1.1 to 4.5 times the best approximation") leaves out P = 800, sigma = 2, where his own file has 6 and 10 solves and ratios 13.6 and 8.3. The script decides; the report says so.

**Readings of the instructions (decided here, stated in the report):**
1. **Item 1 is untimed.** The configurations it repeats (P2-16, P3-2, P3-1) were untimed, and Section 3's per-run report has no time; items 2 and 4a ask for times explicitly. Timing the Beltrami runs by the protocol (warm-up and three timings to the returned iterate) would add about 180 SU (Package 3's logs: about 25 s per iterate at B1 level 1, 157 s at B2 level 1, whole node).
2. **The weights of item 1 are those of Section 3:** 10, 10^3, 10^4, 10^6 (Beltrami: 10^3, 10^4, 10^6 beside the delivered 10). The advisor's scripts also try weights that grow with p (p^3, p^4); they are not in the instructions and are not run.
3. **Where the errors are measured.** The L2 error and delta_L2 stay as delivered (the paper's 201 x 201 evaluation grids), so they are comparable with the Package 2 and 3 rows. The norm of X (H2 for Bratu; H^{2,1} = u, u_x, u_xx, u_t for Burgers and Buckley-Leverett) is integrated by a tensor Gauss-Legendre quadrature, as in the advisor's scripts, with the components weighted equally. delta in the norm of X is the least-squares distance in that norm (all components stacked), as `best` in his scripts.
4. **Item 2's L-BFGS** is the four-method runs' (`lilq.solvers.solve_lil_n`: PyTorch L-BFGS, history 100, strong Wolfe line search), with the tolerances zero and 2,000 iterations. The advisor's scipy L-BFGS-B (maxcor 100) is run beside it, since his expected values come from it.
5. **Beltrami's pressure pins:** `pin_scale` = 1, read literally. The pins' squared weight is `lambda_bc / N_p`, so they follow the new boundary weight. They fix the pressure constant only; Package 3's check K3 showed the errors do not depend on their scale (1e-3 to 1e3).
6. **Runs without expected values** (item 2 at Burgers p = 30 and Poisson-Boltzmann p = 22; item 4b's third draw and P = 1,600; item 1 outside the advisor's sizes) are reported as they come out.
7. **Checks.** Section 6 defines checks for item 3 only. For items 1, 2 and 4: every changed driver at its old settings reproduces the delivered rows (the new options default to the old behaviour), and the LiL-Q runs that repeat Package 4 reproduce its values.
