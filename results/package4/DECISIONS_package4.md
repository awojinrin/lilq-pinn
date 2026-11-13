# DECISIONS.md: the Package 4 entries

Copied from the repository's `DECISIONS.md` at the assembly's commit (e949cc9), newest first.

---

## 2026-11-13 -- Package 4, the assembly: Grace's tree, the checks with the advisor's files, the Section 7 records (`experiments/p4_assemble.py`, `experiments/assemble_common.py`)

The *assembly* turns the extracted `package4.tar.gz` of the report job into the results folder of Section 7. It changes no computed number of a run.

**One shared module for every package's assembly.** `experiments/assemble_common.py` holds what Package 3's and Package 4's assemblies share:
- the commit check;
- copying Grace's tree;
- LF writing;
- `su.csv`;
- the per-job table of `hardware.json`;
- the BLAS, preflight and module lines of `environment.txt`;
- the package's `DECISIONS.md` entries;
- `provenance.json`.

`p3_assemble.py` now imports them. Its output is unchanged: `tests/test_p3_assemble.py` passes as before. The published `package3-results` branch records the commit that assembled it.

**The checks with the advisor's files.** The cluster had neither `expected_burgers.json` nor `expected_pb.json`, so it could not evaluate M1, M8, or M6's values at the five points. The assembly reruns both items' summaries on Grace's files with those files, then `p4_checks collect`.
- Grace's `checks.json`, `checks_item1.json` and `checks_item2.json` are kept as `*_grace.json`.
- Every verdict Grace gave must come out the same, or the assembly stops. A check Grace could not evaluate (M6's `null`) may take any verdict.
- The advisor's two files are not copied. `provenance.json` records their SHA-256.

**The tables stay Grace's.** The summaries also rewrite `terminal.csv` and the other tables from Grace's logs. The assembly compares them with Grace's and puts Grace's bytes back. Every column must agree within 1e-8 relative, except the columns of delta_P (the error of the best approximation in the basis) and the ratios to it. On Grace's tree, every other column agreed exactly.

**delta_P of item 2 is reproducible to about three digits at P >= 484.** delta_P is a least-squares projection (gelsy, rcond = eps) onto the basis restricted to the domain with holes, which is ill-conditioned. Its round-off reaches the fourth digit at the larger sizes. Grace, the laptop and the advisor's `expected_pb.json` ("best") give three values:

| P | delta_L2 spread (Grace, laptop, advisor) | delta_H2 spread |
|---|---|---|
| 100 | 2e-14 | 4e-15 |
| 196 | 2e-11 | 9e-13 |
| 324 | 3e-8 | 2e-10 |
| 484 | 9e-5 | 3e-7 |
| 676 | 9e-4 | 5e-6 |
| 900 | 1.4e-3 (Grace and laptop; the advisor gives no value) | 8.5e-5 |

- The assembly records the spread in `delta_P_across_machines.json` instead of stopping on it.
- The report gives the ratios to delta_P at p >= 22 to three significant figures.
- Item 1's delta_P (Burgers) agrees to 4e-11.

**The report job is not in Grace's `su_per_job.csv`.** It ran after the table was written, so its row comes from `sacct` after it ended (`--sacct-extra`), as for Package 3.

**M9 fails as Grace found:** 1b, E1 at tolerance 1e-5, a 22.6% spread of the three timings. Section 6.2's rule applies: the report gives all three timings and the median. It is a deviation recorded in the report, not a failure of the assembly; `p4_assemble.py` exits 0 once the folder is complete.

---

## 2026-11-12 -- Package 4: the Grace submission (`scripts/cluster/package4/`, `data/package4/`)

The p4 stage in `env.sh`, `sbatch.sh`, `submit_lib.sh` and `package2/stage_report.py` (`results/package4`, one commit, one lock), as for Package 3.

**The jobs** (`_generate_jobs.py` writes all six; `submit_p4.sh` submits them).

| Job | Class | Walltime | What | Laptop |
|---|---|---|---|---|
| `p4_preflight` | cpu | 0:30 | versions, inputs, the test suite, the points folder, `compare-points` | |
| `p4_timed` | timed-cpu | 0:20 | M0; the timings of 1a, 1b and 2a | about 5 min |
| `p4_item1` | cpu | 0:40 | 1a logged, 1b, 1c | 21 min |
| `p4_item2` | cpu | 0:20 | 2a logged, 2b, 2c, 2d, 2e, M7, the solution field | 11 min |
| `p4_item2_constants` | cpu | 1:15 | 2f, after `p4_item2` | 66 min one run after another; about 40 min side by side |
| `p4_report` | cpu | 0:15 | summaries, `checks.json`, sacct, the tarball | |

- **88 SU requested** (walltime x rate: 24 SU/h for cpu, 48 SU/h for timed-cpu), under the cap of 100 even if every job ran to its walltime. About 40 expected.
- Walltimes are about 1.5 times the laptop's stage times. Package 3's runs of the same constants recipe took 0.7 to 1.5 times the laptop's time on Grace.
- **M0 and every timing run on one whole node** at 48 threads, as `package1`'s and as Section 2.3 asks. That node's `hardware.json` and `environment.txt` go to `provenance/timed/` (`p4_checks.py provenance`).

**2f runs the three seeds side by side** (one process per seed, 8 BLAS threads each, 24 in all). Each process goes from p = 10 up, and p = 30 is last. Seed 0's process also takes the two extreme 2d sets, before its p = 30.
- **Measured on the laptop at p = 22:** the three seeds side by side took 321 s, against 567 s one after another at 24 threads (1.8 times faster, the same core count).
- **The per-run constants of `constants.csv` are the same either way:** the smallest c1, the largest c2 and the largest rho_r agree within 1e-6.
- **The per-iterate c1 and c2 are not.** At p = 22 the column-pivoted QR drops 12 or 13 near-dependent columns at the 1e-10 cutoff. Which ones it drops at a plateau iterate depends on round-off, so c1 takes 14 to 19 distinct values across the iterates of one run (0.084 to 0.092 for seed 2), whose coefficients agree to round-off. A different BLAS thread count moves an iterate between these values (up to 10% in c1). This is a property of the recipe, reported under Section 6.2's rule for the constants, not a change of the recipe.
- Each p = 30 run is stopped 5 min before the walltime if it is still running (`until_end`). It keeps every iterate it wrote, since `constants_by_iterate.csv` is written after each one. Section 6.2's rule then applies: the sizes up to p = 26 stand.

**The shipped Halton points are in the repository** (`data/package4/halton_seed{0,1,2}.npz`, 129 kB each, with a README). Every Grace job reads them.
- The preflight copies them into the deliverables' `points/` folder (`points --points data/package4`).
- `compare-points` records Grace's scipy draw against them.
- **The tests that need the advisor's point sets build them from these files.** A test that drew with the local scipy would fail on Grace if scipy 1.13.1 draws other points. That would stop the preflight and cancel every job.

**Checks M1, M6 (values at the five points) and M8 are evaluated at the assembly.** They compare with `expected_burgers.json` and `expected_pb.json`, the advisor's files, which are not in the repository or on Grace. The Grace report summarizes everything else.

**Every logged run of item 2 saves `beta_returned.npy`**, the iterate the rule returns (the last one if it never fires), as Section 7 asks for 2a.

**Review before the runs** (13 November; summaries and records only, no computed number changes).
- **A check with no verdict is not a failure.** Without `expected_pb.json`, M6 has no verdict on the cluster (u\* and rho at the five points), and `checks.json` had read it as failed. It now records `pass: null` with a note, and the area's own verdict as `area_pass`. `collect` combines the items as: false if any part failed, null if any part has no verdict, true otherwise.
- **kappa at the returned iterate.** Section 2.2 asks for kappa by SVD at the iterate the rule returns and at the last iterate. The `*_at_rule` columns of `terminal.csv` are the system of the step at which the rule fires (row k_s, as the reference scripts report the rank). `kappa_retained_returned` and `kappa_raw_returned` now add the system at the returned iterate (row k_s + 1). On the laptop rehearsal the two agree to the digits shown.
- `lilq/provenance.py` and `tests/test_provenance.py` had been committed with CRLF line endings by the UTF-8 fix (35ce117); they are LF again. The bundle normalizes line endings, so no cluster file changed.

---

## 2026-11-12 -- Package 4, item 2: Poisson-Boltzmann around three inclusions (`problems/poisson_boltzmann.py`, `experiments/p4_2_pb_inclusions.py`)

The advisor's instructions of 12 November 2026, Section 4. The reference script `ref_pb_inclusions.py` is the specification. Its `halton` and `boundary` are used unchanged, so that the point sets coincide.

**rho and the derivatives of u\* are in closed form, not SymPy** (the user's choice, 12 November).
- u\* is a sum of three Gaussians, so the derivatives are short and exact, and Grace's environment needs no new package.
- Agreement with the advisor's SymPy values (M6): within 6e-16 relative at his five points.
- `tests/test_poisson_boltzmann.py` compares the closed form with a SymPy derivation where SymPy is installed (skipped otherwise) and with finite differences everywhere.

**The Halton points are drawn on the laptop and shipped** (`data/package4/halton_seed{0,1,2}.npz`, the first 8,000 points in Omega).
- The first n points of a draw are the first n of any longer draw (tested), so every set of the package is a prefix of one draw per seed.
- The laptop's scipy 1.17.0 reproduces the advisor's points (his 1.18.1): the reference script's numbers agree within 0.3% for LiL-Q and 1e-10 for underdetermined collocation.
- Grace's scipy 1.13.1 is compared with the shipped points by the `compare-points` stage. It is recorded, never used.
- Grace's numpy and scipy are not changed, which keeps check M0 safe.

**Errors.** Relative L2 and H2 errors at every iterate go to `errors_by_iterate.csv`. `eps_ref` in `iterations.csv` is the L2 error; the logger's columns are not extended.

**2f, the norm of Y.**
- The 70 x 420 rule inside, and `boundary(4000)` with its weights on the boundary.
- Each block carries its total squared weight in the collocation set: 1 inside, lambda_d on the boundary.
- `experiments.p3_2_bl_certified.section6_row` is used unchanged.

**Timed region.** The construction of the collocation matrices from the shipped points, the assemblies, the solves and the residuals of the rule, stopped where the rule fires.

**Laptop rehearsal** (numpy 2.4.2, scipy 1.17.0; p = 14 and 22, seed 0, and all of 2d and 2e).
- 2a: the rule fires at the advisor's step. Stall levels and both errors are within 0.2% (M8 passes).
- 2d and 2e: every rule step matches. The H2 error / delta ratios match within 0.5%, except at P = 484 with 1.5, 2 and 3 interior points per unknown: 134.6, 52.2 and 8.7 against 136.3, 49.7 and 8.8. These systems are the worst conditioned of the sweep.
- 2b: the square-collocation runs that diverge differ from the advisor's run by run, in status, step count and the errors of the diverged iterate. The two runs that converge are the same (P = 100, approximate Fekete points). This is round-off in a diverging Newton iteration (numpy 2.4 against his 2.5); it is reported side by side, as Section 6.2 directs.
- At P = 484 the stall at the rule is classified conditioning-limited (C). Reported.

---

## 2026-11-12 -- Package 4, item 1: Burgers on Gauss points (`experiments/p4_1_burgers_gauss.py`)

The advisor's instructions of 12 November 2026, Section 3. The reference script `ref_burgers_enrichment.py` is the specification.

**Construction.**
- The trial space is the repository's `sin_cheb` (`Fourier1D(mode='sin')` on (-1, 1) times `Chebyshev1D` on (0, 1)), assembled in tensor form with x-major coefficients.
- On the laptop (numpy 2.4.2, scipy 1.17.0), item 1a matches `expected_burgers.json` at every p: the same number of solves, and stall level, error and delta_P within 1e-11 relative (M1 at p = 10 and 22: within 2e-14 and 1e-11).
- Item 1b matches the reference script's levels within 5e-10.

**Check M3 (the identity of Remark 1) is judged by a scaled difference.**
- The measure is ||(A beta - f) - N(beta)|| / || |A||beta| + |f| ||, at every iterate, against P eps. This is the scale at which the two vectors are formed (Section 4.1 of the manuscript).
- Package 3's K4 reading, the plain relative difference at k = 1, is still recorded. At 1a's k = 1 it is at most 1e-15.
- At the levels of 1b that start from an embedded iterate, the first iterate is already on the plateau. There ||N|| is about the stall level (down to 4e-6), and the plain relative difference reaches 1.8e-10 from cancellation alone. The scaled measure is at most 1.3e-16 everywhere.

**Check M5 is judged on the stall levels.**
- The stall levels of 1b's levels agree with 1a's within 3e-8. The errors of a level's returned iterate differ from 1a's by up to 2.3e-6 (variant E1) and 1.6e-4 (E2, p = 10).
- The reason: a level returns a different step of the plateau than the single solve (E2: n_s = 1).
- The reference script gives the same differences (its own enrichment against its own single solves), so the script decides. Section 3.3's "equal those of the single solves" holds to about 3 digits. Reported, not adjusted.

**Timed region.**
- A timing pass of 1a includes the construction of the level's collocation matrices, as 1b's must ("the assembly at each new level"), so that the comparison of 1b with 1a at the final P is like for like.
- The timed run stops when the rule fires, and its number of solves is recorded and compared with the logged pass (M9).

**`lilq/provenance._run` decodes subprocess output as UTF-8.** On Windows the locale's cp1252 failed on a git diff with non-ASCII text, which made `capture_git_info` raise (found when the documentation rewrite was uncommitted). This makes no difference on Grace; there is a test.
