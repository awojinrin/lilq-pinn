# DECISIONS.md: the Package 5 entries

Copied from the repository's `DECISIONS.md` at the assembly's commit (a61de73), newest first.

---

## 2026-11-17 -- Package 5, item 3: the rerun's result, C4 on Buckley-Leverett LM, and the C4 diagnostic (stage p5e)

**The rerun (p5r)** passed the advisor's stop check: 874 SU (his line: 1,000), every job COMPLETED, no
process stopped, every process loaded the GCC module's OpenMP library. C1 18 of 18 against Package 1
(exact), 108 runs, C2 6 of 6. **C4 failed on two of six:**
- Bratu NiL-N +5.25% (within the ~7% spread the CPU set showed on Grace for that pair).
- **Buckley-Leverett LM +18.1%**: 28.17 s without the error log, 33.28 s with it, 213 iterations
  (the pilot's size on that node); the log's own 2.97 s is off the clock. In the CPU set (the wheel's
  library, 128 iterations) the same pair was +2.5%. The Buckley-Leverett NiL-N pair, the one the advisor
  asked about, timed three times as he asked: reported at the assembly with its spread.

The A100 set's LM runs log the error at every iteration, so if the off-clock log puts time on the clock
(a leak), it does so in every A100 LM run. **The user's decision (option 2):** hold the A100 set until
the cause is shown; then **"CPU + mechanism + A100"**, about 45 SU.

**The hypothesis** (not assumed; this tests it): between LM iterations the error evaluation runs off
the clock; the OpenMP threads idle through part of it, go to sleep (libgomp's default wait policy spins a
while, then sleeps), and the next on-clock step pays their wake-up, or the evaluation evicts the step's
data from cache. Either would depend on the library and the thread count.

**The diagnostic** (`experiments/p5_3_c4_diagnostic.py`; stage p5e, results/package5_item3_c4_diagnostic):
each process imports in its kind's order, records its OpenMP library and its environment, runs the
warm-up of the rerun's checks process (NiL-N, then LM), then times the Buckley-Leverett LM pair at P =
1,024 three times (`p5_3_networks.c2_c4(..., repeats=3)`, which now takes the folder to write into: one
per process). Kinds: `numpy` (the module's library, as the rerun), `torch` (the wheel's, as the CPU
set), `active` (the module's, with `OMP_WAIT_POLICY=ACTIVE` set by the job: the threads spin instead of
sleeping; every other process must run with it unset, checked before any import).
- **CPU** (p5e_cpu, timed-cpu, 1 h walltime): nine processes, three rounds, each kind once in each
  position (a Latin square, against drift over the job). Each pair is the rerun's length, 213
  iterations, so the times compare with the rerun's directly. 9 pairs per kind.
- **A100** (p5e_gpu, gpu-pair, 20 min): two processes side by side, one per A100, 24 threads each, the
  module's library: as the A100 set runs. The length from C4's pilot (about 30 s), as the A100 set's
  check would size it. 6 pairs.
- **Report** (p5e_report, afterany): the summary, then the tarball. 118 SU requested, about 60
  expected (a change from the 45 given to the user: the A100 part runs on a whole two-A100 node, 192
  SU/h, to match the A100 set's threads and pairing; one A100 with 8 cores would cost about 13 SU less
  but would not test the A100 set's layout).

**Decided before the job.** For each group, the median over its pairs of the signed relative difference
(with the log against without):
- **leaks** if the median is above 5% (C4's cutoff); **clean** if within 5%; more than 5% faster with
  the log is not foreseen and is reported to be explained.
- cpu numpy leaks, cpu torch clean: the leak depends on the library. Both leak: it leaks with either.
- cpu numpy clean: the rerun's +18% was not reproduced (reported as such, with the 9 pairs).
- cpu numpy leaks and cpu active clean: the threads going to sleep are the cause; active leaks too:
  not the sleeping threads.
- **The A100 set's gate is the A100 group alone:** both processes' 6 pairs present, each with the
  module's library and no wait policy, and clean: the A100 set goes (step 7 of the steps file).
  Leaks: stop; a fix is brought to the user before the A100 set. C2 (the log does not change the
  iterates) and the CPU groups are reported beside the gate, not part of it: C2 has not been run on
  CUDA before, and a GPU run that is not bit-for-bit repeatable would fail it without any leak.

**Differences from the rerun's process, stated:** the rerun's Buckley-Leverett checks process ran the
NiL-N pair three times (about 3 min) and LM's 10-iteration pilot before the LM pair; these processes go
from the warm-up to the LM pair. The CPU set's process (which passed) also ran NiL-N first, so that
history is common to the pass and the fail; leaving it out saves about 22 SU.

**The scan before the rehearsal** (the user's rule: before Grace, again after the rehearsals): torch's
threads are pinned through `lilq.utils` as in the rerun's process (`pin_torch`, OMP_NUM_THREADS: 48 on
the CPU node, 24 per process on the A100 node); `solve` sets float64; the kind's import order comes
first (`p5_3_openmp.start_process`), and the wait policy is checked before any import (libgomp reads it
at load). Found and changed: the summary first gated the A100 set on every group and on C2; now on the
A100 group alone (above).

**The laptop rehearsal** (code as committed in 1a88feb; BL LM at P = 1,024, the real lengths): the
torch kind 213 iterations, pairs +0.0%, -2.0%, -0.3%; the active kind (OMP_WAIT_POLICY=ACTIVE
recorded) -1.6%, +1.5%, +0.1%; CUDA (one GPU) 441 iterations from the pilot, -0.1%, -0.1%, +1.0%, **C2
passed on CUDA** (the first time it ran there). About 3.5 min per process. The summary said NO, as it
should on the laptop: Windows loads the wheel's library in either order, so the numpy kinds' library is
not the module's, and one GPU gives 3 of the 6 A100 pairs.

**The scan after the rehearsal:** a resubmitted job's process could leave an earlier attempt's
`result.json` beside its own new records if it stopped before writing its own; each process now removes
it first (tested). No CRLF in the changed files. Repo suite 1,058 passed; no-git bundle (1a88feb) 1,027;
WSL bash tests 39 of 39 in each of three modes.

**The result** (jobs 20590230/31/33, 18 November 2026; sha256 66097fcc...; 66 SU: CPU 40.9, A100 25.3).
Every process loaded its kind's library and wait policy; C2 passed in all 11, on CUDA too. As fixed
before the job: **cpu numpy clean (median -1.9%): the rerun's +18% was not reproduced**; cpu torch
clean (-1.4%); cpu active "faster with the log" (-6.1%); **A100 clean (-0.0%): the A100 set may go.**
What the numbers show (read after the verdicts, which stand as fixed):
- No leak with either library. The spread is the clock's: on the CPU node a run of the same 213
  iterations took 29.3 to 34.7 s with the module's library in this job, so the rerun's pair (28.17 s
  without, 33.28 s with) is one fast run and one ordinary one. A single pair on that node cannot
  resolve 5%: for the item 3 report.
- The wheel's library is about 65% slower on this pair (about 51 s against 31 s), as p5o found (+61%).
- The active kind's negative median comes from its first pair: in all three active processes the first
  run without the log took 41 to 51 s against about 31 s afterwards (only that kind shows it). Without
  log runs always go first, so a one-time cost lands on "without". Its other six pairs have a median
  near -1.5%. Not foreseen; reported as such.
- A100: five of six pairs within 0.3%. The +12.3% pair is a slow stretch on GPU 1 (with the log
  31.01 s), and the next run, without the log, was as slow (31.02 s): the node, not the log. It
  bears on the A100 set's pairing gate (5%), not on C4.

**The A100 set (p5g)** ran on bundle 1a88feb (jobs 20592822-25, 18 November 2026; sha256 825ab4b6...):
all four jobs COMPLETED, **1,866 SU** (expected about 2,270: preflight 5.6, Bratu 445.5, Burgers 758.5,
Buckley-Leverett 656.6). The failure, gate and library greps printed nothing. Pairing gates passed
(Buckley-Leverett +3.1%, Bratu +0.5%, Burgers +2.1%) and the report's pairing check too (+2.8%, -2.1%,
+1.8%). 108 network budget runs plus the 9 solo and gate runs, all at 1a88feb; 14 LiL-N L-BFGS runs;
24 hardware.json, all the module's OpenMP library; nothing flagged. Runs that stopped before the budget
stopped on an optimizer stall (11 of 108; the CPU rerun had 13). Package 5 so far: about 3,934 SU of
the 4,400 cap.

---

## 2026-11-17 -- Package 5, item 3: the advisor's reply to the cost check; C4's three timings; Component A's OpenMP library (stage p5k)

**The advisor's reply** (17 November 2026; `Post-JCP/Computational Package 5/Package5_Reply_to_item3_cost_check.md`):
- The rerun of all five CPU network jobs, then the A100 set. The cap of Package 5 is now 4,400 SU. If
  the rerun uses more than 1,000 SU, or a job fails, stop and tell him before the A100 set; otherwise go
  on without asking. The LiL runs, the protocol runs and their checks stay as they ran.
- Pretraining on the clock, for every method: the time to an error is from the starting data to the
  iterate, including what computes the starting point (a network's pretraining; the LiL methods'
  least-squares fit of the initial profile on Buckley-Leverett), excluding what is paid once per
  session (imports, device start-up, compilation, the warm-up run). Both columns kept.
- C4 on Buckley-Leverett NiL-N: rerun the pair; if again outside 5% either way, report both times and
  the spread of the three timings.
- For the release: the library check and the import order go in `REPRODUCE.md`, with the names and
  versions of the two OpenMP libraries.

**Pretraining on the clock: no code or runs needed.** `p5_3_table.py` already writes both columns:
`time_s` (the optimizer's clock) and `with_setup_s` (that plus `setup_s` and `pretrain_s`). Every
record's `setup_s` is per solve, measured after the warm-up, imports and device start-up (the drivers'
clocks start at the solve), so `with_setup_s` is his rule. For the LiL methods on Buckley-Leverett,
`setup_s` holds the system's construction and the initial-profile fit together (`p5_3_lil.Problem`),
not split: the report says so. CPU set values: up to 0.39 s at P = 1,024, against LiL-Q's 4.6 s
solve. The networks' `pretrain_s` is under 1 s everywhere. The assembly makes `with_setup_s` the
headline column.

**C4's three timings (the user's decision, 17 November 2026: three timings always).**
`p5_3_networks.c2_c4(..., repeats=N)` times the pair N times in all, the same length each time, and
records every timing (`timings`: both times and the signed difference, with the log against without),
their spread and how many fall outside 5%. The verdicts stay the first pair's, as fixed before.
`checks --c4-repeats nil_n=3`. In `p5r_checks` only Buckley-Leverett carries it. About 2 min more,
inside the job's walltime. Not chosen: repeating only after a failure (that reruns on a failure);
three timings for all six pairs (about 10 SU, more than asked).

**Component A's OpenMP library (the user's decision, 17 November 2026: measure it).** Package 1's
Component A processes imported torch first and loaded the wheel's library. Their CPU runs
(`A_calibration/full_cpu/`, the representatives F1_18 and F2_05, seeds 0-4, 3,600 s each) are timed
against LiL-Q's CPU runs, which loaded the module's (`kovasznay_comparison.py`). A budget run that is
slower per iteration reaches less in its hour, so the library could make the baselines look worse.
- `experiments/p5_3_openmp_kovasznay.py`, as p5o: three rounds, the two orders alternating, each
  process checking its library (`p5_3_openmp.start_process`, now shared by both drivers). Each process
  runs F1_18 (seed 0; 201 Adam iterations, then L-BFGS, 140 s) and F2_05 (seed 0; 90 s), on the CPU,
  as `component_a.run_one` runs them (`pin_torch` at `component_a`'s import; float64; `test_every=1`,
  test errors off the clock; `component_a.lm_args`, factored out of `run_one`).
- Measures: F1's seconds per Adam iteration (log rows 100 to 200), per L-BFGS iteration (second to last
  L-BFGS row), F2's seconds per evaluation (second to last row). The first row of each phase carries
  set-up and is left out.
- **Criteria, fixed before the job:** the library changes a measure if the medians over the rounds
  differ by more than 5% and the two orders' rounds do not overlap (p5o's). The torch-first runs
  reproduce Package 1's seed-0 run if the loss on every shared log row agrees within 1e-6 (C1's
  tolerance; F1 compares only its three Adam rows, F2 every LM step, so this checks the setting more
  weakly than C1 did: on the laptop the F1 rows agree with Grace's to 7.5e-13 even with another machine
  and library). The summary also gives Package 1's own values of the three measures (all five seeds,
  whole logs: Adam 0.327 s, L-BFGS 0.353 s per iteration, F2 5.86 s per evaluation, medians) and the
  thread counts (Package 1: 48).
- **What follows is not decided here.** If the library changes a measure, the effect on the Kovasznay
  CPU comparison is estimated from these numbers and put to the user before anything else. Rerunning
  the ten CPU runs would cost about 10 node-hours, about 480 SU: more than the cap leaves (below).
- Stage p5k (`results/package5_item3_openmp_kovasznay`; `p5k_openmp_kovasznay.slurm`, timed-cpu, 40
  min; `submit_p5_item3_openmp_kovasznay.sh`): 32 SU requested, about 22 expected. It runs beside the
  rerun. It is not part of the rerun's 1,000 SU line.

**The cap.** Used: 153.5 (items 1, 2, 4 and Addendum 2) + 938.3 (the CPU set) + about 15 (p5o) =
about 1,107. Left of 4,400: about 3,293. Expected: the rerun about 925, p5k about 22, the A100 set
about 2,270 = about 3,217, which leaves about 76.

**The Grace steps** (`Post-JCP/Grace_steps_item3_rerun_and_gpu.md`): p5r and p5k first; dry runs of
all three stages before; then the check of his stop rule (the rerun's `report_log.txt` total, the logs'
failure lines, every job `COMPLETED`); only then p5g.

**Tests:** `tests/test_p5_3_openmp_kovasznay.py` (the measures; a whole process with small networks;
the check stops before any run; the summary's criteria, reproduction, Package 1 values and threads; a
missing order or reference); `tests/test_p5_3_networks.py` (the repeats keep the first pair's verdicts
and record six timed solves; `--c4-repeats` reaches its method only and rejects bad values);
`tests/test_p5_submission.py` (p5k's job, wiring, dry run and submission; p5r's checks job with the
repeats on Buckley-Leverett NiL-N only; the checks loops expanded per job).
- One failure in 13 runs of the new tests, explained before going on: the whole-process test gave F1 an
  8 s budget; with the rehearsal loading the laptop, Adam ran at 0.038 s per iteration instead of
  0.022, so its 201 iterations used 7.7 s and L-BFGS logged too few rows to measure (the failing
  assertion: the L-BFGS value empty). The test now ends F1 by one L-BFGS call
  (`F1_MAX_LBFGS_CALLS`, None on Grace; `f1_train`'s test cap) under a 300 s budget, so its rows no
  longer depend on the load; 3 of 3 passes with the rehearsal running.

**The laptop rehearsal** (bundle at decae59; the check run with 'torch' in p5r_checks, as before):
- p5r_checks rc 0 (513 s). C2 6 of 6. The Buckley-Leverett NiL-N pair three times: C4 1.2%, the
  timings -1.2%, +0.04%, -1.3% (spread 1.4%).
- **C4 5 of 6: Buckley-Leverett LM at 7.9% (with the log 32.0 s, without 29.7 s).** Cause shown, not
  assumed: the same pair three times on the laptop at this commit (`c2_c4(..., repeats=3)`) gave
  -0.7%, +0.6%, -0.4%, every run 28.9 to 29.8 s. The 32.0 s run was a slow stretch of the laptop;
  the log puts no time on the clock (its 1.5 to 1.7 s are off it). On Grace the CPU set's pair was
  +2.5%.
- p5k as written, rc 0 (745 s): the three numpy-first processes stopped at the check (Windows loads
  the wheel's library in either order), the three torch-first ones ran (16 threads on the laptop;
  Package 1's runs: 48); their losses match Package 1's Grace runs within 1.6e-8 (F2) and 1.5e-12
  (F1); the tarball written. An earlier run without the check (bundle de4df64): all six processes;
  with one library on both sides the medians differed by 7 to 8% (laptop noise) but the rounds
  overlapped, so the criterion says no change, as it should.
- Repo suite 1,037 passed; no-git 1,006; WSL bash tests 37 of 37 in each of three modes.

**The scan after the rehearsal** (the user's rule: before Grace, again after the rehearsals), over the
code changed since bundle 03bd5ab:
- **Found and fixed:** `component_a.py` had been saved with CRLF line endings by an edit (the repo is
  LF), so the bundle carried a whole-file rewrite for a 16-line change. Python reads both the same;
  converted back, and the diff against 03bd5ab is the 16 lines again. Earlier in this round (before the
  rehearsal): p5k's provenance was written before `pin_torch` (now after); the summary had no thread
  check (now both runs' and Package 1's); a missing Package 1 reference crashed the summary (now
  reported).
- **Implicit state looked at, none acting:** the C4 repeats build a fresh seed-0 network each time
  (C2 shows the solves are deterministic); they run after the first pair, so warmer, which affects only
  the spread beside the fixed verdict; an interrupted checks job leaves no `result.json` and reruns the
  pair whole. The default `repeats=1` path is the old code, so the budget jobs of p5r and p5g are
  unchanged. In p5k each process imports in its own order before anything else, then
  `component_a` (`pin_torch`: OMP_NUM_THREADS, 48 on a whole node, as Package 1); F1 sets its dtype
  per run and F2 sets float64 after F1 has run; each run seeds itself; F2 runs second in its process
  (in Package 1 it ran first in its own), the same for both orders, so it does not bias the
  comparison. A process that fails its check is caught by `until_end`, the job goes on, and the
  summary says "not measured" for that order (p5o's design).
- **Native libraries:** every Grace process checks its OpenMP library itself (p5r, p5g: 'module';
  p5k: the one its first import should give).

**The p5k result** (job 20581996 on 17 November 2026, bundle 3df7b7b; tarball sha256 688e646c...,
matches Grace; sacct: COMPLETED, 26 min 9 s on 48 cores = 20.9 SU, expected about 22; no 'STOPPED OR
FAILED' or 'NOT RUN' in its log. The report ran inside the job and left it out, so the row is the
user's sacct, as for p5o). Every process loaded the library its first import should give; 48 threads in every
run, as Package 1. Judged by the criteria fixed before the job:
- **F1: the library changes it, and the module's library is the slower one.** Seconds per Adam
  iteration, medians over the three rounds: wheel's 0.336, module's 0.458 (wheel/module 0.73); per
  L-BFGS iteration 0.360 against 0.450 (0.80). The rounds do not overlap. The torch-first values are
  Package 1's own (0.327 and 0.353, all five seeds), within 3%.
- **F2: no change beyond the cutoff** (seconds per evaluation, wheel/module 0.89, the rounds overlap).
- **So Package 1's Kovasznay CPU baselines ran with the faster library for F1.** LiL-Q's CPU times
  (Part B) loaded the module's. Under the module's library F1 would have been about 25 to 36% slower
  per iteration. The mixed libraries favour the baselines, so the same-hardware comparison is, if
  anything, conservative for LiL-Q. Rerunning them under the module's library would only slow the
  baselines down (about 480 SU, more than the cap leaves). What to tell the advisor: put to the user.
- **The reproduction check fails on its letter, cause shown.** Its rule compares every log row that
  Package 1's seed-0 run also has. In rounds 0 and 2 of torch first, F2's time ran out inside step 10;
  that row is the rejected try (accepted 0), logged with the step-9 loss under iteration 10, while
  Package 1's row 10 is the accepted step (13% apart). On every accepted step the torch-first F2 losses
  equal Package 1's exactly (0.0), and F1's Adam rows too (0.0). The numpy-first runs differ from
  Package 1 by up to 1e-12 (F1) and 4e-7 (F2): the library changes the round-off, as in item 3. The
  rule should have left out rows with accepted 0; not changed after the job, reported as is.
- **F2's measure is weaker than F1's.** Its seconds per evaluation are not constant over a run:
  Package 1's whole-log value (5.86) mixes cheap rejected tries late in the run, so it cannot be set
  beside these 90 s runs. Per accepted step over rows 2 to 8 (not a criterion, for the record): wheel
  9.34, module 10.50 (the rounds overlap), Package 1 10.97 (five seeds, same rows). So this job's F2
  was about 15% faster than Package 1's own runs with the same library: F2 ran second in a process
  that had run F1, where Package 1's F2 ran first in its own. The same for both orders, so the
  comparison stands; its absolute values do not reproduce Package 1's.

---

## 2026-11-17 -- Package 5, item 3: the diagnostic's result; numpy first, the OpenMP library check, and the CPU network runs again (stage p5r)

**The diagnostic** (job on 17 November, about 18 min, about 15 SU; tarball sha256 5aa68002...). Every
process loaded the library its first import should give. Judged by the criteria fixed before the job
(the entry below):
- **The library is the cause of C1's failure.** With numpy first, the three C1 runs (NiL-N, seed 0, each
  problem's largest size) equal Package 1's losses over iterations 0 to 100 exactly (difference 0.0).
  With torch first, they equal the CPU set's runs exactly.
- **It changes the timing on Buckley-Leverett, not on Bratu.** The median time per iteration over the
  three rounds, wheel's library against the module's: Bratu P = 225, NiL-N +3.9% and LM +1.0% (under
  the 5% cutoff); Buckley-Leverett P = 1,024, NiL-N +73.5% and LM +60.6% (over it; the rounds do not
  overlap). Burgers was not in the diagnostic.

**Decided (the user, 17 November 2026):** the network runs import numpy first, every Grace job checks
the library, and the CPU set's five network jobs run again, with checks C2 and C4 for the networks:
about 925 SU expected (the CPU set's own SU less its LiL runs), 1,128 requested. The A100 set gets the
same order and check. Both wait for the advisor's approval in the item 3 cost check
(`Post-JCP/Package5_item3_cost_check.md`): item 3 then costs about 4,150 SU against the 3,146 left in
his cap.
- **The CPU set stays as it ran** (`results/package5_item3_cpu`, its scripts unchanged). Its LiL runs
  loaded the module's library (`p5_3_lil.py` imports numpy first) and are kept; its network runs and
  their C2/C4 are replaced by stage p5r's at the assembly.
- **Not chosen:** rerunning only Buckley-Leverett (Burgers was not measured), or keeping Bratu (within
  5%, saves about 220 SU, but its C1 stays failed and the table would mix two libraries). The second is
  offered to the advisor as the cheaper alternative.

**The code.**
- `p5_3_networks.py` imports numpy before torch, as Package 1's `four_method_tables.py` and `p5_3_lil.py`.
- `lilq.provenance`: `openmp_libraries`, `openmp_library_kind` ('torch' if the wheel's library is
  loaded, 'module' if only another), `require_openmp(kind)`, which stops the process (SystemExit,
  "OPENMP LIBRARY CHECK FAILED") otherwise. `p5_3_openmp.py` uses them instead of its own copies.
- `--require-openmp {module,torch}` on `p5_3_networks.py run | solo | checks` and `p5_3_lil.py run`:
  the check before any run or record.
- In the job scripts of p5r and p5g (`OPENMP_CHECK`, `_generate_jobs.py`): before the first run of every
  job, each driver is imported in a fresh process and `require_openmp('module')` must pass, or the job
  ends (`set -e`). In the preflights this stops the whole stage (afterok). Every driver call also
  carries `--require-openmp module`. The check runs in each process because the library is chosen
  per process, by its first import.
- On the laptop (Windows) both orders load the wheel's library (numpy's wheel brings no OpenMP), so the
  laptop rehearsal runs the check with 'torch' in place of 'module'.

**Stage p5r** (`results/package5_item3_cpu_rerun`, its own commit lock; `submit_p5_item3_rerun.sh`):
the preflight (cpu: the library, the inputs, the test suite); then on whole CPU nodes the checks job
(C2/C4 for NiL-N and LM) and the five budget jobs in the CPU set's parts, network runs only; then the
report (afterany). Walltimes from the same model as p5c.

**Tests:** `tests/test_p5_3_openmp.py` (the classification; `require_openmp` on each library; a driver
with `--require-openmp` stops before any record; in a fresh process, numpy is loaded before torch by
every item 3 entry point: the two drivers, the table, `p4_checks.py`, `lilq.provenance`; with the old
order the networks case fails); `tests/test_p5_submission.py` (p5r's runs, records, classes,
walltimes, SU requested, dry run and chain; the check in every p5r and p5g job before its first run;
p5c's scripts without it).

**The scans before each hand-off** now also diff the OpenMP and BLAS libraries in the rehearsal's
`hardware.json` against the reference package's.

**The laptop rehearsal** (bundle at 710f49c; 5 s budgets, one GPU; the check run with 'torch', since
the laptop loads the wheel's library in either order):
- p5r: every job rc 0; 108 network runs; the resubmitted job skipped its 18 runs; the tarball written.
- p5g: the pair jobs rc 0 (108 network and 14 LiL runs); the gate stopped the BL job as written (one
  GPU); the tarball written.
- Run as written (with 'module'), the check stopped p5r_bl_b and the p5g preflight before any run
  ("OPENMP LIBRARY CHECK FAILED").
- C2 6 of 6. **C4 5 of 6: Bratu NiL-N at 5.4%.** Cause shown, not assumed: the same pair repeated three
  times on the laptop at this commit gave +3.3%, -2.5% and +2.4%. Both signs, and the run without the
  log alone ranged 33.9 to 34.8 s, against 35.9 s in the rehearsal. So it is the laptop's clock spread
  during a slow stretch. On Grace the same pair gave 1.6% (CPU set).
- Repo suite 1,028 passed; no-git 997; WSL bash tests 35 of 35 in each of three modes.

**The first such diff, over every Grace record** (`hardware.json`, grouped by loaded libraries and
versions):
- Every record has the same BLAS (FlexiBLAS 3.4.4 over OpenBLAS 0.3.27, the GCC 13.3.0 modules) and the
  same numpy 1.26.4, scipy 1.13.1 and torch 2.10.0+cu126. Only the OpenMP library differs.
- Package 1: Parts B (the four-method runs among them) and C loaded the module's library (313 records).
  **Component A, the Kovasznay calibration, loaded the torch wheel's (116 records: screening 48, full
  45, full_cpu 20, float32 2, checks 1).** These are the tuned network baselines, timed against LiL-Q;
  the 20 `full_cpu` records are CPU runs. How much the library changed their CPU times is not known
  (in item 3 it was +1 to +4% on Bratu and +61 to +74% on Buckley-Leverett). Open; put to the user.
- Packages 2 to 5 on Grace: the module's library in every record. The only others are laptop previews
  (Windows) and laptop assembly records with no OpenMP library.
- The CPU set: the LiL processes loaded the module's library, the network processes the wheel's.

---

## 2026-11-17 -- Package 5, item 3: the CPU set's checks, C1's failure, and the OpenMP library diagnostic (stage p5o)

**The CPU set** (bundle 0f74bdf, jobs 20564295-20564302). All seven compute jobs completed, about
938 SU used against about 1,110 expected. There are 108 network runs, 14 LiL-N L-BFGS budget runs
and 31 protocol runs, all with `timing_matches`. No job was stopped early.
- C2: 6 of 6 pass.
- C3 (LiL-Q on Burgers against Package 4): 7 of 7 pass, to 1e-11 or better, with every rule step equal.
- C4: 5 of 6 pass. BL NiL-N failed at 5.9%: the run *with* the error log was the faster one, which the
  log cannot cause. It is reported and not rerun.
- **C1 failed on 15 of the 18 runs it compares** (cutoff 1e-6, fixed on 15 November and not moved).
  - The losses agree to round-off at first (about 1e-16 at iterations 0 to 10). The gap then grows
    about tenfold every few iterations: about 1e-12 by iteration 20, 1e-8 by 50, and 1e-4 to 0.66 by
    100.
  - BL differs by 2e-9 already at iteration 0, through its pretraining.
  - This is the same algorithm from the same start, with arithmetic that differs in the last digit
    from the first step.
- `p5_3_networks.py c1` (`c1_all`) now runs C1 on every run that has a Package 1 counterpart; it was
  added at this assembly.

**The likely cause, missed by the three implicit-state reviews: which OpenMP library a process loads.**
- `p5_3_networks.py` imports torch before numpy, so its processes loaded the PyTorch wheel's own
  OpenMP library (`torch/lib/libgomp.so.1`).
- Package 1's four-method runs (`four_method_tables.py`, numpy first) loaded the GCC module's
  (`GCCcore/13.3.0/lib64/libgomp.so.1.0.0`). So did every Grace job recorded since, including the LiL
  driver of this item, which imports numpy first.
- The thread counts were the same (48, one pool). The reviews checked counts, not libraries.
- If the two libraries also differ in speed, the network times are not on Package 1's footing, nor
  on the LiL runs'.

**The diagnostic** (the user's decision, 17 November 2026): stage p5o, `results/package5_item3_openmp`
under its own commit lock. It is one timed-cpu job, 30 min, 24 SU requested and about 16 expected.
- Six processes of `experiments/p5_3_openmp.py`, three rounds, alternating which package is imported
  first.
- Each process checks which library it loaded and stops if it is not the expected one
  (`--require-library`).
- In round 0, the C1 runs: NiL-N, seed 0, at each problem's largest size, 110 iterations.
- In every round, NiL-N and LM at Bratu P = 225 and BL P = 1,024, 30 s each, without the error log.

**Decided before the job:**
- The library is the cause if the numpy-first C1 runs pass C1 against Package 1 and the torch-first
  ones reproduce the CPU set's own runs (C1's comparison, against those runs).
- The library changes a method's timing on a problem if the medians over the rounds of the time per
  iteration differ by more than 5% (C4's and the pairing cutoff) and the two orders' rounds do not
  overlap.
- What follows from each outcome (keep the CPU set or rerun its network jobs; the import order of the
  A100 set) is the user's decision, with these numbers. The A100 set and the cost-check memo wait
  for it.

---

## 2026-11-16 -- Package 5, item 3: the job layout (batch 5; stages p5c and p5g)

**Two stages**, each with its own results folder under its own commit lock: `p5c`, the CPU set
(`results/package5_item3_cpu`), and `p5g`, the A100 set (`results/package5_item3_gpu`). The A100 set is
submitted only after the cost check has told the advisor its class and cost (his rule 3).

**The CPU set** (`submit_p5_item3_cpu.sh`, 8 jobs):
- `p5c_preflight` (cpu): the three references, the test suite. The others wait for it.
- `p5c_protocol` (timed-cpu): LiL-Q, `trf` and the square collocation by the usual protocol on the three
  problems, then C2 and C4 (NiL-N and LM at the largest size, about 30 s each; see the review below).
- Five budget jobs (timed-cpu, 48 threads, one run at a time): `p5c_bratu` (27 network runs and LiL-N
  L-BFGS at the 3 sizes), `p5c_burgers_a` / `_b` (23 + 3, 22 + 4), `p5c_bl_a` / `_b` (18 + 2 each). The
  network runs are split round robin (`--part i/n`) over the list sorted by size, so each job gets about
  the same mix of sizes, methods and seeds.
- `p5c_report` (afterany): the time-to-error table, then the tarball.

**The A100 set** (`submit_p5_item3_gpu.sh`, 5 jobs):
- `p5g_preflight` (shared-gpu, one A100): the references, and the item 3 tests with CUDA visible (the torch
  residuals and the off-clock timer on the GPU; the CPU preflight skips them).
- One `gpu-pair` job per problem (both A100s, 48 cores, 360 GB, 192 SU/h): first stops unless it sees two
  GPUs. Then two processes side by side, process i on GPU i (`CUDA_VISIBLE_DEVICES`), 24 threads each,
  network part i/2 and then its share of the LiL-N L-BFGS sizes.
- **The solo check** (the plan's paired-against-solo check): per problem, NiL-N seed 0 at the largest size
  (the run with the most work on the GPU), run alone for 60 s before the pair starts, on the GPU its paired
  run uses. The report compares the clock at the last iteration both logged; within 5% (cutoff set before
  any run). Three solo runs cost about 15 SU.
- **No CPU pinning.** Each process has 24 threads and the operating system places them. If the pairing
  check fails, binding each process to its GPU's socket is the first thing to try.
- `p5g_report` (afterany): the pairing check, the table, the tarball.

**Every process has its own tag** (`--tag`, e.g. `cpu-burgers-a`, `gpu-bl-1`): it names the process's
warm-up and provenance records and is kept in each `run.json`, so no two processes write the same file.
Each run's folder is fixed by problem, size, method, seed and device, and the tests check that every run
is in exactly one job per stage.

**Plan changes (flagged to the user):**
- **C1 and C3 run at the assembly on the laptop**, not on Grace. C1 compares with the Package 1 histories
  and C3 with the delivered Package 4 logs, which are not on Grace. Items 1, 2 and 4 did the same.
- **Report steps under their own time limits** (table 10 min, tarball 15 min, pairing 5 min): the
  weakness found in Addendum 2's lost hour.

**Walltimes and cost, from a laptop probe** (16 November 2026, 30 s budgets at each problem's largest
size, CPU and CUDA). The error log is off the clock but not off the walltime:
- LM logs the error at every iteration: 7 to 21% of the budget (Bratu 21% CPU / 15% CUDA, Burgers
  12% / 11%, BL 7% / 5%).
- The L-BFGS methods log every 10 iterations: under 2%. Pretraining is under 2 s.
- Each run is costed at 600 s x (1 + its share, rounded up: 0.22, 0.13, 0.08; 0.02) + 30 s fixed, and
  LiL-N L-BFGS as a full budget run. Walltime = expected x 1.10, rounded up to 15 min.
- **CPU set: 1,260 SU requested, about 1,110 expected. A100 set: 2,535 requested, about 2,220 expected.**
  Both are upper bounds: LiL-N L-BFGS stalls early at the small sizes, and the fixed part was 2 to 5 s on
  the laptop. Against the plan's 1,070 and 2,040, the difference is the LM error log. With both sets, the
  upper bound is about 3,330 against the 3,177 left in the advisor's cap. The cost check gives him the
  A100 figure after the CPU set has run, when its real charge is known.
- The probe's BL NiL-N CUDA run took 1,498 s for a 30 s budget: the laptop slept from 19:34:58 to
  19:59:22 (Windows Kernel-Power events 42 and 107) during one iteration. Not the code: the budget is
  checked after each iteration, and an iteration cannot be interrupted.

**The second review for implicit state and missed bugs** (the user's request, after the rehearsal):
- Fixed: `pairing()` stopped the report if a solo or paired run had been cut before its `run.json`; such
  a row now has no verdict and says why. Each row also records both runs' evaluation counts at the
  compared iteration, so a different iterate path is visible.
- Fixed, a missed plan item: C2 and C4 were still 300 NiL-N and 30 LM iterations (2 to 10 s), not the
  "tens of seconds" the 15 November entry asked for. Fixed counts did not work either: in the rehearsal
  1,000 NiL-N iterations took 5 to 7 s on the laptop (the probe's rates had predicted 17 to 33 s), and C4
  failed on two such runs (12.8% and 5.7%; one overlapped a test run). **Both runs now have the count
  that takes about 30 s at the rate of an untimed pilot on the same node** (NiL-N 100 iterations, LM 10).
  The 5% cutoff is unchanged. A wall-clock budget for the first run was rejected: it can end an LM
  iteration in the middle of its rejected steps, so a run capped at the same count would not repeat it.
  With the pilot, the laptop's six C2/C4 runs took 27 to 36 s; C2 passed in all six, C4 in four (Bratu LM
  +7.7%, Burgers NiL-N +8.2%). Repeated three times each on a quiet laptop, those two gave -2.9, +2.6,
  -2.8% and +0.6, +0.3, -0.8%: noise on the laptop, not a cost of the log that the clock misses. On
  Grace C4 is one run per method and problem, as planned; a failure is reported, never rerun to a pass.
- Fixed: the pair jobs wrote `hardware.json` before setting their 24 threads.
- **The rehearsal** (laptop, 5 s budgets, both stages, every job script's own body): every job ended
  with rc 0; 108 network and 14 LiL-N L-BFGS runs per device, each in one job; one warm-up and one
  provenance record per process; the 31 protocol runs all timed with `timing_matches`; tables with no
  run flagged; both tarballs (12.6 and 3.2 MB at 5 s budgets). The pairing check failed (paired runs
  70 to 124% slower) because both processes shared the laptop's one GPU: the check tells the cases
  apart, and the evaluation counts at the compared iterations were equal (the same path).
- Checked, by design: each run calls `set_seed` and draws its collocation set from its own seed, and the
  weights are built on the CPU, so a run's result does not depend on the runs before it in its process
  (the parts give each process a different sequence). The network driver sets torch's default dtype to
  float64 for the whole process, as P2-8 and the four-method runs do; every run in it wants float64.
- Checked: the table reads `networks/*_P*/*/run.json`, so the solo runs under `checks/` never enter it.

**The third review: implicit state in everything the CPU set touches** (the user's request, "I don't have
SUs to waste"). Read: the job scripts, `env.sh`, the resource classes, `lilq/blas_threads.py`, both
drivers, P2-8's `setup` and `lm`, the four-method runners and the solver path.
- Fixed, the costliest: **a resubmitted job ran every run again.** Nothing looked at what `$PKG`
  already held, so a job resubmitted after a node failure 4 h in would have repeated 4 h of runs
  (about 190 SU on a whole CPU node). Each driver now skips a run whose `run.json` is from this commit
  with the same settings (`finished()`: the networks' budget; LiL-N L-BFGS's budget; the protocol runs'
  timed/untimed). It skips the warm-up when nothing is left. A run stopped part way has no `run.json`,
  because it is written last, so it runs again from the start. C2 and C4 skip a `result.json` of this
  commit. This is the rule `experiments/component_a.py` already uses.
- Fixed: a resubmitted job overwrote the first job's `hardware.json`, the record of the node its
  finished runs ran on. `lilq.provenance.job_dir` now writes a second job's record beside the first
  (`<name>_job<SLURM_JOB_ID>`), in both drivers and in `p4_checks.py provenance`. Every `run.json` and
  C2/C4 result records its `slurm_job_id`, so each run points to its node.
- Fixed, wording: the item 3 jobs' `until_end` comment said a stop keeps every iterate. That is true of
  items 1 and 2's drivers, but not of these, which write a run's files when it ends. The item 3 jobs now
  say that a stop keeps only the finished runs. The scripts of items 1, 2 and Addendum 2, which have
  run, are unchanged.
- Checked, by design: threads. `env.sh` sets OMP/OPENBLAS/MKL to `SLURM_CPUS_PER_TASK` (48 on a
  timed-cpu node) before Python starts. `lilq.utils` sets torch's threads to the same count, and every
  `run.json` records both (`thread_env`, `torch_threads`).
- Checked, by design: the budget is training time with the error log off the clock, in both L-BFGS
  (`RunProbe`) and LM (`lm`'s `excluded`). LM checks it after each iteration and inside the rejected
  steps. One LM iteration at the largest sizes took 0.06 to 0.25 s on the laptop, so a run overshoots
  600 s by well under the walltime model's 30 s fixed allowance.
- Checked, by design: no file is written outside `$PKG`. The warm-up's scratch log is under `$PKG`,
  named per tag, and deleted afterwards. The four-method runners and the solver path write nothing.
  Matplotlib is not used.
- Checked, by design: C2 compares bit for bit. Grace has reproduced runs bit for bit across jobs before
  (entry of 2026-10-05, Package 2 Stage 2, check C8). A failure is reported, never
  rerun.
- Checked: the preflight (class `cpu`, 24 cores, 32 GB) ran Package 4's suite in 10 min 33 s. The suite
  takes 10.5 min on the laptop now, inside the 30 min walltime.
- To resubmit one job after a failure: `LILQ_WAVE=p5c bash scripts/cluster/sbatch.sh <job>.slurm`, then the
  report job again. The commit lock refuses a different bundle.
- The CPU set was submitted on 17 November 2026 from bundle 0f74bdf (jobs 20564295-20564302).

**The pairing gate (the user's decision, 16 November 2026).** Before, the paired-against-solo check
ran only in the report, after all three A100 jobs. A failure would have been found after about
2,220 SU, with every A100 time in the table biased. Now each pair job checks first:
- **How.** After the solo check, the same run (NiL-N, seed 0, largest size, 60 s) runs on both GPUs at
  once (`solo --gate 0` and `--gate 1`). `p5_3_networks.py gate` compares the run on the solo's GPU with
  the solo by `compare_clocks`, which the report's `pairing` now shares: the clock at the last
  iteration both logged, within 5% either way. The cutoff is the plan's (`PAIR_RTOL`), not a new one.
- **The job goes on only if `passed` is True.** It stops before any budget run (exit 1) if the clock
  is off by more than 5%, or if there is no verdict: a gate run that did not finish (the run on the
  other GPU is the load, so it must finish too), or a record from an earlier job (each record's
  `slurm_job_id` must be this job's). The report still runs (afterany) and packs what there is.
- **Recorded:** `checks/gate_<bench>.json` with both clocks, the evaluation counts and
  `overlap_fraction`. That is the share of the judged run's solve during which the other GPU was
  solving too, from the solve windows (`solver_window_epoch_s`, now in every network `run.json`). It
  is recorded, not judged.
- **Cost:** `GATE_EXTRA_S` = 300 s per job in the walltime model. Bratu's walltime goes from 3:15 to
  3:30; the other two are unchanged after rounding. The A100 set is now 2,583 SU requested and about
  2,270 expected, up from 2,535 and 2,220. The model counts 5 min per job; the gate itself should take
  about 2 (warm-up, pretraining, 60 s). If the gate fails, a job costs about 5 min (about 16 SU)
  instead of 3 to 5 h.
- **Rehearsal** (laptop, `rehearse_item3_v2.sh`; solo and gate at their 60 s, budget runs at 5 s):
  - With the gate as written, `p5g_bl` stopped before its budget runs, rc 1. On one GPU the two gate
    runs share it, so the clock was +107%, with an overlap of 0.999: the stop works.
  - With the stop made non-fatal, all three pair jobs and the report ended with rc 0.
  - The CPU stage, rehearsed on bundle 0f74bdf itself, ended with rc 0 in every job. A resubmitted
    `p5c_bl_a` took 9 s and `p5c_protocol` 22 s, every run and C2/C4 result skipped as held.
- **Scan after the rehearsal:**
  - The gate and solo folders are under `networks/checks/`, which the table's glob
    (`networks/*_P*/*/run.json`) and `pairing`'s (`checks/solo_*`) do not reach.
  - Both gate processes exit before the pair starts, so their GPU memory is free.
  - `until_end` hides a gate run's failure, but the missing record then gives no verdict, and the job
    stops.
  - The gate command has its own 5 min `timeout`.
  - On a resubmission the solo and gate runs are redone on the new node; the job-id check refuses
    the old records.

---

## 2026-11-16 -- Package 5, Addendum 2: the Grace run, a lost hour, and the assembly (`experiments/p5_assemble.py --a2`)

**The first job (20549881, node c224, 11:27-12:28) produced nothing.** All six processes were stopped by
the 55-minute guard (exit 124) and none created its run folder. The job used 26 s of CPU in all, and
2.2 GB at most for the six. So they were idle right after starting, not computing. The same start takes
8 s on the laptop with Grace's own item 1 files (7.8 GiB peak). It ran on the day Grace's Slurm was
timing out. A login-node check after the outage read the five input files in 80 s wall-clock for
3.6 s of CPU, and in 7.7 s at the next try. **Reading: a stall of the scratch filesystem, not the
code.** It cost 24.5 SU (the full hour). The second job (20553982, node c119, the same bundle and
lock) ran in 954 s: every part done, K1 and K2 pass with zero difference.
- Weakness found: the summary and the tarball steps of the job script have no time guard, so a stall
  there also ends without a tarball. Item 3's jobs get one; the Addendum 2 script is not changed (its
  rerun used the bundle as is).

**The assembly** (`--a2 <extracted>/package5_a2 [--a2-logs <job log>]`), as items 1, 2 and 4:
- The 18 runs, `terminal.csv`, `stretches.csv`, the checks and K1's runs go to `P5_A2_initial_profile/`,
  the advisor's Section 6 path. The stage's own records (`COMMIT`, `provenance/a2`, `slurm_logs`,
  `sacct.txt`, `su_per_job.csv`) go to `package5_a2/`. The second job's log is added there: the
  tarball is written before its own job ends.
- Every run must have `run.json`, `iterations.csv`, `betas.npy`, `errors_by_iterate.csv` and
  `updates_by_iterate.csv`, or the assembly stops.
- The summary is rerun on Grace's stored iterates. Its tables and JSON files must equal Grace's within
  1e-8 relative but delta_P's (round-off, recorded in `package5_a2_rerun_differences.json`), or the
  assembly stops. Grace's are kept. K1 and K2 (Grace's verdicts) join the exit code.
- Section 3a's update histories are written at the assembly (they are post-processing of delivered
  coefficients). They match the advisor's two hand values to the figures he quoted.
- Both Addendum 2 jobs go into `su.csv` through `--sacct-extra`, the lost hour included.

**The delivered packages the assembly reads (found at this assembly).** The 14 November assembly was
given Grace's raw trees for Packages 2, 3 and 4 (`.../grace/...`). This one is given the delivered
folders (`package2_results`, `package3_results`, `package4_results`). Every Grace table, `checks.json`
and C1 to C3 come out the same either way. One file differs:
`P3_1_beltrami_certified/beside_delivered.csv`, whose ratios use the delivered delta_P of each field.
Package 3's delivered pressure delta_P is the value re-evaluated after the advisor's reply of
10 October (`delta_P_reevaluated.json`, 1e-4 to 2e-4 below Grace's first value). Grace's raw tree has
the first value. **Decided:** the delivered folders, as the column's name says. The p and combined
ratios move by at most 2e-4 relative. One figure printed in the 14 November memo changes: B2 at weight
10^3, p (pin gauge) / delta_P, 169 -> 170. The memo as sent stays as it is; the next memo says so in
one line.

---

## 2026-11-16 -- Package 5, item 3: the A100 class -- two runs side by side on a whole two-A100 node (the user's decision)

The A100 set (the three network methods and LiL-N L-BFGS: 122 runs of up to 600 s) runs two at a time on a whole two-A100 node, one run per GPU: 192 SU/h for the node, 96 SU/h per run, about 2,040 SU as an upper bound; with the CPU set (about 1,070 SU) about 3,110 against the 3,177 left in the advisor's cap. The node holds only our runs, as every timed GPU run so far. The alternatives were one A100 with 8 cores on a shared node (80 SU/h, about 1,700 SU; other users' jobs on the node and 8 cores put its timings on another footing) and a whole node per run (192 SU/h, about 4,070 SU; over the cap and the account's balance).
- **Check:** on a few runs, a paired run's time is within 5% of the same run alone on the node (batch 5 sets which runs, before any comparison).
- **The advisor's rule 3 still holds:** the CPU set first, then the cost check tells him this class and its cost before the A100 set. The cost check's remaining question is whether pretraining belongs on the clock.

---

## 2026-11-16 -- Package 5, item 3: review of batches 3 and 4 for implicit state and missed bugs (the user's request, before any Grace job)

Fixed (each with a test):
1. **`setup_s` counted the reference.** `Problem` timed the loading of the reference and the error evaluator with the system, so `with_setup_s` would have charged every LiL method for reading a file. It now covers the system and the initial coefficients only (for Buckley-Leverett the fit of the initial condition is part of it).
2. **A global left behind.** `solve_lbfgs` set torch's default dtype to float64 for the rest of the process. The residuals do not depend on it (every tensor is made from float64 arrays), but later code in the same process would have. It is now set for the run and restored.
3. **The L-BFGS coefficients were read through a side channel.** The error is read from the parameter tensor the loss was last called with (`TorchLoss.param`), which is the optimizer's own tensor at the accepted point. That holds by how `LBFGSObjective` calls the loss, so each run now records `last_logged_is_returned` (the last logged coefficients equal the returned ones).
4. **`trf`'s `evaluations` column was the iterate index.** It now counts the residual evaluations up to each accepted iterate (rejected steps included), as the network runs' column counts evaluations.
5. **numpy numbers became strings in `run.json`** (`default=str`). They are now written as numbers.
6. **The square collocation assumed its reference grid.** It reads the p = 48 field and compares it on P2-3's 201 x 201 axis; it now checks that the file's axes are that grid.
7. **C3 crashed on a missing Package 4 log.** It now writes a failed row with the reason.
8. **An untimed LiL run could enter the table.** A `--no-timing` run has times from its logged pass. Every LiL record now says `timed`, and the table leaves out and lists a run with `timed` false.

Implicit state kept, by design, and recorded:
- **Threads.** numpy's BLAS and torch take their thread counts from `OMP_NUM_THREADS` and the other variables, which `lilq.blas_threads` sets (by `setdefault`) before numpy loads; the job script sets them explicitly. Every record carries `thread_env`, and L-BFGS also `torch_threads`.
- **Determinism of the protocol.** A timed pass must give the logged pass's iterate count and coefficients. If gelsy with many threads ever gave a different rule step, the run is kept with no times and flagged in the table, rather than timed on another path.
- **Output paths.** A run's folder is fixed by problem, size, method and device; two jobs that run the same combination into one results folder would overwrite each other. The job layout (batch 5) must give each combination to one job.
- **Device names** become folder names (`lbfgs_cuda`); the jobs pass `cuda`, not `cuda:0`.
- **Size of `betas_logged.npy`.** One vector per 10 L-BFGS iterations. A fast run that does not stall could keep tens of thousands (Burgers P = 900 on the CPU ran 35,000 iterations in 52 s before stalling: 3,500 vectors, 25 MB). It is measured in the rehearsal before deciding what goes on the results branch.

---

## 2026-11-16 -- Package 5, item 3: the time-to-error table (`experiments/p5_3_table.py`)

Section 6's table, from every history file under `P5_3_cost/` (network and LiL runs alike), at the errors 1e-2 to 1e-6.
- **Per run:** the clock time of the first logged iterate whose error is at most the level ("at" counts). L-BFGS logs every 10 iterations, so its time has that resolution; LM, `trf`, LiL-Q and the square collocation log every iterate.
- **Per size:** the median over the seeds, a seed that never reaches the level counting as infinitely late. So the median exists only when two of three seeds reach it (the plan's reading 7). The LiL methods have one run per size.
- **Per method, problem, device and level:** the smallest over the sizes, with the size that gives it; a dash where none reaches the level.
- **Setup and pretraining beside the clock** (`with_setup_s`): the same time plus the run's `setup_s` and `pretrain_s`, taken through the same median and minimum (which may then pick another size). The question of what the clock covers goes to the advisor; either answer is then in the table without a rerun.
- **Left out and listed in `flagged.csv`:** a run with no time in its history (a timed pass that did not reproduce the logged pass). A run that ended in `failure` is listed too, but its history up to the failure counts.
- **Fixed on the way:** the square collocation's timestamps were numpy floats, which numpy 2 writes as `np.float64(...)` in `history.csv`; they are now plain floats, and `write_history` converts any float to a plain one.

---

## 2026-11-16 -- Package 5, item 3: the LiL runs (`experiments/p5_3_lil.py`; torch residuals in `p2_16_certified`, `p4_1_burgers_gauss`, `p3_2_bl_certified`)

**The methods on LiL-Q's trial space** (Section 6): LiL-Q, LiL-N by `trf`, LiL-N by L-BFGS, and, for Bratu, the square collocation of P2-3. **The systems** are item 1's at the weight 10^4 (the advisor's reply of 15 November 2026, answer 1): Bratu P2-16 tensor Chebyshev on the CGL set, N/P = 10, P = 25, 100, 225; viscous Buckley-Leverett, the lifted sine x Chebyshev of Package 3 (its split), P = 64 to 1,024. Burgers is Package 4's sine x Chebyshev on the Gauss-Legendre set, p = 6 to 30, its weight 10 on the initial rows (his answer: "as in Package 4"). **The error** is the network runs' (one reference per problem, the entry of 15 November below).

**LiL-N by L-BFGS on the A100: one torch residual per system, the same code on both devices** (the user's decision, 16 November 2026). Item 2's LiL-N evaluates N(beta) and 2 A^T N in numpy, which runs on the processor only; on the A100 the GPU would sit idle. Each system's `Rows` now has a torch form (`torch_residual(device)`; for P2-16 the module function `torch_residual(rows, device)`): the same matrices, moved to the device once, and the same pointwise nonlinearity. The gradient is autograd's. The alternatives were torch on the A100 with item 2's numpy on the CPU (the two devices' times would then mix code and hardware, and the numpy path forms the M x P Jacobian at every evaluation) and the numpy path inside the A100 job (the run would be labelled A100 without using it). For Buckley-Leverett, f'(S) is the physics' `flux_derivative_differentiable`, so the gradient carries the f''(S) S_x term; the detached `flux_derivative` drops it (the entry on LiL-N's plateau). Without a graph the detached form is used, under `enable_grad`, for the same values.
- **Checks** (cutoffs set before any comparison): the torch residual and the autograd gradient of ||N||^2 against numpy's N and 2 A(beta)^T N, at beta0 plus a perturbation, on the CPU and CUDA, within 1e-10 relative: measured 1e-16 to 8e-16 for all three systems.
- **The run check, changed after it failed (the user's decision, 16 November 2026).** The planned check, the first 100 L-BFGS losses against item 2's numpy path on Burgers p = 10 within 1e-6, failed at 2.2e-5. Two numpy runs agree exactly. The torch and numpy runs agree within 3e-12 through iteration 30, while the loss falls from 5 to 2.3e-3. Near the minimum (1.0464e-3) L-BFGS amplifies the 1e-16 difference of the gradients: 1e-8 at iteration 40, 2.2e-6 at 50, then back to 1e-12 by iteration 100. Both stop by the stall rule (at 106 and 132 iterations) at the same minimum, final losses 4e-14 apart. A window short enough to pass would be chosen from this answer, so the check is now what does not depend on the path: both runs stop by the stall rule, and their final losses agree within 1e-8 relative.

**Starting point.** LiL-N (`trf` and L-BFGS) starts where LiL-Q starts, since it minimizes the same objective: zero for Bratu and Burgers, Package 3's fit of the initial condition for Buckley-Leverett (`p3_2_bl_certified.initial_guess(..., 'ic')`).

**`trf` as in item 2:** tolerances 1e-12, at most 200 evaluations, the analytic Jacobian. Section 6 says the trust-region runs "end by their own criteria", and scipy refuses tolerances that are all zero. Its iterates are logged at the Jacobian's evaluations, which scipy 1.13 makes at the start and at every accepted iterate; the final `x` is the last logged one (checked per run, `final_x_is_last_logged`).

**The clocks start at the solver.** For every method the collocation set, the basis matrices and, for L-BFGS, their move to the device are built before the clock and timed beside it (`setup_s`), as item 2's clocks. Package 4's LiL-Q timings included the construction of the matrices; C3 compares values (errors and the rule step), not times.
- **LiL-Q, `trf`, square collocation: "the usual protocol", per iterate.** One logged pass keeps every iterate and serves as the warm-up (as item 2's protocol runs). Then three timed passes record a timestamp at every iterate and nothing else, each the median of five if a pass is under 1 s (Package 4's protocol). An iterate's time is the median of its timestamps. Every timed pass must give the logged pass's iterate count and final coefficients within 1e-10 relative (`timing_matches`); the laptop runs give a difference of 0. The square collocation keeps P2-3's measure: the assembly and solve times summed over the Newton steps before the iterate, its settings (K_max 20, tolerance 1e-9) unchanged, SQ-CGL at p = 7, 12, 17 ((p - 2)^2 = 25, 100, 225 free coefficients).
- **LiL-N by L-BFGS:** one logged pass carries the timing (Section 6), the error every 10 iterations off the clock (`RunProbe`), the 600 s budget, tolerances 0 (`stall_rule='f1'`), no target, no cap.
- **The Bratu residual alone.** `p2_16_certified.assemble(rows, beta, system=False)` returns the residual without building A and f. Item 1's `ScalarRows.residual` built the whole system to return it; on a timed run that would put an unneeded matrix on LiL-Q's and `trf`'s clocks. The arithmetic of R is unchanged (the same expressions; tested equal to the full assembly's within 1e-14).

**Stored coefficients.** LiL-Q and `trf`: every iterate (`betas.npy`), as Section 2 asks of the LiL runs. L-BFGS: the coefficients at every logged iterate (`betas_logged.npy`, the ones its errors come from) and the last. Every iterate of a 600 s L-BFGS run would be up to some 10^5 vectors.

**C3** (LiL-Q values of Burgers equal Package 4): at every p, the same rule step as Package 4's item 1a log, and the error of every iterate up to the returned one within 1e-8 relative of that log's `eps_ref` (the cutoff of Addendum 2's K1, set before the runs).

**First laptop look (untimed, smallest sizes).** LiL-N L-BFGS stalls at the minimum within a fraction of a second (Bratu P = 25: 20 iterations; Burgers P = 36: 47; Buckley-Leverett P = 64: 113 on the CPU, 114 on CUDA), and `trf` and L-BFGS end at the same error (Buckley-Leverett P = 64: 1.510e-2; LiL-Q returns 1.396e-2 by its rule). **At the largest sizes** (laptop, logged passes untimed, L-BFGS to a 120 s budget): LiL-Q 0.18 s (Bratu 225), 2.8 s (Burgers 900), 4.6 s (Buckley-Leverett 1,024); `trf` 0.44 s, 5.0 s, 14.7 s (19 evaluations), at LiL-Q's error to 4 figures; L-BFGS on the CPU stalls at 2.7 s and 52 s on Bratu and Burgers and reaches the budget on Buckley-Leverett (4.85e-5 against `trf`'s 4.844e-5); on the laptop GPU it stalls at 72 s on Bratu (8,361 iterations against the CPU's 1,347: another round-off path) and reaches the budget on Burgers and Buckley-Leverett. So the 14 L-BFGS runs per device are costed at 600 s as an upper bound that the larger sizes reach, and the protocol runs of LiL-Q and `trf` are a few minutes in all.

---

## 2026-11-16 -- Package 5, Addendum 2, Sections 2 and 3: the 18 runs and their summary (`experiments/p5_a2_plateau.py run | k1 | summarize`)

The addendum asks for 18 runs of gravity Buckley-Leverett, every one from the least-squares fit of the initial profile held constant in time, K_max = 60, the rule read from the log. Item 1's convention throughout: k_s is the rule's step and the returned iterate is k_s + 1.
- **A:** item 1's configuration, both of its trial spaces, P = 256, 576, 1,024, weight 10^4.
- **B:** P = 576 and 1,024 at diffusion 0.05 and 0.02.
- **C:** P = 1,024 at weight 10.
- **D:** the paper's run (cosine x Fourier, random tensor grid seed 42) at weights 10 and 10^4, every iterate stored.

**The initial profile.** For A to C this is the guess item 1 already uses for viscous Buckley-Leverett: Package 3's `'ic'`, the initial saturation minus the lifting, fitted by `pretrain_lil`. Part D uses the delivered run's own `initial_guess='ic'`. Its iterate 0 is captured by wrapping `pretrain_lil`; every later iterate comes from the solver's `diagnostics_callback`. Check: the stored coefficients must match the log, the same count and each iterate's L2 error equal to the log's `eps_ref` within 1e-8. Otherwise the run raises.

**Diffusion.** `p3_2.physics_for(case, nu)` and `p5_1.bl_setup(..., nu)` take the diffusion; 0.1 is the default, so nothing that existed changes. The references for 0.05 and 0.02 are Package 2's finite-difference refinements. **The reference error** (his rule: errors are reported only if the reference's error is below a third of the run's) is the relative difference of each reference's last refinement step: 3.4e-7, 7.3e-7 and 5.7e-7 for diffusion 0.1, 0.05 and 0.02.

**The norm of X** is used only at diffusion 0.1, against item 1's fine reference (p_ref from item 1's `chosen_gravity.json`, on its quadrature, gated as item 1 is). Item 1 has no fine reference at 0.05 or 0.02, so part B reports eL2 and delta_L2 only. Section 3 asks for the error and its ratio to delta_P, which L2 gives.
- Part D's eX is on the same quadrature, with the paper space's derivatives.
- Its delta_P is given in L2 (the reference's grid) and in H^{2,1}, each by gelsy and by gelsd at rcond = eps, with the ranks. The cosine x Fourier space may be rank-deficient; the ranks say whether the two drivers agree.

**The stretch** (Section 3): "the longest stretch of consecutive steps before the stop over which the nonlinear residual changes by less than 5% per step".
- Steps k -> k + 1 with k + 1 <= k_s, judged on `norm_R_h`. With no stop, all 60 steps count.
- On his own run this gives two stretches of two steps: 4..6 (the plateau) and 11..13 (the rule's approach). His plateau, k = 4 to 8, is not a single stretch, because the step 6 -> 7 changes the residual by 8.4%.
- So `terminal.csv` reports the longest stretch (the earliest on a tie) and the number tied (`n_longest`). `stretches.csv` lists every stretch of two or more steps, so the plateau and the approach to the stop are both visible.
- Errors are withheld wherever the one-third rule withholds them.

**Threads.** Parts A to C and K1 run with item 1's 4 BLAS threads, and part D with the delivered run's 24. That way K1 and K2 compare like with like (item 1's C1 compared runs on the same configuration). This is why the script has separate steps (`run --part`, `k1`, `summarize`): the thread count is fixed when BLAS loads.

**The Grace stage** is `p5a` (`results/package5_a2`, its own commit lock): one cpu-mem job, 1 h, 24 SU requested and about 10 expected (`scripts/cluster/package5/p5_a2.slurm`, generated by `_generate_jobs.py`; submitted by `submit_p5_a2.sh`).
- Every input is checked first.
- Then five processes (A and B by trial space, C) and K1 run side by side at 4 threads each; each process holds about 8 GB while it builds item 1's norm of X (measured).
- Then part D runs at 24 threads, followed by the summary and the tarball.
- A failed process does not stop part D, the summary or the tarball.

**The checks.**
- **K1:** item 1's two gravity runs at P = 1,024, weight 10^4 (zero start), rerun in the stage. It requires the same rule step, and eps_ref and eX at iterates 0..k_s + 1 within 1e-8 relative of the delivered runs.
- **K2:** part D at weight 10 against the delivered `P2_9_bases/bl_gravity_paper_P1024_ic`: norm_R_h, norm_Rlin_h and eps_ref at every iterate within 1e-8. The largest difference up to the delivered first stall is reported too.
- The tolerance is item 1's C1 (`p5_checks.py`). If K2 fails beyond the stall only, the memo says so: the plateau lies before it.

**Tests:** `tests/test_p5_a2_plateau.py`.
- The 18 runs; the stretches.
- The summary on the delivered run's own first 16 iterates: rule step 13, returned iterate 14, 0.0116, the tie 4..6 and 11..13.
- The one-third rule withholding errors; the gate on the X ratio; K2 passing on a copy and failing at the iterate that was moved.

---

## 2026-11-16 -- Package 5, Addendum 2, Section 3a: the update histories of delivered runs (`experiments/p5_a2_plateau.py updates`, `lilq.xnorm.relative_updates`)

The advisor's Addendum 2 of 15 November 2026 asks for the relative update of the function per iterate, ||u^(k+1) - u^(k)|| / ||u^(k+1)||, in L2 and in the norm of X, from the stored coefficients of 168 delivered runs: item 1 (96), the Poisson-Boltzmann runs with the Chebyshev space (Package 4 item 2a, 18; item 4a, 9), and the random-feature runs of item 4b (45). Post-processing only.

**The definition** (checked against his two values before anything was built on it). Each item's own norm and quadrature:
- item 1: H2 (Bratu) or H^{2,1} (Burgers, Buckley-Leverett) on item 1's Gauss-Legendre rule, `n_gl` from each run;
- Poisson-Boltzmann and random features: H2 on `Norms`'s fine quadrature, the one their errors used.

Row k of `updates_by_iterate.csv` is the update into iterate k (k = 1..K). For the lifted sine space the lifting is part of u: it cancels in the difference and stays in the denominator. His values agree to every figure he gives: Poisson-Boltzmann p = 22, seed 0, H2 1.0, 0.063, 2.3e-5, 1.4e-6, then 1.2e-6 (his "near 1e-6"); viscous Buckley-Leverett, tensor Chebyshev, P = 64, weight 10^4, H^{2,1} 0.99, 2.2, 0.38, 0.25, 0.15, 0.093.

**Where the files go.** The 150 Package 5 runs get the file in their own run folders (on `package5-results`), as he asks. Package 4's 18 runs get it on `package5-results` too, under `P5_A2_initial_profile/updates_package4/<their Package 4 path>/`. That way the delivered `package4-results` branch stays as delivered. Our call; the note says so.

**Tests:** `tests/test_p5_a2_plateau.py` (the function by hand and against a direct sum, the folder names of the Poisson-Boltzmann spaces, item 1's runs in place).

---

## 2026-11-15 -- Package 5, item 3: the network runs to the 600 s budget (`experiments/p5_3_networks.py`)

Section 6 of the instructions: NiL-N (L-BFGS), NiL-Q (L-BFGS) and NiL-N (Levenberg-Marquardt), "as in the four-method runs" and "as in P2-8", seeds 0, 1, 2; "Optimizer tolerances zero. No target loss. Each optimizer run ends at a wall-clock budget of 600 s."

**The same runs, by construction** (as P2-8). The driver calls each benchmark's own `run_nil_n` / `run_nil_q` with `paper_setup`'s configuration and the seed. It replaces only the solver by a wrapper, so the network, pretraining, collocation set, data and loss weights are the Package 1 code. The wrapper sets:
- `R_tol = 0` and `stall_rule='f1'` (tolerances 0, one restart, then `optimizer_stall`);
- no iteration, evaluation or outer-iteration cap (10^9). The budget (`RunProbe`) is the only limit.
- **NiL-Q keeps its inner cap per linearization** (Bratu 300/400, Burgers and BL as in `paper_setup`). It sets how many L-BFGS steps NiL-Q takes before it relinearizes, so it is part of the method. The outer cap (25 and so on) only stopped the run, so the budget replaces it. This is my reading of "ends at a wall-clock budget"; it does not need the advisor.

LM is P2-8's `lm()` with target 0, no iteration cap and its wall-clock cap set to the budget. `lm()` now also returns `off_clock_s`, the time of its error log; it changes nothing else in P2-8.

**The short runs** (the warm-up, checks C2 and C4) stop at a number of L-BFGS iterations through the probe (`RunProbe(max_iterations=...)`, added here). For NiL-Q that counts its inner iterations over the run, so the cap means the same for every method; the solvers' own caps stay at 10^9.

**The error** is `eps_ref`, the four-method runs' relative L2 error on the test grid against the Section 6.2 reference (BL: the finite-difference reference), through P2-8's `eps_ref_function`. **One reference per problem for every method of item 3.** Bratu and Burgers: the p = 48 spectral reference and Cole-Hopf. The networks and item 1's LiL runs already share them, on the same 201 x 201 grid. BL: the stored `bl_fd_ref_viscous_nu0.1.npz` of Packages 2 and 3, which the LiL runs use. The four-method network runs computed theirs (`reference_solution`): the same physics (nu = 0.1, M = 0.5, T = 0.4), the same grid, and a relative difference of 1.5e-7, which would matter only near the 1e-6 level. The error is logged off the clock at iteration 0, every 10 L-BFGS iterations (every LM iteration), and at the last iterate. **The clock** starts at the first optimizer iteration, as `training_time`. Pretraining (`pretrain_s`) and setup (`setup_s`) are timed beside it, not on it. Whether pretraining belongs on the clock is a question for the advisor, asked once in the cost check.

**Warm-up.** Each job runs every method for 5 iterations at the smallest size, untimed, before its first run. On the laptop the first of these passes in a fresh process (NiL-N, 5 iterations at Burgers P = 25) took 5 s, against 3.5 ms per iteration in the budget run that followed (2,830 iterations in 10 s). Item 2's re-time saw a one-time cost of about 29 s on Grace.

**C1** (one run per problem against its v2.0.0 run, first 100 iterations).
- **Which runs.** Package 1 (`results/package1_v2.0.0`) ran on the CPU node only at each problem's largest size (Bratu 225, Burgers 625, BL 1,024), and on the A100 at every size. C1 is NiL-N, seed 0, at that size on the CPU node. Every other run that has a Package 1 counterpart is compared too, and reported beside it.
- **The comparison.** The loss at iterations 0 to 100, relative difference, cutoff **1e-6**. This was set before any comparison, from what separates "the same run" from "another run":
  - A different setup changes the loss at iteration 0 by 1e-3 or more.
  - Package 1's 54 stall-rule controls, each a separate Grace job, matched their original runs from iteration 0 with no departure over thousands of iterations, on the CPU and on the A100.
  - On different hardware, round-off grows through L-BFGS. The laptop against Grace CPU: 1e-16 at iteration 0 and 1e-13 at iteration 10, but 1e-2 by iteration 100 (Bratu, Burgers).
  - BL's pretraining fit amplifies round-off across hardware: Package 1's own BL P = 1,024 seed 0 starts at 23.21 on the CPU node and 22.06 on the A100, and the laptop gives 22.68. So C1 is meaningful only on the same device class, which is how it is run.

**C2 and C4** (`c2_c4`): NiL-N (300 iterations) and LM (30 iterations), seed 0, at the largest size, each without and then with the error log, after the warm-up.
- C2: the same parameters and the same loss history, bit for bit.
- C4: the clocks within 5% (the instructions).

The laptop rehearsal at Burgers P = 100 gave identical iterates for both methods. Its runs were under 1 s, too short to judge 5%, so the Grace runs are sized to tens of seconds (batch 5).

**Stalls before 600 s** (what the cost depends on): Package 1's 54 controls with tolerances 0 ran to 10,000 to 20,000 iterations (100 to 950 s on Grace) and none stalled. So the cost estimate keeps every budget run at its full 600 s.

**Tests:** `tests/test_p5_3_networks.py` (9) and `tests/test_run_probe.py` (11).

---

## 2026-11-15 -- Package 5, item 3: the error along a run and the wall-clock budget, in the shared L-BFGS loops (`lilq/metrics.py` `RunProbe`, `lilq/solvers.py`)

Item 3 runs every method for a fixed wall-clock budget and reads off the time at which each run first reaches a common accuracy. So the network and LiL-N runs need two things they did not have: the error of the current iterate logged along the run, and a stop when the run's clock reaches the budget. Both go into the existing loops as one optional argument, `probe`. There is no copy of a loop.

**What changed.**
- `MetricsTracker` gains a pausable clock. `off_clock(fn)` runs `fn` and adds its time to `off_clock_s`. `elapsed()` and `wall_time` exclude that time. On a GPU the queued work is synchronized before and after, so the solver's own work stays on the clock and the evaluation's work does not.
- `RunProbe(error_fn, every=10, budget_s=None)` logs a row `(iteration, evaluations, time_s, loss, error)` at iteration 0, every `every` iterations, and at the last iterate. `time_s` is the `wall_time` the run recorded for that iterate. The error is evaluated off the clock. Once the clock reaches `budget_s`, the probe says so and the loop stops after that iteration. The iteration that crosses the budget completes and is logged.
- `_lbfgs_loop`, `solve_nil_n`, `solve_lil_n` and `solve_nil_q` take `probe=None`. NiL-Q checks the budget in its inner loop and stops its outer loop too. The summaries gain `budget_reached` (False without a probe).
- The budget is checked after the converged test, so a run that meets its own stopping rule in the same iteration reports converged, not the budget.

**What does not change.** Without a probe every loop is as before: same iterates, same records, same summaries apart from the new `budget_reached` field. With a probe the iterates are the same bit for bit (check C2 of the item 3 plan), because `error_fn` must not touch the parameters or their gradients.

**Tests** (`tests/test_run_probe.py`, 10):
- C2 on small NiL-N, NiL-Q and LiL-N problems: identical parameters and loss histories with and without a probe;
- the cadence of the rows;
- the error time excluded from `wall_time`, the rows' times and the budget;
- the budget ending NiL-N and NiL-Q's outer loop;
- `off_clock` waiting for queued GPU work (skipped without CUDA).

**Revisit if** an error evaluation must see the gradients, or a method outside these loops needs the same log: LM for networks and trust-region LiL-N log from their own drivers (later batches).

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
