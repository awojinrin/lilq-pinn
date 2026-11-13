# Computational Package 4 results

This is `package4_results/`, in the Section 7 layout of Computational Package 4 (the advisor's
instructions of 12 November 2026). The branch holds `v3-dev` at `e949cc9`, the commit the results
were assembled at, and adds only `results/package4/`.

**Start with `report.md`.** It ends with a list of its files.

**Commits:**
- **The runs:** every Grace job ran at `8a1e849` (`COMMIT`), under the source lock.
- **After the runs:** `e949cc9` adds the assembly, `experiments/p4_assemble.py`, and the helpers
  it shares with Package 3's (`experiments/assemble_common.py`). No code a run uses changed.
- **Where they are recorded:** `provenance.json` records the runs' commit, the assembly's and
  the commits between. `DECISIONS_package4.md` gives the reasons.

**What is Grace's.** Every run folder, table, log and timing is as Grace wrote it. The cluster
did not have the advisor's `expected_burgers.json` and `expected_pb.json`, so the assembly
re-evaluated the checks on Grace's files with them:
- `checks.json`, `checks_item1.json` and `P4_2_pb_inclusions/checks_item2.json` are the
  re-evaluated checks (M0 to M9, with M1, M8 and M6's values at the five points added);
- Grace's own are beside them as `*_grace.json`. Every verdict Grace gave is unchanged;
- the summaries also rewrote the tables from Grace's logs. The tables kept here are Grace's;
  every column agreed exactly, except delta_P (the error of the best approximation) of item 2 and
  the ratios to it, whose round-off differs from machine to machine at P >= 484
  (`delta_P_across_machines.json`).

**What is not here:**
- the raw cluster tarball (`package4.tar.gz`). Everything in it is here, assembled. The tarball is
  kept by the authors;
- the advisor's `expected_burgers.json` and `expected_pb.json`. `provenance.json` records their
  SHA-256.

**Redactions.** Before publishing, these were replaced by placeholders in 10 text files
(listed in `REDACTED_FILES.txt`):
- the cluster user's scratch path (`$SCRATCH`);
- the HPRC allocation account (`<account>`);
- the user name (`<user>`);
- the laptop's host name, where the bundle was built (`<laptop-host>`).

Nothing else changed. Every number, every binary file (`*.npy`, `*.npz`) and every CSV is as
assembled; the only numbers removed are the account numbers. The unredacted originals are kept by
the authors.
