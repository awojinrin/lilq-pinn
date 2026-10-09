# Computational Package 3 results

This is `package3_results/`, in the Section 9 layout of Computational Package 3 and Section 6
of its Addendum 1. It was sent to the advisor with `report.md`. The branch holds `v3-dev`
at `086461c`, the commit the results were assembled at, and adds only `results/package3/`.

**Start with `report.md`.** It ends with a list of its files.

**Commits:**
- **The runs:** every Grace job ran at `97e463d` (`COMMIT`), under the source lock.
- **After the runs:**
  - `86cc344` corrects check L1's comparison of P2-12's terminal row (`report.md`, Section 4.1);
  - `3e13572` adds the assembly, `experiments/p3_assemble.py`;
  - `086461c` lets it take the hardware record written after the runs (`provenance_cpu/`).
- **Where they are recorded:** `provenance.json` records the runs' commit, the assembly's and
  the commits between. `DECISIONS_package3.md` gives the reasons.

**What is Grace's.** Every run folder and every summary is as Grace wrote it, except L1.
L1 was re-evaluated at the assembly, on Grace's files; Grace's own verdict is kept beside
the new one in `P3_4_burgers_large_P/checks_item4.json`.

**What is not here:** the raw cluster tarballs (`package3.tar.gz`, after wave 1 and after
B3 level 2). Everything in them is here, assembled. The tarballs are kept by the authors.

**Redactions.** Before publishing, these were replaced by placeholders in 11 text files
(listed in `REDACTED_FILES.txt`):
- the cluster user's scratch path (`$SCRATCH`);
- the HPRC allocation account (`<account>`);
- the user name (`<user>`);
- the laptop's home folder (`<laptop-home>`).

Nothing else changed. Every number, every binary file (`*.npy`) and every CSV's shape is as
assembled; the only numbers removed are the account numbers. The unredacted originals are kept
by the authors.

**Laptop previews:** `laptop_preview/` holds the laptop rehearsals of 8 October, including
the batch 0 check K1 against the advisor's `expected_constants.py`.
