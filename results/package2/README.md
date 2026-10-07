# Computational Package 2 results (Stages 1 and 2)

This is `package2_results/`, the Section 12.1 layout of Computational Package 2, as sent
to the advisor with `report.md`. The branch starts at `7af8855` on `v3-dev`, the commit the
results were assembled at. The branch's own commit adds only `results/package2/`.

**Start with `report.md`** (Stage 2) and `report_stage1.md` (Stage 1). Each ends with a
list of its files.

**Commits:**
- **Stage 1:**
  - the release `v2.0.0` (5339f60), checked by gate G1;
  - Stage 1's Grace job at `750f0b6` (`code/COMMIT_stage1`);
  - check C1 at `c1e7f4c`.
- **Stage 2:** `4f327f7` (`code/COMMIT_stage2`). That covers Grace's jobs and FASTER's
  item 5.
- **Item 7's A100 series:** the GPU addendum at `e5eb447` (`code/COMMIT_stage2_gpu`).
- **The assembly:** `7af8855`, after the runs, with `--later-commit`.
- **The changes between them** are listed in `code/FILES_CHANGED_IN_GPU_ADDENDUM.txt` and
  `code/FILES_CHANGED_FOR_ASSEMBLY.txt`. `provenance.json` records them all, and
  `DECISIONS.md` on `v3-dev` gives the reasons.

**What is not here:**
- **The raw cluster tarballs** (`package2_stage1.tar.gz`, `package2_stage2.tar.gz`,
  `package2_stage2_faster.tar.gz`, `package2_stage2_gpu.tar.gz`). Everything in them that
  the results use is here, assembled. The tarballs are kept by the authors.
- **The Zenodo staging of `package1`.** That record is published: DOI
  10.5281/zenodo.23147620.

**Redactions.** Before publishing, these were replaced by placeholders in 261 text files
(listed in `REDACTED_FILES.txt`):
- the cluster user's scratch path (`$SCRATCH`);
- the HPRC allocation accounts (`<account>`);
- the laptop's home folder (`<laptop-home>`);
- the user name (`<user>`);
- NumPy's own build-machine paths in its configuration dump (`<numpy-build-runner>`).

Nothing else changed:
- every number, every binary file (`*.npz`, `*.pt`, figures) and every CSV's shape is as
  assembled;
- the only numbers removed are the account numbers.

The unredacted originals are kept by the authors.

**Laptop previews:** `laptop_preview/`, with its own README. These are the runs that
`report.md` sets beside the Grace results under each item.
