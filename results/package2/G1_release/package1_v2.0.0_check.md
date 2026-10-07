# package1 rebuilt at v2.0.0 on Grace: comparison with the final package1

**The job.** Package 2, Section 2, item 2.
- **Run:** Grace job 19963045, 4 Oct 2026, 7 min 26 s on 24 cores, about
  3 SU. The job file is `grace_jobs/package1_v2.0.0.slurm`.
- **Method:** it rebuilt `package1` by copying the four wave folders, with
  the tag's own `90_finalize` steps, at `5339f60` (`v2.0.0`). Nothing was
  rerun.
- **Output:** `results/package1_v2.0.0/package1` (413 MB), with
  `provenance.json`. That file records the release, its commit and the wave
  commits: 8a3f5f7, 905e58c, 905e58c and 17b3539, status final.
- **Backup:** `$HOME/lilq-results/package1_v2.0.0.tar.gz`.

**The comparison** with the final `package1` assembled on the laptop
(`package1_local`, from the downloaded waves) used every file's SHA-256
(`manifest.csv`). For the files that differ, the Grace copies were
downloaded (`check_files.tar.gz`) and compared by content.

| Files | Result |
|---|---|
| 3,282 | byte-identical: every run log, table, model and summary |
| 28 | identical once line endings are converted (written on Windows vs Linux) |
| `B_instrumentation/runs_index.csv`, `stopping_rule/stall_columns.csv` | the same rows (63 and 306) in a different order. The rows are sorted by path, which Windows does without regard to case and Linux does not. |
| 11 reference solutions (`reference/*.npz`, recomputed by the finalize step) | Bratu and Burgers bit-identical; the others agree to 2e-16 – 1.4e-13 relative (round-off of the finite-difference and finite-volume solves on another BLAS) |
| 17 residual-band figures and `eps_u_vs_time.pdf` | the same curves and values. Grace has no Times New Roman, so matplotlib draws the text in DejaVu Serif, and the axes shift by a pixel or two. These are package-side figures; the manuscript's are made by its own scripts and regenerate pixel-identical. |
| `code/` (100 files) | the saved source at the tag; the laptop assembly saved an earlier commit's |
| `WAVES.json` | the commit that assembled the package (now the tag) and line endings |
| `provenance.json` | new |
| 16 `collocation.npz` (B10) | only on Grace. The downloaded wave archives left out `.npz` files, so the Grace copy is the more complete one. |

**Conclusion.** The rebuild at the tag reproduces the final `package1`.
Every result is identical or agrees to round-off, and the only other
differences are where the files were written and with which fonts.
