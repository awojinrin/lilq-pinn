# Package 3: the corrections after your reply

**To:** R. Younis. **From:** G. Awojinrin. **Date:** 10 October 2026.
**Refers to:** your reply of 10 October.

**Code:** `awojinrin/lilq-pinn`, branch `v3-dev` at bfbac47. This is one commit after 086461c,
the commit of the report; `DECISIONS.md` describes it in the entry "after the advisor's reply".

**Results:** the branch `package3-results`, under `results/package3/`. `report.md` is revised in
place: a "Revised" line, and its new Section 4.5.

## Summary

The three corrections are made, without reruns. Each one recomputes from Grace's files:
- **Item 4's ranks:** the rank at the paper's threshold, beside gelsy's.
- **Beltrami's pressure δ_P:** the least squares is now truncated at the paper's threshold.
- **The BLAS:** OpenBLAS 0.3.27, through FlexiBLAS 3.4.4.

Nothing else changed in the results, and nothing is needed from you. As you asked, the release
waits for the freeze, and for you to add me to the organization on GitHub. That covers the
merge into `main`, the tag `v2.1.0` and the Zenodo version.

## 1. Item 4's ranks

`P3_4_burgers_large_P/terminal.csv` now has two columns beside gelsy's `rank` and
`kappa_retained`:
- `num_rank_svd`, at the last iterate;
- `num_rank_svd_min`, over the run.

At the paper's threshold, max(N, P) σ₁ ε_mach, the ranks are 625, 889, 996 and 1,155 at
P = 625, 900, 1,024 and 1,225, against your 889, 996 and 1,155. Our minimum over the run equals
the last iterate's; your 890 at k = 0 for P = 900 is above it. The assembly recomputes every
row from Grace's files and stops if any column Grace wrote would change; none did.

The table in the report's Section 3 and the sentence in its Section 4.2 are corrected. Items 1
and 2 are full rank at both thresholds (κ at most 3e4), so their statements and K7 stand.

## 2. Beltrami's pressure δ_P

**What we used.** gelsy with rcond = ε_mach, as in the solves.

**Why it was high.** The matrix [pressure basis | 11 time-level columns] has N_p dependent
columns, as you found: 5, 6 and 8 at B1, B2 and B3. At rcond = ε_mach gelsy kept them, the
coefficients grew to 10⁸–10¹¹, and round-off raised the residual.

**The fix.** `delta_P` now truncates the pressure least squares at the paper's threshold,
max(m, n) ε_mach. On the laptop, an SVD projection and a pivoted-QR projection agree with it
to 1e-11 at all three sizes. A test checks it against the truncated SVD on a small case, where
the old setting is off by 1.7e-4.

The assembly re-evaluates the values. `terminal.csv` keeps Grace's beside them
(`delta_P_p_grace`, `delta_P_combined_grace`, and `delta_P_reevaluated.json`):

| size | δ_P (p), Grace | δ_P (p), now | change |
|---|---|---|---|
| B1 | 4.131611e-3 | 4.131174e-3 | −1.06e-4 |
| B2 | 4.987758e-4 | 4.986755e-4 | −2.01e-4 |
| B3 | 1.554235e-5 | 1.554234e-5 | −8.1e-7 |

The velocity δ_P, recomputed in the same pass, equals Grace's to 4e-13. The pressure ratio
error/δ_P moves from 30.451 to 30.454 (B1) and from 47.845 to 47.854 (B2). At B3 it stays
145.884 to that digit. The combined ratio moves by at most 1.2e-4 relative.

## 3. The BLAS library

The provenance job ran in the runs' environment (job 20075205, `provenance_cpu/`). The thread
pools it loaded (threadpoolctl) are:
- FlexiBLAS 3.4.4 (`libflexiblas.so.3.4`);
- dispatching to OpenBLAS 0.3.27 (`libopenblas_skylakexp-r0.3.27.so`, the Skylake-X kernel);
- at 48 threads.

So the manuscript's "OpenBLAS 0.3.27" is right; FlexiBLAS is the layer in between.
`environment.txt` and the report's Section 4.4 now name both.

## Files

In `results/package3/` on the branch:
- `report.md`: revised; Section 4.5 has the corrections.
- `P3_4_burgers_large_P/terminal.csv`: with the two rank columns.
- `P3_1_beltrami_certified/terminal.csv`: the pressure δ_P and its ratios, with Grace's
  values beside them.
- `P3_1_beltrami_certified/delta_P_reevaluated.json`.
- `environment.txt`.
- `DECISIONS_package3.md`, `provenance.json`, `assembly_log.txt`.
