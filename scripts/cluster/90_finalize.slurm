#!/bin/bash
# After wave 4 (not timed): assemble results/package1 (package1_results/)
# from the four wave roots, each locked to its own commit, with WAVES.json
# naming them (scripts/cluster/assemble_package.py); then reference/ and
# code/, the merged four-method table, check B1, the residual-band figures
# (task B5), the Section 4.6 Kovasznay comparison, the merged B8 and B9
# tables (B9's allocation and resume columns filled first), and the run
# index. Submitted with wave 4 (LILQ_WAVE=4). Minutes.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-finalize
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python scripts/cluster/assemble_package.py --results "$RESULTS" --out "$RESULTS/package1"
export PKG="$RESULTS/package1"
export A="$PKG/A_calibration" B="$PKG/B_instrumentation" C="$PKG/C_oversampling"

python experiments/component_b.py --save-reference --save-code --out-root "$PKG"
python experiments/four_method_tables.py --out-dir "$B" --merge-from "$B"/four_method_jobs/*/
python experiments/component_b.py --reproduction-check --out-root "$PKG"
python experiments/residual_band_figures.py --from-logs "$PKG" --pass paper --out-dir "$B/residual_band_figures/paper"
python experiments/residual_band_figures.py --from-logs "$PKG" --pass kmax  --out-dir "$B/residual_band_figures/kmax"
python experiments/kovasznay_comparison.py --package "$PKG"
# B9 tables written before the allocation / resumed_at columns (wave 3): fill them
# (the advisor's follow-up of 1 October, item 4).
python experiments/darcy_fv_comparison.py --annotate "$B/darcy_fv"
python - <<'PY'
import csv, glob, os
from pathlib import Path
from experiments.component_b import write_index
B = Path(os.environ["B"])
write_index(B)                                    # every run job wrote it; rebuild once
rows, lil_seen = [], set()
for path in sorted(glob.glob(str(B / "darcy_fv" / "*" / "darcy_fv_comparison.csv"))):
    for r in csv.DictReader(open(path)):
        if r["method"] == "LiL":                  # recomputed by every task: keep one per field
            if r["field"] in lil_seen:
                continue
            lil_seen.add(r["field"])
        rows.append(r)
if rows:
    with open(B / "darcy_fv_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
b8 = [r for path in sorted(glob.glob(str(B / "b8_jobs" / "*" / "b8_initial_guess.csv")))
      for r in csv.DictReader(open(path))]
# Wave 3's tables have no K_max column, wave 4's K_max = 60 reruns do: the union
# of the columns, and wave 3's LiL-Q K_max from its iterations_cap.
b8_cols = []
for r in b8:
    if r["method"] == "LiL-Q" and not r.get("K_max"):
        r["K_max"] = r.get("iterations_cap", "")
    b8_cols += [c for c in r if c not in b8_cols]
if b8:
    with open(B / "b8_initial_guess.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=b8_cols, restval="")
        w.writeheader(); w.writerows(b8)
index = list(csv.DictReader(open(B / "runs_index.csv")))
fm = list(csv.DictReader(open(B / "four_method_tables.csv")))
print(f"Section 3.3 runs: {len(index)} ({sum(r['status'] != 'ok' for r in index)} failed)")
print(f"four-method rows: {len(fm)} ({sum(r['stopping_reason'] == 'failure' for r in fm)} failures)")
print(f"B9 rows: {len(rows)} (NiL: {sum(r['method'] == 'NiL' for r in rows)} of 24: 4 fields x 3 seeds x 2 precisions)")
print(f"B8 rows: {len(b8)}: 128 of wave 3 and its LiL-Q reruns with K_max = 60 "
      f"({sum(r['stopping_reason'] == 'failure' for r in b8)} failures)")
PY
# LiL-Q's four-method rows from the clean times, last: it fails the job if a
# clean run's iterations differ from its logged pass's (reply on wave 3, 2.2).
python experiments/four_method_tables.py --out-dir "$B" --lilq-from-clean-timing
