#!/bin/bash
# After everything else (not timed): reference/ and code/, the merged
# four-method table, check B1, the residual-band figures (task B5), the
# Section 4.6 Kovasznay comparison, the merged B8 and B9 tables, and the run index.
# Minutes.
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-finalize
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/component_b.py --save-reference --save-code --out-root "$PKG"
python experiments/four_method_tables.py --out-dir "$B" --merge-from "$B"/four_method_jobs/*/
python experiments/component_b.py --reproduction-check --out-root "$PKG"
python experiments/residual_band_figures.py --from-logs "$PKG" --pass paper --out-dir "$B/residual_band_figures/paper"
python experiments/residual_band_figures.py --from-logs "$PKG" --pass kmax  --out-dir "$B/residual_band_figures/kmax"
python experiments/kovasznay_comparison.py --package "$PKG"
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
if b8:
    with open(B / "b8_initial_guess.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b8[0]))
        w.writeheader(); w.writerows(b8)
index = list(csv.DictReader(open(B / "runs_index.csv")))
fm = list(csv.DictReader(open(B / "four_method_tables.csv")))
print(f"Section 3.3 runs: {len(index)} ({sum(r['status'] != 'ok' for r in index)} failed)")
print(f"four-method rows: {len(fm)} ({sum(r['stopping_reason'] == 'failure' for r in fm)} failures)")
print(f"B9 rows: {len(rows)} (NiL: {sum(r['method'] == 'NiL' for r in rows)} of 12)")
print(f"B8 rows: {len(b8)} of 128 ({sum(r['stopping_reason'] == 'failure' for r in b8)} failures)")
PY
