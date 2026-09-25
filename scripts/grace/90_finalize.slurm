#!/bin/bash
# After everything else (not timed): reference/ and code/, the merged
# four-method table, check B1, the residual-band figures (task B5), the
# Section 4.6 Kovasznay comparison, the merged B9 table, and the run index.
# Minutes. Partition: set by submit_all.sh (CPU_PARTITION).
#SBATCH --job-name=lilq-finalize
#SBATCH --time=01:00:00
#SBATCH --account=132698954494
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x.%j.out

source scripts/grace/env.sh
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
index = list(csv.DictReader(open(B / "runs_index.csv")))
fm = list(csv.DictReader(open(B / "four_method_tables.csv")))
print(f"Section 3.3 runs: {len(index)} ({sum(r['status'] != 'ok' for r in index)} failed)")
print(f"four-method rows: {len(fm)} ({sum(r['stopping_reason'] == 'failure' for r in fm)} failures)")
print(f"B9 rows: {len(rows)} (NiL: {sum(r['method'] == 'NiL' for r in rows)} of 12)")
PY
