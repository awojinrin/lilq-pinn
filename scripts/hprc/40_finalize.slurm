#!/bin/bash
# After everything else: merge the four-method job CSVs into
# four_method_tables.csv, write check B1 (reproduction_check.csv), and draw
# the Section 3.5 residual-band figures from the Section 3.3 logs. Seconds.
#SBATCH --job-name=lilq-finalize
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --account=142698950992
#SBATCH --output=logs/%x.%j.out

source scripts/hprc/env.sh
set -euo pipefail

B="$PKG/B_instrumentation"
python experiments/four_method_tables.py --out-dir "$B" --merge-from "$B"/four_method_jobs/*/
python experiments/component_b.py --reproduction-check --out-root "$PKG"
python experiments/residual_band_figures.py --from-logs "$PKG" --pass paper --out-dir "$B/residual_band_figures/paper"
python experiments/residual_band_figures.py --from-logs "$PKG" --pass kmax  --out-dir "$B/residual_band_figures/kmax"
python - <<'EOF'
import csv, os
from pathlib import Path
from experiments.component_b import write_index
write_index(Path(os.environ["PKG"]) / "B_instrumentation")  # both run jobs wrote it; rebuild once
b = os.path.join(os.environ["PKG"], "B_instrumentation")
rows = list(csv.DictReader(open(os.path.join(b, "runs_index.csv"))))
failed = [r["run"] for r in rows if r["status"] != "ok"]
fm = list(csv.DictReader(open(os.path.join(b, "four_method_tables.csv"))))
fm_failed = [(r["benchmark"], r["P"], r["method"], r["seed"], r["device"]) for r in fm if r["stopping_reason"] == "failure"]
print(f"Section 3.3 runs: {len(rows)} ({len(failed)} failed) {failed}")
print(f"four-method rows: {len(fm)} ({len(fm_failed)} failures) {fm_failed}")
EOF
