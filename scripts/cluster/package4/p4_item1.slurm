#!/bin/bash
# Package 4, item 1 (Section 3): 1a's seven logged runs (K_max = 60), 1b's four enrichment
# runs, 1c's constants at every iterate. Laptop: 3 min, 16 s and 17 min.
# Submit: through scripts/cluster/package4/submit_p4.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p4-item1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p4_1_burgers_gauss.py logged --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference"
python experiments/p4_1_burgers_gauss.py enrich --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference"
python experiments/p4_1_burgers_gauss.py constants --out "$PKG"
