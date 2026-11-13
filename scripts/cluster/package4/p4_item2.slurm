#!/bin/bash
# Package 4, item 2 (Section 4): 2a's 18 logged runs, 2b, 2c, the 2d and 2e sweeps, M7 and the
# solution field. Laptop: 7 min, 10 s, 9 s, 1.5 min, 1.5 min, 30 s and 6 s.
# Submit: through scripts/cluster/package4/submit_p4.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p4-item2
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p4_2_pb_inclusions.py lilq --out "$PKG" --points data/package4
python experiments/p4_2_pb_inclusions.py square --out "$PKG" --points data/package4
python experiments/p4_2_pb_inclusions.py under --out "$PKG" --points data/package4
python experiments/p4_2_pb_inclusions.py rows --out "$PKG" --points data/package4
python experiments/p4_2_pb_inclusions.py weight --out "$PKG" --points data/package4
python experiments/p4_2_pb_inclusions.py m7 --out "$PKG"
python experiments/p4_2_pb_inclusions.py solution --out "$PKG"
