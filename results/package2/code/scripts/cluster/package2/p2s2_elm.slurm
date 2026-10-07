#!/bin/bash
# Package 2, Stage 2, item 4 (P2-14 and P2-4; the advisor's instructions of 4 October
# 2026, Section 7): the ELM hidden-weight scale sweep on Burgers (12 sigma x 5 seeds,
# and the two Table 3 rows reproduced), then the ELM basis on Kovasznay (check C5,
# the sigma sweep, seeds 0-4 by QR and by the two normal-equation variants of the advisor's
# reply of 5 October, and the Chebyshev surrogate). Untimed. About an hour and a half
# (laptop). Results in results/package2_stage2.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_elm.slurm
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-elm
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
[[ -f "$PKG/reference/burgers_cole_hopf.npz" ]] || python experiments/p2_references.py burgers --out "$PKG"
python experiments/p2_14_elm.py sweep --out "$PKG" --reference-dir "$PKG/reference" \
    --package1 "$RESULTS/package1_v2.0.0/package1"
python experiments/p2_14_elm.py kovasznay --out "$PKG"
