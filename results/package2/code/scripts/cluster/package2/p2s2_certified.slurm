#!/bin/bash
# Package 2, Stage 2, item 3 (P2-16; the advisor's instructions of 4 October 2026,
# Section 6.3, and his Stage-1 reply): LiL-Q on Bratu and Burgers on certified
# Clenshaw-Curtis-weighted CGL grids (P x N/P = 24 runs, K_max = 60), with the
# sampling constants, the realized ratio per iterate, delta_P and checks C4 and C6.
# Untimed; about 15 min on a laptop. Results in results/package2_stage2/P2_16_certified.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_certified.slurm
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-certified
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
[[ -f "$PKG/reference/bratu_ref_p48.npz" ]] || python experiments/p2_3_classical.py reference --out "$PKG"
[[ -f "$PKG/reference/burgers_cole_hopf.npz" ]] || python experiments/p2_references.py burgers --out "$PKG"
python experiments/p2_16_certified.py --out "$PKG" --reference-dir "$PKG/reference"
