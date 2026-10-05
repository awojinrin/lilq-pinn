#!/bin/bash
# Package 2, Stage 2, item 2 (P2-8; the advisor's instructions of 4 October 2026,
# Section 5): NiL-N trained by Levenberg-Marquardt (the F2 optimizer), Bratu, P = 25, 100, 225,
# seeds 0, 1, 2. Caps per run: the target, 2,000 LM iterations, 15 min of training,
# or a stall (f1). Times are reported, so an exclusive CPU node. A configuration the
# 15-min cap ends before the target and the iteration cap is rerun on an A100
# (`p2_8_lm_networks.py gpu-list`). About 4 min on a laptop; finished runs are
# skipped on resubmission. Results in results/package2_stage2/P2_8_lm_networks.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_lm_networks_bratu.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-lm-networks-bratu
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
[[ -f "$PKG/reference/bratu_ref_p48.npz" ]] || python experiments/p2_3_classical.py reference --out "$PKG"
python experiments/p2_8_lm_networks.py run --benchmark bratu --out "$PKG" --reference-dir "$PKG/reference" \
    --package1 "$RESULTS/package1_v2.0.0/package1"
