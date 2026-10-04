#!/bin/bash
# Package 2, Stage 1 (the advisor's instructions of 4 October 2026, Sections 3,
# 6.2 and 12.3): on an exclusive CPU node with the thread setting of the
# package1 runs (48), the Bratu (SQ-CGL p = 48, with p = 64 for its error) and
# Burgers (Cole-Hopf, with the Basdevant check) references; the reported LiL-Q
# configurations of Bratu (P = 25, 100, 225) and Burgers (P = 25-625) rerun
# with eps_ref logged against them (component_b, paper pass); and check C4:
# norm_R_h equal to package1's (wave 2's runs) to 1e-10 at every k. Minutes.
# Results in results/package2_stage1, locked to this commit.
# Submit: LILQ_WAVE=p2s1 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s1_scalar_reruns.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s1-scalar
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_3_classical.py reference --out "$PKG"
python experiments/p2_references.py burgers --out "$PKG"
python experiments/component_b.py --benchmarks bratu burgers --passes paper --devices cpu \
    --out-root "$PKG/P2_12_reference_errors" --reference-dir "$PKG/reference"
python experiments/p2_c4_check.py --reruns "$PKG/P2_12_reference_errors/B_instrumentation" \
    --package1 "$RESULTS/wave2/B_instrumentation" --out "$PKG/P2_12_reference_errors"
