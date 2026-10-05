#!/bin/bash
# Package 2, Stage 2, item 6b (P2-9; the advisor's instructions of 4 October 2026,
# Section 9.2): boundary-conforming bases. Bratu with sin x sin and with Chebyshev and
# weak rows (P = 25, 100, 225); Buckley-Leverett, both cases, with the lifted sine basis
# S = (1 - x) + x(1 - x) sum beta sin(i pi x) phi_j(t), P = 64 .. 1,024, from both
# initial guesses; K_max = 60 on the paper's grids and weights; the paper's bases rerun
# beside them on the same references, and BL with the lifting and plain sines (an addition;
# DECISIONS.md, batch 2). Untimed. About 30 min on a laptop.
# Results in results/package2_stage2/P2_9_bases.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_bases.slurm
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-bases
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
[[ -f "$PKG/reference/bratu_ref_p48.npz" ]] || python experiments/p2_3_classical.py reference --out "$PKG"
python experiments/p2_9_bases.py --out "$PKG"
