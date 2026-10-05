#!/bin/bash
# Package 2, Stage 2, item 6a (P2-2; the advisor's instructions of 4 October 2026,
# Section 9.1): Buckley-Leverett nu-refinement. First the finite-difference references,
# refined at each (case, nu) until two successive refinements agree to 1e-6 (check
# C7); then LiL-Q at nu = 0.1, 0.05, 0.02, 0.01 (viscous, P = 576 and 1,024, and
# P = 1,600 at nu <= 0.02) and the gravity case at nu = 0.05, 0.02 (P = 576, 1,024;
# it runs while 80 SU remain under item 6's cap), each with the paper pass and the
# K_max = 60 pass; check C8 against package1 at nu = 0.1. Untimed, but on a whole
# 48-thread node, package1's setting: at kappa about 1e16-1e17, C8 can only be met,
# if at all, on the hardware and thread count of the runs it compares with
# (DECISIONS.md, batch 1). About 40 min on a laptop. Results in
# results/package2_stage2/P2_2_nu_refinement and reference/.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_nu_refinement.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-nu-refinement
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_2_nu_refinement.py reference --out "$PKG"
python experiments/p2_2_nu_refinement.py runs --out "$PKG" --package1 "$RESULTS/package1_v2.0.0/package1"
