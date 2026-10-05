#!/bin/bash
# Package 2, Stage 2 on FASTER: item 5 (the Darcy network with hard Dirichlet
# conditions; the advisor's instructions of 4 October 2026, Section 8: "FASTER,
# untimed job class, shared GPU"). From the same commit as the Grace jobs
# (submit_p2s2.sh), into FASTER's own results/package2_stage2, locked to it:
#   p2s2_preflight (versions, the GPU, the test suite)
#   -> p2s2_darcy_hardbc (array of 12: 4 fields x 3 seeds)
#   -> p2s2_report (afterany): sacct, and results/package2_stage2_faster.tar.gz.
# Requested: the array 12 x 45 min of shared A100 at about 137 SU/h (FASTER's measured
# rate: 128 per A100-hour plus the cores), 1,233 SU; the preflight 137 (1 h, shared A100)
# and the report 12 (30 min, 24 cores): 1,382 SU, within item 5's 1,500 on FASTER.
# Charged by elapsed time: a run stops at 2,000 iterations or 30 min of training,
# so about 700 expected (su_plan.csv); the request is the most it can be.
# Run from a FASTER login node:
#   DRY_RUN=1 bash scripts/cluster/package2/submit_p2s2_faster.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package2/submit_p2s2_faster.sh
export CLUSTER=faster
LILQ_WAVE=p2s2
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package2
echo "FASTER: item 5 only (Grace runs the rest of Stage 2, submit_p2s2.sh)."
confirm_balance "1,382 SU requested, about 700 expected (su_plan.csv; 3 jobs, the array of 12 among them)"

submit pre $P/p2s2_preflight.slurm
submit dar $P/p2s2_darcy_hardbc.slurm  --dependency=afterok:$pre
submit rep $P/p2s2_report.slurm        --dependency=afterany:$dar
[[ "$DRY_RUN" == 1 ]] || echo "Item 5 submitted on FASTER. When job $rep finishes: results/package2_stage2_faster.tar.gz to download."
