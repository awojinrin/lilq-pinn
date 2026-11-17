#!/bin/bash
# Package 5, item 3 (Section 6: cost at a common accuracy), the A100 set: after the CPU set and the
# cost check (the advisor's rule 3). Into results/package5_item3_gpu, locked to one commit:
#   p5g_preflight (shared-gpu: the OpenMP library each driver loads, the inputs, the item 3 tests
#   with CUDA visible)
#   -> each waiting for the preflight, each on one whole two-A100 node (gpu-pair, 192 SU/h), each
#      checking the OpenMP library before its first run (the GCC module's, as Package 1):
#        p5g_bratu, p5g_burgers, p5g_bl: the solo check, the pairing gate (a job stops before its
#          budget runs unless its paired clock is within 5% of the solo's), then two processes side
#          by side, one per A100, 24 threads each (122 budget runs of 600 s in all)
#   -> p5g_report (afterany): the paired-against-solo check, the time-to-error table, sacct,
#      su_per_job.csv and results/package5_item3_gpu.tar.gz.
# 5 jobs: 2,583 SU requested (walltime x rate), about 2,270 expected.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_item3_gpu.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_item3_gpu.sh
LILQ_WAVE=p5g
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "2,583 SU requested, about 2,270 expected (5 jobs, 3 of them whole two-A100 nodes)"

submit pre $P/p5g_preflight.slurm
after="--dependency=afterok:$pre"
submit bra $P/p5g_bratu.slurm    $after
submit bur $P/p5g_burgers.slurm  $after
submit bl  $P/p5g_bl.slurm       $after
submit rep $P/p5g_report.slurm   --dependency=afterany:$bra:$bur:$bl
[[ "$DRY_RUN" == 1 ]] || echo "Item 3, the A100 set, submitted. When the report has run (squeue -u \$USER), download results/package5_item3_gpu.tar.gz."
