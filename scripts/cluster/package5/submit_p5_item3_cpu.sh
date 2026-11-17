#!/bin/bash
# Package 5, item 3 (Section 6: cost at a common accuracy), the CPU set. Into
# results/package5_item3_cpu, locked to one commit:
#   p5c_preflight (cpu: the inputs, the test suite)
#   -> each waiting for the preflight, each on one whole CPU node (timed-cpu, 48 threads):
#        p5c_protocol: LiL-Q, trf and the square collocation by the usual protocol; checks C2 and C4
#        p5c_bratu, p5c_burgers_a, p5c_burgers_b, p5c_bl_a, p5c_bl_b: the 122 budget runs of 600 s
#          (108 network, 14 LiL-N L-BFGS), one at a time
#   -> p5c_report (afterany): the time-to-error table, sacct, su_per_job.csv and
#      results/package5_item3_cpu.tar.gz.
# 8 jobs: 1,260 SU requested (walltime x rate), about 1,110 expected.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_item3_cpu.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_item3_cpu.sh
LILQ_WAVE=p5c
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "1,260 SU requested, about 1,110 expected (8 jobs, 6 of them whole CPU nodes)"

submit pre $P/p5c_preflight.slurm
after="--dependency=afterok:$pre"
submit pro $P/p5c_protocol.slurm   $after
submit bra $P/p5c_bratu.slurm      $after
submit bua $P/p5c_burgers_a.slurm  $after
submit bub $P/p5c_burgers_b.slurm  $after
submit bla $P/p5c_bl_a.slurm       $after
submit blb $P/p5c_bl_b.slurm       $after
submit rep $P/p5c_report.slurm     --dependency=afterany:$pro:$bra:$bua:$bub:$bla:$blb
[[ "$DRY_RUN" == 1 ]] || echo "Item 3, the CPU set, submitted. When the report has run (squeue -u \$USER), download results/package5_item3_cpu.tar.gz."
