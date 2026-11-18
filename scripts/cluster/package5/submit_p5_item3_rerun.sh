#!/bin/bash
# Package 5, item 3: the CPU set's network runs again, with numpy imported first (the user's decision,
# 17 November 2026, after the OpenMP library diagnostic; the advisor's approval in the item 3 cost
# check). Into results/package5_item3_cpu_rerun, locked to one commit; the CPU set
# (results/package5_item3_cpu) stays as it ran and keeps the LiL runs:
#   p5r_preflight (cpu: the OpenMP library each driver loads, the inputs, the test suite)
#   -> each waiting for the preflight, each on one whole CPU node (timed-cpu, 48 threads), each
#      checking the OpenMP library before its first run:
#        p5r_checks: checks C2 and C4 for NiL-N and LM
#        p5r_bratu, p5r_burgers_a, p5r_burgers_b, p5r_bl_a, p5r_bl_b: the 108 network budget runs of
#          600 s, one at a time, in the CPU set's parts
#   -> p5r_report (afterany): the time-to-error table, sacct, su_per_job.csv and
#      results/package5_item3_cpu_rerun.tar.gz.
# 8 jobs: 1,128 SU requested (walltime x rate), about 925 expected (the CPU set's own SU).
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_item3_rerun.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_item3_rerun.sh
LILQ_WAVE=p5r
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "1,128 SU requested, about 925 expected (8 jobs, 6 of them whole CPU nodes)"

submit pre $P/p5r_preflight.slurm
after="--dependency=afterok:$pre"
submit chk $P/p5r_checks.slurm     $after
submit bra $P/p5r_bratu.slurm      $after
submit bua $P/p5r_burgers_a.slurm  $after
submit bub $P/p5r_burgers_b.slurm  $after
submit bla $P/p5r_bl_a.slurm       $after
submit blb $P/p5r_bl_b.slurm       $after
submit rep $P/p5r_report.slurm     --dependency=afterany:$chk:$bra:$bua:$bub:$bla:$blb
[[ "$DRY_RUN" == 1 ]] || echo "Item 3, the CPU network runs again, submitted. When the report has run (squeue -u \$USER), download results/package5_item3_cpu_rerun.tar.gz."
