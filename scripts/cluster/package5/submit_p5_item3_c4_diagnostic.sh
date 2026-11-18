#!/bin/bash
# Package 5, item 3: the C4 diagnostic (the user's decision, 17 November 2026). In the CPU rerun, check
# C4 failed for Buckley-Leverett LM (+18.1% with the error log); the A100 set waits for this. Into
# results/package5_item3_c4_diagnostic under its own commit lock:
#   p5e_cpu (timed-cpu, 1 h): nine processes on one whole CPU node, the LM pair three times each: the
#     module's OpenMP library, the wheel's, and the module's with OMP_WAIT_POLICY=ACTIVE
#   p5e_gpu (gpu-pair, 20 min): two processes side by side on one two-A100 node, 24 threads each, as
#     the A100 set (both at once with p5e_cpu)
#   -> p5e_report (afterany): the summary (A100 SET MAY GO: yes or NO), sacct, su_per_job.csv and
#      results/package5_item3_c4_diagnostic.tar.gz.
# 3 jobs: 118 SU requested (walltime x rate: 48 + 64 + 6), about 60 expected.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh
LILQ_WAVE=p5e
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "118 SU requested, about 60 expected (a whole CPU node, a whole two-A100 node, a report)"

submit cpu $P/p5e_cpu.slurm
submit gpu $P/p5e_gpu.slurm
submit rep $P/p5e_report.slurm --dependency=afterany:$cpu:$gpu
[[ "$DRY_RUN" == 1 ]] || echo "Submitted. When the report has run (squeue -u \$USER), see the last lines of logs/lilq-p5e-report.<jobid>.out and download results/package5_item3_c4_diagnostic.tar.gz."
