#!/bin/bash
# Package 4 on Grace (the advisor's instructions of 12 November 2026), CPU only.
# Into results/package4, locked to one commit:
#   p4_preflight (versions, the inputs, the test suite, the points folder)
#   -> each waiting for the preflight:
#        p4_timed (timed-cpu): M0, the timings of 1a, 1b and 2a
#        p4_item1 (cpu): 1a, 1b, 1c
#        p4_item2 (cpu): 2a to 2e, M7, the solution field
#          -> p4_item2_constants (cpu): 2f, after p4_item2
#   -> p4_report (afterany): the summaries, checks.json, sacct, su_per_job.csv and
#      results/package4.tar.gz.
# 6 jobs: 88 SU requested (walltime x rate), about 40 expected. Cap 100 SU (charged).
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package4/submit_p4.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package4/submit_p4.sh
LILQ_WAVE=p4
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package4
confirm_balance "88 SU requested, about 40 expected (6 jobs; cap 100 SU)"

submit pre  $P/p4_preflight.slurm
after="--dependency=afterok:$pre"
submit tim  $P/p4_timed.slurm             $after
submit it1  $P/p4_item1.slurm             $after
submit it2  $P/p4_item2.slurm             $after
submit c2   $P/p4_item2_constants.slurm   --dependency=afterok:$it2
submit rep  $P/p4_report.slurm            --dependency=afterany:$tim:$it1:$it2:$c2
[[ "$DRY_RUN" == 1 ]] || echo "Package 4 submitted. When the report has run (squeue -u \$USER), download results/package4.tar.gz."
