#!/bin/bash
# Package 5 on Grace, items 1, 2 and 4 (the advisor's instructions of 13 November 2026 and his
# reply to the cost check of 14 November 2026), CPU only. Into results/package5, locked to one
# commit:
#   p5_preflight (cpu-mem: its billing, the inputs, the test suite, the points)
#   -> each waiting for the preflight:
#        p5_ref_viscous, p5_ref_gravity (cpu-mem): the fine Buckley-Leverett references
#          -> p5_item1 (cpu-mem): the 96 runs, after both references whatever their outcome
#        p5_beltrami (cpu-mem): the six Beltrami runs
#        p5_timed (timed-cpu): the timings of 4a, then item 2
#        p5_item4 (cpu-mem): 4a logged and 2c, 4b, delta_P two ways
#   -> p5_report (afterany): the summaries, sacct, su_per_job.csv and results/package5.tar.gz.
# 8 jobs: 330 SU requested (walltime x rate), about 145 expected. The advisor approved about
# 135, and up to about 170 if the Beltrami runs are slower on 24 cores.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package5/submit_p5.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package5/submit_p5.sh
LILQ_WAVE=p5
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package5
confirm_balance "330 SU requested, about 145 expected (8 jobs; the advisor approved about 135 to 170)"

submit pre  $P/p5_preflight.slurm
after="--dependency=afterok:$pre"
submit rv   $P/p5_ref_viscous.slurm   $after
submit rg   $P/p5_ref_gravity.slurm   $after
submit it1  $P/p5_item1.slurm         --dependency=afterok:$pre,afterany:$rv:$rg
submit bel  $P/p5_beltrami.slurm      $after
submit tim  $P/p5_timed.slurm         $after
submit it4  $P/p5_item4.slurm         $after
submit rep  $P/p5_report.slurm        --dependency=afterany:$rv:$rg:$it1:$bel:$tim:$it4
[[ "$DRY_RUN" == 1 ]] || echo "Package 5 submitted. When the report has run (squeue -u \$USER), download results/package5.tar.gz."
