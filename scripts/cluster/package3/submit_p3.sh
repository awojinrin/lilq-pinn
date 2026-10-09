#!/bin/bash
# Package 3 on Grace, wave 1 (the advisor's instructions of 8 October 2026, and Addendum 1).
# Into results/package3, locked to one commit:
#   p3_preflight (versions, the inputs, the test suite, K1, K2)
#   -> every compute job, each waiting for the preflight:
#        K0 (timed-cpu)                         item 1: B1 and the checks K3, K6, K8 (timed-cpu)
#        item 1: B2 (timed-cpu)                 item 1: B3 level 1 (timed-cpu)
#        item 2: BL 2a and 2b, K5, K6 (cpu)     item 3: elasticity and Darcy, K5 (timed-cpu)
#        item 4 (Addendum 1): Burgers at larger P, clean timings (timed-cpu)
#   -> p3_report (afterany): sacct, su_per_job.csv, results/package3.tar.gz.
# Not submitted here: B3 level 2 (wave 2, submit_p3_b3l2.sh), after B3 level 1 and K0, only if
# the launch rule and the budget guard hold.
# 9 jobs: 828 SU requested (walltime x rate, su_plan.csv), about 376 expected; with B3 level 2,
# 1,164 requested and about 617 expected. Cap 850 SU (charged).
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package3/submit_p3.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package3/submit_p3.sh
LILQ_WAVE=p3
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package3
confirm_balance "828 SU requested, about 376 expected (su_plan.csv; 9 jobs; B3 level 2 is wave 2)"

submit pre  $P/p3_preflight.slurm
after="--dependency=afterok:$pre"
submit k0   $P/p3_k0.slurm                $after
submit b1   $P/p3_beltrami_small.slurm    $after
submit b2   $P/p3_beltrami_b2.slurm       $after
submit b3   $P/p3_beltrami_b3_l1.slurm    $after
submit bl   $P/p3_bl.slurm                $after
submit aff  $P/p3_affine.slurm            $after
submit bur  $P/p3_burgers_large_P.slurm   $after
submit rep  $P/p3_report.slurm            --dependency=afterany:$k0:$b1:$b2:$b3:$bl:$aff:$bur
[[ "$DRY_RUN" == 1 ]] || echo "Package 3, wave 1 submitted. When $b3 (B3 level 1) and $k0 (K0) have finished:"
[[ "$DRY_RUN" == 1 ]] || echo "  bash scripts/cluster/package3/submit_p3_b3l2.sh   (checks the launch rule and the budget first)"
