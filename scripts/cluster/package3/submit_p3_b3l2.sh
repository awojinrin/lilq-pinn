#!/bin/bash
# Package 3, wave 2: B3 level 2 (665,331 x 7,984; 7 h, 336 SU requested), then the report
# again (afterany). Submitted only if b3l2_gate.py says so: wave 1 over; K0 passed; B3 level 1
# complete and the launch rule 1.91 x (8 x s/iteration + kappa time) < 6.5 h (Section 3.4);
# the SU charged so far + 336 <= 850. Otherwise nothing is submitted (Section 8.2: level 1
# stands), or it says to wait.
# Run from a login node with the project's Python (module load ...; source the venv):
#   DRY_RUN=1 bash scripts/cluster/package3/submit_p3_b3l2.sh   (the gate's verdict, then
#       sbatch --test-only on both jobs whatever the verdict; submits nothing)
#   bash scripts/cluster/package3/submit_p3_b3l2.sh
LILQ_WAVE=p3
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package3
gate=0
python $P/b3l2_gate.py --results "$RESULTS" --logs "${LILQ_LOGS:-logs}" || gate=$?   # LILQ_LOGS: tests only
if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY RUN: the gate's exit code is $gate; checking both jobs anyway."
elif [[ $gate -ne 0 ]]; then
    echo "Nothing submitted."
    exit $gate
fi
confirm_balance "336 SU requested (B3 level 2, 7 h), 12 for the report; about 240 and 2 expected"
submit l2   $P/p3_beltrami_b3_l2.slurm
submit rep  $P/p3_report.slurm             --dependency=afterany:$l2
[[ "$DRY_RUN" == 1 ]] || echo "B3 level 2 submitted ($l2); the report ($rep) repacks results/package3.tar.gz after it."
