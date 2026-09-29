#!/bin/bash
# Wave 3 of 3 (Addendum v2.2 Section 4.2): B8 and B9, requested ~4,800 SU
# (the advisor estimated ~7,700 with the larger B9 network),
# once wave 2's charges have posted. Into results/wave3:
#   [00 preflight, only if the code differs from wave 1's]
#   40 B9 NiL Darcy (12 tasks), 41 B8 (4 tasks)
#   -> 91 wave report, and 90 finalize: results/package1 assembled from the
#      three waves (WAVES.json names each wave's commit).
LILQ_WAVE=3
source "$(dirname "$0")/submit_lib.sh"
now=$(code_commit); w1=$(wave_commit 1)
confirm_balance "~4,800 SU (from the walltimes)"

deps=()
if [[ "$now" != "$w1" ]]; then
    echo "Code $now differs from wave 1's ($w1): the preflight runs again."
    submit pre $S/00_preflight.slurm
    deps=(--dependency=afterok:$pre)
fi
submit b9  $S/40_b9_nil_darcy.slurm      ${deps[@]+"${deps[@]}"}
submit b8  $S/41_b8_initial_guess.slurm  ${deps[@]+"${deps[@]}"}
submit rep $S/91_wave_report.slurm       --dependency=afterany:$b9:$b8
submit fin $S/90_finalize.slurm          --dependency=afterany:$b9:$b8:$rep
echo "Wave 3 submitted. When job $fin finishes: results/package1 (package1_results) and results/wave3_report.tar.gz."
