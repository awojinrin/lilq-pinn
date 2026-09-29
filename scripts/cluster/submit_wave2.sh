#!/bin/bash
# Wave 2 of 3 (Addendum v2.2 Section 4.2): Component A and the rest of B4,
# requested ~10,000 SU at 120 SU/h. Into results/wave2:
#   [00 preflight and 29 A1 gate, only if the code differs from wave 1's]
#   30 A screen -> 31 A full -> 32 A CPU, 33 A float32
#   20 / 21 for Burgers, viscous BL, gravity BL (--array=1-3)
#   -> 91 wave report: results/wave2_report.tar.gz.
# With wave 1's code, wave 1's Component A search and checks (A1, A2, F2) are
# copied in and gate Component A; with new code they are redone here.
LILQ_WAVE=2
source "$(dirname "$0")/submit_lib.sh"
now=$(code_commit); w1=$(wave_commit 1)
[[ -n "$w1" ]] || echo "WARNING: results/wave1 has no COMMIT (wave 1 not run?)" >&2
confirm_balance "~10,000 SU at 120 SU/h (advisor's estimate)"

deps=()
if [[ "$now" != "$w1" ]]; then
    echo "Code $now differs from wave 1's ($w1): the preflight and the A1 gate run again."
    submit pre $S/00_preflight.slurm
    deps=(--dependency=afterok:$pre)
fi
A1_JSON="$RESULTS/wave1/A_calibration/checks/a1.json"
a1_passed=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['passed'])" "$A1_JSON" 2>/dev/null || echo False)
if [[ "$now" == "$w1" && "$a1_passed" == True ]]; then
    echo "Same code as wave 1, and check A1 passed there: its search and checks are reused."
    if [[ "$DRY_RUN" != 1 ]]; then
        mkdir -p "$RESULTS/wave2/A_calibration"
        for item in search checks tuning_log.md; do
            [[ -e "$RESULTS/wave2/A_calibration/$item" ]] || cp -r "$RESULTS/wave1/A_calibration/$item" "$RESULTS/wave2/A_calibration/"
        done
    fi
    adeps=(${deps[@]+"${deps[@]}"})
else
    if [[ "$now" == "$w1" ]]; then
        echo "Check A1 did not pass in wave 1 ($A1_JSON): stop and report it." >&2
        exit 1
    fi
    submit gate $S/29_A1_gate.slurm ${deps[@]+"${deps[@]}"}
    adeps=(--dependency=afterok:$gate)
fi
submit scr  $S/30_A_screen.slurm         ${adeps[@]+"${adeps[@]}"}
submit full $S/31_A_full.slurm           --dependency=afterok:$scr
submit acpu $S/32_A_cpu.slurm            --dependency=afterok:$full
submit af32 $S/33_A_float32.slurm        --dependency=afterok:$full
submit fmg  $S/20_four_method_gpu.slurm  ${deps[@]+"${deps[@]}"} --array=1-3
submit fmc  $S/21_four_method_cpu.slurm  ${deps[@]+"${deps[@]}"} --array=1-3
submit rep  $S/91_wave_report.slurm      --dependency=afterany:$scr:$full:$acpu:$af32:$fmg:$fmc
[[ "$DRY_RUN" == 1 ]] || echo "Wave 2 submitted. When job $rep finishes: results/wave2_report.tar.gz."
[[ "$DRY_RUN" == 1 ]] || echo "Submit wave 3 once wave 2's charges have posted."
