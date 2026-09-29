#!/bin/bash
# Wave 2 of 3 (Addendum v2.2 Section 4.2, and the advisor's reply to wave 1):
# Component A, the rest of B4 with its stall controls, and the timing reruns.
# Expected ~5,900-6,700 SU (the advisor's estimate, one GPU per timed job at
# 120 SU/h); ~11,700 SU if every job ran to its walltime. Into results/wave2:
#   [00 preflight and 29 A1 gate, only if the code differs from wave 1's]
#   30 A screen -> 31 A full -> 32 A CPU, 33 A float32
#   20 / 21 for Burgers, viscous BL, gravity BL, each with its stall controls;
#     task 0 (Bratu): the controls of wave 1's stalled Bratu runs (--array=0-3)
#   11a / 11b timing reruns (warm-up runs; item 2.2); 12 Beltrami and Darcy
#     (the advisor's reply of 30 Sept; in the wave that ran, submitted by hand)
#   -> 91 wave report: results/wave2_report.tar.gz.
# With wave 1's code, wave 1's Component A search and checks (A1, A2, F2) are
# copied in and gate Component A; with new code they are redone here.
LILQ_WAVE=2
source "$(dirname "$0")/submit_lib.sh"
now=$(code_commit); w1=$(wave_commit 1)
[[ -n "$w1" ]] || echo "WARNING: results/wave1 has no COMMIT (wave 1 not run?)" >&2
confirm_balance "~5,900-6,700 SU expected (advisor's estimate); ~11,700 SU if every job hit its walltime"

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
submit fmg  $S/20_four_method_gpu.slurm  ${deps[@]+"${deps[@]}"} --array=0-3
submit fmc  $S/21_four_method_cpu.slurm  ${deps[@]+"${deps[@]}"} --array=0-3
submit tcpu $S/11a_timing_reruns_cpu.slurm ${deps[@]+"${deps[@]}"}
submit tgpu $S/11b_timing_reruns_gpu.slurm ${deps[@]+"${deps[@]}"}
submit tbd  $S/12_timing_reruns_beltrami_darcy.slurm ${deps[@]+"${deps[@]}"}   # added 30 Sept (submitted by hand in wave 2)
submit rep  $S/91_wave_report.slurm      --dependency=afterany:$scr:$full:$acpu:$af32:$fmg:$fmc:$tcpu:$tgpu:$tbd
[[ "$DRY_RUN" == 1 ]] || echo "Wave 2 submitted. When job $rep finishes: results/wave2_report.tar.gz."
[[ "$DRY_RUN" == 1 ]] || echo "When the first timed job starts, check sacct shows gres/gpu=1 (item 2.3):"
[[ "$DRY_RUN" == 1 ]] || echo "  sacct -X -j <job> --format=JobID,JobName%24,AllocTRES%70  -- and after it ends, its charge in myproject."
[[ "$DRY_RUN" == 1 ]] || echo "Submit wave 3 once wave 2's charges have posted."
