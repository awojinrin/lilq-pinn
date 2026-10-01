#!/bin/bash
# Wave 4 of 4 (the advisor's reply to wave 2 and his follow-up, 1 October
# 2026; Addendum v2.3). Expected ~3,900 SU (his estimate; ~5,300 if every
# job ran to its walltime). Into results/wave4:
#   00 preflight -> 29 A1 gate (new commit: the search, A1, A2, F2 Jacobian)
#   Component A, from wave 2's runs (copied in below; decision 1, option 3):
#     34 F1 re-pick: wave 2's finalists kept, the representative by
#        validation residual (no training) -> 32 A CPU (F1), 33 float32
#     35 F2 full stage: the validation top three (F2_10, F2_05, F2_14)
#        x seeds 0-4 x 60 min -> 32 A CPU (F2)
#   13a / 13b clean timing (CPU, with the gravity BL P = 64 logged rerun; GPU)
#   43 Table 3 with the default-init ELM row, 44 B10, 45 B8's K_max = 60 reruns
#   -> 91 wave report: results/wave4_report.tar.gz
#   -> 90 finalize: results/package1 assembled from the four waves.
LILQ_WAVE=4
source "$(dirname "$0")/submit_lib.sh"
last=$(wave_commit 3)
echo "Code $(code_commit); wave 3 ran at ${last:-?}: the preflight and the gate run again."
confirm_balance "~3,900 SU expected (advisor's estimate); ~5,300 SU if every job hit its walltime"

# Component A starts from wave 2's search, screening runs and F1's finalist
# full runs. The tuning log is copied too, so that wave 4's runs are appended
# to it rather than starting a new one; wave 2's selection and representative
# files stay in results/wave2 and wave 4 writes its own.
W2A="$RESULTS/wave2/A_calibration"
A4="$RESULTS/wave4/A_calibration"
[[ -d "$W2A/screening" && -d "$W2A/full" ]] || { echo "wave 2's Component A runs are missing ($W2A)" >&2; exit 1; }
if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$A4/full"
    for item in search screening tuning_log.md; do
        [[ -e "$A4/$item" ]] || cp -r "$W2A/$item" "$A4/"
    done
    for run in "$W2A"/full/F1_04_s* "$W2A"/full/F1_15_s* "$W2A"/full/F1_18_s*; do
        [[ -e "$A4/full/$(basename "$run")" ]] || cp -r "$run" "$A4/full/"
    done
    echo "Copied wave 2's Component A search, screening runs and F1 finalist runs into $A4."
fi

submit pre  $S/00_preflight.slurm
submit gate $S/29_A1_gate.slurm           --dependency=afterok:$pre
submit f1   $S/34_A_f1_repick.slurm       --dependency=afterok:$gate
submit f2   $S/35_A_f2_full.slurm         --dependency=afterok:$gate
submit cpu1 $S/32_A_cpu.slurm             --dependency=afterok:$f1 --array=0
submit cpu2 $S/32_A_cpu.slurm             --dependency=afterok:$f2 --array=1
submit f32  $S/33_A_float32.slurm         --dependency=afterok:$f1
submit tcpu $S/13a_clean_timing_cpu.slurm --dependency=afterok:$pre
submit tgpu $S/13b_clean_timing_gpu.slurm --dependency=afterok:$pre
submit t3   $S/43_basis_study.slurm       --dependency=afterok:$pre
submit b10  $S/44_b10_cgl_cc.slurm        --dependency=afterok:$pre
submit b8   $S/45_b8_kmax60.slurm         --dependency=afterok:$pre
submit rep  $S/91_wave_report.slurm       --dependency=afterany:$f1:$f2:$cpu1:$cpu2:$f32:$tcpu:$tgpu:$t3:$b10:$b8
submit fin  $S/90_finalize.slurm          --dependency=afterany:$rep
[[ "$DRY_RUN" == 1 ]] || echo "Wave 4 submitted. When job $rep finishes: results/wave4_report.tar.gz;"
[[ "$DRY_RUN" == 1 ]] || echo "when job $fin finishes: results/package1 (package1_results/) from the four waves."
