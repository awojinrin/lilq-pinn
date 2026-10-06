#!/bin/bash
# Package 2, Stage 2, the GPU addendum on Grace (batch 8): item 7's GPU scaling series
# again, in the corrected environment. The first run (job 20017337, and its retry
# 20018277) failed at Kovasznay P = 3,675: with the PyTorch module's CUDA 12.6.0
# libraries ahead of the torch wheel's own on LD_LIBRARY_PATH, cuSOLVER's QR failed for a
# band of shapes (job 20028701). env.sh now puts the wheel's libraries first. Into
# results/package2_stage2_gpu, locked to this commit (the main stage stays at its own):
#   p2s2_preflight (as lilq-p2s2g-preflight; the library record and the QR probe, the tests)
#   -> p2s2_scaling_gpu (as lilq-p2s2g-scaling-gpu: every size, 1,875 and 2,700 included)
#   -> p2s2_report (as lilq-p2s2g-report, afterany): results/package2_stage2_gpu.tar.gz.
# Requested: preflight 80 (1 h shared A100), scaling 288 (1.5 h, a whole A100 node at 192/h),
# report 12: 380 SU. Expected about 160 (scaling about 45 min).
# Run from a Grace login node:
#   DRY_RUN=1 bash scripts/cluster/package2/submit_p2s2g.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package2/submit_p2s2g.sh
LILQ_WAVE=p2s2g
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package2
confirm_balance "380 SU requested, about 160 expected (su_plan.csv; 3 jobs)"

submit pre $P/p2s2_preflight.slurm     --job-name=lilq-p2s2g-preflight
submit gpu $P/p2s2_scaling_gpu.slurm   --job-name=lilq-p2s2g-scaling-gpu --dependency=afterok:$pre
submit rep $P/p2s2_report.slurm        --job-name=lilq-p2s2g-report --dependency=afterany:$gpu
[[ "$DRY_RUN" == 1 ]] || echo "GPU addendum submitted. When job $rep finishes: results/package2_stage2_gpu.tar.gz to download."
