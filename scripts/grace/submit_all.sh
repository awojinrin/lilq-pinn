#!/bin/bash
# Submit Package 1's Grace jobs, from the repository root on a Grace login
# node:   bash scripts/grace/submit_all.sh
# The pre-flight must pass before anything else starts (afterok); the
# Component A stages chain; finalize waits for all. Timed jobs each hold a
# whole A100 node (--exclusive). CPU_PARTITION: a Grace partition without
# GPUs for the finalize job -- check `sinfo -s` (default below).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs   # sbatch does not create the --output directory
CPU_PARTITION="${CPU_PARTITION:-short}"

# sbatch can fail without a usable job ID (e.g. "Socket timed out"); stop at
# the first such failure. A timeout may still have queued the job, so check
# `squeue -u $USER` (and `sacct`) before resubmitting anything.
submitted=""
submit() {   # submit <variable> <sbatch args...>: sets <variable> to the job ID
    local var=$1 id; shift
    id=$(sbatch --parsable "$@" | cut -d';' -f1) || true
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "sbatch failed for ${*: -1}; submitted so far:${submitted:- none}." >&2
        echo "Check squeue -u \$USER before resubmitting." >&2
        exit 1
    fi
    submitted="$submitted $id"
    printf -v "$var" '%s' "$id"
}

S=scripts/grace
submit pre   $S/00_preflight.slurm
submit lilq  --dependency=afterok:$pre   $S/10_timed_lilq.slurm
submit fmg   --dependency=afterok:$pre   $S/20_four_method_gpu.slurm
submit fmc   --dependency=afterok:$pre   $S/21_four_method_cpu.slurm
submit ascr  --dependency=afterok:$pre   $S/30_A_screen.slurm
submit afull --dependency=afterok:$ascr  $S/31_A_full.slurm
submit acpu  --dependency=afterok:$afull $S/32_A_cpu.slurm
submit af32  --dependency=afterok:$afull $S/33_A_float32_and_a1.slurm
submit b9    --dependency=afterok:$pre   $S/40_b9_nil_darcy.slurm
submit fin   --dependency=afterany:$lilq:$fmg:$fmc:$ascr:$afull:$acpu:$af32:$b9 \
             --partition="$CPU_PARTITION" $S/90_finalize.slurm

echo "preflight $pre -> LiL-Q timed $lilq, four-method GPU $fmg / CPU $fmc,"
echo "  Component A screen $ascr -> full $afull -> CPU $acpu, float32+A1 $af32,"
echo "  B9 NiL $b9 -> finalize $fin"
