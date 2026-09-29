#!/bin/bash
# Submit all of Package 1's jobs, from the repository root on a login node:
#   CLUSTER=grace CPU_PARTITION=<from sinfo -s> bash scripts/cluster/submit_all.sh
# (CLUSTER=faster uses FASTER's profile.) The pre-flight must pass before
# anything else starts; Component A's stages chain; finalize waits for all.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs   # sbatch does not create the --output directory
S=scripts/cluster

# Check the profile before submitting anything (a failure halfway would leave
# part of the chain queued).
CLUSTER="${CLUSTER:-grace}"
[[ -f $S/profiles/$CLUSTER.sh ]] || { echo "no profile $S/profiles/$CLUSTER.sh" >&2; exit 1; }
source $S/profiles/$CLUSTER.sh
[[ -n "$CPU_PARTITION" ]] || { echo "set CPU_PARTITION to a partition without GPUs (sinfo -s)" >&2; exit 1; }
export CLUSTER CPU_PARTITION

# sbatch can fail without a usable job ID (e.g. "Socket timed out"); stop at
# the first such failure. A timeout may still have queued the job, so check
# `squeue -u $USER` (and `sacct`) before resubmitting anything.
submitted=""
submit() {   # submit <variable> <job script> [sbatch options...]: sets <variable> to the job ID
    local var=$1 script=$2 id; shift 2
    id=$(bash $S/sbatch.sh "$script" --parsable "$@" | cut -d';' -f1) || true
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "submission failed for $script; submitted so far:${submitted:- none}." >&2
        echo "Check squeue -u \$USER before resubmitting." >&2
        exit 1
    fi
    submitted="$submitted $id"
    printf -v "$var" '%s' "$id"
}

submit pre   $S/00_preflight.slurm
submit lilq  $S/10_timed_lilq.slurm           --dependency=afterok:$pre
submit fmg   $S/20_four_method_gpu.slurm      --dependency=afterok:$pre
submit fmc   $S/21_four_method_cpu.slurm      --dependency=afterok:$pre
submit b9    $S/40_b9_nil_darcy.slurm         --dependency=afterok:$pre
submit b8    $S/41_b8_initial_guess.slurm     --dependency=afterok:$pre
submit cc    $S/42_component_c.slurm          --dependency=afterok:$pre
submit b6    $S/43_basis_study.slurm          --dependency=afterok:$pre
# Component A (SKIP_A=1 holds it back; submit it later with submit_component_a.sh).
adeps=""
if [[ "${SKIP_A:-0}" != 1 ]]; then
    submit agate $S/29_A1_gate.slurm          --dependency=afterok:$pre
    submit ascr  $S/30_A_screen.slurm         --dependency=afterok:$agate
    submit afull $S/31_A_full.slurm           --dependency=afterok:$ascr
    submit acpu  $S/32_A_cpu.slurm            --dependency=afterok:$afull
    submit af32  $S/33_A_float32.slurm        --dependency=afterok:$afull
    adeps=":$agate:$ascr:$afull:$acpu:$af32"
fi
submit fin   $S/90_finalize.slurm              --dependency=afterany:$lilq:$fmg:$fmc:$b9:$b8:$cc:$b6$adeps

echo "preflight $pre -> timed LiL-Q $lilq, four-method GPU $fmg / CPU $fmc,"
echo "  B9 NiL $b9, B8 $b8, Component C $cc, B6 $b6 -> finalize $fin"
[[ -n "$adeps" ]] && echo "  Component A: A1 gate $agate -> screen $ascr -> full $afull -> CPU $acpu, float32 $af32"                   || echo "  Component A held back (SKIP_A=1): bash scripts/cluster/submit_component_a.sh"
