#!/bin/bash
# Submit Component A on its own (after submit_all.sh ran with SKIP_A=1), then
# a second finalize that includes it:
#   CLUSTER=grace bash scripts/cluster/submit_component_a.sh
# The A1 gate job (search, Component A tests, checks F2, A2, A1) runs first.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
S=scripts/cluster
submit() {   # submit <variable> <job script> [sbatch options...]
    local var=$1 script=$2 id; shift 2
    id=$(bash $S/sbatch.sh "$script" --parsable "$@" | cut -d';' -f1) || true
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "submission failed for $script; check squeue -u \$USER" >&2; exit 1; }
    printf -v "$var" '%s' "$id"
}
submit agate $S/29_A1_gate.slurm
submit ascr  $S/30_A_screen.slurm         --dependency=afterok:$agate
submit afull $S/31_A_full.slurm           --dependency=afterok:$ascr
submit acpu  $S/32_A_cpu.slurm            --dependency=afterok:$afull
submit af32  $S/33_A_float32.slurm        --dependency=afterok:$afull
submit fin   $S/90_finalize.slurm         --dependency=afterany:$agate:$ascr:$afull:$acpu:$af32
echo "Component A: A1 gate $agate -> screen $ascr -> full $afull -> CPU $acpu, float32 $af32 -> finalize $fin"
