#!/bin/bash
# Submit the Component B jobs in order, from the repository root on a login
# node:   bash scripts/hprc/submit_all.sh
# The pre-flight job must pass before anything else starts (afterok); the
# finalize job waits for all the others. Check progress with: squeue -u $USER
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs   # sbatch does not create the --output directory

# sbatch can fail (e.g. "Socket timed out" when the scheduler is overloaded)
# without a usable job ID; stop at the first such failure rather than chain
# later jobs onto an empty dependency. A timeout may still have queued the
# job, so check `squeue -u $USER` before resubmitting anything.
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

submit pre scripts/hprc/00_preflight.slurm
submit cpu --dependency=afterok:$pre scripts/hprc/10_logged_reruns_cpu.slurm
submit gpu --dependency=afterok:$pre scripts/hprc/11_logged_reruns_gpu.slurm
submit fmg --dependency=afterok:$pre scripts/hprc/20_four_method_gpu.slurm
submit fmc --dependency=afterok:$pre scripts/hprc/21_four_method_cpu.slurm
submit bas --dependency=afterok:$pre scripts/hprc/30_basis_study_and_beltrami_pinned.slurm
submit fin --dependency=afterany:$cpu:$gpu:$fmg:$fmc:$bas scripts/hprc/40_finalize.slurm

echo "preflight $pre -> cpu $cpu, gpu $gpu, four-method gpu $fmg / cpu $fmc, 3.6+3.7 $bas -> finalize $fin"
