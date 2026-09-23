#!/bin/bash
# Submit the Component B jobs in order, from the repository root on a login
# node:   bash scripts/hprc/submit_all.sh
# The pre-flight job must pass before anything else starts (afterok); the
# finalize job waits for all the others. Check progress with: squeue -u $USER
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs   # sbatch does not create the --output directory

pre=$(sbatch --parsable scripts/hprc/00_preflight.slurm)
cpu=$(sbatch --parsable --dependency=afterok:$pre scripts/hprc/10_logged_reruns_cpu.slurm)
gpu=$(sbatch --parsable --dependency=afterok:$pre scripts/hprc/11_logged_reruns_gpu.slurm)
fmg=$(sbatch --parsable --dependency=afterok:$pre scripts/hprc/20_four_method_gpu.slurm)
fmc=$(sbatch --parsable --dependency=afterok:$pre scripts/hprc/21_four_method_cpu.slurm)
bas=$(sbatch --parsable --dependency=afterok:$pre scripts/hprc/30_basis_study_and_beltrami_pinned.slurm)
fin=$(sbatch --parsable --dependency=afterany:$cpu:$gpu:$fmg:$fmc:$bas scripts/hprc/40_finalize.slurm)

echo "preflight $pre -> cpu $cpu, gpu $gpu, four-method gpu $fmg / cpu $fmc, 3.6+3.7 $bas -> finalize $fin"
