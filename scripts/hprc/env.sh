# Sourced by every job script (before `set -euo pipefail`: Lmod's own
# functions are not safe under `set -u`): software environment, thread
# counts, paths. Assumes the layout from HPRC_FASTER_Workflow_Guide.md:
#   $SCRATCH/lilq-run/venv           (python -m venv --system-site-packages)
#   $SCRATCH/lilq-run/lilq-pinn      (the extracted upload bundle)

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0 || { echo "module load failed"; exit 1; }
source "$SCRATCH/lilq-run/venv/bin/activate" || { echo "venv missing: $SCRATCH/lilq-run/venv"; exit 1; }
cd "$SCRATCH/lilq-run/lilq-pinn" || { echo "repo missing: $SCRATCH/lilq-run/lilq-pinn"; exit 1; }

# One thread per core the job was given -- never more. (lilq/blas_threads.py
# derives the same value from SLURM itself; setting it here as well makes
# the job's thread count visible in its own script.)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

export MPLBACKEND=Agg        # compute nodes have no display
export PYTHONUNBUFFERED=1    # progress lines reach the .out file as they happen

export PKG="$SCRATCH/lilq-run/lilq-pinn/results/package1"   # Section 6's package1_results/
mkdir -p "$PKG"

echo "job ${SLURM_JOB_ID:-?} on $(hostname): $OMP_NUM_THREADS threads, $(nproc) cores usable"
