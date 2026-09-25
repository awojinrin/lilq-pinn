# Sourced by every Grace job script (before `set -euo pipefail`: Lmod's own
# functions are not safe under `set -u`): software environment, threads,
# paths. Layout, as on FASTER:
#   $SCRATCH/lilq-run/venv           (python -m venv --system-site-packages)
#   $SCRATCH/lilq-run/lilq-pinn      (the extracted upload bundle)
#
# CHECK ONCE on a Grace login node before the first submission:
#   module spider PyTorch
# The names below are FASTER's. If Grace's differ, change LILQ_MODULES here
# (or export it before sbatch); nothing else refers to them.
LILQ_MODULES="${LILQ_MODULES:-GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0}"

module purge
module load $LILQ_MODULES || { echo "module load failed: $LILQ_MODULES"; exit 1; }
source "$SCRATCH/lilq-run/venv/bin/activate" || { echo "venv missing: $SCRATCH/lilq-run/venv"; exit 1; }
cd "$SCRATCH/lilq-run/lilq-pinn" || { echo "repo missing: $SCRATCH/lilq-run/lilq-pinn"; exit 1; }

# One thread per core the job holds: all of the node's for the timed
# (--exclusive) jobs. BLAS, OpenMP and PyTorch all use this count
# (lilq/blas_threads.py, pin_torch), and hardware.json records it.
NCORES="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export OMP_NUM_THREADS="$NCORES"
export OPENBLAS_NUM_THREADS="$NCORES"
export MKL_NUM_THREADS="$NCORES"

export MPLBACKEND=Agg        # compute nodes have no display
export PYTHONUNBUFFERED=1    # progress lines reach the .out file as they happen

export PKG="$SCRATCH/lilq-run/lilq-pinn/results/package1"   # package1_results/
export A="$PKG/A_calibration" B="$PKG/B_instrumentation" C="$PKG/C_oversampling"
mkdir -p "$PKG"

echo "job ${SLURM_JOB_ID:-?} on $(hostname): $OMP_NUM_THREADS threads, $(nproc) cores usable," \
     "$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'OverSubscribe=[A-Z]*' || echo 'OverSubscribe=?')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
