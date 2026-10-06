# Sourced by every job script (before `set -euo pipefail`: Lmod's own
# functions are not safe under `set -u`): software environment, threads,
# paths. Layout, on either cluster:
#   $SCRATCH/lilq-run/venv           (python -m venv --system-site-packages)
#   $SCRATCH/lilq-run/lilq-pinn      (the extracted upload bundle)
# LILQ_MODULES comes from the cluster profile (scripts/cluster/profiles/),
# exported by sbatch.sh.
LILQ_MODULES="${LILQ_MODULES:-GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0}"

module purge
module load $LILQ_MODULES || { echo "module load failed: $LILQ_MODULES"; exit 1; }
source "$SCRATCH/lilq-run/venv/bin/activate" || { echo "venv missing: $SCRATCH/lilq-run/venv"; exit 1; }
# The PyTorch module puts its own torch 2.9.1 on PYTHONPATH, which Python
# searches before the venv; the venv's torch 2.10.0 (Addendum v2.2 Section
# 2.4) must come first. numpy and scipy still come from the modules.
export PYTHONPATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRATCH/lilq-run/lilq-pinn" || { echo "repo missing: $SCRATCH/lilq-run/lilq-pinn"; exit 1; }
# The wheel's own NVIDIA libraries first (Package 2, Stage 2, batch 8). The module put
# CUDA 12.6.0's libraries on LD_LIBRARY_PATH, which the loader searches before the
# wheel's RUNPATH, so some of torch 2.10.0+cu126's companions came from the module.
# With that mix, cuSOLVER's QR failed for a band of shapes around 11,036 x 3,675 on
# Grace (job 20028701); with the wheel's folders first, every shape passed.
# LILQ_MODULE_LD_LIBRARY_PATH keeps the module's path (the preflight records both).
export LILQ_MODULE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
_lilq_nv="$(python -m lilq.cuda_libraries --ld-path)" || { echo "lilq.cuda_libraries failed"; exit 1; }
[[ -n "$_lilq_nv" ]] && export LD_LIBRARY_PATH="$_lilq_nv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _lilq_nv

# One thread per core the job holds: all of the node's for the timed
# (--exclusive) jobs. BLAS, OpenMP and PyTorch all use this count
# (lilq/blas_threads.py, pin_torch), and hardware.json records it.
NCORES="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export OMP_NUM_THREADS="$NCORES"
export OPENBLAS_NUM_THREADS="$NCORES"
export MKL_NUM_THREADS="$NCORES"

# Tests that concern only Component A: run by 29_A1_gate, skipped by the preflight.
export LILQ_COMPONENT_A_TESTS="tests/test_f1_pinn.py tests/test_lm_kovasznay.py tests/test_component_a.py tests/test_baseline_search.py tests/test_kovasznay_comparison.py"

export MPLBACKEND=Agg        # compute nodes have no display
export PYTHONUNBUFFERED=1    # progress lines reach the .out file as they happen

# Each wave writes its own package root, locked to its own commit (Addendum
# v2.2 Section 4.2; results/wave<N>); 90_finalize assembles results/package1
# (package1_results/) from the three.
export RESULTS="$SCRATCH/lilq-run/lilq-pinn/results"
: "${LILQ_WAVE:?LILQ_WAVE is not set: submit through scripts/cluster/sbatch.sh or submit_wave<N>.sh}"
# Package 2's two stages (p2s1, p2s2) write results/package2_stage1 and
# results/package2_stage2, each locked to its own commit like a wave.
case "$LILQ_WAVE" in
    p2s1) export PKG="$RESULTS/package2_stage1" ;;
    p2s2) export PKG="$RESULTS/package2_stage2" ;;
    p2s2g) export PKG="$RESULTS/package2_stage2_gpu" ;;   # item 7's GPU series in the corrected environment (batch 8)
    *)    export PKG="$RESULTS/wave$LILQ_WAVE" ;;
esac
export A="$PKG/A_calibration" B="$PKG/B_instrumentation" C="$PKG/C_oversampling"
mkdir -p "$PKG"

# Provenance lock (Addendum v2.2 Section 2.8.4): the first job records the
# commit and the source-tree hash in $PKG/COMMIT; every job refuses to run
# if the code differs from it, if the files on disk differ from the bundle
# as uploaded, or if the code's version is unknown.
python -m lilq.source_lock --pkg "$PKG" || exit 1
# PyTorch 2.10.0 in every job, not only in the preflight, which waves 2 and 3
# skip on unchanged code (Addendum v2.2 Section 2.4; the advisor's reply, item 2.3).
python -c "from lilq.provenance import assert_torch_version; assert_torch_version()" || exit 1

echo "job ${SLURM_JOB_ID:-?} on $(hostname): $OMP_NUM_THREADS threads, $(nproc) cores usable," \
     "$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'OverSubscribe=[A-Z]*' || echo 'OverSubscribe=?')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
