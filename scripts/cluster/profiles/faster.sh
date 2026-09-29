# TAMU FASTER. A100s sit in composable nodes of 4-16 GPUs with 64 cores; an
# exclusive node holds all of its GPUs (check the "Requested SUs" line sbatch
# prints before relying on the timed jobs here).
ACCOUNT=142698950992
GPU_PARTITION=gpu
GPU_GRES=gpu:a100:1
NODE_CORES=64
CPU_NODE_CORES=64
CPU_PARTITION="${CPU_PARTITION:-cpu}"
LILQ_MODULES="${LILQ_MODULES:-GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0}"   # verified on FASTER, 2026-09-23
