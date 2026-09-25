# TAMU Grace. A100 nodes: 2 x A100, 48 cores (check: sinfo -p gpu -o "%N %c %m %G %l").
ACCOUNT=132698954494
GPU_PARTITION=gpu
GPU_GRES=gpu:a100:1
NODE_CORES=48                # all cores of an A100 node (the timed jobs hold the whole node)
CPU_PARTITION="${CPU_PARTITION:-}"   # a partition without GPUs, from `sinfo -s` (required; no safe default)
LILQ_MODULES="${LILQ_MODULES:-GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0}"   # FASTER's; check `module spider PyTorch`
