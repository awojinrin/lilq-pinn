# TAMU Grace. A100 nodes g001-g100: 2 x A100, 48 cores, 360 GB, gpu partition limit 4 days
# (sinfo, 2026-09-25).
ACCOUNT="${LILQ_ACCOUNT:-}"     # your HPRC allocation (myproject -l): export LILQ_ACCOUNT=<account>
GPU_PARTITION=gpu
GPU_GRES=gpu:a100:1
NODE_CORES=48                # all cores of an A100 node: with all its memory, a timed job holds the node (one GPU charged)
NODE_MEM=360G                # all its memory (RealMemory 368,640 MB = 360 GiB); Grace's job_submit refuses --mem=0
CPU_NODE_MEM=360G            # the same memory as the A100 nodes (scontrol, 2026-09-29)
CPU_NODE_CORES=48            # regular compute nodes c001-c800: 2 x Xeon Gold 6248R, the A100 nodes' CPU (lscpu, 2026-09-29)
CPU_PARTITION="${CPU_PARTITION:-medium}"   # CPU nodes, 1-day limit (short: 2 h, too short for Component C)
LILQ_MODULES="${LILQ_MODULES:-GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0}"   # present on Grace (module spider, 2026-09-25)
