#!/bin/bash
# Package 5, item 3, the A100 set: the preflight, on one A100 (shared-gpu: 8 cores). The first job of
# stage p5g: env.sh locks results/package5_item3_gpu to this commit here. The inputs and the item 3
# tests with CUDA visible. Every other job of the stage waits for it.
# Submit: through scripts/cluster/package5/submit_p5_item3_gpu.sh
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5g-preflight
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
inputs=("$RESULTS/package2_stage2/reference"/bratu_ref_p48.npz
        "$RESULTS/package2_stage2/reference"/burgers_cole_hopf.npz
        "$RESULTS/package2_stage2/reference"/bl_fd_ref_viscous_nu0.1.npz)
for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
echo "inputs OK (${#inputs[@]} files)"
# The item 3 tests with the A100 visible (the torch residuals and the off-clock timer on CUDA)
python -m pytest tests/test_p5_3_lil.py tests/test_p5_3_networks.py tests/test_run_probe.py tests/test_p5_3_table.py -q
echo "PREFLIGHT OK"
