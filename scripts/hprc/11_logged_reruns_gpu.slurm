#!/bin/bash
# Sections 3.2/3.3, GPU: Kovasznay on the GPU (five sizes, both passes) and
# check B3 (GPU vs CPU at every size, plus gelsy/gels/GPU-QR timings on the
# same system). Minutes of work. The job fails (non-zero exit) if B3 fails
# at any size -- Section 3.2: "if it fails, stop and report".
# float64 throughout: use an A100 (full-rate FP64), not a T4 (1/32 rate).
#SBATCH --job-name=lilq-b33-gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=142698950992
#SBATCH --output=logs/%x.%j.out

source scripts/hprc/env.sh
set -euo pipefail

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python experiments/component_b.py --benchmarks kovasznay --devices cuda --out-root "$PKG"
python experiments/component_b.py --gpu-equivalence --out-root "$PKG"
