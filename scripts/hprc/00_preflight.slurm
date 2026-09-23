#!/bin/bash
# Pre-flight: the test suite and a smoke run of every Section 3.3 benchmark,
# on a compute node with the cluster's own library versions. Run this first;
# if it fails, nothing else is worth submitting.
#SBATCH --job-name=lilq-preflight
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:t4:1   # any CUDA device will do for tests
#SBATCH --partition=gpu
#SBATCH --account=142698950992
#SBATCH --output=logs/%x.%j.out

source scripts/hprc/env.sh
set -euo pipefail

python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m pytest tests/ -q
python experiments/component_b.py --smoke --out-root "$SCRATCH/lilq-run/preflight"
python experiments/component_b.py --gpu-equivalence --smoke --out-root "$SCRATCH/lilq-run/preflight"
python experiments/four_method_tables.py --quick --out-dir "$SCRATCH/lilq-run/preflight/four_method"
echo "PREFLIGHT OK"
