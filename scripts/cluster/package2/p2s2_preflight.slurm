#!/bin/bash
# Package 2, Stage 2: the preflight (as Package 1's 00_preflight). On a shared A100 (not
# timed): PyTorch 2.10.0, the thread-pool library, the GPU, and the test suite -- with
# the GPU visible, so the GPU tests run too (Beltrami's repaired GPU path, Package 2
# batch 4, among them). Every other Stage 2 job waits for it. The first Stage 2 job:
# env.sh locks results/package2_stage2 to this commit here.
# Also the GPU addendum's preflight (LILQ_WAVE=p2s2g, submit_p2s2g.sh): results/package2_stage2_gpu.
# Submit: through scripts/cluster/package2/submit_p2s2.sh
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-preflight
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python -c "from lilq.provenance import assert_torch_version; print('torch', assert_torch_version(), 'OK')"
python -c "import threadpoolctl; print('threadpoolctl', threadpoolctl.__version__, 'OK')"
python -c "import numpy, scipy, torch; assert torch.cuda.is_available(), 'no GPU visible'; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"
# The NVIDIA libraries (batch 8): those the jobs load, with cuSOLVER's QR on the shape that failed
# with the module's libraries first (item 7, P = 3,675) and on the paper's P = 1,875; then the
# module-first environment once, for the record.
python -m lilq.cuda_libraries --json "$PKG/cuda_libraries.json" --probe 11036x3675 5564x1875 > /dev/null
python -c "import json, sys; r = json.load(open(sys.argv[1])); bad = {k: v for k, v in r['geqrf_probe'].items() if v != 'ok'}; assert not bad, bad; print('cuSOLVER QR probe OK:', ', '.join(r['geqrf_probe']))" "$PKG/cuda_libraries.json"
LD_LIBRARY_PATH="$LILQ_MODULE_LD_LIBRARY_PATH" python -m lilq.cuda_libraries --json "$PKG/cuda_libraries_module_first.json"     --probe 11036x3675 > /dev/null || echo "(the module-first record failed: not needed to run)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
echo "PREFLIGHT OK"
