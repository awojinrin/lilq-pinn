#!/bin/bash
# Package 3 (the advisor's instructions of 8 October 2026, and Addendum 1): the preflight.
# The first Package 3 job: env.sh locks results/package3 to this commit here. Versions,
# every input the jobs read (fail now, not hours in), the test suite, checks K1 and K2.
# Every other Package 3 job waits for it.
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-preflight
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python -c "from lilq.provenance import assert_torch_version; print('torch', assert_torch_version(), 'OK')"
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
# K8 (package1), item 3's logged k = 0 residuals (package1; P2-10), item 2's references, item 4's
# reference and its L1 control (P2-12)
P2_10="$RESULTS/package2_stage2/P2_10_elasticity_manufactured"
inputs=("$RESULTS/package1_v2.0.0/package1"/B_instrumentation/beltrami_pinned/iterations.csv "$RESULTS/package1_v2.0.0/package1"/B_instrumentation/bl_P576_cpu_paper/iterations.csv)
for d in S1 S2 S3 SPE10; do inputs+=("$RESULTS/package1_v2.0.0/package1"/B_instrumentation/darcy_${d}_cpu_paper/iterations.csv); done
for P in 50 200 450 800 1250; do
    inputs+=("$RESULTS/package1_v2.0.0/package1"/B_instrumentation/elasticity_P${P}_cpu_paper/iterations.csv
             "$P2_10/compatible/P$P/iterations.csv" "$P2_10/specified/P$P/iterations.csv")
done
inputs+=("$RESULTS/package2_stage2/reference/bl_fd_ref_viscous_nu0.1.npz"
         "$RESULTS/package2_stage2/reference/bl_fd_ref_gravity_nu0.1.npz"
         "$RESULTS/package2_stage1/reference/burgers_cole_hopf.npz"
         "$RESULTS/package2_stage1/P2_12_reference_errors/B_instrumentation/burgers_P625_cpu_paper/iterations.csv")
for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
echo "inputs OK (${#inputs[@]} files)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
python experiments/p3_checks.py k1 --out "$PKG"
python experiments/p3_checks.py k2 --out "$PKG"
echo "PREFLIGHT OK"
