#!/bin/bash
# Package 5, items 1, 2 and 4 (the advisor's instructions of 13 November 2026 and his reply to
# the cost check of 14 November 2026): the preflight. The first Package 5 job: env.sh locks
# results/package5 to this commit here. The billing of the cpu-mem class, every input the
# jobs read (fail now, not later), the test suite, Grace's scipy against the shipped points.
# Every other Package 5 job waits for it.
# Submit: through scripts/cluster/package5/submit_p5.sh
# lilq-resources: cpu-mem   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5-preflight
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
# This job is cpu-mem (24 cores, 96 GB): the plan assumes Grace bills it as 24 cores. Stop
# everything (the other jobs wait for this one) if it is billed higher.
billing=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'billing=[0-9]*' | head -1 | cut -d= -f2 || true)
echo "cpu-mem billing: ${billing:-unknown} per hour (expected 24)"
[[ -n "$billing" ]] || echo "WARNING: the billing could not be read; check sacct's AllocTRES of this job"
[[ "${billing:-0}" -le 24 ]] || { echo "CPU-MEM IS BILLED AT $billing, NOT 24: STOPPING (nothing else runs)"; exit 1; }
# Every input the jobs read: item 1's references (Package 2, stage 2), item 2's Cole-Hopf
# reference (stage 1, as Package 4), the shipped Halton points of Packages 4 and 5
inputs=("$RESULTS/package2_stage2/reference"/bratu_ref_p48.npz
        "$RESULTS/package2_stage2/reference"/burgers_cole_hopf.npz
        "$RESULTS/package2_stage2/reference"/bl_fd_ref_viscous_nu0.1.npz
        "$RESULTS/package2_stage2/reference"/bl_fd_ref_gravity_nu0.1.npz
        "$RESULTS/package2_stage1/reference"/burgers_cole_hopf.npz
        data/package4/halton_seed0.npz
        data/package4/halton_seed1.npz
        data/package4/halton_seed2.npz
        data/package5/halton_seed0.npz
        data/package5/halton_seed1.npz
        data/package5/halton_seed2.npz)
for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
echo "inputs OK (${#inputs[@]} files)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
# Grace's own scipy against the shipped 16,000 points (a record: every set is built from the shipped points)
python experiments/p5_4_pb.py compare-points --out "$PKG" --points data/package5
echo "PREFLIGHT OK"
