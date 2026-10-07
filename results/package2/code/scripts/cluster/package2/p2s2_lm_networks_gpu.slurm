#!/bin/bash
# Package 2, Stage 2, item 2 (P2-8; the advisor's instructions of 4 October 2026,
# Section 5), contingent: the A100 reruns. Every configuration that the 15-min cap
# ended on the CPU before the target and before 2,000 iterations
# (`p2_8_lm_networks.py gpu-list`) is rerun on the A100, all three seeds, with
# the same caps; the run folders end in _cuda. Submit only after the three CPU
# jobs, and only if gpu-list names something (the laptop rehearsal names
# nothing). About 48 SU per run at most (15 min at 192 SU/h). The walltime,
# 2.5 h (480 SU requested), keeps item 2 within its 600-SU cap with the three
# CPU jobs (120 SU requested) and covers 10 runs. The configurations run in
# gpu-list order (sorted: BL, Bratu, then Burgers by increasing P), so if the
# walltime ends first, the reruns dropped are the largest Burgers sizes, as
# Section 5 requires.
# Submit: LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_lm_networks_gpu.slurm
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-lm-networks-gpu
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
mapfile -t configs < <(python experiments/p2_8_lm_networks.py gpu-list --out "$PKG")
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "gpu-list is empty: no A100 reruns needed"
    exit 0
fi
for c in "${configs[@]}"; do
    read -r bench P <<< "$c"
    python experiments/p2_8_lm_networks.py run --benchmark "$bench" --P "$P" --device cuda --out "$PKG" \
        --reference-dir "$PKG/reference" --package1 "$RESULTS/package1_v2.0.0/package1"
done
