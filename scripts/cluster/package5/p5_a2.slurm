#!/bin/bash
# Package 5, Addendum 2 (Sections 2 and 3): the 18 runs of gravity Buckley-Leverett from the initial
# profile, K1 and K2, the summary, then results/package5_a2.tar.gz. cpu-mem: six processes, each
# about 8 GB while it builds item 1's norm of X. Laptop rehearsal at K_max = 4: 8 min; the 60
# iterations of every run about 11 min of solves at 16 threads.
# Submit: through scripts/cluster/package5/submit_p5_a2.sh
# lilq-resources: cpu-mem   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5a-a2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=3600   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps every iterate it wrote (the drivers write after each one); the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
# Every input: the finite-difference references (Package 2, stage 2), item 1's fine reference
# and its two gravity runs at P = 1,024, weight 10^4 (K1), the delivered P2-9 run (K2)
inputs=("$RESULTS/package2_stage2/reference"/bl_fd_ref_gravity_nu0.1.npz
        "$RESULTS/package2_stage2/reference"/bl_fd_ref_gravity_nu0.05.npz
        "$RESULTS/package2_stage2/reference"/bl_fd_ref_gravity_nu0.02.npz
        "$RESULTS/package5/P5_1_weights"/bl_reference/chosen_gravity.json
        "$RESULTS/package5/P5_1_weights"/bl_reference/gravity_p88/beta_reference.npy
        "$RESULTS/package5/P5_1_weights"/bl_reference/gravity_p80/beta_reference.npy
        "$RESULTS/package5/P5_1_weights"/bl_cheb_gravity/P1024_lam10000/iterations.csv
        "$RESULTS/package5/P5_1_weights"/bl_cheb_gravity/P1024_lam10000/errors_by_iterate.csv
        "$RESULTS/package5/P5_1_weights"/bl_sine_gravity/P1024_lam10000/iterations.csv
        "$RESULTS/package5/P5_1_weights"/bl_sine_gravity/P1024_lam10000/errors_by_iterate.csv
        "$RESULTS/package2_stage2/P2_9_bases/bl_gravity_paper_P1024_ic"/iterations.csv)
for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
echo "inputs OK (${#inputs[@]} files)"
python experiments/p4_checks.py provenance --out "$PKG" --name a2
A2=(--out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" --item1-root "$RESULTS/package5")
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# Parts A to C (16 runs) in five processes and K1 beside them, 4 threads each (24 in all): item 1's
# thread count, so that K1 compares like with like
pids=()
until_end python experiments/p5_a2_plateau.py run --part A --space cheb "${A2[@]}" & pids+=($!)
until_end python experiments/p5_a2_plateau.py run --part A --space sine "${A2[@]}" & pids+=($!)
until_end python experiments/p5_a2_plateau.py run --part B --space cheb "${A2[@]}" & pids+=($!)
until_end python experiments/p5_a2_plateau.py run --part B --space sine "${A2[@]}" & pids+=($!)
until_end python experiments/p5_a2_plateau.py run --part C "${A2[@]}" & pids+=($!)
until_end python experiments/p5_a2_plateau.py k1 "${A2[@]}" & pids+=($!)
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] && echo "PARTS A TO C AND K1 DONE" || echo "PARTS A TO C OR K1: A PROCESS FAILED (see above)"
# Part D at the delivered run's 24 threads (K2 compares the repeat at weight 10 with it)
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
until_end python experiments/p5_a2_plateau.py run --part D "${A2[@]}"
python experiments/p5_a2_plateau.py summarize "${A2[@]}" --delivered "$RESULTS/package2_stage2/P2_9_bases/bl_gravity_paper_P1024_ic" || echo "SUMMARY FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p5a --results "$RESULTS"
