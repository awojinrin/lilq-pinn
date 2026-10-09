"""Writes Package 3's job scripts (LF line endings). Kept beside them, so the job
scripts can be regenerated and reviewed from one place.

Usage: python scripts/cluster/package3/_generate_jobs.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEAD = '#!/bin/bash\n'
TAIL = '#SBATCH --output=logs/%x.%j.out\n\nsource scripts/cluster/env.sh\nset -euo pipefail\n'
CAPPED = ("# capped <limit> <command...>: the run under its own wall cap (the advisor's, Section 3.4). A run\n"
          "# stopped by its cap keeps what it logged (Section 8.2); the job goes on to the next run.\n"
          'capped() { local cap=$1; shift; timeout "$cap" "$@" || echo "STOPPED OR FAILED (exit $?, cap $cap): $*"; }\n')
P1 = '"$RESULTS/package1_v2.0.0/package1"'

PREFLIGHT_BODY = f'''python -c "from lilq.provenance import assert_torch_version; print('torch', assert_torch_version(), 'OK')"
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
# K8 (package1), item 3's logged k = 0 residuals (package1; P2-10), item 2's references, item 4's
# reference and its L1 control (P2-12)
P2_10="$RESULTS/package2_stage2/P2_10_elasticity_manufactured"
inputs=({P1}/B_instrumentation/beltrami_pinned/iterations.csv {P1}/B_instrumentation/bl_P576_cpu_paper/iterations.csv)
for d in S1 S2 S3 SPE10; do inputs+=({P1}/B_instrumentation/darcy_${{d}}_cpu_paper/iterations.csv); done
for P in 50 200 450 800 1250; do
    inputs+=({P1}/B_instrumentation/elasticity_P${{P}}_cpu_paper/iterations.csv
             "$P2_10/compatible/P$P/iterations.csv" "$P2_10/specified/P$P/iterations.csv")
done
inputs+=("$RESULTS/package2_stage2/reference/bl_fd_ref_viscous_nu0.1.npz"
         "$RESULTS/package2_stage2/reference/bl_fd_ref_gravity_nu0.1.npz"
         "$RESULTS/package2_stage1/reference/burgers_cole_hopf.npz"
         "$RESULTS/package2_stage1/P2_12_reference_errors/B_instrumentation/burgers_P625_cpu_paper/iterations.csv")
for f in "${{inputs[@]}}"; do
    [[ -f "$f" ]] || {{ echo "MISSING INPUT: $f"; exit 1; }}
done
echo "inputs OK (${{#inputs[@]}} files)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
python experiments/p3_checks.py k1 --out "$PKG"
python experiments/p3_checks.py k2 --out "$PKG"
echo "PREFLIGHT OK"
'''

JOBS = {
    'p3_preflight.slurm': ('preflight', 'cpu', '01:00:00', '''# Package 3 (the advisor's instructions of 8 October 2026, and Addendum 1): the preflight.
# The first Package 3 job: env.sh locks results/package3 to this commit here. Versions,
# every input the jobs read (fail now, not hours in), the test suite, checks K1 and K2.
# Every other Package 3 job waits for it.
# Submit: through scripts/cluster/package3/submit_p3.sh
''', PREFLIGHT_BODY),
    'p3_k0.slurm': ('k0', 'timed-cpu', '02:00:00', '''# Package 3, check K0 (Section 7): gelsy and the column-pivoted QR on a random 600,000 x 7,984
# matrix (4.79e9 entries, above 2^32; 38 GB) with a consistent right-hand side. About 1 h
# (two factorizations at Package 2's measured 44 GFLOP/s). B3 level 2 (5.3e9 entries) is
# launched only if K0 passed (submit_p3_b3l2.sh).
# Submit: through scripts/cluster/package3/submit_p3.sh
''', 'python experiments/p3_k0_k8.py k0 --out "$PKG"\n'),
    'p3_beltrami_small.slurm': ('beltrami-small', 'timed-cpu', '03:00:00', '''# Package 3, item 1 (Section 3) and checks K3, K6, K8, on one whole node (48 threads):
# - the face and slab constants of all six grids (minutes);
# - B1 levels 1 and 2 (1 h cap each), with rho_r;
# - K6: B1 level 1 again (bit-identical); K3: B1 level 1 with the pin weight x 1e-3, x 1e3;
# - K8: the paper's pinned Beltrami run and viscous BL P = 576 against package1 (48 threads,
#   as package1).
# Submit: through scripts/cluster/package3/submit_p3.sh
''', CAPPED + f'''B=experiments/p3_1_beltrami_certified.py
for s in B1 B2 B3; do for l in 1 2; do python $B constants --size $s --level $l --out "$PKG"; done; done
capped 1h python $B run --size B1 --level 1 --out "$PKG"
capped 1h python $B run --size B1 --level 1 --out "$PKG" --tag _rerun
capped 1h python $B run --size B1 --level 1 --out "$PKG" --pin-scale 1e-3 --tag _pin1e-3
capped 1h python $B run --size B1 --level 1 --out "$PKG" --pin-scale 1e3 --tag _pin1e3
capped 1h python $B run --size B1 --level 2 --out "$PKG"
python experiments/p3_k0_k8.py k8 --out "$PKG" --package1 {P1}
'''),
    'p3_beltrami_b2.slurm': ('beltrami-b2', 'timed-cpu', '04:00:00', '''# Package 3, item 1: B2 (3,171 coefficients) at levels 1 and 2, 2 h cap each (Section 3.4);
# kappa by SVD (P < 3,200). About 20 and 45 min.
# Submit: through scripts/cluster/package3/submit_p3.sh
''', CAPPED + '''B=experiments/p3_1_beltrami_certified.py
capped 2h python $B run --size B2 --level 1 --out "$PKG"
capped 2h python $B run --size B2 --level 2 --out "$PKG"
'''),
    'p3_beltrami_b3_l1.slurm': ('beltrami-b3-l1', 'timed-cpu', '04:00:00', '''# Package 3, item 1: B3 (the paper's 7,984 coefficients) at level 1, 348,168 x 7,984, 4 h cap
# (Section 3.4; the run stops at 3 h 55 min so the job ends cleanly). About 2.6 h; peak about
# 105 GB. Its seconds per iteration and kappa time feed B3 level 2's launch rule.
# Submit: through scripts/cluster/package3/submit_p3.sh
''', CAPPED + 'capped 235m python experiments/p3_1_beltrami_certified.py run --size B3 --level 1 --out "$PKG"\n'),
    'p3_beltrami_b3_l2.slurm': ('beltrami-b3-l2', 'timed-cpu', '07:00:00', '''# Package 3, item 1: B3 at level 2, 665,331 x 7,984 (42.5 GB; peak about 150-200 GB), 7 h cap
# (Section 3.4; the run stops at 6 h 55 min). Wave 2: submitted by submit_p3_b3l2.sh only if
# K0 passed, the launch rule holds and the budget guard holds (b3l2_gate.py).
# Submit: through scripts/cluster/package3/submit_p3_b3l2.sh
''', CAPPED + 'capped 415m python experiments/p3_1_beltrami_certified.py run --size B3 --level 2 --out "$PKG"\n'),
    'p3_bl.slurm': ('bl', 'cpu', '02:00:00', '''# Package 3, item 2 (Section 4): Buckley-Leverett 2a and 2b, 48 runs (and the other-guess
# reruns of Section 8.2), six side by side with 4 BLAS threads each, with K5 and K6 in the
# same pool; then 2a's a priori constants and the summary. Laptop: 67 min for the sweep.
# Submit: through scripts/cluster/package3/submit_p3.sh
''', '''python experiments/p3_2_bl_certified.py all --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" \\
    --workers 6 --threads 4 --with-checks
'''),
    'p3_affine.slurm': ('affine', 'timed-cpu', '01:30:00', '''# Package 3, item 3 (Section 5): the certificates for elasticity (3 solutions x 5 sizes) and
# Darcy (4 fields), each re-solve in its own process at the original run's thread count (48:
# package1; 24: P2-10), then K5 (elasticity 144 points, Darcy SPE10 n_q = 6) and the summary
# against package1 and P2-10. Laptop: 22 min.
# Submit: through scripts/cluster/package3/submit_p3.sh
''', f'''python experiments/p3_3_affine_certificates.py all --out "$PKG" --package1 {P1} \\
    --p2-10 "$RESULTS/package2_stage2/P2_10_elasticity_manufactured"
'''),
    'p3_burgers_large_P.slurm': ('burgers-large-p', 'timed-cpu', '01:00:00', '''# Package 3, Addendum 1 (item 4): Burgers LiL-Q at P = 900, 1,024, 1,225 and the P = 625 control
# (1,600 only if none reaches 1.6e-5): the logged passes, then the clean timings (an exclusive
# node, 48 threads; warm-up, median of three) and checks L1-L4. Cap 50 SU.
# Submit: through scripts/cluster/package3/submit_p3.sh
''', '''python experiments/p3_4_burgers_large_P.py all --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference" \\
    --p2-12 "$RESULTS/package2_stage1/P2_12_reference_errors"
'''),
    'p3_report.slurm': ('report', 'cpu', '00:30:00', '''# Package 3: after the jobs (afterany). Item 1's summary (terminal.csv, checks K3, K4, K6, K7,
# over every Beltrami run so far: items 2-4 summarize in their own jobs), then sacct,
# su_per_job.csv and results/package3.tar.gz (the whole package folder and the Slurm logs) to
# download (scripts/cluster/package2/stage_report.py). A failed summary does not stop the tarball.
# Submit: through submit_p3.sh, and again by submit_p3_b3l2.sh after B3 level 2.
''', '''python experiments/p3_1_beltrami_certified.py summarize --out "$PKG" || echo "ITEM 1 SUMMARY FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p3 --results "$RESULTS"
'''),
}


def main():
    for name, (short, cls, wall, doc, body) in JOBS.items():
        text = (HEAD + doc + f'# lilq-resources: {cls}   (applied by scripts/cluster/sbatch.sh from the cluster profile)\n'
                + f'#SBATCH --job-name=lilq-p3-{short}\n#SBATCH --time={wall}\n' + TAIL + body)
        (HERE / name).write_bytes(text.encode())
    print(f'{len(JOBS)} job scripts written in {HERE}')


if __name__ == '__main__':
    main()
