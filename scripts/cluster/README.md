# Running Package 1 on TAMU HPRC (Grace or FASTER)

Every job of Package 1 v2.0 and Addendums v2.1-v2.2 runs on the cluster, in
three waves (Addendum v2.2 Section 4). Timed work (everything whose times go
into the paper) holds a whole node: on an A100 node, one A100 with all 48
cores and all 360G of memory, without `--exclusive` -- no other job fits on
the node (the advisor's reply to wave 1, item 2.3), though Grace still
charges 192 SU/h for it, as for wave 1's `--exclusive` node with both A100s
(waves 2-3); a CPU node (`--exclusive`) -- the same CPU, no GPU surcharge -- when it
does not use the GPU.
Untimed work runs on a shared A100 or on CPU cores. The same job scripts
serve both clusters: the cluster-specific values live in `profiles/grace.sh`
and `profiles/faster.sh`, and `sbatch.sh` applies them according to each
script's `# lilq-resources:` class (`timed`, `timed-cpu`, `shared-gpu`, `cpu`).

## 1. Before the first submission (login node)

```bash
module spider PyTorch                    # module names (profile's LILQ_MODULES)
sinfo -p gpu -o "%N %c %m %G %l"         # A100 nodes: cores (Grace profile: 48), memory, GPUs, time limit
sinfo -s                                 # a partition without GPUs, for CPU_PARTITION
myproject -l                             # the account and its balance
export LILQ_ACCOUNT=<account>            # the account jobs are charged to (add it to ~/.bashrc)
```

Edit the profile if the module names or the nodes' core counts differ. On
Grace the CPU nodes (`medium`, 1-day limit) have the A100 nodes' CPU, 2 x Xeon
Gold 6248R (checked with `lscpu`, 2026-09-29); every job records its CPU
model in `hardware.json`.

## 2. Build and upload (laptop), then the environment (cluster, once)

```bash
python scripts/make_hprc_bundle.py          # -> ../lilq-pinn-<sha>.tar.gz; refuses uncommitted changes
```

```bash
mkdir -p $SCRATCH/lilq-run && tar -xzf lilq-pinn-<sha>.tar.gz -C $SCRATCH/lilq-run
cd $SCRATCH/lilq-run
module load <LILQ_MODULES from the profile>
python -m venv --system-site-packages venv
source venv/bin/activate
echo "numpy==1.26.4" > constraints.txt   # keep the module's numpy: the newest matplotlib pulls numpy 2.x,
                                         # which the module's scipy 1.13.1 cannot import
pip install --no-cache-dir -c constraints.txt matplotlib pytest threadpoolctl   # threadpoolctl: thread pools in hardware.json (2.12)
pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126   # about 3 GB
export PYTHONPATH=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):$PYTHONPATH
python -c "import numpy, scipy.linalg, torch; print(numpy.__version__, numpy.__file__, torch.__version__)"
# expect 1.26.4 from /sw/eb/sw/SciPy-bundle/... (not the venv) and 2.10.0+cu126. If the venv has its own
# numpy, remove it with the venv first on PYTHONPATH (pip uninstall -y numpy), or pip finds the module's.
```

The PyTorch module puts its own torch (2.9.1) on `PYTHONPATH`, ahead of the
venv. `env.sh` puts the venv first in every job, and the preflight fails
unless torch is 2.10.0 (Addendum v2.2 Section 2.4: 2.9.1's line search
ignores `max_eval`, which changes the L-BFGS evaluation counts).

## 3. Submit, one wave at a time

```bash
cd $SCRATCH/lilq-run/lilq-pinn
DRY_RUN=1 bash scripts/cluster/submit_wave1.sh    # every job through sbatch --test-only; submits nothing
bash scripts/cluster/submit_wave1.sh
```

Run the dry run first, before every wave: it puts each job through the
scheduler's own checks (on Grace, `--mem=0` is refused, so whole-node jobs
request `--mem=360G`, all of a node's 368,640 MB) and prints an estimated
start, without submitting, charging or writing anything.

Each script shows the balance (`myproject -l`) and asks before submitting
anything (`YES=1` skips the question). Grace printed no "Requested SUs" line
in wave 1: its booking covers cores only, and the GPU surcharge is charged
at run time. When wave 2's first timed GPU job starts, check that it holds
one GPU (`sacct -X -j <job> --format=JobID,JobName%24,AllocTRES%70` shows
`gres/gpu=1`), and after it ends, its charge in `myproject`.

| Wave | Jobs | Requested (advisor's estimate) | Then |
|---|---|---|---|
| 1 | preflight; A1 gate; 10a, 10b; Component C; B6; B4 for Bratu (20 `--array=0 --time=03:00:00`, 21 `--array=0`) | ~1,410 SU (charged: 332) | done at `8a3f5f7`; results on branch `wave1-results` |
| 2 | preflight and A1 gate (new code); Component A (30 -> 31 -> 32, 33); B4 for the other three benchmarks with stall controls, and wave 1's Bratu controls (20, 21 `--array=0-3`); timing reruns (11a, 11b) | ~5,900-6,700 SU expected; ~11,700 if every job hit its walltime | `results/wave2_report.tar.gz`; check `gres/gpu=1` on the first timed job; submit wave 3 once wave 2's charges have posted |
| 3 | B9 (40); B8 (41) | ~2,500-4,200 SU (charged: 1,019) | done at `905e58c`; results on branch `wave3-results` |
| 4 | preflight and A1 gate (new code); Component A: F1 re-pick (34), F2 full stage (35), CPU reruns (32, one task each), F1 float32 (33); clean timing (13a, 13b); Table 3 (43); B10 (44); B8 K_max = 60 reruns (45); finalize | ~3,900 SU (advisor's estimate); ~5,300 if every job hit its walltime | `results/wave4_report.tar.gz`; `results/package1` (package1_results/) from the four waves |

**One results folder per wave.** Wave N writes `results/wave<N>` only, and
the first job of a wave locks that folder to the code's commit and
source-tree hash (`results/wave<N>/COMMIT`, `lilq/source_lock.py`). Every job
then refuses to run if the code differs from its wave's lock, if a file was
edited after upload, or if the bundle was built from uncommitted changes. A
code change between waves is therefore allowed and recorded: the next wave
locks to the new commit, and, when the code differs from wave 1's, also
reruns the preflight (and, in wave 2, the A1 gate; with wave 1's code it
reuses wave 1's search and checks instead). `90_finalize` assembles
`results/package1` from the three folders and records each wave's commit in
`WAVES.json`.

**Each wave ends with `91_wave_report`**: the wave's run index, check B1's
reproduction table, the merged four-method table, `sacct` for its jobs, and
`results/wave<N>_report.tar.gz` with everything the advisor asked to see
(COMMIT, the check tables, the K_max and Beltrami logs, the four-method rows
with their histories, the Component A checks, `oversampling.csv`, every
`hardware.json`, the Slurm logs). Models stay on the cluster. It also
writes `su_per_job.csv`: each job's SUs from `sacct` at Grace's rates
(`wave_report.su_rate`), which reproduce the charges of waves 2 and 3. In
wave 4 it adds the B8 reruns beside wave 3 (`b8_kmax60_vs_wave3.csv`) and
every quoted LiL-Q time clean against logged
(`clean_timing/clean_vs_logged.csv`). `90_finalize` marks `package1`
provisional (`WAVES.json`'s `status`) until the advisor has reviewed the
wave 4 report; after his approval, `assemble_package.py --final` marks it
final.

**If the preflight fails.** Every job of the wave depends on it, and `sbatch.sh`
submits with `--kill-on-invalid-dep=yes`, so they are cancelled rather than
left queued with their SUs booked. The preflight writes only
`results/wave<N>/COMMIT` into the wave folder (its smoke outputs go to
`$SCRATCH/lilq-run/preflight/<job id>/`). To recover at a new commit: upload
the fixed bundle, remove `results/wave<N>/COMMIT` (nothing else of that wave
exists yet), and submit the wave again. Change walltimes only on the command
line (`--time=...`): editing a `.slurm` file on the cluster changes the tree
hash, and the lock refuses the job.

To resubmit one script (after a walltime kill, say), use the wrapper with the
wave set, so the profile's resources and the wave's folder apply:
`LILQ_WAVE=2 bash scripts/cluster/sbatch.sh scripts/cluster/20_four_method_gpu.slurm --array=2`.
Every job resumes where it stopped: completed runs are skipped, and a B9
Darcy network resumes from its last 5,000-epoch checkpoint.

Every run saves its trained model next to its logs (`solution.pt`,
`network.pt`, F1's `model.pt`, F2's `theta.pt`); see "Reloading trained
models" in the top-level README. Keep the `results/` folders whole when
copying them off the cluster: the models are what later figures are made from.

| Script | What | Class | Walltime |
|---|---|---|---|
| `00_preflight` | torch version, tests (not the Component A ones), smoke runs of B and C | shared-gpu | 1 h |
| `29_A1_gate` | Component A search and tests; checks F2, A2 and A1 (the plain PINN, 1 h budget); gates Component A only | shared-gpu | 2 h |
| `10a_timed_lilq_cpu` | Section 3.3 CPU runs incl. the K_max = 60 and Beltrami K_max = 8 passes; Section 3.7 and its K_max pass | timed-cpu | 4 h |
| `10b_timed_lilq_gpu` | Section 3.3 Kovasznay GPU runs (both passes); check B3 with its timings | timed | 2 h |
| `11a_timing_reruns_cpu` | wave 2: the paper passes of Bratu, Burgers, both BL, elasticity, Kovasznay, with warm-up runs | timed-cpu | 1 h |
| `11b_timing_reruns_gpu` | wave 2: the Kovasznay GPU paper passes and check B3, with warm-up runs | timed | 1 h |
| `13a_clean_timing_cpu` | wave 4: gravity BL P = 64 logged paper pass (K_max = 60); clean timing (warm-up, diagnostics off) of every quoted CPU LiL-Q time, the pinned Beltrami run included | timed-cpu | 1.5 h |
| `13b_clean_timing_gpu` | wave 4: clean timing of the Kovasznay GPU paper passes | timed | 1 h |
| `20_four_method_gpu` | B4 GPU pass, one task per benchmark, then its stall controls; wave 2's Bratu task: wave 1's Bratu controls | timed | 8 h each |
| `21_four_method_cpu` | B4 CPU pass at the largest sizes, then its stall controls; wave 2's Bratu task: wave 1's | timed-cpu | 4 h each |
| `30_A_screen` | Component A: 24 configurations x 10 min per family | timed x 2 | 6 h each |
| `31_A_full` | selection, top 3 x 5 seeds x 60 min per family, representative | timed x 2 | 18 h each |
| `32_A_cpu` | each representative on the CPU, 5 seeds x 60 min | timed-cpu x 2 | 7 h each |
| `33_A_float32` | F1 float32-Adam run | timed | 2 h |
| `40_b9_nil_darcy` | B9: NiL Darcy, the manuscript's network (3,555 parameters), float64 and float32, 4 fields x 3 seeds (LiL alongside) | shared-gpu x 12 | 3 h each |
| `41_b8_initial_guess` | B8: 128 runs, one task per (case, guess); networks on the A100 | shared-gpu x 4 | 6 h each |
| `42_component_c` | Component C: 196 LiL-Q runs | cpu | 6 h |
| `43_basis_study` | B6: the Burgers basis study (wave 4: with the default-init ELM row, the test error and both kappas) | cpu | 3 h |
| `34_A_f1_repick` | wave 4: F1's representative re-picked among wave 2's finalists by validation residual (no training) | shared-gpu | 1 h |
| `35_A_f2_full` | wave 4: F2's validation top three x 5 seeds x 60 min, representative by validation residual | timed | 18 h |
| `44_b10_cgl_cc` | wave 4: B10, Kovasznay on CGL grids with Clenshaw-Curtis weights (16 runs) | cpu | 2 h |
| `45_b8_kmax60` | wave 4: B8's capped LiL-Q rows of wave 3, rerun with K_max = 60 | cpu | 1 h |
| `91_wave_report` | the wave's tables and report tarball | cpu | 45 min |
| `90_finalize` | package1 from the waves; reference/, code/, merged tables (B4, B8, B9), B1, B5 figures, Section 4.6 comparison | cpu | 1 h |

Walltimes are about 1.5 x the expected runtimes (the advisor's patch 0001):
HPRC books pending SUs from the requested walltimes at submission. Revise
them from wave 1's measured runtimes.

**SUs.** Charged for time used: cores x hours plus a GPU surcharge per GPU-hour
(Grace A100 72; FASTER A100 128). On Grace a timed A100 job was expected to
be 48 core-SU plus 72 for its one A100, 120 SU per node-hour, but waves 2 and
3 were charged 192, as wave 1's `--exclusive` node with both A100s was
(`sacct`, `myproject`; the wave 2 report). An exclusive CPU node
is 48 SU per hour; a shared A100 job 80; a 24-core CPU job 24. Wave 1 was
charged 332 SU; the advisor expects 5,900-6,700 for wave 2 and 2,500-4,200
for wave 3. On
FASTER, A100s sit in 4-16-GPU nodes, so an exclusive node holds several A100s
at 128 SU each: the timed GPU jobs there would cost several times more.

## 4. Package 2, Stage 2 (Grace; item 5 on FASTER)

Stage 2 writes `results/package2_stage2`, locked to one commit like a wave.
Build the bundle at that commit, upload it and extract it over the previous
code (`results/` is not in the bundle, so Package 1 and Stage 1 stay as they
are), then:

```bash
cd $SCRATCH/lilq-run/lilq-pinn
DRY_RUN=1 bash scripts/cluster/package2/submit_p2s2.sh    # all 17 jobs through sbatch --test-only; submits nothing
bash scripts/cluster/package2/submit_p2s2.sh
```

**The chain:**
1. `p2s2_preflight` checks versions, the GPU and the test suite, and locks the folder.
2. `p2s2_references` makes every reference once.
3. The 14 compute jobs (items 1-4 and 6-8) wait for the references.
4. `p2s2_report` runs after all of them (afterany). It writes `sacct.txt` and `su_per_job.csv`,
   and packs `results/package2_stage2.tar.gz`: the whole stage and its Slurm logs.

The plan is 1,274 SU requested, about 618 expected (`package2_results/su_plan.csv`).

**Item 2's A100 reruns are not in the chain.** When the three LM jobs are done, run:

```bash
python experiments/p2_8_lm_networks.py gpu-list --out results/package2_stage2
```

Submit `p2s2_lm_networks_gpu.slurm` only if it names a configuration
(`LILQ_WAVE=p2s2 bash scripts/cluster/sbatch.sh scripts/cluster/package2/p2s2_lm_networks_gpu.slurm`).

**Item 5 (Darcy) runs on FASTER, from the same commit.** Upload the same bundle to FASTER's
`$SCRATCH/lilq-run/lilq-pinn` and extract it there. FASTER's venv needs the same PyTorch 2.10.0;
the preflight checks it. Then, on a FASTER login node:

```bash
cd $SCRATCH/lilq-run/lilq-pinn
DRY_RUN=1 bash scripts/cluster/package2/submit_p2s2_faster.sh    # 3 jobs through sbatch --test-only
bash scripts/cluster/package2/submit_p2s2_faster.sh
```

The script sets `CLUSTER=faster` itself. The chain is:
1. `p2s2_preflight` checks the environment and locks FASTER's `results/package2_stage2` to the
   commit.
2. `p2s2_darcy_hardbc` runs 12 array tasks (4 fields x 3 seeds), each capped at 2,000 LM
   iterations or 30 min of training.
3. `p2s2_report` packs `results/package2_stage2_faster.tar.gz`.

That is 1,382 SU requested and about 700 expected, within item 5's 1,500 on FASTER.

**Assembly, on the laptop.** Download the tarball(s), FASTER's too, and check out the stage's
commit. Then:

```bash
python experiments/p2_assemble.py stage2 --stage <extracted>/package2_stage2 --out ../package2_results \
    --package1 <package1> --faster <extracted FASTER>/package2_stage2
```

This copies the item folders and references into the Section 12.1 layout, writes `code/`, and
runs the summaries that need Stage 1's laptop-side CSVs. `--faster` refuses a FASTER folder
locked to another commit, and adds item 5's rows. Back up both tarballs to `$HOME/lilq-results`
(on each cluster) before deleting anything on `$SCRATCH`.

## Resource use

Timed jobs hold the whole node and start one thread per core for BLAS,
OpenMP and PyTorch; `hardware.json` in every run directory records the node's
CPU model and GPUs, the thread counts in effect, whether the job held the
node exclusively (`exclusive`), whether it held every core and all the memory
(`holds_whole_node`) and its GPU count (`gpus_allocated`). Never run
experiments or tests on a login node.
