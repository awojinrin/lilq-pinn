# Running Package 1 on TAMU HPRC (Grace or FASTER)

Every job of Package 1 v2.0 and Addendums v2.1-v2.2 runs on the cluster, in
three waves (Addendum v2.2 Section 4). Timed work (everything whose times go
into the paper) holds a whole node (`--exclusive`): an A100 node when it uses
the GPU, a CPU node -- the same CPU, no GPU surcharge -- when it does not.
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
source venv/bin/activate && pip install matplotlib pytest threadpoolctl   # threadpoolctl: thread pools in hardware.json (Addendum v2.2 2.12)
pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126   # about 3 GB
export PYTHONPATH=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):$PYTHONPATH
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"   # 2.10.0+cu126, and the module's numpy
```

The PyTorch module puts its own torch (2.9.1) on `PYTHONPATH`, ahead of the
venv. `env.sh` puts the venv first in every job, and the preflight fails
unless torch is 2.10.0 (Addendum v2.2 Section 2.4: 2.9.1's line search
ignores `max_eval`, which changes the L-BFGS evaluation counts).

## 3. Submit, one wave at a time

```bash
cd $SCRATCH/lilq-run/lilq-pinn
bash scripts/cluster/submit_wave1.sh
```

Each script shows the balance (`myproject -l`) and asks before submitting
anything (`YES=1` skips the question). Read the "Requested SUs" line sbatch
prints for each job: it shows whether an exclusive A100 node is charged for
one GPU or both (120 or 192 SU/h), which changes every estimate below.

| Wave | Jobs | Requested (advisor's estimate) | Then |
|---|---|---|---|
| 1 | preflight; A1 gate; 10a, 10b; Component C; B6; B4 for Bratu (20, 21 `--array=0`) | ~2,000 SU | send `results/wave1_report.tar.gz` and the SUs charged per job; wait for the advisor's reply |
| 2 | Component A (30 -> 31 -> 32, 33); B4 for the other three benchmarks (20, 21 `--array=1-3`) | ~10,000 SU | `results/wave2_report.tar.gz`; submit wave 3 once wave 2's charges have posted |
| 3 | B9 (40); B8 (41); finalize | ~4,800 SU (from the walltimes; the advisor estimated ~7,700 with the larger B9 network) | `results/package1` (package1_results/), `results/wave3_report.tar.gz` |

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
`hardware.json`, the Slurm logs). Models stay on the cluster.

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
| `20_four_method_gpu` | B4 GPU pass, one task per benchmark | timed | 8 h each |
| `21_four_method_cpu` | B4 CPU pass at the largest sizes | timed-cpu | 3 h each |
| `30_A_screen` | Component A: 24 configurations x 10 min per family | timed x 2 | 6 h each |
| `31_A_full` | selection, top 3 x 5 seeds x 60 min per family, representative | timed x 2 | 18 h each |
| `32_A_cpu` | each representative on the CPU, 5 seeds x 60 min | timed-cpu x 2 | 7 h each |
| `33_A_float32` | F1 float32-Adam run | timed | 2 h |
| `40_b9_nil_darcy` | B9: NiL Darcy, the manuscript's network (3,555 parameters), float64 and float32, 4 fields x 3 seeds (LiL alongside) | shared-gpu x 12 | 3 h each |
| `41_b8_initial_guess` | B8: 128 runs, one task per (case, guess); networks on the A100 | shared-gpu x 4 | 6 h each |
| `42_component_c` | Component C: 196 LiL-Q runs | cpu | 6 h |
| `43_basis_study` | B6: the Burgers basis study | cpu | 3 h |
| `91_wave_report` | the wave's tables and report tarball | cpu | 45 min |
| `90_finalize` | package1 from the waves; reference/, code/, merged tables (B4, B8, B9), B1, B5 figures, Section 4.6 comparison | cpu | 1 h |

Walltimes are about 1.5 x the expected runtimes (the advisor's patch 0001):
HPRC books pending SUs from the requested walltimes at submission. Revise
them from wave 1's measured runtimes.

**SUs.** Charged for time used: cores x hours plus a GPU surcharge per GPU-hour
(Grace A100 72; FASTER A100 128). On Grace an exclusive A100 node is 48 core-SU
plus 72 or 144 per hour, depending on whether both of its A100s are charged:
120-192 SU per node-hour. An exclusive CPU node is 48 SU per hour; a shared
A100 job 80; a 24-core CPU job 24. Expected total: about 10,000-15,000 SU at
120 SU/h, about 17,000 if both GPUs of an exclusive node are charged. On
FASTER, A100s sit in 4-16-GPU nodes, so an exclusive node holds several A100s
at 128 SU each: the timed GPU jobs there would cost several times more.

## Resource use

Timed jobs hold the whole node and start one thread per core for BLAS,
OpenMP and PyTorch; `hardware.json` in every run directory records the node's
CPU model and GPUs, the thread counts in effect, and whether the job held the
node exclusively. Never run experiments or tests on a login node.
