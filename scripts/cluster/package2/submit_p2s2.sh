#!/bin/bash
# Package 2, Stage 2 on Grace (the advisor's instructions of 4 October 2026; Stage 2
# released in his Stage-1 reply). Into results/package2_stage2, locked to one commit:
#   p2s2_preflight (versions, the GPU, the test suite)
#   -> p2s2_references (every reference, once)
#   -> every compute job, each waiting for the references:
#        item 1: classical Bratu, Kovasznay, Burgers (timed-cpu)
#        item 2: NiL-N by Levenberg-Marquardt, Bratu, Burgers, BL (timed-cpu)
#        item 3: certified grids (cpu)          item 4: ELM (cpu)
#        item 6: nu-refinement (timed-cpu), boundary-conforming bases (cpu)
#        item 7: scaling, CPU (two timed-cpu jobs) and A100 (timed)
#        item 8: manufactured elasticity (cpu)
#   -> p2s2_report (afterany): sacct, su_per_job.csv, results/package2_stage2.tar.gz.
# Not submitted here:
#   - item 2's contingent A100 reruns (p2s2_lm_networks_gpu.slurm): only if
#     `python experiments/p2_8_lm_networks.py gpu-list --out results/package2_stage2`
#     names a configuration once the three LM jobs are done (Section 5);
#   - item 5 (Darcy) runs on FASTER, from its own script.
# 17 jobs: 1,274 SU requested (walltime x rate, su_plan.csv), about 618 expected (charged by
# elapsed time); 1,730 requested if the contingent A100 job is submitted too.
# Run from a login node:
#   DRY_RUN=1 bash scripts/cluster/package2/submit_p2s2.sh     (checks every job; submits nothing)
#   bash scripts/cluster/package2/submit_p2s2.sh
LILQ_WAVE=p2s2
source "$(dirname "$0")/../submit_lib.sh"
P=$S/package2
confirm_balance "1,274 SU requested, about 618 expected (su_plan.csv; 17 jobs)"

submit pre  $P/p2s2_preflight.slurm
submit refs $P/p2s2_references.slurm             --dependency=afterok:$pre
after="--dependency=afterok:$refs"
submit c1   $P/p2s2_classical_bratu.slurm        $after
submit c2   $P/p2s2_classical_kovasznay.slurm    $after
submit c3   $P/p2s2_classical_burgers.slurm      $after
submit l1   $P/p2s2_lm_networks_bratu.slurm      $after
submit l2   $P/p2s2_lm_networks_burgers.slurm    $after
submit l3   $P/p2s2_lm_networks_bl.slurm         $after
submit cert $P/p2s2_certified.slurm              $after
submit elm  $P/p2s2_elm.slurm                    $after
submit nu   $P/p2s2_nu_refinement.slurm          $after
submit bas  $P/p2s2_bases.slurm                  $after
submit sa   $P/p2s2_scaling_cpu_a.slurm          $after
submit sb   $P/p2s2_scaling_cpu_b.slurm          $after
submit sg   $P/p2s2_scaling_gpu.slurm            $after
submit ela  $P/p2s2_elasticity_manufactured.slurm $after
submit rep  $P/p2s2_report.slurm                 --dependency=afterany:$c1:$c2:$c3:$l1:$l2:$l3:$cert:$elm:$nu:$bas:$sa:$sb:$sg:$ela
[[ "$DRY_RUN" == 1 ]] || echo "Stage 2 submitted. When the LM jobs ($l1 $l2 $l3) finish, check item 2's gpu-list;"
[[ "$DRY_RUN" == 1 ]] || echo "when job $rep finishes: results/package2_stage2.tar.gz to download."
