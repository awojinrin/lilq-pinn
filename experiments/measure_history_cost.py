"""
Cost of the four-method loss history (the advisor's reply to wave 2, Section 3, item 3)
=======================================================================================

Does a four-method run's ``training_time`` include the cost of the loss
history logged every 10 iterations, and how large is it? The history
(``loss_history_every_10``) is subsampled after the run from the
per-iteration record of ``lilq.metrics.MetricsTracker``; that record is on
the clock. This times, inside real runs (2,000 iterations, target 0 so none
stops early):

* ``MetricsTracker.record`` -- the recording itself;
* the loop's ``LBFGSObjective.value()`` at the accepted point -- a stored
  value, no evaluation; it is also the stopping test, which the method needs
  without any history (NiL-Q's stopping test is its full-loss monitor
  evaluation, counted separately as ``monitor_evaluations``).

On this laptop (RTX 5080, 2026-10-01; DECISIONS.md): the recording is
0.01% of the training time on the GPU and 0.05-0.06% on the CPU; the
accepted-point lookup 0.44% and 0.7%.

Usage::

    python experiments/measure_history_cost.py
"""
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import lilq.solvers as solvers  # noqa: E402
from lilq.metrics import MetricsTracker  # noqa: E402

acc = {'record': 0.0, 'value': 0.0, 'n_record': 0, 'n_value': 0}
orig_record, orig_value = MetricsTracker.record, solvers.LBFGSObjective.value


def record(self, *a, **k):
    t = time.perf_counter()
    out = orig_record(self, *a, **k)
    acc['record'] += time.perf_counter() - t
    acc['n_record'] += 1
    return out


def value(self, with_grad=False):
    t = time.perf_counter()
    out = orig_value(self, with_grad)
    acc['value'] += time.perf_counter() - t
    acc['n_value'] += 1
    return out


def main():
    MetricsTracker.record, solvers.LBFGSObjective.value = record, value
    from experiments.run_bratu import paper_setup
    from experiments.run_burgers import paper_setup as burgers_setup
    from problems.bratu import run_nil_n, run_nil_q
    from problems.burgers import run_nil_n as burgers_nil_n
    cases = [('bratu P=100 NiL-N', paper_setup(10), run_nil_n), ('bratu P=100 NiL-Q', paper_setup(10), run_nil_q),
             ('burgers P=225 NiL-N', burgers_setup(15), burgers_nil_n)]
    devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    for device in devices:
        for name, (config, opt), runner in cases:
            opt = dataclasses.replace(opt, max_iterations=2000, max_quasi_iters_nn=8, max_inner_iters_nn=250,
                                      R_tol=0.0)
            config = dataclasses.replace(config, init_seed=0)
            for k in acc:
                acc[k] = 0 if k.startswith('n_') else 0.0
            s = runner(config, opt, device=torch.device(device), verbose=False)[-1]
            t = s['training_time']
            print(f"{device:4s} {name:22s} iterations {s['total_iterations']:5d}  training_time {t:7.2f} s  "
                  f"record {acc['record']:.3f} s ({100 * acc['record'] / t:.2f}%)  "
                  f"loop value() {acc['value']:.3f} s ({100 * acc['value'] / t:.2f}%)  "
                  f"[{acc['n_record']} records, {acc['n_value']} value calls]", flush=True)


if __name__ == '__main__':
    main()
