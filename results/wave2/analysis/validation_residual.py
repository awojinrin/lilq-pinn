"""Validation residual of every Component A model in wave 2: the momentum and
continuity mean squares (plus the soft-BC mean square, F1 soft only) on
20,000 fresh uniform interior points and 400 points per boundary face --
F1's unweighted loss, on points no run trained on. No exact solution is
used except the boundary data, which is problem data. Compared with the
recorded training loss and test errors."""
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(r'C:\Users\awoji\Documents\LiL-Q\Post-JCP\lilq-pinn')
sys.path.insert(0, str(REPO))
import baselines.f1_pinn as f1  # noqa: E402
from baselines.lm_kovasznay import coons, ell, g_u, g_v  # noqa: E402
from lilq.saved_models import load_f1, load_f2  # noqa: E402

A = Path(sys.argv[1])
OUT = Path(sys.argv[2])
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen = torch.Generator().manual_seed(20261001)          # independent of every training seed
XY_INT = f1.interior_points(20000, gen, torch.float64, dev)
XY_BC = f1.boundary_points(400, torch.float64, dev)


def val_terms(fields_fn, soft_bc):
    sums = {'xmom': 0.0, 'ymom': 0.0, 'cont': 0.0}
    for chunk in XY_INT.split(2500):
        r1, r2, r3 = f1.ns_residuals(fields_fn, chunk)
        for k, r in zip(sums, (r1, r2, r3)):
            sums[k] += float((r.detach() ** 2).sum())
    terms = {k: v / len(XY_INT) for k, v in sums.items()}
    if soft_bc:
        with torch.no_grad():
            out = fields_fn(XY_BC)
            ue, ve, _ = f1.exact(XY_BC[:, 0], XY_BC[:, 1])
            terms['bc'] = float(((out[:, 0] - ue) ** 2).mean() + ((out[:, 1] - ve) ** 2).mean())
    return terms


def f2_fields(model):
    def fn(xy):
        out = model(xy)
        x, y = xy[:, 0], xy[:, 1]
        L = ell(x, y)
        return torch.stack([coons(g_u, x, y) + L * out[:, 0], coons(g_v, x, y) + L * out[:, 1], out[:, 2]], 1)
    return fn


rows = []
for run_json in sorted(A.glob('*/F[12]_*/run.json')):
    d = run_json.parent
    stage, name = d.parent.name, d.name
    run = json.loads(run_json.read_text())
    if name.startswith('F1'):
        model, _ = load_f1(d, device=dev)
        model = model.double()
        soft = run['config']['bc'] == 'soft'
        terms = val_terms(model, soft)
        train = run.get('final_loss_unweighted')
    else:
        model, _ = load_f2(d, device=dev)
        terms = val_terms(f2_fields(model), False)
        train = run.get('final_loss')
    rows.append({'stage': stage, 'run': name, 'config': name.rsplit('_s', 1)[0], 'family': name[:2],
                 'train_loss': train, 'val_residual': sum(terms.values()), **{f'val_{k}': v for k, v in terms.items()},
                 'eps_u': run.get('eps_u'), 'eps_v': run.get('eps_v'), 'eps_p_meanfree': run.get('eps_p_meanfree'),
                 'end_reason': run.get('end_reason')})
    print(f"{stage:10s} {name:10s} train {train:.2e}  val {rows[-1]['val_residual']:.2e}  eps_u {run.get('eps_u'):.2e}",
          flush=True)

keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT} ({len(rows)} runs, device {dev})")
