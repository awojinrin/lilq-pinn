"""
Field data for the manuscript's solution-field figures, from the saved models
=============================================================================

The advisor's reply to wave 4, Section 3: the solution-field figures are
still images from the superseded code. This evaluates the **saved models of
the regenerated runs** in a package (nothing is retrained or re-solved) on
each figure's grid and writes one ``.npz`` per figure into ``--out``:

=====================  ===================================================  =============================
file                   runs                                                 grid
=====================  ===================================================  =============================
``bratu.npz``          LiL-Q, LiL-N, NiL-N, NiL-Q (seed 0) at P = 25, 225     201 x 201 on [0,1]^2
``burgers.npz``        the four formulations at P = 625                     201 x 201 in (x, t)
``buckley_leverett.npz`` LiL-Q and NiL-N (seed 0) at P = 1,024, viscous and   201 x 201 in (x, t)
                       gravity, with the finite-difference reference
``kovasznay.npz``      LiL-Q at P = 1,875 (u, v, p), with the exact solution  301 x 401, the evaluation grid
``beltrami.npz``       the pinned LiL-Q run at t = 1 on the six faces of      41 x 41 per face
                       [-1,1]^3 (u, v, w, p), with the exact fields
``darcy.npz``          finite-volume, LiL and NiL (B9, seed 0, float64 and    the finite-volume cell centres
                       float32) pressures, S1, S2, S3, SPE10
``elasticity.npz``     LiL at P = 50 (u_x, u_y), with the exact solution     200 x 200
=====================  ===================================================  =============================

LiL-Q runs are the Section 3.3 CPU paper passes; LiL-N and NiL the CPU
four-method runs (``four_method_jobs/<benchmark>_cpu/models/``). Where a CPU
model does not exist, the GPU run's is used and labelled so (Bratu at P =
25: the CPU four-method job ran P = 225 only). Every file holds the
coordinate arrays, each method's field (array ``[i, j]`` over the first and
second coordinate, ``indexing='ij'``), the exact or reference field where
there is one, and ``meta``: a JSON string with, per method, the run folder
(relative to the package), the device and the commit that produced it (from
the nearest ``hardware.json``), plus the checks below. ``manifest.json``
lists everything, and the models that are missing.

**Evaluation.** A LiL field on a grid is evaluated as the tensor product
(``lilq.test_errors.tensor_grid_values``), the same code path the runs'
logged test errors used; ``meta['evaluation_roundoff']`` gives, per field,
the largest difference from the pointwise evaluation (``basis.evaluate`` at
every point). They differ only by round-off, which is large where the
coefficients are: the viscous Buckley-Leverett LiL-Q solution at P = 1,024
has coefficients up to 2.5e9, and its two evaluations differ by up to
1.7e-5, so its error reproduces to about three digits.

**Checks** (in ``meta`` and the manifest): where the run logged an error,
the same error is recomputed from the evaluated fields, which confirms the
evaluation -- Buckley-Leverett's and Kovasznay's LiL-Q test errors
(``summary.json``), Darcy's delta_FV of every LiL and NiL model (B9's table),
elasticity's test error; and Darcy's LiL pressure re-evaluated from its
basis against the pressure the run stored.

Usage::

    python experiments/figure_data.py --package <package1> --out results/figure_data
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np
import torch

from lilq.saved_models import evaluate_field, load_network, load_solution
from lilq.source_lock import current_commit
from lilq.test_errors import tensor_grid_values


class Context:
    def __init__(self, package):
        self.package = Path(package)
        self.B = self.package / 'B_instrumentation'
        self.missing = []
        self.roundoff = {}

    def field(self, label, sol, name, axes):
        """``name`` of a saved LiL solution on the tensor grid ``axes``
        (tensor-product evaluation, as the logged test errors), recording
        the largest difference from the pointwise evaluation under
        ``label``."""
        entry = sol['fields'][name]
        values = tensor_grid_values(entry['basis'], entry['coefficients'], axes)
        mesh = np.meshgrid(*axes, indexing='ij')
        self.roundoff[label] = float(np.abs(values - evaluate_field(sol, name, *mesh)).max())
        return values

    def rel(self, path):
        return Path(path).relative_to(self.package).as_posix()

    def commit(self, run_dir):
        """The commit that produced a run: the nearest hardware.json's git
        record, then summary.json's or run.json's ``commit``."""
        d = Path(run_dir)
        while True:
            hw = d / 'hardware.json'
            if hw.exists():
                c = (json.loads(hw.read_text()).get('git') or {}).get('commit')
                if c:
                    return c
            for name in ('summary.json', 'run.json'):
                if (d / name).exists():
                    c = json.loads((d / name).read_text()).get('commit')
                    if c:
                        return c
            if d == self.package or d.parent == d:
                return None
            d = d.parent

    def source(self, run_dir, device, note=None):
        s = {'run': self.rel(run_dir), 'device': device, 'commit': self.commit(run_dir)}
        if note:
            s['note'] = note
        return s

    def four_method(self, bench, P, method, seed):
        """A CPU four-method model, or the GPU one (labelled) if there is none."""
        tag = 'sna' if method.startswith('LiL') else f's{seed}'
        cpu = self.B / 'four_method_jobs' / f'{bench}_cpu' / 'models' / f'{bench}_P{P}_{method}_{tag}_cpu'
        gpu = self.B / 'four_method_jobs' / f'{bench}_gpu' / 'models' / f'{bench}_P{P}_{method}_{tag}_cuda'
        if cpu.exists():
            return cpu, 'cpu', None
        if gpu.exists():
            self.missing.append({'figure': bench, 'P': P, 'method': method, 'seed': seed,
                                 'missing': self.rel(cpu), 'used_instead': self.rel(gpu)})
            return gpu, 'cuda', f'no CPU model ({self.rel(cpu)}): the GPU run of the same method and seed'
        self.missing.append({'figure': bench, 'P': P, 'method': method, 'seed': seed,
                             'missing': self.rel(cpu), 'used_instead': None})
        return None, None, None


def _nn(model, a, b):
    dtype = next(model.parameters()).dtype
    pts = torch.tensor(np.stack([np.ravel(a), np.ravel(b)], axis=1), dtype=dtype)
    with torch.no_grad():
        return model(pts).reshape(np.shape(a)).double().numpy()


def _rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def _save(out, name, arrays, meta, ctx=None):
    meta = dict(meta, written_by_commit=current_commit())
    if ctx is not None and ctx.roundoff:
        meta['evaluation_roundoff'] = dict(ctx.roundoff)
        ctx.roundoff.clear()
    np.savez_compressed(out / name, meta=np.array(json.dumps(meta)), **arrays)
    return {'file': name, 'arrays': {k: list(np.shape(v)) for k, v in arrays.items()}, **meta}


def _scalar_figure(ctx, bench, sizes, methods, n=201, second='y'):
    """Bratu / Burgers: the four formulations on a 201 x 201 grid."""
    arrays, sources = {}, {}
    for P in sizes:
        lilq_dir = ctx.B / f'{bench}_P{P}_cpu_paper'
        sol = load_solution(lilq_dir)
        cfg = sol['config']
        a = np.linspace(*cfg.x_domain, n)
        b = np.linspace(*(cfg.y_domain if second == 'y' else (0.0, cfg.T_final)), n)
        A, Bm = np.meshgrid(a, b, indexing='ij')
        arrays['x'], arrays[second] = a, b
        arrays[f'P{P}_LiL-Q'] = ctx.field(f'{bench}_P{P}_LiL-Q', sol, 'u', [a, b])
        sources[f'P{P}_LiL-Q'] = ctx.source(lilq_dir, 'cpu')
        for method in methods:
            d, dev, note = ctx.four_method(bench, P, method, 0)
            if d is None:
                continue
            key = f'P{P}_{method}'
            if method.startswith('LiL'):
                arrays[key] = ctx.field(f'{bench}_{key}', load_solution(d), 'u', [a, b])
            else:
                arrays[key] = _nn(load_network(d)['model'], A, Bm)
            sources[key] = ctx.source(d, dev, note)
    return arrays, sources


def bratu(ctx, out):
    arrays, sources = _scalar_figure(ctx, 'bratu', (25, 225), ('LiL-N', 'NiL-N', 'NiL-Q'))
    return _save(out, 'bratu.npz', arrays, ctx=ctx, meta={'figure': 'Bratu solution fields', 'grid': '201 x 201 on [0,1]^2',
                                            'exact': 'none (no closed form)', 'sources': sources})


def burgers(ctx, out):
    arrays, sources = _scalar_figure(ctx, 'burgers', (625,), ('LiL-N', 'NiL-N', 'NiL-Q'), second='t')
    return _save(out, 'burgers.npz', arrays, ctx=ctx, meta={'figure': 'Burgers solution fields', 'grid': '201 x 201 in (x, t)',
                                              'exact': 'none', 'sources': sources})


def buckley_leverett(ctx, out, P=1024):
    from problems.buckley_leverett import TEST_GRID, reference_solution
    arrays, sources, checks = {}, {}, {}
    for case in ('bl', 'bl_gravity'):
        lilq_dir = ctx.B / f'{case}_P{P}_cpu_paper'
        sol = load_solution(lilq_dir)
        cfg = sol['config']
        x = np.linspace(*cfg.x_domain, TEST_GRID[0])
        t = np.linspace(0.0, cfg.T_final, TEST_GRID[1])
        X, T = np.meshgrid(x, t, indexing='ij')
        ref = reference_solution(cfg)
        arrays['x'], arrays['t'] = x, t
        arrays[f'{case}_reference'] = ref
        arrays[f'{case}_LiL-Q'] = ctx.field(f'{case}_LiL-Q', sol, 'u', [x, t])
        sources[f'{case}_LiL-Q'] = ctx.source(lilq_dir, 'cpu')
        logged = json.loads((lilq_dir / 'summary.json').read_text()).get('test_eps_u')
        checks[f'{case}_LiL-Q_eps_u'] = {'recomputed': _rel_l2(arrays[f'{case}_LiL-Q'], ref), 'logged': logged}
        d, dev, note = ctx.four_method(case, P, 'NiL-N', 0)
        if d is not None:
            arrays[f'{case}_NiL-N'] = _nn(load_network(d)['model'], X, T)
            sources[f'{case}_NiL-N'] = ctx.source(d, dev, note)
            checks[f'{case}_NiL-N_eps_u'] = {'recomputed': _rel_l2(arrays[f'{case}_NiL-N'], ref)}
    return _save(out, 'buckley_leverett.npz', arrays, ctx=ctx, meta={
        'figure': 'Buckley-Leverett comparison, viscous (bl) and gravity (bl_gravity)', 'P': P, 'N': 32,
        'grid': '201 x 201 in (x, t)',
        'reference': 'finite-difference method-of-lines solve, problems.buckley_leverett.reference_solution',
        'sources': sources, 'checks': checks})


def kovasznay(ctx, out, P=1875):
    from problems.kovasznay import TEST_GRID, KovasznayPhysics
    run = ctx.B / f'kovasznay_P{P}_cpu_paper'
    sol = load_solution(run)
    phys = KovasznayPhysics(sol['config'])
    x = np.linspace(*phys.x_domain, TEST_GRID[0])
    y = np.linspace(*phys.y_domain, TEST_GRID[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    arrays = {'x': x, 'y': y}
    for f in 'uvp':
        arrays[f'LiL-Q_{f}'] = ctx.field(f'kovasznay_LiL-Q_{f}', sol, f, [x, y])
        arrays[f'exact_{f}'] = getattr(phys, f'exact_{f}')(X, Y)
    s = json.loads((run / 'summary.json').read_text())
    checks = {f'eps_{f}': {'recomputed': _rel_l2(arrays[f'LiL-Q_{f}'], arrays[f'exact_{f}']),
                           'logged': s.get(f'test_eps_{f}')} for f in 'uv'}
    pm, em = arrays['LiL-Q_p'] - arrays['LiL-Q_p'].mean(), arrays['exact_p'] - arrays['exact_p'].mean()
    checks['eps_p_meanfree'] = {'recomputed': _rel_l2(pm, em), 'logged': s.get('test_eps_p_meanfree')}
    return _save(out, 'kovasznay.npz', arrays, ctx=ctx, meta={
        'figure': 'Kovasznay fields and errors', 'P': P, 'grid': '301 x 401 on [-0.5,1] x [-0.5,1.5]',
        'pressure': 'LiL-Q_p in the run\'s pin gauge; the mean-free error is in checks',
        'sources': {'LiL-Q': ctx.source(run, 'cpu')}, 'checks': checks})


FACES = (('x', -1.0), ('x', 1.0), ('y', -1.0), ('y', 1.0), ('z', -1.0), ('z', 1.0))


def beltrami(ctx, out, n=41, t=1.0):
    from problems.beltrami import BeltramiPhysics
    run = ctx.B / 'beltrami_pinned'
    sol = load_solution(run)
    phys = BeltramiPhysics(sol['config'])
    s = np.linspace(-1.0, 1.0, n)
    S1, S2 = np.meshgrid(s, s, indexing='ij')
    arrays = {'s': s}
    for axis, value in FACES:
        fixed = np.full_like(S1, value)
        coords = {'x': (fixed, S1, S2), 'y': (S1, fixed, S2), 'z': (S1, S2, fixed)}[axis]
        T = np.full_like(S1, t)
        face = f'{axis}{"m" if value < 0 else "p"}1'
        one = np.array([value])
        axes = {'x': [one, s, s], 'y': [s, one, s], 'z': [s, s, one]}[axis] + [np.array([t])]
        for f in 'uvwp':
            arrays[f'{face}_LiL-Q_{f}'] = ctx.field(f'beltrami_{face}_{f}', sol, f, axes).reshape(n, n)
            arrays[f'{face}_exact_{f}'] = getattr(phys, f'exact_{f}')(*coords, T)
    return _save(out, 'beltrami.npz', arrays, ctx=ctx, meta={
        'figure': 'Beltrami', 'run': 'the pinned LiL-Q run (Section 3.7)', 't': t, 'grid': f'{n} x {n} per face',
        'faces': {'xm1': 'x = -1, (y, z) = (s, s)', 'xp1': 'x = +1, (y, z)', 'ym1': 'y = -1, (x, z)',
                  'yp1': 'y = +1, (x, z)', 'zm1': 'z = -1, (x, y)', 'zp1': 'z = +1, (x, y)'},
        'pressure': 'LiL-Q_p in the run\'s pinned gauge',
        'sources': {'LiL-Q': ctx.source(run, 'cpu')}})


DARCY_FIELDS = ('S1', 'S2', 'S3', 'SPE10')


def darcy(ctx, out):
    from problems.darcy import DarcyPhysics, delta_fv, load_darcy_pinn
    from experiments.darcy_fv_comparison import _config
    arrays, sources, checks = {}, {}, {}
    for field in DARCY_FIELDS:
        job = ctx.B / 'darcy_fv' / f'{field}_s0'
        lil_dir = job / 'models' / f'LiL_{field}'
        sol = load_solution(lil_dir)
        cfg = sol['config']
        phys = DarcyPhysics(cfg, verbose=False)
        x_c = (np.arange(cfg.NX_CELLS) + 0.5) * (phys.LX / cfg.NX_CELLS)
        y_c = (np.arange(cfg.NY_CELLS) + 0.5) * (phys.LY / cfg.NY_CELLS)
        X, Y = np.meshgrid(x_c, y_c, indexing='ij')
        # LiL: h = y_nd + h_tilde(x_nd, y_nd), P = h * DELTA_P + P_BOTTOM (problems.darcy.solve_lilq_darcy)
        xn = (np.arange(cfg.NX_CELLS) + 0.5) / cfg.NX_CELLS
        yn = (np.arange(cfg.NY_CELLS) + 0.5) / cfg.NY_CELLS
        Xn, Yn = np.meshgrid(xn, yn, indexing='ij')
        P_lil = (Yn + ctx.field(f'darcy_{field}_h_tilde', sol, 'h_tilde', [xn, yn])) * phys.DELTA_P + cfg.P_BOTTOM
        P_fvm = np.asarray(sol['extra']['P_fvm'])
        arrays[f'{field}_x'], arrays[f'{field}_y'] = x_c, y_c
        arrays[f'{field}_FV'], arrays[f'{field}_LiL'] = P_fvm, P_lil
        sources[f'{field}_LiL'] = ctx.source(lil_dir, 'cpu')
        sources[f'{field}_FV'] = {'from': f"{ctx.rel(lil_dir)}/solution.pt, extra['P_fvm']",
                                  'commit': ctx.commit(lil_dir)}
        with open(job / 'darcy_fv_comparison.csv', newline='') as f:
            table = list(csv.DictReader(f))
        logged = {(r['method'], r['seed'], r['dtype']): float(r['delta_fv']) for r in table}
        checks[f'{field}_LiL'] = {'P_lil_vs_stored_max_abs': float(np.abs(P_lil - sol['extra']['P_lil']).max()),
                                  'delta_fv_recomputed': delta_fv(P_lil, P_fvm, cfg.P_BOTTOM),
                                  'delta_fv_logged': logged.get(('LiL', '', 'float64'))}
        for dtype in ('float64', 'float32'):
            nil_dir = job / 'models' / f'NiL_{field}_s0_{dtype}'
            if not (nil_dir / 'network.pt').exists():
                ctx.missing.append({'figure': 'darcy', 'field': field, 'method': 'NiL', 'seed': 0,
                                    'dtype': dtype, 'missing': ctx.rel(nil_dir), 'used_instead': None})
                continue
            pinn, _ = load_darcy_pinn(nil_dir, config=cfg, physics=phys)
            P_nil = np.asarray(pinn.predict(X, Y)['P'], dtype=np.float64)
            arrays[f'{field}_NiL_{dtype}'] = P_nil
            sources[f'{field}_NiL_{dtype}'] = ctx.source(nil_dir, 'cuda (B9, shared A100)')
            checks[f'{field}_NiL_{dtype}'] = {'delta_fv_recomputed': delta_fv(P_nil, P_fvm, cfg.P_BOTTOM),
                                              'delta_fv_logged': logged.get(('NiL', '0', dtype))}
    return _save(out, 'darcy.npz', arrays, ctx=ctx, meta={
        'figure': 'Darcy pressure fields', 'grid': 'the finite-volume cell centres (physical coordinates)',
        'units': 'pressure as in B9 (psi)', 'sources': sources, 'checks': checks})


def elasticity(ctx, out, P=50):
    from problems.elasticity import TEST_GRID, ElasticityPhysics
    run = ctx.B / f'elasticity_P{P}_cpu_paper'
    sol = load_solution(run)
    cfg = sol['config']
    phys = ElasticityPhysics(cfg)
    x = np.linspace(*cfg.x_domain, TEST_GRID[0])
    y = np.linspace(*cfg.y_domain, TEST_GRID[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    arrays = {'x': x, 'y': y, 'LiL_ux': ctx.field('elasticity_ux', sol, 'u', [x, y]),
              'LiL_uy': ctx.field('elasticity_uy', sol, 'v', [x, y]),
              'exact_ux': phys.exact_ux(X, Y), 'exact_uy': phys.exact_uy(X, Y)}
    s = json.loads((run / 'summary.json').read_text())
    checks = {'eps_ux': {'recomputed': _rel_l2(arrays['LiL_ux'], arrays['exact_ux']), 'logged': s.get('test_eps_u')},
              'eps_uy': {'recomputed': _rel_l2(arrays['LiL_uy'], arrays['exact_uy']), 'logged': s.get('test_eps_v')}}
    return _save(out, 'elasticity.npz', arrays, ctx=ctx, meta={
        'figure': 'Elasticity fields and errors (the superseded image: elasticity_fields_and_errors_N5.png)',
        'P': P, 'grid': '200 x 200 on [0,1]^2', 'sources': {'LiL': ctx.source(run, 'cpu')}, 'checks': checks})


FIGURES = {'bratu': bratu, 'burgers': burgers, 'buckley_leverett': buckley_leverett, 'kovasznay': kovasznay,
           'beltrami': beltrami, 'darcy': darcy, 'elasticity': elasticity}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Field data for the manuscript's solution-field figures.")
    ap.add_argument('--package', required=True, help='package1 (with the saved models)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--figures', nargs='+', choices=list(FIGURES), default=list(FIGURES))
    args = ap.parse_args(argv)
    ctx = Context(args.package)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'package': str(ctx.package), 'files': []}
    for name in args.figures:
        entry = FIGURES[name](ctx, out)
        manifest['files'].append(entry)
        print(f"Wrote {entry['file']}: {len(entry['arrays'])} arrays")
        for k, c in (entry.get('checks') or {}).items():
            print(f"  check {k}: {c}")
    manifest['missing_models'] = ctx.missing
    manifest['written_by_commit'] = current_commit()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Missing models: {len(ctx.missing)}" + ''.join(f"\n  {m}" for m in ctx.missing))


if __name__ == '__main__':
    main()
