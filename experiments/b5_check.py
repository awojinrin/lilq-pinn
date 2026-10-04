"""
The evidence for the Appendix B.5 check (Package 2, Section 2.3)
================================================================

For each row of the manuscript's Table ``tab:run_settings`` (Appendix B.5;
``run_settings_B5.csv``), what the runs of ``package1`` actually used, read
from their own records (``run.json``, the four-method and B9/B10 tables, the
saved networks, the Component A configurations) and, where a run records
nothing, from the code at this commit. Writes ``b5_actual.csv``, one row per
table row with the same columns as the table; ``b5_check.md`` compares the two.

Usage::

    python experiments/b5_check.py --package <package1> --out <package2_results>/G1_release/b5_actual.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

COLUMNS = ('experiment', 'trial_space_or_network', 'P', 'N_over_P', 'collocation_set', 'aux_conditions', 'stop',
           'seeds', 'K_max', 'runs', 'source')


def _rng(values, fmt='{:g}'):
    values = sorted(set(values))
    if not values:
        return ''
    return fmt.format(values[0]) if len(values) == 1 else f"{fmt.format(values[0])}-{fmt.format(values[-1])}"


def _runs(B, pattern):
    def size(p):
        m = re.search(r'_P(\d+)', p)
        return (int(m.group(1)) if m else 0, p)
    paths = sorted(glob.glob(str(B / pattern)), key=size)
    return [json.loads((Path(p) / 'run.json').read_text()) for p in paths], [Path(p).name for p in paths]


def section33(B, experiment, pattern):
    recs, names = _runs(B, pattern)
    fam = sorted({json.dumps(r['basis_description'].get('family') or
                             {k: v.get('family') for k, v in r['basis_description'].items()}) for r in recs})
    aux = sorted({k for r in recs for k in r['N_composition'] if not re.match(r'(pde|.*momentum|continuity|darcy)', k)})
    stop = sorted({f"{r['stopping_rule']['type']}" + (f" {r['stopping_rule'].get('tolerance')}" if 'tolerance' in r['stopping_rule'] else '')
                   for r in recs})
    coll = sorted({r['collocation_construction'].get('method', '') for r in recs})
    seeds = sorted({str(r['collocation_construction'].get('seed')) for r in recs if r['collocation_construction'].get('seed') is not None})
    return {'experiment': experiment, 'trial_space_or_network': '; '.join(fam),
            'P': _rng(r['P_total'] for r in recs), 'N_over_P': _rng((round(r['N_total'] / r['P_total'], 2) for r in recs), '{:.2f}'),
            'collocation_set': '; '.join(coll) + (f" (collocation seed {', '.join(seeds)})" if seeds else ''),
            'aux_conditions': 'rows: ' + ', '.join(aux) if aux else 'none (in the basis)', 'stop': '; '.join(stop),
            'seeds': '-', 'K_max': _rng(r.get('K_max') for r in recs), 'runs': ' '.join(names), 'source': 'run.json'}


def four_method(B):
    with open(B / 'four_method_tables.csv', newline='') as f:
        rows = [r for r in csv.DictReader(f) if not r.get('variant')]
    from lilq.saved_models import load_network
    nets = sorted(glob.glob(str(B / 'four_method_jobs' / '*' / 'models' / '*NiL-N_s0_*')))
    params = {sum(p.numel() for p in load_network(n)['model'].parameters()) for n in nets if 'f1_stall' not in n}
    m = load_network(nets[0])['model']
    layers = [type(x).__name__ for x in m.network]
    return {'experiment': 'Four-method comparison (6.1)',
            'trial_space_or_network': f"NiL: {'-'.join(layers)} (two hidden layers); LiL-N: the LiL-Q basis",
            'P': f"LiL {_rng(int(r['P']) for r in rows)}; NiL parameters {min(params)}-{max(params)}",
            'N_over_P': 'as LiL-Q (same collocation sets)', 'collocation_set': 'as LiL-Q', 'aux_conditions': 'rows',
            'stop': ', '.join(sorted({r['stopping_reason'] for r in rows})),
            'seeds': ', '.join(sorted({r['seed'] for r in rows if r['seed']})) + ' (NiL)', 'K_max': '',
            'runs': f'{len(rows)} rows', 'source': 'four_method_tables.csv, saved networks'}


def basis_study(B):
    from experiments.run_burgers_basis_comparison import ComparisonConfig, generate_collocation
    with open(B / 'basis_study' / 'table3_basis_study.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    cfg = ComparisonConfig(N_x=25, N_t=25)
    pts = generate_collocation(cfg)
    N = pts['n_pde'] + pts['n_ic'] + 2 * pts['n_bc']
    n_dim = int(round(pts['n_pde'] ** 0.5))
    return {'experiment': 'Basis study (6.3)', 'trial_space_or_network': ', '.join(r['basis_key'] for r in rows),
            'P': _rng(int(r['P']) for r in rows), 'N_over_P': f'{N / 625:.2f}',
            'collocation_set': f'equispaced {n_dim} x {n_dim} interior', 'aux_conditions': 'rows: ic, bc_left, bc_right',
            'stop': f"{_rng(int(r['K_max']) for r in rows)} iterations (stopping rule off)", 'seeds': '-',
            'K_max': _rng(int(r['K_max']) for r in rows), 'runs': f'{len(rows)} bases',
            'source': 'table3_basis_study.csv; generate_collocation at this commit'}


def darcy_nil(B):
    import torch
    with open(B / 'darcy_fv_comparison.csv', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['method'] == 'NiL']
    nets = sorted(glob.glob(str(B / 'darcy_fv' / '*' / 'models' / 'NiL_*' / 'network.pt')))
    d = torch.load(nets[0], map_location='cpu', weights_only=False)
    params = sorted({r['n_params'] for r in rows})
    return {'experiment': 'Darcy NiL (6.8)',
            'trial_space_or_network': f"three networks, {d['networks']['num_layers']} hidden layers x {d['networks']['hidden_dim']}",
            'P': ', '.join(params), 'N_over_P': f"{39600 / int(params[0]):.2f}", 'collocation_set': 'cell centers (60 x 220)',
            'aux_conditions': 'approximate (penalty rows)',
            'stop': f"{', '.join(sorted({str(torch.load(n, map_location='cpu', weights_only=False)['max_epochs']) for n in nets}))} epochs Adam",
            'seeds': ', '.join(sorted({r['seed'] for r in rows})) + f" (dtypes {', '.join(sorted({r['dtype'] for r in rows}))})",
            'K_max': '', 'runs': f'{len(rows)} NiL rows', 'source': 'darcy_fv_comparison.csv, network.pt'}


def oversampling(C):
    runs = sorted(Path(C, 'runs').iterdir())
    P = sorted({int(re.search(r'_P(\d+)', r.name).group(1)) for r in runs})
    ratios = sorted({float(r.name.split('_r')[1].split('_')[0]) for r in runs})
    dists = sorted({r.name.split('_')[3] for r in runs})
    seeds = sorted({r.name.split('_s')[-1] for r in runs if '_random_' in r.name})
    return {'experiment': 'Oversampling (6.10)', 'trial_space_or_network': 'as Bratu and Kovasznay',
            'P': ', '.join(map(str, P)), 'N_over_P': f'{ratios[0]:g}-{ratios[-1]:g} ({", ".join(f"{x:g}" for x in ratios)})',
            'collocation_set': ', '.join(dists), 'aux_conditions': 'rows', 'stop': 'as the paper passes',
            'seeds': f"random: {len(seeds)} draws ({', '.join(seeds)})", 'K_max': '', 'runs': f'{len(runs)} runs',
            'source': 'C_oversampling/runs'}


def b10(B):
    with open(B / 'b10' / 'b10.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    return {'experiment': 'CGL with CC weights (6.10)', 'trial_space_or_network': 'as Kovasznay',
            'P': _rng(int(r['P']) for r in rows), 'N_over_P': ', '.join(sorted({r['ratio_target'] for r in rows}, key=float)),
            'collocation_set': 'CGL; weights ' + ', '.join(sorted({r['weights'] for r in rows})),
            'aux_conditions': 'rows; one pressure pin', 'stop': 'paper pass: rel. coefficient change 1e-9; kmax pass: 60 iterations',
            'seeds': '-', 'K_max': _rng(int(r['K_max']) for r in rows), 'runs': f'{len(rows)} runs', 'source': 'b10.csv'}


def component_a(A):
    out = []
    for fam, label in (('F1', 'F1 (6.6)'), ('F2', 'F2 (6.6)')):
        rep = json.loads((A / 'full' / f'{fam}_representative.json').read_text())
        cfg = rep['representative_config']
        runs = sorted(A.glob(f"full/{rep['representative']}_s*/run.json"))
        r0 = json.loads(runs[0].read_text())
        coll = f"{cfg['n_int']} random" + (', redrawn' if cfg.get('resample') else '')
        nop = '' if fam == 'F1' else f"{3 * cfg['n_int'] / cfg['n_theta']:.2f} (3 equations x {cfg['n_int']} points)"
        out.append({'experiment': label, 'trial_space_or_network': f"Fourier-feature MLP {rep['representative']}",
                    'P': cfg['n_theta'], 'N_over_P': nop, 'collocation_set': coll,
                    'aux_conditions': cfg.get('bc', 'exact (Coons lifting)') + (' (penalty)' if cfg.get('bc') == 'soft' else ''),
                    'stop': f"{r0['budget_s'] / 60:g} min", 'seeds': _rng(json.loads(p.read_text())['seed'] for p in runs),
                    'K_max': '', 'runs': ' '.join(p.parent.name for p in runs), 'source': 'run.json, representative'})
    return out


def collect(package):
    package = Path(package)
    B, A, C = package / 'B_instrumentation', package / 'A_calibration', package / 'C_oversampling'
    rows = [section33(B, 'Bratu (6.2)', 'bratu_P*_cpu_paper'),
            section33(B, 'Burgers (6.3)', 'burgers_P*_cpu_paper'),
            section33(B, 'Buckley-Leverett Ng=0 (6.4)', 'bl_P*_cpu_paper'),
            section33(B, 'Buckley-Leverett Ng=-5 (6.4)', 'bl_gravity_P*_cpu_paper'),
            four_method(B), basis_study(B),
            section33(B, 'Elasticity (6.5)', 'elasticity_P*_cpu_paper'),
            section33(B, 'Kovasznay (6.6)', 'kovasznay_P*_cpu_paper')]
    belt = json.loads((B / 'beltrami_pinned' / 'run.json').read_text())
    rep = json.loads((B / 'beltrami_pinned' / 'report.json').read_text())
    rows.append({'experiment': 'Beltrami (6.7)', 'trial_space_or_network':
                 f"chebyshev 4D ({rep['N_vel']}^4 per velocity; {rep['N_p']}^4 pressure)", 'P': belt['P_total'],
                 'N_over_P': f"{belt['N_total'] / belt['P_total']:.2f}",
                 'collocation_set': 'equispaced {N_x}^4 interior'.format(**belt['collocation_construction']),
                 'aux_conditions': f"rows; {belt['N_composition']['pressure_pin']} pressure pins",
                 'stop': f"{belt['stopping_rule']['type']} {belt['stopping_rule'].get('tolerance')}", 'seeds': '-',
                 'K_max': belt.get('K_max'), 'runs': 'beltrami_pinned', 'source': 'run.json, report.json'})
    rows.append(section33(B, 'Darcy (6.8)', 'darcy_*_cpu_paper'))
    rows += [darcy_nil(B), oversampling(C), b10(B)] + component_a(A)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="The evidence for the Appendix B.5 check.")
    ap.add_argument('--package', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rows = collect(args.package)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['experiment']}: {r['trial_space_or_network']} | P {r['P']} | N/P {r['N_over_P']} | "
              f"{r['collocation_set']} | {r['aux_conditions']} | {r['stop']} | seeds {r['seeds']} | K_max {r['K_max']}")


if __name__ == '__main__':
    main()
