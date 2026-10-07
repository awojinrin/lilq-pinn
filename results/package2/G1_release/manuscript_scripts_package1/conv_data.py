"""Data loading for the training-loss convergence figures (four formulations).

Gradient-based runs (NiL-N, NiL-Q, LiL-N): primary rows of four_method_tables.csv
(wave 1 for Bratu, wave 2 for Burgers / BL / gravity BL), GPU (device == 'cuda'),
full per-iteration loss from four_method_jobs/<group>/models/<run>/history.csv
(the run directory without the '_f1_stall_rule' suffix; those are the controls).
Each history is checked against the table's total_iterations, final_loss and
loss_history_every_10.

LiL-Q: iterations.csv of the CPU 'paper' pass, loss = norm_R_h**2 (checked against
summary.json final_loss); gravity BL P = 64 uses the wave-1 K_max pass up to the
first k with norm_R_h**2 < target (k = 43).
"""
import ast
import json
import os

import numpy as np
import pandas as pd

W1 = '<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/B_instrumentation'
W2 = '<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/B_instrumentation'

PS = {
    'bratu': [25, 100, 225],
    'burgers': [25, 100, 225, 400, 625],
    'bl': [64, 256, 576, 1024],
    'bl_gravity': [64, 256, 576, 1024],
}
METHODS = ['NiL-N', 'NiL-Q', 'LiL-N', 'LiL-Q']


def table(bench):
    f = f'{W1}/four_method_tables.csv' if bench == 'bratu' else f'{W2}/four_method_tables.csv'
    d = pd.read_csv(f)
    return d[(d.benchmark == bench) & (d.device == 'cuda')].copy()


def _hist_path(bench, P, method, seed):
    s = 'sna' if method == 'LiL-N' else f's{int(seed)}'
    run = f'{bench}_P{P}_{method}_{s}_cuda'
    root = f'{W1}/four_method_jobs' if bench == 'bratu' else f'{W2}/four_method_jobs'
    hits = [f'{root}/{g}/models/{run}/history.csv' for g in sorted(os.listdir(root))
            if os.path.exists(f'{root}/{g}/models/{run}/history.csv')]
    if len(hits) != 1:
        raise RuntimeError(f'{run}: {hits}')
    return hits[0]


def target(bench, P):
    s = json.load(open(f'{W2}/{bench}_P{P}_cpu_paper/summary.json'))
    return float(s['R_tol'])


def gradient_run(bench, P, method, seed=0):
    """Return dict(it, loss, reason, n_iter, final_loss, row) for one GPU run."""
    d = table(bench)
    sel = (d.P == P) & (d.method == method)
    if method != 'LiL-N':
        sel &= (d.seed == seed)
    row = d[sel]
    assert len(row) == 1, (bench, P, method, seed, len(row))
    row = row.iloc[0]
    if 'variant' in row and isinstance(row['variant'], str):
        raise RuntimeError('control row selected')
    h = pd.read_csv(_hist_path(bench, P, method, seed))
    it = h.iteration.values.astype(int)
    loss = h.loss.values.astype(float)
    # checks against the table
    assert it[-1] == int(row.total_iterations), (bench, P, method, it[-1], row.total_iterations)
    assert np.isclose(loss[-1], row.final_loss, rtol=1e-10, atol=0), (loss[-1], row.final_loss)
    sub = ast.literal_eval(row.loss_history_every_10)
    lk = dict(zip(it, loss))
    for e in sub:
        assert np.isclose(lk[int(e[0])], e[1], rtol=1e-10, atol=0), (bench, P, method, e)
    tgt = target(bench, P)
    if row.stopping_reason == 'target':
        assert loss[-1] <= tgt and loss[:-1].min() > tgt
    else:
        assert loss.min() > tgt
    return dict(it=it, loss=loss, reason=row.stopping_reason, n_iter=int(it[-1]),
                final_loss=float(loss[-1]), cap=int(row.iterations_cap), row=row)


def all_seeds(bench, P, method):
    d = table(bench)
    r = d[(d.P == P) & (d.method == method)]
    return r[['seed', 'total_iterations', 'final_loss', 'stopping_reason']].reset_index(drop=True)


def lilq_run(bench, P):
    """Return dict(k, loss, n_iter, final_loss, reason, source)."""
    tgt = target(bench, P)
    src = f'{W2}/{bench}_P{P}_cpu_paper'
    s = json.load(open(f'{src}/summary.json'))
    df = pd.read_csv(f'{src}/iterations.csv')
    loss = df.norm_R_h.values ** 2
    assert np.isclose(loss[-1], s['final_loss'], rtol=1e-12)
    assert int(df.k.values[-1]) == int(s['iterations'])
    if bench == 'bl_gravity' and P == 64 and s['converged']:
        # package1 (release v2.0.0): the paper pass is the wave-4 K_max = 60 rerun,
        # which reaches the target at k = 43 (D4); used as is.
        assert int(s['iterations']) == 43 and loss[-1] < tgt and (loss[:-1] >= tgt).all()
        reason = 'target'
    elif bench == 'bl_gravity' and P == 64:
        # wave-2 layout: the paper pass is the K_max = 20 file; splice the wave-1 K_max = 60 pass
        assert not s['converged'] and int(s['iterations']) == int(s['K_max']) == 20
        paper = loss
        src = f'{W1}/{bench}_P{P}_cpu_kmax'
        df = pd.read_csv(f'{src}/iterations.csv')
        L = df.norm_R_h.values ** 2
        kstop = int(np.argmax(L < tgt))
        assert L[kstop] < tgt and kstop == 43 and (L[:kstop] >= tgt).all()
        # the K_max pass retraces the paper pass over its 20 iterations
        assert np.allclose(L[:len(paper)], paper, rtol=1e-8)
        df = df[df.k <= kstop]
        loss = df.norm_R_h.values ** 2
        reason = 'target'
    else:
        assert s['converged'] and loss[-1] < tgt and (loss[:-1] >= tgt).all()
        reason = 'target'
    return dict(k=df.k.values.astype(int), loss=loss, n_iter=int(df.k.values[-1]),
                final_loss=float(loss[-1]), reason=reason, source=src,
                paper_iters=int(s['iterations']), paper_final=float(s['final_loss']))
