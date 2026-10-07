"""Training-loss histories of the four formulations (NiL-N, NiL-Q, LiL-N, LiL-Q).

Outputs (in <laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts):
  fig_conv_bratu_by_size      one panel per P, all four formulations overlaid
  fig_conv_<bench>_by_method  one panel per formulation, all P overlaid
Gradient-based runs: GPU, seed 0 for NiL-N / NiL-Q (LiL-N has no seed).
LiL-Q: CPU paper pass (gravity BL P = 64: K_max pass up to k = 43).
"""
import importlib.util
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, '<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts')
spec = importlib.util.spec_from_file_location('style', '<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts/style.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
style.apply()
import conv_data as C

OUT = '<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts'
METHODS = C.METHODS
TITLES = {'bratu': 'Bratu', 'burgers': 'Burgers', 'bl': 'BL', 'bl_gravity': 'BL gravity'}
END = {  # stopping reason -> marker spec
    'target': dict(marker='*', ms=9.5, mec='black', mew=0.8, label='target reached'),
    'iteration_cap': dict(marker='|', ms=9.0, mec='black', mew=1.5, label='iteration budget'),
    'optimizer_stall': dict(marker='x', ms=6.5, mec='black', mew=1.3, label='optimizer stall'),
}
TGT_LS = (0, (1, 1.6))
# method colours for the by-size figure: kept distinct from the P colours of the other figures
MCOL = {'NiL-N': '0.15', 'NiL-Q': '0.6', 'LiL-N': '#56B4E9', 'LiL-Q': '#882255'}
MMK = {'NiL-N': 'o', 'NiL-Q': 's', 'LiL-N': '^', 'LiL-Q': 'D'}


def blended(ax):
    return blended_transform_factory(ax.transAxes, ax.transData)


def save(fig, stem):
    fig.savefig(f'{OUT}/{stem}.pdf')
    fig.savefig(f'{OUT}/{stem}.png', dpi=200)
    plt.close(fig)


def fmt_P(P):
    return f'$P = {P:,}$'.replace(',', '{,}')


def log_markevery(x, phase, per_decade=3.0):
    """Indices of points closest to log-spaced positions (x = iteration + 1)."""
    lo, hi = np.log10(x[0]), np.log10(x[-1])
    pos = np.arange(lo + phase / per_decade, hi + 1e-9, 1.0 / per_decade)
    idx = np.unique(np.searchsorted(x, 10.0 ** pos).clip(0, len(x) - 1))
    return idx[idx < len(x) - 1]          # the end point carries the stop marker


def plot_gradient(ax, run, color, marker, phase):
    x = run['it'] + 1.0
    y = run['loss']
    ax.loglog(x, y, '-', color=color, lw=0.9, zorder=3)
    mi = log_markevery(x, phase)
    ax.loglog(x[mi], y[mi], ls='none', marker=marker, ms=3.4, mfc=color, mec=color, zorder=4)
    end_marker(ax, x[-1], y[-1], run['reason'], color)


def end_marker(ax, x, y, reason, color, z=6):
    e = END[reason]
    if e['marker'] == '*':
        ax.plot([x], [y], ls='none', marker='*', ms=e['ms'], mfc=color, mec='black', mew=e['mew'], zorder=z)
    else:   # line markers: black underlay, series colour on top
        ax.plot([x], [y], ls='none', marker=e['marker'], ms=e['ms'] + 0.6, mec='black', mew=e['mew'] + 1.1, zorder=z)
        ax.plot([x], [y], ls='none', marker=e['marker'], ms=e['ms'], mec=color, mew=e['mew'], zorder=z + 0.5)


def set_log_y(ax, lo, hi):
    ax.set_ylim(lo, hi)
    dec = np.log10(hi / lo)
    step = 1 if dec <= 7 else 2
    e0 = step * np.floor(np.log10(lo) / step)
    ax.yaxis.set_major_locator(plt.FixedLocator(10.0 ** np.arange(e0, np.ceil(np.log10(hi)) + 1, step)))
    ax.yaxis.set_minor_locator(plt.FixedLocator(10.0 ** np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1, 1))
                               if step > 1 else plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(lo, hi)


def set_log_x(ax, xmax):
    ax.set_xlim(0.75, xmax)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def ylims(values):
    v = np.concatenate([np.asarray(a, float) for a in values])
    v = v[np.isfinite(v) & (v > 0)]
    lo = 10.0 ** np.floor(np.log10(v.min()) - 0.15)
    hi = 10.0 ** np.ceil(np.log10(v.max()) + 0.15)
    return lo, hi


def end_handles():
    return [Line2D([], [], ls='none', marker=e['marker'], ms=e['ms'] * (0.9 if k == 'target' else 1.0),
                   mfc='0.6' if e['marker'] == '*' else 'none', mec=e['mec'], mew=e['mew'], label=e['label'])
            for k, e in END.items()]


# ---------------------------------------------------------------- by method
def by_method(bench, stats):
    Ps = C.PS[bench]
    runs = {m: {P: C.gradient_run(bench, P, m, 0) for P in Ps} for m in METHODS[:3]}
    lq = {P: C.lilq_run(bench, P) for P in Ps}
    tg = {P: C.target(bench, P) for P in Ps}
    lo, hi = ylims([r['loss'] for m in runs for r in runs[m].values()] + [r['loss'] for r in lq.values()]
                   + [list(tg.values())])
    xmax = max(r['n_iter'] for m in runs for r in runs[m].values()) * 1.6

    fig, axs = plt.subplots(2, 2, figsize=(style.TEXTWIDTH_IN, 4.75), sharey=True,
                            gridspec_kw=dict(wspace=0.08, hspace=0.42))
    axs = axs.ravel()
    logk = True    # LiL-Q panels (d) on a log (k + 1) axis, as in the by-size figure
    for j, m in enumerate(METHODS):
        ax = axs[j]
        for i, P in enumerate(Ps):
            c, mk = style.COLORS[i], style.MARKERS[i]
            ax.axhline(tg[P], color=c, lw=0.8, ls=TGT_LS, zorder=1)
            # hollow series marker at the left end of each target line (identifies P in greyscale)
            ax.plot([0.035], [tg[P]], transform=blended(ax), ls='none', marker=mk, ms=3.6, mfc='white',
                    mec=c, mew=0.8, zorder=2, clip_on=False)
            if m != 'LiL-Q':
                plot_gradient(ax, runs[m][P], c, mk, phase=i / len(Ps))
            else:
                r = lq[P]
                xk = r['k'] + (1.0 if logk else 0.0)
                me = [i for i, k in enumerate(r['k'][:-1]) if k <= 12 or k % 4 == 0]
                ax.semilogy(xk, r['loss'], '-', color=c, marker=mk, ms=3.4, mfc=c, mec=c, lw=0.9, zorder=3,
                            markevery=me)
                e = END['target']
                ax.plot([xk[-1]], [r['loss'][-1]], ls='none', marker=e['marker'], ms=e['ms'], mfc=c,
                        mec=e['mec'], mew=e['mew'], zorder=6)
        set_log_y(ax, lo, hi)
        if m != 'LiL-Q':
            ax.set_xscale('log')
            set_log_x(ax, xmax)
            ax.set_xlabel(r'Iteration $+\,1$')
        elif logk:
            kx = max(r['n_iter'] for r in lq.values())
            ax.set_xscale('log')
            ax.set_xlim(0.85, (kx + 1) * 1.18)
            tk = list(range(1, kx + 2)) if kx + 1 <= 6 else ([1, 2, 3, 5, 10] if kx + 1 <= 12 else [1, 2, 5, 10, 20, 50])
            ax.xaxis.set_major_locator(plt.FixedLocator(tk))
            ax.xaxis.set_major_formatter(plt.FixedFormatter([str(t) for t in tk]))
            ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=60))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_xlabel(r'Outer iteration $k + 1$')
        else:
            kx = max(r['n_iter'] for r in lq.values())
            ax.set_xlim(-0.04 * kx - 0.2, kx * 1.04 + 0.2)
            step = 1 if kx <= 6 else (2 if kx <= 12 else (5 if kx <= 25 else 10))
            ax.set_xticks(range(0, kx + 1, step))
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_xlabel(r'Outer iteration $k$')
        ax.set_title(f'({"abcd"[j]}) {m}', loc='left', pad=3)
        if j % 2 == 0:
            ax.set_ylabel('Training loss (weighted MSE)')
    # legend above
    hP = [Line2D([], [], color=style.COLORS[i], marker=style.MARKERS[i], ms=3.4, lw=0.9, label=fmt_P(P))
          for i, P in enumerate(Ps)]
    hS = end_handles() + [Line2D([], [], color='0.25', lw=0.8, ls=TGT_LS, marker='o', ms=3.6, mfc='white',
                                 mec='0.25', mew=0.8, markevery=[0], label='target MSE')]
    fig.canvas.draw()
    ytop = axs[0].get_position().y1
    fig.legend(handles=hP, loc='lower center', bbox_to_anchor=(0.5, ytop + 0.072), ncol=len(Ps),
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)
    fig.legend(handles=hS, loc='lower center', bbox_to_anchor=(0.5, ytop + 0.03), ncol=4,
               columnspacing=1.4, handletextpad=0.4, borderaxespad=0.0)
    save(fig, f'fig_conv_{bench}_by_method')

    st = {}
    for P in Ps:
        d = dict(target=tg[P])
        for m in METHODS[:3]:
            r = runs[m][P]
            d[m] = dict(n_iter=r['n_iter'], final_loss=r['final_loss'], reason=r['reason'], cap=r['cap'],
                        loss0=float(r['loss'][0]), min_loss=float(r['loss'].min()),
                        final_over_target=r['final_loss'] / tg[P],
                        seeds=C.all_seeds(bench, P, m).to_dict('records'))
        r = lq[P]
        d['LiL-Q'] = dict(n_iter=r['n_iter'], final_loss=r['final_loss'], loss=r['loss'].tolist(),
                          source=r['source'], paper_iters=r['paper_iters'], paper_final=r['paper_final'],
                          final_over_target=r['final_loss'] / tg[P])
        st[P] = d
    stats[bench] = st


# ---------------------------------------------------------------- by size (Bratu)
def by_size(bench='bratu'):
    Ps = C.PS[bench]
    runs = {P: {m: C.gradient_run(bench, P, m, 0) for m in METHODS[:3]} for P in Ps}
    lq = {P: C.lilq_run(bench, P) for P in Ps}
    tg = {P: C.target(bench, P) for P in Ps}
    lo, hi = ylims([runs[P][m]['loss'] for P in Ps for m in METHODS[:3]] + [lq[P]['loss'] for P in Ps]
                   + [list(tg.values())])
    xmax = max(runs[P][m]['n_iter'] for P in Ps for m in METHODS[:3]) * 1.6
    fig, axs = plt.subplots(1, 3, figsize=(style.TEXTWIDTH_IN, 2.55), sharey=True,
                            gridspec_kw=dict(wspace=0.08))
    for j, P in enumerate(Ps):
        ax = axs[j]
        ax.axhline(tg[P], color='0.25', lw=0.8, ls=TGT_LS, zorder=1)
        for i, m in enumerate(METHODS):
            c, mk = MCOL[m], MMK[m]
            if m != 'LiL-Q':
                plot_gradient(ax, runs[P][m], c, mk, phase=i / 3)
            else:
                r = lq[P]
                x = r['k'] + 1.0
                ax.loglog(x, r['loss'], '-', color=c, marker=mk, ms=3.4, mfc=c, mec=c, lw=0.9, zorder=5,
                          markevery=list(range(len(x) - 1)))
                e = END['target']
                ax.plot([x[-1]], [r['loss'][-1]], ls='none', marker=e['marker'], ms=e['ms'], mfc=c,
                        mec=e['mec'], mew=e['mew'], zorder=7)
        set_log_y(ax, lo, hi)
        set_log_x(ax, xmax)
        ax.set_xlabel(r'Iteration $+\,1$')
        ax.set_title(f'({"abc"[j]}) ' + fmt_P(P), loc='left', pad=3)
    axs[0].set_ylabel('Training loss (weighted MSE)')
    hM = [Line2D([], [], color=MCOL[m], marker=MMK[m], ms=3.4, lw=0.9, label=m)
          for i, m in enumerate(METHODS)]
    hS = end_handles() + [Line2D([], [], color='0.25', lw=0.8, ls=TGT_LS, label='target MSE')]
    fig.canvas.draw()
    ytop = axs[0].get_position().y1
    fig.legend(handles=hM, loc='lower center', bbox_to_anchor=(0.5, ytop + 0.135), ncol=4,
               columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0)
    fig.legend(handles=hS, loc='lower center', bbox_to_anchor=(0.5, ytop + 0.065), ncol=4,
               columnspacing=1.4, handletextpad=0.4, borderaxespad=0.0)
    save(fig, f'fig_conv_{bench}_by_size')


if __name__ == '__main__':
    stats = {}
    for b in ['bratu', 'burgers', 'bl', 'bl_gravity']:
        by_method(b, stats)
    by_size('bratu')

    def conv(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        raise TypeError(type(o))
    json.dump(stats, open(f'{OUT}/conv_stats.json', 'w'), indent=1, default=conv)
