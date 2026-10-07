"""Component C: dependence on N/P and on the point distribution (Kovasznay, Bratu)."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import style
style.apply()

OS = pd.read_csv('<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/C_oversampling/results/oversampling.csv')
CC = pd.read_csv('<laptop-home>/Documents/LiL-Q/Post-JCP/Computational_Package2/Package2v2/attachments/S85_sampling_constants/componentC_kovasznay_c1c2.csv')
CC['ratio_actual'] = CC.N / CC.P
DIST = [  # key, label, colour, marker, linestyle
    ('paper', None, style.COLORS[0], 'o', '-'),
    ('cgl', 'Chebyshev–Gauss–Lobatto', style.COLORS[1], 's', '--'),
    ('random', 'uniform random (median, range)', style.COLORS[2], '^', ':'),
]
PAPER_LABEL = {'kovasznay': 'equispaced grid', 'bratu': 'random tensor grid'}
TICKS = [1, 1.5, 2, 3, 5, 10, 20]

def agg(df, col):
    g = df.groupby(['ratio_nominal' if 'ratio_nominal' in df else 'ratio', 'distribution' if 'distribution' in df else 'dist'])
    return g.agg(x=('ratio_actual', 'first'), med=(col, 'median'), lo=(col, 'min'), hi=(col, 'max'), n=(col, 'size')).reset_index()

def panel(ax, df, col, dkey):
    t = agg(df, col)
    dcol = 'distribution' if 'distribution' in t else 'dist'
    for key, lab, c, m, ls in DIST:
        s = t[t[dcol] == key].sort_values('x')
        if key == 'random':
            ax.fill_between(s.x, s.lo, s.hi, color=c, alpha=0.18, lw=0, zorder=1)
            ax.plot(s.x, s.lo, color=c, lw=0.4, alpha=0.7, zorder=1)
            ax.plot(s.x, s.hi, color=c, lw=0.4, alpha=0.7, zorder=1)
            ax.plot(s.x, s.med, ls=ls, color=c, marker=m, ms=4, mfc='white', mec=c, mew=0.9, lw=1.1, zorder=3)
        else:
            ax.plot(s.x, s.med, ls=ls, color=c, marker=m, ms=4, mfc=c, mec=c, lw=1.0, zorder=2 if key == 'cgl' else 4)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(TICKS); ax.set_xticklabels([f'{v:g}' for v in TICKS])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(0.9, 23)
    return t

def tidy_log_y(ax):
    lo, hi = ax.get_ylim(); dec = np.log10(hi / lo)
    step = 1 if dec <= 7 else (2 if dec <= 14 else 3)
    e0 = step * np.floor(np.log10(lo) / step) - step
    ax.yaxis.set_major_locator(plt.FixedLocator(10.0 ** np.arange(e0, np.ceil(np.log10(hi)) + step + 1, step)))
    if dec <= 4:
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
    elif step == 1:
        ax.yaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.yaxis.set_minor_locator(plt.FixedLocator(10.0 ** np.arange(e0, np.ceil(np.log10(hi)) + 2, 1)))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(lo, hi)

def legend(fig, axes_top, bench):
    h = []
    for key, lab, c, m, ls in DIST:
        lab = PAPER_LABEL[bench] if key == 'paper' else lab
        if key == 'random':
            h.append((Patch(facecolor=c, alpha=0.18, lw=0),
                      Line2D([], [], ls=ls, color=c, marker=m, ms=4, mfc='white', mec=c, mew=0.9, lw=1.1)))
        else:
            h.append(Line2D([], [], ls=ls, color=c, marker=m, ms=4, mfc=c, mec=c, lw=1.0))
    labs = [PAPER_LABEL[bench], DIST[1][1], DIST[2][1]]
    fig.legend(h, labs, loc='lower center', bbox_to_anchor=(0.5, axes_top + 0.01), ncol=3,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0, handlelength=2.8)

stats = {}
for bench, Ps, rows in [('kovasznay', [300, 1200], ['eps_u', 'kappa', 'realized_ratio']),
                        ('bratu', [100, 225], ['eps_u', 'kappa'])]:
    nr = len(rows)
    fig, axs = plt.subplots(nr, 2, figsize=(style.TEXTWIDTH_IN, 1.95 * nr + 0.35), sharex=True,
                            gridspec_kw=dict(wspace=0.25, hspace=0.24))
    st = {}
    for j, P in enumerate(Ps):
        d = OS[(OS.benchmark == bench) & (OS.P == P)]
        for i, col in enumerate(rows):
            ax = axs[i, j]
            if col == 'realized_ratio':
                t = panel(ax, CC[CC.P == P], col, bench)
                ax.axhline(1.0, color='0.5', lw=0.6, ls='-', zorder=0)
            else:
                t = panel(ax, d, col, bench)
            st[f'{col}_P{P}'] = t.to_dict(orient='records')
            tidy_log_y(ax)
            if i == 0:
                ax.set_title(f'({"abcdef"[i*2+j]})  $P = {P:,}$'.replace(',', '{,}'), loc='left', pad=3)
            else:
                ax.set_title(f'({"abcdef"[i*2+j]})', loc='left', pad=3)
            if i == nr - 1:
                ax.set_xlabel(r'Oversampling ratio $N/P$')
        # iterations / stopping summary for figures.md
        st[f'runs_P{P}'] = d.groupby(['ratio_nominal', 'distribution']).agg(
            iters_med=('iterations', 'median'), iters_min=('iterations', 'min'), iters_max=('iterations', 'max'),
            n_kmax=('stopping', lambda s: int((s == 'K_max').sum())), n=('stopping', 'size'),
            eps_p_mf_med=('eps_p_meanfree', 'median'), eps_v_med=('eps_v', 'median'),
            rank_min=('num_rank_svd', 'min'), N=('N_rows', 'first')).reset_index().to_dict(orient='records')
    ylab = {'eps_u': (r'$E_u$' if bench == 'kovasznay' else r'$\varepsilon_{\mathrm{test}}$'),
            'kappa': r'$\kappa(\mathbf{A})$',
            'realized_ratio': r'$\varrho_{\mathrm{r}}$'}
    for i, col in enumerate(rows):
        axs[i, 0].set_ylabel(ylab[col])
    fig.canvas.draw()
    legend(fig, axs[0, 0].get_position().y1 + 0.03 / (nr / 2), bench)
    style.save(fig, f'fig_np_{bench}')
    stats[bench] = st
json.dump(stats, open('<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/np_stats.json', 'w'), indent=1, default=float)
