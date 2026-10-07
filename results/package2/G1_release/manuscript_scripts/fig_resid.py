"""Residual histories and phase indicator for the scalar benchmarks (K_max = 60 passes)."""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style
style.apply()

W1 = '<laptop-home>/AppData/Local/Temp/claude/C--Users-<user>-Documents-LiL-Q/b870b5bf-bc15-4f34-8be6-4c7449cc20a2/scratchpad/waves/results/wave1/B_instrumentation'
W2 = '<laptop-home>/AppData/Local/Temp/claude/C--Users-<user>-Documents-LiL-Q/b870b5bf-bc15-4f34-8be6-4c7449cc20a2/scratchpad/waves/results/wave2/B_instrumentation'
TAU_CHI, TAU_R = 0.1, 0.01
BENCH = {  # name: (P list, x-axis limit, title)
    'bratu':      ([25, 100, 225], 10),
    'burgers':    ([25, 100, 225, 400, 625], 10),
    'bl':         ([64, 256, 576, 1024], 20),
    'bl_gravity': ([64, 256, 576, 1024], 50),
}


def ME(kk):
    """All markers up to k = 10, then every third (long histories only)."""
    import numpy as _np
    kk = _np.asarray(kk)
    if kk.max() <= 20:
        return None
    return [i for i, v in enumerate(kk) if v <= 10 or v % 3 == 0]

def rule_stop(df, ns=2):
    """First k >= ns at which chi_j <= tau_chi and |Rlin_j - Rlin_{j-1}| <= tau_r Rlin_j for j = k-ns+1..k."""
    rl = df.norm_Rlin_h.values; chi = df.chi.values
    ok = [False] + [bool(np.isfinite(chi[j]) and np.isfinite(rl[j]) and chi[j] <= TAU_CHI
                         and abs(rl[j] - rl[j-1]) <= TAU_R * rl[j]) for j in range(1, len(df))]
    for k in range(ns, len(df)):
        if all(ok[k - ns + 1:k + 1]):
            return k
    return None

stats = {}
for bench, (Ps, kx) in BENCH.items():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(style.TEXTWIDTH_IN, 2.55),
                                   gridspec_kw=dict(wspace=0.28))
    st = {}
    for i, P in enumerate(Ps):
        c, m = style.COLORS[i], style.MARKERS[i]
        ms = 3.6 if kx <= 20 else 2.6
        df = pd.read_csv(f'{W1}/{bench}_P{P}_cpu_kmax/iterations.csv')
        dp = pd.read_csv(f'{W2}/{bench}_P{P}_cpu_paper/iterations.csv')
        k = df.k.values; R = df.norm_R_h.values; Rl = df.norm_Rlin_h.values; chi = df.chi.values
        ks = rule_stop(df); kr = ks + 1
        sel = k <= kx
        axa.semilogy(k[sel], R[sel], '-', color=c, marker=m, ms=ms, mfc=c, mec=c, lw=1.0, zorder=3, markevery=ME(k[sel]))
        okl = sel & np.isfinite(Rl)
        axa.semilogy(k[okl], Rl[okl], '--', color=c, marker=m, ms=ms, mfc='white', mec=c, mew=0.8, lw=0.9, zorder=2, markevery=ME(k[okl]))
        axa.semilogy([kr], [R[kr]], marker='*', ms=11, mfc=c, mec='black', mew=0.9, ls='none', zorder=5)
        okc = sel & np.isfinite(chi)
        axb.semilogy(k[okc], chi[okc], '-', color=c, marker=m, ms=ms, mfc=c, mec=c, lw=1.0, zorder=3, markevery=ME(k[okc]))
        axb.semilogy([ks], [chi[ks]], marker='*', ms=11, mfc=c, mec='black', mew=0.9, ls='none', zorder=5)
        # statistics for figures.md
        Kp = int(dp.k.max())
        first_below = int(next(j for j in range(len(chi)) if np.isfinite(chi[j]) and chi[j] <= TAU_CHI))
        st[P] = dict(k_stop=ks, k_return=kr, R_return=R[kr], Rlin_stop=Rl[ks], chi_stop=chi[ks],
                     R_final=R[-1], K=int(k[-1]), R_return_over_final=R[kr]/R[-1],
                     R_min=float(np.nanmin(R)), k_Rmin=int(np.nanargmin(R)),
                     chi0=chi[0], chi_max=float(np.nanmax(chi)), k_chi_max=int(np.nanargmax(chi)),
                     chi_max_before_stop=float(np.nanmax(chi[:ks])) if ks > 0 else None,
                     first_k_chi_le_tau=first_below,
                     chi_median_after_return=float(np.nanmedian(chi[kr:])),
                     paper_iters=Kp, paper_R_final=float(dp.norm_R_h.values[-1]),
                     R0=R[0], Rlin0=Rl[0],
                     R_last10_relspread=float((R[-11:].max()-R[-11:].min())/R[-1]),
                     shown_upto=kx)
    stats[bench] = st
    axb.axhline(TAU_CHI, color='0.25', lw=0.8, ls=':', zorder=1)
    if bench == 'bl_gravity':
        axb.text(0.55 * kx, TAU_CHI / 1.6, r'$\tau_\chi = 0.1$', ha='center', va='top', fontsize=8, color='0.15')
    elif bench == 'bl':
        axb.text(kx * 0.985, TAU_CHI * 1.6, r'$\tau_\chi = 0.1$', ha='right', va='bottom', fontsize=8, color='0.15')
    else:
        axb.text(kx * 0.985, TAU_CHI / 1.6, r'$\tau_\chi = 0.1$', ha='right', va='top', fontsize=8, color='0.15')
    for ax in (axa, axb):
        ax.set_xlim(-0.03 * kx, kx * 1.03)
        ax.set_xlabel(r'Outer iteration $k$')
        if kx <= 10:
            ax.set_xticks(range(0, kx + 1, 2))
        elif kx <= 20:
            ax.set_xticks(range(0, kx + 1, 4))
        else:
            ax.set_xticks(range(0, kx + 1, 10))
    for ax in (axa, axb):
        lo, hi = ax.get_ylim(); dec = np.log10(hi / lo)
        step = 1 if dec <= 7 else (2 if dec <= 14 else 3)
        e0 = step * np.floor(np.log10(lo) / step) - step
        ax.yaxis.set_major_locator(plt.FixedLocator(10.0 ** np.arange(e0, np.ceil(np.log10(hi)) + step + 1, step)))
        if dec <= 5:
            ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
        elif step == 1:
            ax.yaxis.set_minor_locator(plt.NullLocator())
        else:
            ax.yaxis.set_minor_locator(plt.FixedLocator(10.0 ** np.arange(np.floor(np.log10(lo)) - 1, np.ceil(np.log10(hi)) + 2, 1)))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        if dec < 2.5:
            ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=[2.0, 5.0], numticks=50))
            ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, p: ('%g' % v)))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: ('%g' % v)))
        ax.set_ylim(lo, hi)
    axa.set_ylabel(r'$\|\mathbf{R}^{(k)}\|_h,\ \|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$')
    axb.set_ylabel(r'$\chi_k$')
    axa.set_title('(a)', loc='left', pad=3)
    axb.set_title('(b)', loc='left', pad=3)
    # legends: P (colour + marker) and series style, placed above the panels
    hP = [Line2D([], [], color=style.COLORS[i], marker=style.MARKERS[i], ms=3.6, lw=1.0,
                 label=f'$P = {P:,}$'.replace(',', '{,}')) for i, P in enumerate(Ps)]
    hS = [Line2D([], [], color='0.3', marker='o', ms=3.6, lw=1.0, label=r'$\|\mathbf{R}^{(k)}\|_h$'),
          Line2D([], [], color='0.3', marker='o', ms=3.6, mfc='white', lw=0.9, ls='--',
                 label=r'$\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$'),
          Line2D([], [], color='0.3', marker='o', ms=3.6, lw=1.0, label=r'$\chi_k$ (b)'),
          Line2D([], [], marker='*', ms=10, mfc='0.7', mec='black', mew=0.9, ls='none',
                 label='termination rule')]
    hS = [hS[0], hS[1], hS[3]]
    fig.canvas.draw()
    ytop = axa.get_position().y1
    l1 = fig.legend(handles=hP, loc='lower center', bbox_to_anchor=(0.5, ytop + 0.135),
                    ncol=len(Ps), columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)
    fig.legend(handles=hS, loc='lower center', bbox_to_anchor=(0.5, ytop + 0.065),
               ncol=3, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0)
    style.save(fig, f'fig_resid_{bench}')

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    raise TypeError(type(o))
json.dump(stats, open('<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts/resid_stats.json', 'w'), indent=1, default=conv)
