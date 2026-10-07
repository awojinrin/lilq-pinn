"""Kovasznay: velocity error against wall-clock time for the calibrated network
baselines (F1_18, F2_05; five seeds each, GPU) and LiL-Q (one run per basis size,
GPU and CPU clean times). Reads wave-4 output files only."""
import sys; sys.path.insert(0, '<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1')
import style; style.apply()
import matplotlib.pyplot as plt, pandas as pd, numpy as np

W4 = '<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/'
A = W4 + 'A_calibration/full/'
cmp = pd.read_csv(W4 + 'A_calibration/results/kovasznay_comparison.csv')

fig, ax = plt.subplots(figsize=(style.TEXTWIDTH_IN * 0.62, style.TEXTWIDTH_IN * 0.42))
fam = [('F1_18', 'F1 (Fourier-feature PINN)', style.COLORS[0], '-'),
       ('F2_05', 'F2 (Levenberg–Marquardt)', style.COLORS[1], '-')]
for cfg, lab, col, ls in fam:
    for s in range(5):
        d = pd.read_csv(f'{A}{cfg}_s{s}/log.csv')
        d = d[d.eps_u.notna()]
        ax.loglog(d.t_cum_s, d.eps_u, ls=ls, color=col, lw=0.7, alpha=0.85,
                  label=lab if s == 0 else None)

lil = cmp[cmp.family == 'LiL-Q'].copy()
lil['P'] = lil.config.str.replace('P=', '').astype(int)
g = lil[lil.device == 'gpu'].sort_values('P'); c = lil[lil.device == 'cpu'].sort_values('P')
ax.loglog(g.time_s, g.eps_u_median, 'o', color='k', ms=4.2, label='LiL-Q, GPU')
ax.loglog(c.time_s, c.eps_u_median, 'o', color='k', mfc='white', ms=4.2, label='LiL-Q, CPU')
for (_, rg), (_, rc) in zip(g.iterrows(), c.iterrows()):
    ax.plot([rg.time_s, rc.time_s], [rg.eps_u_median, rc.eps_u_median], ':', color='k', lw=0.6)
    P = int(rg.P)
    txt = f'$P={P:,}$'.replace(',', '{,}')
    off = (-6, 0)
    ax.annotate(txt, (rg.time_s, rg.eps_u_median), textcoords='offset points', xytext=off,
                ha='right', va='center', fontsize=7)

ax.set_xlabel('Wall-clock time (s)'); ax.set_ylabel(r'$E_{u}$')
ax.set_xlim(3e-3, 5e3); ax.set_ylim(1e-13, 3)
ax.yaxis.set_major_locator(plt.FixedLocator(10.0 ** np.arange(-13, 1, 2)))
ax.yaxis.set_minor_locator(plt.FixedLocator(10.0 ** np.arange(-13, 1, 1)))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.axvline(3600, color='0.6', lw=0.6, ls='--')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)
style.save(fig, 'fig_compA_kovasznay')
print(g[['P', 'time_s', 'eps_u_median']]); print(c[['P', 'time_s', 'eps_u_median']])
