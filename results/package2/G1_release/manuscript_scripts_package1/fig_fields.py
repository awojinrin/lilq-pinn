"""Solution-field figures redrawn from the figure-data exports (branch figure-data, 9ff7b16).
Reads the .npz files only; nothing is recomputed except differences of the stored fields."""
import sys; sys.path.insert(0, '<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1')
import style; style.apply()
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

D = '<laptop-home>/AppData/Local/Temp/claude/C--Users-<user>-Documents-LiL-Q/b870b5bf-bc15-4f34-8be6-4c7449cc20a2/scratchpad/figdata/results/figure_data/'
W = style.TEXTWIDTH_IN
FIELD = 'viridis'; ERR = 'magma'
def L(name): return np.load(D + name + '.npz', allow_pickle=True)

def cbar(fig, ax, im, label=None, **kw):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, **kw)
    cb.ax.tick_params(labelsize=7)
    if label: cb.set_label(label, fontsize=8)
    return cb

def panel(ax, X, Y, F, title=None, **kw):
    im = ax.pcolormesh(X, Y, F.T, shading='auto', rasterized=True, **kw)
    if title: ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7, top=False, right=False)
    return im

def logerr(E, lo=None):
    E = np.abs(E); floor = lo if lo is not None else max(E.max() * 1e-6, 1e-300)
    return np.maximum(E, floor)

def errnorm(*arrs):
    hi = max(np.abs(a).max() for a in arrs)
    lo = hi * 1e-4
    return LogNorm(vmin=lo, vmax=hi)

METH = ['NiL-N', 'NiL-Q', 'LiL-N', 'LiL-Q']
SC = {m: r'\textsc{' + m + '}' for m in METH}
LAB = {'NiL-N': 'NiL-N', 'NiL-Q': 'NiL-Q', 'LiL-N': 'LiL-N', 'LiL-Q': 'LiL-Q'}

# ---------------- Bratu ----------------
z = L('bratu'); x, y = z['x'], z['y']
fig, axs = plt.subplots(2, 4, figsize=(W, W * 0.50), constrained_layout=True)
for r, P in enumerate([25, 225]):
    Fs = [z[f'P{P}_{m}'] for m in METH]
    vmin = min(F.min() for F in Fs); vmax = max(F.max() for F in Fs)
    for c, (m, F) in enumerate(zip(METH, Fs)):
        ax = axs[r, c]
        im = panel(ax, x, y, F, title=f'{LAB[m]}, $P = {P}$', cmap=FIELD, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal'); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if c == 0: ax.set_ylabel('$y$', fontsize=8)
        else: ax.set_yticklabels([])
        if r == 1: ax.set_xlabel('$x$', fontsize=8)
        else: ax.set_xticklabels([])
    cbar(fig, axs[r, :], im, '$u$')
style.save(fig, 'fig_fields_bratu')

# ---------------- Burgers ----------------
z = L('burgers'); x, t = z['x'], z['t']
fig, axs = plt.subplots(1, 4, figsize=(W, W * 0.27), constrained_layout=True)
for c, m in enumerate(METH):
    ax = axs[c]
    im = panel(ax, x, t, z[f'P625_{m}'], title=LAB[m], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('$x$', fontsize=8); ax.set_xticks([-1, 0, 1]); ax.set_yticks([0, 0.5, 1])
    if c == 0: ax.set_ylabel('$t$', fontsize=8)
    else: ax.set_yticklabels([])
cbar(fig, axs, im, '$u$')
style.save(fig, 'fig_fields_burgers')

# ---------------- Buckley-Leverett ----------------
z = L('buckley_leverett'); x, t = z['x'], z['t']
fig, axs = plt.subplots(2, 5, figsize=(W, W * 0.44), constrained_layout=True)
for r, (case, lab) in enumerate([('bl', '$N_g = 0$'), ('bl_gravity', '$N_g = -5$')]):
    ref = z[f'{case}_reference']; F = {m: z[f'{case}_{m}'] for m in ['NiL-N', 'LiL-Q']}
    cols = [('Reference', ref), ('NiL-N', F['NiL-N']), ('LiL-Q', F['LiL-Q'])]
    for c, (ttl, G) in enumerate(cols):
        im = panel(axs[r, c], x, t, G, title=f'{ttl}, {lab}', cmap=FIELD, vmin=0, vmax=1)
    cbar(fig, axs[r, 2], im, '$S$')
    E = {m: np.abs(F[m] - ref) for m in F}
    nrm = errnorm(*E.values())
    for c, m in enumerate(['NiL-N', 'LiL-Q']):
        imE = panel(axs[r, 3 + c], x, t, np.maximum(E[m], nrm.vmin), title=f'$|S - S_{{\\mathrm{{ref}}}}|$, {m}', cmap=ERR, norm=nrm)
    cbar(fig, axs[r, 4], imE)
    for c in range(5):
        ax = axs[r, c]; ax.set_xticks([0, 0.5, 1])
        if c == 0: ax.set_ylabel('$t$', fontsize=8)
        else: ax.set_yticklabels([])
        if r == 1: ax.set_xlabel('$x$', fontsize=8)
        else: ax.set_xticklabels([])
style.save(fig, 'fig_fields_bl')
m = json.loads(str(z['meta'])); print('BL checks', {k: v['recomputed'] for k, v in m['checks'].items()})

# ---------------- Kovasznay ----------------
z = L('kovasznay'); x, y = z['x'], z['y']
fig, axs = plt.subplots(3, 3, figsize=(W * 0.80, W * 0.90), constrained_layout=True)
for r, f in enumerate(['u', 'v', 'p']):
    ex, lq = z[f'exact_{f}'], z[f'LiL-Q_{f}']
    if f == 'p':
        ex = ex - ex.mean(); lq = lq - lq.mean()
    vmin, vmax = ex.min(), ex.max()
    panel(axs[r, 0], x, y, ex, title=f'Exact ${f}$', cmap=FIELD, vmin=vmin, vmax=vmax)
    im = panel(axs[r, 1], x, y, lq, title=f'\\textsc{{LiL-Q}} ${f}$'.replace('\\textsc{LiL-Q}', 'LiL-Q'), cmap=FIELD, vmin=vmin, vmax=vmax)
    cbar(fig, axs[r, 1], im)
    E = np.abs(lq - ex); nrm = LogNorm(vmin=max(E.max() * 1e-4, 1e-17), vmax=E.max())
    imE = panel(axs[r, 2], x, y, np.maximum(E, nrm.vmin), title=f'$|{f} - {f}^{{*}}|$', cmap=ERR, norm=nrm)
    cbar(fig, axs[r, 2], imE)
    print('Kovasznay max abs err', f, E.max(), 'rel to max|ex|', E.max() / np.abs(ex).max())
    for c in range(3):
        ax = axs[r, c]; ax.set_aspect('equal'); ax.set_xticks([-0.5, 0.25, 1]); ax.set_yticks([-0.5, 0.5, 1.5])
        if c == 0: ax.set_ylabel('$y$', fontsize=8)
        else: ax.set_yticklabels([])
        if r == 2: ax.set_xlabel('$x$', fontsize=8)
        else: ax.set_xticklabels([])
style.save(fig, 'fig_fields_kovasznay')

# ---------------- Elasticity ----------------
z = L('elasticity'); x, y = z['x'], z['y']
fig, axs = plt.subplots(2, 3, figsize=(W * 0.80, W * 0.56), constrained_layout=True)
for r, (f, lab) in enumerate([('ux', 'u_x'), ('uy', 'u_y')]):
    ex, lq = z[f'exact_{f}'], z[f'LiL_{f}']
    vmin, vmax = ex.min(), ex.max()
    panel(axs[r, 0], x, y, ex, title=f'Exact ${lab}$', cmap=FIELD, vmin=vmin, vmax=vmax)
    im = panel(axs[r, 1], x, y, lq, title=f'LiL ${lab}$', cmap=FIELD, vmin=vmin, vmax=vmax)
    cbar(fig, axs[r, 1], im)
    E = np.abs(lq - ex); nrm = LogNorm(vmin=1e-18, vmax=max(E.max(), 1e-17))
    imE = panel(axs[r, 2], x, y, np.maximum(E, 1e-18), title=f'$|{lab} - {lab}^{{*}}|$', cmap=ERR, norm=nrm)
    cbar(fig, axs[r, 2], imE)
    print('Elasticity max abs err', f, E.max())
    for c in range(3):
        ax = axs[r, c]; ax.set_aspect('equal'); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if c == 0: ax.set_ylabel('$y$', fontsize=8)
        else: ax.set_yticklabels([])
        if r == 1: ax.set_xlabel('$x$', fontsize=8)
        else: ax.set_xticklabels([])
style.save(fig, 'fig_fields_elasticity')

# ---------------- Darcy ----------------
z = L('darcy')
cases = ['S1', 'S2', 'S3', 'SPE10']
rows = [('FV', 'FV'), ('NiL_float64', 'NiL'), ('LiL', 'LiL')]
allF = [z[f'{c}_{k}'] for c in cases for k, _ in rows]
vmin = min(F.min() for F in allF); vmax = max(F.max() for F in allF)
fig, axs = plt.subplots(3, 4, figsize=(W * 0.78, W * 0.98), constrained_layout=True)
for c, cs in enumerate(cases):
    xx, yy = z[f'{cs}_x'], z[f'{cs}_y']
    for r, (k, lab) in enumerate(rows):
        ax = axs[r, c]
        im = panel(ax, xx, yy, z[f'{cs}_{k}'], title=f'{lab}, {cs}', cmap=FIELD, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal'); ax.set_xticks([0, 1200]); ax.set_yticks([0, 1100, 2200])
        if c == 0: ax.set_ylabel('$y$ (ft)', fontsize=8)
        else: ax.set_yticklabels([])
        if r == 2: ax.set_xlabel('$x$ (ft)', fontsize=8)
        else: ax.set_xticklabels([])
cbar(fig, axs, im, 'Pressure (psi)')
style.save(fig, 'fig_fields_darcy')

# ---------------- Beltrami: unfolded faces ----------------
z = L('beltrami'); s = z['s']
faces = [('xm1', '$x=-1$'), ('xp1', '$x=+1$'), ('ym1', '$y=-1$'), ('yp1', '$y=+1$'), ('zm1', '$z=-1$'), ('zp1', '$z=+1$')]
fig, axs = plt.subplots(4, 6, figsize=(W, W * 0.80), constrained_layout=True)
for r, f in enumerate(['u', 'v', 'w', 'p']):
    E = [np.abs(z[f'{fc}_LiL-Q_{f}'] - z[f'{fc}_exact_{f}']) for fc, _ in faces]
    nrm = errnorm(*E)
    for c, ((fc, lab), Ec) in enumerate(zip(faces, E)):
        ax = axs[r, c]
        im = panel(ax, s, s, np.maximum(Ec, nrm.vmin), title=(lab if r == 0 else None), cmap=ERR, norm=nrm)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_ylabel(f'$|{f} - {f}^{{*}}|$', fontsize=8.5)
    cbar(fig, axs[r, :], im)
    print('Beltrami max abs err', f, max(e.max() for e in E), 'max|exact|', max(np.abs(z[f'{fc}_exact_{f}']).max() for fc, _ in faces))
style.save(fig, 'fig_fields_beltrami_err')

# ---------------- Beltrami: cube views ----------------
from matplotlib import cm
from matplotlib.colors import Normalize
def cube(ax, getF, cmap, norm):
    S1, S2 = np.meshgrid(s, s, indexing='ij'); one = np.ones_like(S1)
    for fc, X, Y, Z in [('xp1', one, S1, S2), ('yp1', S1, one, S2), ('zp1', S1, S2, one)]:
        F = getF(fc)
        ax.plot_surface(X, Y, Z, facecolors=plt.get_cmap(cmap)(norm(F)), rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False, rasterized=True)
    ax.view_init(elev=24, azim=40); ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False); a.line.set_color((0, 0, 0, 0))
    ax.grid(False)
vis = ['xp1', 'yp1', 'zp1']
fig = plt.figure(figsize=(W * 0.90, W * 1.08))
gs = fig.add_gridspec(4, 3, wspace=0.35, hspace=0.08)
for r, f in enumerate(['u', 'v', 'w', 'p']):
    exs = [z[f'{fc}_exact_{f}'] for fc in vis]; lqs = [z[f'{fc}_LiL-Q_{f}'] for fc in vis]
    nf = Normalize(min(a.min() for a in exs + lqs), max(a.max() for a in exs + lqs))
    E = {fc: np.abs(z[f'{fc}_LiL-Q_{f}'] - z[f'{fc}_exact_{f}']) for fc in vis}
    hi = max(e.max() for e in E.values()); ne = LogNorm(hi * 1e-3, hi)
    for c, (getF, cmap, norm, ttl) in enumerate([
            (lambda fc: z[f'{fc}_exact_{f}'], FIELD, nf, f'Exact ${f}$'),
            (lambda fc: z[f'{fc}_LiL-Q_{f}'], FIELD, nf, f'LiL-Q ${f}$'),
            (lambda fc: np.maximum(E[fc], ne.vmin), ERR, ne, f'$|{f} - {f}^{{*}}|$')]):
        ax = fig.add_subplot(gs[r, c], projection='3d')
        cube(ax, getF, cmap, norm)
        ax.set_title(ttl, fontsize=8.5, pad=-2)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.7); cb.ax.tick_params(labelsize=6.5)
        if norm is nf: cb.set_ticks(np.round(np.linspace(nf.vmin, nf.vmax, 3), 2))
fig.savefig('<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/fig_fields_beltrami.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
fig.savefig('<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/fig_fields_beltrami.png', bbox_inches='tight', dpi=150)
plt.close(fig)
