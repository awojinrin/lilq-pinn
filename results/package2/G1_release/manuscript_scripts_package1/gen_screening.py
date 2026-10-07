"""Generate the appendix tables of the screened Component A configurations from the
wave-4 package files (screening runs are wave 2's, copied into wave 4 byte for byte)."""
import json
A='<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/A_calibration/screening/'
def sci(x, d=1):
    if x is None: return '--'
    m, e = f'{x:.{d}e}'.split('e'); e = int(e)
    return f'${m}{{\\times}}10^{{{e}}}$' if e != 0 else f'${m}$'
def lr(x): return {1e-3: '$10^{-3}$', 3e-4: '$3{\\times}10^{-4}$', 1e-4: '$10^{-4}$'}[x]
def k(n): return {2000: '2k', 8000: '8k', 32000: '32k', 5000: '5k', 20000: '20k', 50000: '50k'}[n]
def num(n): return f'{n:,}'.replace(',', '{,}')

out = {}
for fam in ['F1', 'F2']:
    sel = json.load(open(A + f'{fam}_selection.json'))
    rk_loss = {d['id']: i + 1 for i, d in enumerate(sel['ranking_by_training_loss'])}
    rk_val = {d['id']: i + 1 for i, d in enumerate(sel['ranking'])}
    fin = set(sel['top']); rep = {'F1': 'F1_18', 'F2': 'F2_05'}[fam]
    rows = []
    for i in range(24):
        cid = f'{fam}_{i:02d}'; p = A + cid + '_s0/'
        r = json.load(open(p + 'run.json')); v = json.load(open(p + 'validation.json'))
        c = r['config']
        tl = r.get('final_loss_unweighted') if fam == 'F1' else r['final_loss']
        name = cid.replace('_', '\\_')
        if cid in fin: name += '$^{\\ast}$'
        if cid == rep: name = '\\textbf{' + cid.replace('_', '\\_') + '}$^{\\ast}$'
        if fam == 'F1':
            bc = 'exact' if c['bc'] == 'hard' else f"pen.\\ {int(c['lambda_bc'])}"
            cells = [name, f"{c['width']}$\\times${c['depth']}", 'sep.' if c['trunk'] == 'separate' else 'sh.',
                     str(c['m']), f"{c['sigma_ff']:g}", bc, 'y' if c['balancing'] else 'n', lr(c['eta']),
                     k(c['t_adam']), k(c['n_int']), 'y' if c['resample'] else 'n', num(c['n_theta']),
                     sci(tl), sci(v['val_residual']), sci(r['eps_u']), str(rk_loss[cid]), str(rk_val[cid])]
        else:
            end = '$\\mu$' if r.get('end_reason') == 'mu_overflow' else ''
            cells = [name, f"{c['width']}$\\times${c['depth']}", str(c['m']), f"{c['sigma_ff']:g}", k(c['n_int']),
                     num(r['n_rows']), num(c['n_theta']) + ('$^{\\dagger}$' if c['n_theta'] > r['n_rows'] else ''),
                     sci(tl) + (f'$^{{\\mathrm{{o}}}}$' if end else ''), sci(v['val_residual']), sci(r['eps_u']),
                     str(rk_loss[cid]), str(rk_val[cid])]
        rows.append(' & '.join(cells) + ' \\\\')
    out[fam] = '\n'.join(rows)

F1 = r"""\begin{table}[!htb]
\centering
\caption{Kovasznay flow: the 24 screened configurations of the Fourier-feature PINN family F1 (Section~\ref{sec:kovasznay}), each trained with seed 0 for 10 minutes on the GPU. Net: width $\times$ number of hidden layers; trunk: one network shared by the three fields (sh.) or one per field (sep.); $m$, $\sigma$: number and scale of the random Fourier features; BC: Dirichlet conditions imposed exactly or by a penalty with weight $\lambda_{\mathrm{bc}}$; Bal.: gradient-norm balancing of the loss terms; $\eta$: Adam learning rate; $T_{\mathrm{A}}$: Adam iterations before L-BFGS; $N_{\mathrm{int}}$: interior collocation points; Res.: interior points redrawn during Adam; $n_{\theta}$: trainable parameters; Loss: final unweighted training loss; Val.: validation residual on $20{,}000$ fresh points; $E_{u}$: velocity error at the end of the screening run; ranks by the final training loss and by the validation residual. $^{\ast}$Finalists, selected by the final training loss; the representative, chosen among them by the median validation residual over five seeds of 60 minutes, is in bold.}
\label{tab:screening}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}lllrrlclrrcrlllrr@{}}
\toprule
& & & & & & & & & & & & & & & \multicolumn{2}{c@{}}{Rank} \\
\cmidrule(l){16-17}
Config. & Net & Trunk & $m$ & $\sigma$ & BC & Bal. & $\eta$ & $T_{\mathrm{A}}$ & $N_{\mathrm{int}}$ & Res. & $n_{\theta}$ & Loss & Val. & $E_{u}$ & Loss & Val. \\
\midrule
""" + out['F1'] + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering
\caption{Kovasznay flow: the 24 screened configurations of the Levenberg--Marquardt family F2 (Section~\ref{sec:kovasznay}), each trained with seed 0 for 10 minutes on the GPU; Dirichlet conditions are imposed exactly. Rows: collocated residuals ($3N_{\mathrm{int}}$ and one pressure pin); Loss: final training loss; other columns as in Table~\ref{tab:screening}. $^{\dagger}$More parameters than collocated residuals. $^{\mathrm{o}}$Run ended when the damping parameter overflowed. $^{\ast}$Finalists, the three configurations with the smallest validation residual; the representative, chosen among them by the median validation residual over five seeds of 60 minutes, is in bold.}
\label{tab:screening_f2}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrllllrr@{}}
\toprule
& & & & & & & & & & \multicolumn{2}{c@{}}{Rank} \\
\cmidrule(l){11-12}
Config. & Net & $m$ & $\sigma$ & $N_{\mathrm{int}}$ & Rows & $n_{\theta}$ & Loss & Val. & $E_{u}$ & Loss & Val. \\
\midrule
""" + out['F2'] + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open('<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/screening_tables.tex', 'w').write(F1)
print(F1[:3000])
