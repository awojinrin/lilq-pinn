"""Rule with persistence: both detector conditions hold at n_s consecutive steps (tau_chi=0.1, tau_r=0.01)."""
import pandas as pd, numpy as np, glob, os, json
TC, TR, EPS = 0.1, 0.01, 2.220446049250313e-16
def flags(chi, rl, tc=TC, tr=TR):
    b = np.zeros(len(chi), bool)
    for i in range(1, len(chi)):
        b[i] = np.isfinite(chi[i]) and chi[i] <= tc and abs(rl[i] - rl[i-1]) <= tr * rl[i]
    return b
def fire_ns(chi, rl, ns, tc=TC, tr=TR):
    b = flags(chi, rl, tc, tr)
    for i in range(1, len(chi)):
        if i - ns + 1 >= 1 and b[i - ns + 1:i + 1].all(): return i
    return None
B = '<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/B_instrumentation'
runs = ['bratu_P25','bratu_P100','bratu_P225','burgers_P25','burgers_P100','burgers_P225','burgers_P400','burgers_P625',
        'bl_P64','bl_P256','bl_P576','bl_P1024','bl_gravity_P64','bl_gravity_P256','bl_gravity_P576','bl_gravity_P1024',
        'kovasznay_P75','kovasznay_P300','kovasznay_P675','kovasznay_P1200','kovasznay_P1875','beltrami_P7984']
def cls(d, ks):
    kr = d.kappa_retained[ks] if 'kappa_retained' in d and np.isfinite(d.kappa_retained[ks]) else d.kappa[ks]
    return 'C' if d.roundoff_ratio[ks] < 10 * kr * EPS else 'A'
print('== Table 16 (n_s = 1, 2, 3)')
early = {1: 0, 2: 0, 3: 0}; errstop = {1: 0, 2: 0, 3: 0}; Rmax = {1: 0, 2: 0, 3: 0}
for r in runs:
    d = pd.read_csv(f'{B}/{r}_cpu_kmax/iterations.csv'); p = pd.read_csv(f'{B}/{r}_cpu_paper/iterations.csv')
    K = int(d.k.max()); R = d.norm_R_h.values; e = d.eps_u.values
    stop = 43 if r == 'bl_gravity_P64' else int(p.k.max())
    ep = e[43] if r == 'bl_gravity_P64' else p.eps_u.values[-1]
    line = f'{r:18s} stop={stop:2d}'
    for ns in (1, 2, 3):
        i = fire_ns(d.chi.values[:K], d.norm_Rlin_h.values[:K], ns); kr = i + 1
        rr = R[kr] / np.nanmin(R); Rmax[ns] = max(Rmax[ns], rr)
        emin = np.nanmin(e[1:]); kmin = int(np.nanargmin(e[1:])) + 1
        if rr > 1.02: early[ns] += 1
        if e[kr] > 2 * emin and kr < kmin: errstop[ns] += 1
        line += f' | ns={ns} ret={kr:2d} {cls(d, i)} R={rr:.3f} E={e[kr]/ep:.2f}'
        if r.startswith('bl_gravity_P64') or r.startswith('bl_gravity_P256'): line += f' (R {rr:.2f})'
    print(line)
print('early', early, 'errstop', errstop, 'Rmax', {k: round(v, 4) for k, v in Rmax.items()})
print('== held out')
OS = '<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/C_oversampling/runs'
for ns in (1, 2, 3):
    fired = cens = 0; Rr = []; kov = []; k12 = {'A': [], 'C': []}
    for dd in sorted(glob.glob(OS + '/*')):
        n = os.path.basename(dd)
        if (n.startswith('kovasznay') and '_r3_paper' in n) or (n.startswith('bratu') and '_r10_paper' in n): continue
        d = pd.read_csv(dd + '/iterations.csv'); K = int(d.k.max())
        chi, rl, R = d.chi.values, d.norm_Rlin_h.values, d.norm_R_h.values
        i = fire_ns(chi[:K], rl[:K], ns); i1 = fire_ns(chi[:K], rl[:K], 1)
        if i is None:
            if i1 is not None: cens += 1
            continue
        fired += 1; kr = i + 1; Rr.append(R[kr] / np.nanmin(R))
        if n.startswith('kovasznay'):
            e = d.eps_u.values; kov.append((e[kr] / np.nanmin(e[1:]), e[kr] / e[K]))
            if 'P1200' in n: k12[cls(d, i)].append((e[kr], n))
    kv = np.array(kov)
    print(f'ns={ns}: fired {fired}, base-fired-but-not {cens}, R max {max(Rr):.5f}, kov n={len(kv)} Emin max {kv[:,0].max():.3f} med {np.median(kv[:,0]):.4f}, Estop {kv[:,1].min():.3f}-{kv[:,1].max():.3f}')
    for c in 'AC':
        v = sorted(k12[c]); print('   ', c, len(v), ('%.2e-%.2e' % (v[0][0], v[-1][0])) if v else '', [x[1] for x in v[:1]])
print('== pilot (LS runs, p>=10)')
res = json.load(open('<laptop-home>/Documents/LiL-Q/Post-JCP/Computational_Package2/Package2v2/attachments/bratu_pilot/study.json'))
for ns in (1, 2, 3):
    worst = []; ret = []
    for m, rr in res.items():
        if m.startswith('SQ'): continue
        for r in rr:
            h = r['hist']; n = len(h)
            chi = np.array([h[k+1].get('chi', np.nan) if k+1 < n else np.nan for k in range(n)])
            rl = np.array([x['Rlin'] for x in h]); E = np.array([x['E'] for x in h])
            i = fire_ns(chi, rl, ns)
            if i is None: continue
            worst.append(E[i+1] / np.nanmin(E[1:])); ret.append(i + 1)
    print(f'ns={ns}: runs {len(worst)}, worst E ratio {max(worst):.2f}, n>2: {sum(w>2 for w in worst)}, returned iterates {sorted(set(ret))}')
