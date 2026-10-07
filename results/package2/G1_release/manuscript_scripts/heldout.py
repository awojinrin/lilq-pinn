import pandas as pd, numpy as np, glob, os, json
EPS=2.220446049250313e-16
def load(f):
    d=pd.read_csv(f); return d
def rule(d,tc=0.1,tr=0.01):
    K=d.k.max(); b=d[d.k<K]; rl=b.norm_Rlin_h.values; chi=b.chi.values
    for i in range(1,len(b)):
        if np.isfinite(chi[i]) and chi[i]<=tc and abs(rl[i]-rl[i-1])<=tr*rl[i]: return i
    return None
rows=[]
B8='<laptop-home>/AppData/Local/Temp/claude/C--Users-<user>-Documents-LiL-Q/b870b5bf-bc15-4f34-8be6-4c7449cc20a2/scratchpad/waves/results/wave3/B_instrumentation/b8_jobs'
for f in sorted(glob.glob(B8+'/*/lilq_logs/*/iterations.csv')):
    name=f.split('/lilq_logs/')[1].split('/')[0]
    if not (name.startswith('viscous_zero') or name.startswith('gravity_ic')): continue
    rows.append(('B8',name,f))
OS='<laptop-home>/AppData/Local/Temp/claude/C--Users-<user>-Documents-LiL-Q/b870b5bf-bc15-4f34-8be6-4c7449cc20a2/scratchpad/waves/results/wave1/C_oversampling/runs'
for d in sorted(glob.glob(OS+'/*')):
    n=os.path.basename(d)
    if (n.startswith('kovasznay') and '_r3_paper' in n) or (n.startswith('bratu') and '_r10_paper' in n): continue
    rows.append(('C',n,d+'/iterations.csv'))
out=[]
for src,n,f in rows:
    d=load(f); K=int(d.k.max()); ks=rule(d)
    R=d.norm_R_h.values; Rmin=np.nanmin(R)
    r=dict(src=src,run=n,K=K,k_s=ks)
    if ks is not None:
        kr=ks+1; r.update(k_ret=kr,R_ratio=R[kr]/Rmin, at_end=(kr==K))
        if 'eps_u' in d and d.eps_u.notna().any() and n.startswith('kovasznay'):
            e=d.eps_u.values; e1=np.nanmin(e[1:]); r.update(E_ratio_min=e[kr]/e1, E_ratio_stop=e[kr]/e[K])
        cls = 'C' if d.roundoff_ratio[ks] < 10*(d.kappa_retained[ks] if 'kappa_retained' in d and np.isfinite(d.kappa_retained[ks]) else d.kappa[ks])*EPS else 'A'
        r['cls']=cls
    out.append(r)
o=pd.DataFrame(out); o.to_csv('heldout.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
print(o[o.src=='B8'].to_string())
c=o[o.src=='C']
c=c.assign(bench=c.run.str.split('_').str[0], ratio=c.run.str.extract(r'_r([0-9.]+)_')[0].astype(float), dist=c.run.str.extract(r'_r[0-9.]+_([a-z]+)')[0])
print('C runs',len(c),'fired',c.k_s.notna().sum())
print(c.groupby(['bench',c.ratio>1]).agg(n=('run','size'),fired=('k_s',lambda s:s.notna().sum()),Rmax=('R_ratio','max'),Rmed=('R_ratio','median'),early=('R_ratio',lambda s:(s>1.02).sum()),atend=('at_end','sum')))
print(c[(c.R_ratio>1.02)][['run','K','k_s','R_ratio','E_ratio_min','cls']].to_string())
print(c[c.k_s.isna()][['run','K']].to_string())
kc=c[c.bench=='kovasznay']
print('kov E_ratio_stop range', kc.E_ratio_stop.min(), kc.E_ratio_stop.max(), 'E_ratio_min max', kc.E_ratio_min.max())
