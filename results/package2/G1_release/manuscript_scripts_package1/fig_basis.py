import sys; sys.path.insert(0,'<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1')
import style; style.apply()
import matplotlib.pyplot as plt, pandas as pd, numpy as np
R='<laptop-home>/Documents/LiL-Q/Post-JCP/package1_local/package1/B_instrumentation/basis_study/runs/'
order=[('sin_cheb','Sin$\\times$Cheb'),('sin_fourier','Sin$\\times$Fourier'),('augsin_cheb','AugSin$\\times$Cheb'),
       ('fourier_cheb','Fourier$\\times$Cheb'),('fourier_fourier','Fourier$\\times$Fourier'),('cheb_cheb','Cheb$\\times$Cheb'),
       ('elm_default','ELM (tanh, default init.)'),('elm','ELM (tanh, Xavier)'),('cos_cheb','Cos$\\times$Cheb'),('sin_sin','Sin$\\times$Sin')]
cols=style.COLORS[:6]+['#882255','#000000','#999999','#661100']
mk=['o','s','^','D','v','p','*','h','<','>']
ls=['-','--','-.',':','-','--','-.',':','-','--']
fig,ax=plt.subplots(figsize=(style.TEXTWIDTH_IN*0.62,style.TEXTWIDTH_IN*0.42))
K=10
for i,(k,lab) in enumerate(order):
    d=pd.read_csv(R+k+'/iterations.csv')
    y=d.norm_R_h**2
    me={'sin_cheb':(0,2),'sin_fourier':(1,2),'fourier_cheb':(0,2),'fourier_fourier':(1,2)}.get(k,1)
    ax.semilogy(d.k[:K+1],y[:K+1],ls=ls[i],marker=mk[i],color=cols[i],mfc='white' if i%2 else cols[i],ms=5.0 if mk[i]=='*' else 3.8,lw=0.9,label=lab,markevery=me)
ax.set_xlabel('Outer iteration $k$'); ax.set_ylabel('Training loss (weighted MSE)')
ax.set_xlim(-0.3,K+0.3); ax.set_xticks(range(0,K+1,2))
lo,hi=ax.get_ylim(); ax.set_ylim(1e-9,hi)
ax.yaxis.set_major_locator(plt.FixedLocator(10.0**np.arange(-9,4,2)))
ax.yaxis.set_minor_locator(plt.FixedLocator(10.0**np.arange(-9,4,1)))
ax.yaxis.set_minor_formatter(plt.NullFormatter()); ax.set_ylim(1e-9,hi)
ax.legend(loc='upper left',bbox_to_anchor=(1.01,1.0),ncol=1,fontsize=8,handlelength=2.6)
fig.savefig('<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/fig_basis_burgers.pdf'); fig.savefig('<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/fig_basis_burgers.png',dpi=200)
# report last-iteration values for check
for k,lab in order:
    d=pd.read_csv(R+k+'/iterations.csv'); print(k, len(d)-1, (d.norm_R_h**2).iloc[-1], (d.norm_R_h**2).iloc[K])
