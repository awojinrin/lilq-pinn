import numpy as np
from numpy.polynomial import chebyshev as C
Re=40.0; z=Re/2-np.sqrt(Re**2/4+4*np.pi**2)
xs=np.linspace(-0.5,1.0,301); ys=np.linspace(-0.5,1.5,401)
X,Y=np.meshgrid(xs,ys,indexing='ij')
ue=1-np.exp(z*X)*np.cos(2*np.pi*Y); ve=z/(2*np.pi)*np.exp(z*X)*np.sin(2*np.pi*Y); pe=0.5*(1-np.exp(2*z*X))
def V(t,a,b,n): return C.chebvander(2*(t-a)/(b-a)-1,n-1)
E={5:(0.3768540617641253,1.2138661056703866,0.558520078869309),10:(0.028966849372465353,0.12971880322750823,0.029119457826986155),15:(1.2304982335107468e-05,9.867624336922714e-05,3.0306920895648072e-05),20:(7.359311300812976e-09,2.5360856483613258e-08,1.8903283934247005e-08),25:(7.002775416216135e-13,5.346822930292403e-12,1.6524600426894258e-12)}
print('pd   d_u      E_u/d_u   d_v      E_v/d_v   d_p(mf)  E_p/d_p')
for pd in [5,10,15,20,25]:
    Vx=V(xs,-0.5,1.0,pd); Vy=V(ys,-0.5,1.5,pd)
    # separable LS projection on tensor grid: (Vx^+) F (Vy^+)^T is the discrete L2 best approx on the grid
    Px=np.linalg.pinv(Vx); Py=np.linalg.pinv(Vy)
    def best(F): return Vx@(Px@F@Py.T)@Vy.T
    def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
    du=rel(best(ue),ue); dv=rel(best(ve),ve)
    pm=pe-pe.mean(); bp=best(pe); dp=np.linalg.norm((bp-bp.mean())-pm)/np.linalg.norm(pm)
    eu,ev,ep=E[pd]
    print(f'{pd:3d} {du:9.2e} {eu/du:7.2f} {dv:9.2e} {ev/dv:7.2f} {dp:9.2e} {ep/dp:7.2f}')
print('combined (u,v,p mean-free) product L2:')
nu,nv,npm=np.linalg.norm(ue),np.linalg.norm(ve),np.linalg.norm(pe-pe.mean())
for pd in [5,10,15,20,25]:
    Vx=V(xs,-0.5,1.0,pd); Vy=V(ys,-0.5,1.5,pd); Px=np.linalg.pinv(Vx); Py=np.linalg.pinv(Vy)
    best=lambda F: Vx@(Px@F@Py.T)@Vy.T
    du=np.linalg.norm(best(ue)-ue); dv=np.linalg.norm(best(ve)-ve); bp=best(pe); dp=np.linalg.norm((bp-bp.mean())-(pe-pe.mean()))
    eu,ev,ep=E[pd]
    num=np.sqrt((eu*nu)**2+(ev*nv)**2+(ep*npm)**2); den=np.sqrt(du**2+dv**2+dp**2); tot=np.sqrt(nu**2+nv**2+npm**2)
    print(f'{pd:3d} P={3*pd*pd:5d} delta_rel={den/tot:9.2e} err_rel={num/tot:9.2e} ratio={num/den:7.2f}')
print('norms',nu,nv,npm)
