import numpy as np, sys
sys.path.insert(0,'ana'); from tests import *
from scipy import stats
rng=np.random.default_rng(2026)
def sim(n,burn=200):
    a,b=0.08,0.90; s2u=1e-4; om=s2u*(1-a-b)
    z=rng.standard_normal(n+burn); s2=np.empty(n+burn); r=np.empty(n+burn); s2[0]=s2u
    for t in range(n+burn):
        if t>0: s2[t]=om+a*r[t-1]**2+b*s2[t-1]
        r[t]=np.sqrt(s2[t])*z[t]
    return r[burn:],np.sqrt(s2[burn:])
def one(n,a,scale=1.0):
    r,s=sim(n); s=s*scale
    q=stats.norm.ppf(a); var=s*q; I=(r<=var).astype(int)
    u=stats.norm.cdf(r/s)
    return kupiec(I,a)[1]<.05, christoffersen(I,a)[3]<.05, dq(I,var,a)[1]<.05, du_escanciano(u,a)[1]<.05
res={}
for a in (0.05,0.025,0.01):
    x=np.array([one(670,a) for _ in range(3000)]); res[a]=x.mean(0)*100
    print('size n=670',a,np.round(res[a],1))
x=np.array([one(265,0.01) for _ in range(3000)]); print('size n=265 1%',np.round(x.mean(0)*100,1))
x=np.array([one(250,0.01) for _ in range(3000)]); print('size n=250 1%',np.round(x.mean(0)*100,1))
for u in (0.10,0.20,0.30):
    x=np.array([one(670,0.01,1-u) for _ in range(1500)]); print('power 1% understate',u,np.round(x.mean(0)*100,1))
