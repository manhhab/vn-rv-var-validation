import numpy as np, pandas as pd
def gam(r,h):
    return np.dot(r[h:],r[:-h]) if h>0 else np.dot(r,r)
def rvac(r,q):
    return gam(r,0)+2*sum(gam(r,h) for h in range(1,q+1)) if len(r)>q else gam(r,0)
def parzen(x):
    x=abs(x)
    return np.where(x<=0.5,1-6*x**2+6*x**3,np.where(x<=1,2*(1-x)**3,0.0))
def rk(r,H):
    H=max(int(np.ceil(H)),1)
    s=gam(r,0)
    for h in range(1,min(H,len(r)-1)+1):
        s+=2*parzen((h-1)/H)*gam(r,h)
    return s
def bpv(r):
    a=np.abs(r); return np.pi/2*np.dot(a[1:],a[:-1])
def tsrv(p,K=5):
    # p: 1-min log price path within a session (regular or record)
    n=len(p)-1
    if n<K+1: return np.sum(np.diff(p)**2)
    rv_all=np.sum(np.diff(p)**2)
    sub=[np.sum(np.diff(p[k::K])**2) for k in range(K)]
    nbar=(n-K+1)/K
    return np.mean(sub)-nbar/n*rv_all
