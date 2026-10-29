import numpy as np, sys
sys.path.insert(0,'ana'); from measures import *
rng=np.random.default_rng(7)
SESS=[135,90]; IVd=0.16**2/252
def session_paths(pnon, w, mode='nonsync', freeze=0.45, om=0.0):
    out=[]
    for L in SESS:
        sd=np.sqrt(IVd*L/sum(SESS)/L)
        pstar=np.concatenate([[0],np.cumsum(rng.normal(0,sd,L))])
        if mode=='nonsync':
            N=len(w); last=np.tile(pstar[0],N); obs=np.empty(L+1); obs[0]=pstar[0]
            for i in range(1,L+1):
                upd=rng.random(N)>pnon; last[upd]=pstar[i]; obs[i]=np.dot(w,last)
        elif mode=='freeze':
            obs=pstar.copy()
            for i in range(1,L+1):
                if rng.random()<freeze: obs[i]=obs[i-1]
            obs[-1]=pstar[-1]
        elif mode=='noise':
            obs=pstar+rng.normal(0,np.sqrt(om),L+1)
        else: obs=pstar
        out.append((obs,pstar))
    return out
def measures(paths):
    acc={}
    for obs,pstar in paths:
        r=np.diff(obs); n=len(r)
        m=dict(rv1=gam(r,0),ac1=rvac(r,1),ac2=rvac(r,2),ac5=rvac(r,5))
        om_=m['rv1']/(2*n); p10=obs[list(range(0,n+1,10))+([n] if n%10 else [])]; iv=np.sum(np.diff(p10)**2)
        H=3.5134*(om_/iv)**(2/5)*n**(3/5) if iv>0 else 1; m['rk']=rk(r,H)
        Hb=3.5134*(om_/np.sqrt(iv))**(2/5)*n**(3/5) if iv>0 else 1; m['rk_draft']=rk(r,Hb)
        n_=len(obs)-1; nb=(n_-4)/5; m['tsrv']=tsrv(obs)/(1-nb/n_); m['bpv']=bpv(r)*n/(n-1); m['g1']=gam(r,1); m['g0']=gam(r,0)
        for k,v in m.items(): acc[k]=acc.get(k,0)+v
    return acc
def run(label,N=1500,**kw):
    S=[];ACs=[]
    for _ in range(N):
        acc=measures(session_paths(**kw)); S.append(acc); ACs.append(acc['g1']/acc['g0'])
    keys=['rv1','ac1','ac2','ac5','rk','rk_draft','tsrv','bpv']
    print(f'{label:28s} AC1 {np.mean(ACs):+.3f} '+' '.join(f"{k} {(np.mean([s[k] for s in S])/IVd-1)*100:+.1f}" for k in keys))
eq=np.ones(30)/30
cw=1/np.arange(1,31)**0.9; cw/=cw.sum()
run('no contamination',pnon=0,w=eq,mode='none')
run('index freeze 45%',pnon=0,w=eq,mode='freeze')
run('indep noise',pnon=0,w=eq,mode='noise',om=6.8e-8)
for p in (0.3,0.5,0.7):
    run(f'nonsync {int(p*100)}% equal',pnon=p,w=eq)
    run(f'nonsync {int(p*100)}% capw',pnon=p,w=cw)
print('top weight capw',cw[:3].round(3))
