import pandas as pd, numpy as np, sys, pickle
sys.path.insert(0,'ana'); from measures import *
MS=[1,2,3,5,10,15,30]
def samp(p,m):
    idx=list(range(0,len(p),m))
    if idx[-1]!=len(p)-1: idx.append(len(p)-1)
    return p[idx]
def sess_measures(p):
    r=np.diff(p); n=len(r); res={}
    for m in MS: res[f'rv{m}']=np.sum(np.diff(samp(p,m))**2)
    for q in (1,2,3,4,5,6,8,10): res[f'ac{q}']=rvac(r,q)
    om=res['rv1']/(2*n); iv=res['rv10']
    xi2=om/iv if iv>0 else 0
    H=3.5134*xi2**(2/5)*n**(3/5); res['H']=H; res['rk']=rk(r,H)
    xi2b=om/np.sqrt(iv) if iv>0 else 0
    Hb=3.5134*xi2b**(2/5)*n**(3/5); res['Hpaper']=Hb; res['rk_paper']=rk(r,Hb)
    K=5; subs=[np.sum(np.diff(samp(p[k:],K))**2) for k in range(K)]
    nbar=(n-K+1)/K; ts=np.mean(subs)-nbar/n*res['rv1']
    res['tsrv']=ts; res['tsrv_adj']=ts/(1-nbar/n)
    res['bpv']=bpv(r)*n/(n-1)
    res['g0']=gam(r,0); res['g1']=gam(r,1)
    res['nzero']=np.sum(r==0); res['n']=n
    res['rs']=p[-1]-p[0]
    return res
for name in ['VNINDEX','VN30']:
    o=pickle.load(open(f'data/{name}_B.pkl','rb'))
    rows=[]
    for d,v in o.items():
        a=sess_measures(v['paths'][0]); b=sess_measures(v['paths'][1])
        row={k:a[k]+b[k] for k in a if k not in('H','Hpaper','rs')}
        row.update(d=d,H=(a['H']+b['H'])/2,Hpaper=(a['Hpaper']+b['Hpaper'])/2,
                   rm=a['rs'],ra=b['rs'],bench=a['rs']**2+b['rs']**2,
                   rct=a['rs']+b['rs'], roc=v['close']-v['open'],
                   lunch=v['paths'][1][0]-v['paths'][0][-1], atc=v['close']-v['paths'][1][-1],
                   ac1day=(a['g1']+b['g1'])/(a['g0']+b['g0']) if (a['g0']+b['g0'])>0 else np.nan,
                   nrec=v['nrec'])
        rows.append(row)
    D=pd.DataFrame(rows).set_index('d'); D.to_pickle(f'data/{name}_daily.pkl')
    B=D.bench.mean()
    print('=====',name, 'days',len(D))
    print(' annvol bench(session) %.2f  rct^2 %.2f  roc^2(9:15->close) %.2f' % tuple(np.sqrt(252*x)*100 for x in [B,(D.rct**2).mean(),(D.roc**2).mean()]))
    print(' share of E[roc^2]: lunch %.1f%% atc %.1f%%' % ((D.lunch**2).mean()/(D.roc**2).mean()*100,(D.atc**2).mean()/(D.roc**2).mean()*100))
    sig={m:round(np.sqrt(252*D[f'rv{m}'].mean())*100,2) for m in MS}; print(' signature',sig)
    cols=[f'rv{m}' for m in MS]+[f'ac{q}' for q in (1,2,3,4,5,6,8,10)]+['rk','rk_paper','tsrv','tsrv_adj','bpv']
    print(' bias',{c:round((D[c].mean()/B-1)*100,1) for c in cols})
    print(' H mean %.2f  H paper %.2f' % (D.H.mean(),D.Hpaper.mean()))
    print(' AC1 mean %.3f  pos share %.1f  zero share %.2f  returns/day %.1f' % (D.ac1day.mean(),(D.ac1day>0).mean()*100,D.nzero.sum()/D.n.sum()*100,D.n.mean()))
    print(' negative ac5 days',(D.ac5<0).sum(),' ac10',(D.ac10<0).sum())
