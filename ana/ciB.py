import pandas as pd, numpy as np
rng=np.random.default_rng(42)
def sbb_idx(T,L,rng):
    idx=np.empty(T,int); i=0
    while i<T:
        s=rng.integers(T); l=rng.geometric(1/L)
        for j in range(l):
            if i>=T: break
            idx[i]=(s+j)%T; i+=1
    return idx
cols=['rv1','rv2','rv3','rv5','rv10','rv15','rv30','ac1','ac2','ac3','ac5','ac10','rk','rk_paper','tsrv','tsrv_adj','bpv']
for n in ['VNINDEX','VN30']:
    D=pd.read_pickle(f'data/{n}_daily.pkl'); T=len(D)
    X=D[cols].values; b=D.bench.values
    pt=X.mean(0)/b.mean()-1
    bs=[]
    for k in range(2000):
        i=sbb_idx(T,10,rng); bs.append(X[i].mean(0)/b[i].mean()-1)
    bs=np.array(bs); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    print('=====',n)
    for c,p,l,h in zip(cols,pt,lo,hi): print(f'  {c:9s} {p*100:6.1f}  [{l*100:6.1f}, {h*100:6.1f}]')
    # differences: rv5 - ac5 ; rk - rv5 ; rv1 - rv5
    for a,c in [('ac5','rv5'),('ac2','rv5'),('rk','rv5'),('rv10','rv5')]:
        ia,ic=cols.index(a),cols.index(c)
        dd=bs[:,ia]-bs[:,ic]; print(f'  diff {a}-{c}: {(pt[ia]-pt[ic])*100:.1f} [{np.percentile(dd,2.5)*100:.1f},{np.percentile(dd,97.5)*100:.1f}]')
    # by year and regime
    D['yr']=D.index.year; D['reg']=np.where(D.index>='2025-05-05','post','pre')
    for key in ['yr','reg']:
        g=D.groupby(key)
        print(' ',key,{k:(round((v.rv1.mean()/v.bench.mean()-1)*100,1),round((v.rv5.mean()/v.bench.mean()-1)*100,1),round((v.ac5.mean()/v.bench.mean()-1)*100,1),round(v.ac1day.mean(),3),len(v)) for k,v in g})
