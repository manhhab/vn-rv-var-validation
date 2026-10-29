import pandas as pd, numpy as np, pickle, json, re
rng=np.random.default_rng(42)
def sbb_idx(T,L):
    idx=np.empty(T,int); i=0
    while i<T:
        s=rng.integers(T); l=rng.geometric(1/L)
        for j in range(l):
            if i>=T: break
            idx[i]=(s+j)%T; i+=1
    return idx
TB={}
EST=[('rv1','Realized variance, 1 minute'),('rv2','Realized variance, 2 minutes'),('rv3','Realized variance, 3 minutes'),('rv5','Realized variance, 5 minutes'),('rv10','Realized variance, 10 minutes'),('rv15','Realized variance, 15 minutes'),('rv30','Realized variance, 30 minutes'),
     ('ac1','RV-AC(1)'),('ac2','RV-AC(2)'),('ac3','RV-AC(3)'),('ac5','RV-AC(5)'),('ac10','RV-AC(10)'),('rk','Realized kernel (Parzen)'),('tsrv_adj','Two-scale estimator'),('bpv','Bipower variation')]
T1=[];T2=[];T3={};T5={}
for n in ['VNINDEX','VN30']:
    D=pd.read_pickle(f'data/{n}_daily.pkl'); raw=pd.read_pickle(f'data/{n}.pkl'); raw=raw[pd.to_datetime(raw.TradingDate)>='2022-06-09']
    B=D.bench.mean()
    T1.append([n,f'{len(D)}','Jun 9, 2022','Sep 25, 2026',f'{len(raw):,}'.replace(',',' '),f'{D.n.mean():.0f}',f'{np.sqrt(252*B)*100:.2f}%'])
    T2.append([n]+[f'{np.sqrt(252*D[f"rv{m}"].mean())*100:.2f}' for m in (1,2,3,5,10,15,30)]+[f'{np.sqrt(252*B)*100:.2f}'])
    X=D[[c for c,_ in EST]].values; b=D.bench.values; T=len(D)
    pt=X.mean(0)/b.mean()-1
    bs=np.array([ (lambda i: X[i].mean(0)/b[i].mean()-1)(sbb_idx(T,10)) for _ in range(2000)])
    lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    T3[n]=[(f'{p*100:+.1f}'.replace('-','−'),f'[{l*100:.1f}, {h*100:.1f}]'.replace('-','−')) for p,l,h in zip(pt,lo,hi)]
    x=D.copy(); x['lag10']=x.rv10.shift(1); x=x.dropna(subset=['lag10']); x['q']=pd.qcut(x.lag10,5,labels=False)
    g=x.groupby('q')
    T5[n]=dict(ac=[f'{v:+.3f}'.replace('-','−') for v in g.ac1day.mean()],
               sf=[f'{v:+.1f}'.replace('-','−') for v in ((g.rv1.mean()/g.bench.mean())**.5-1)*100],
               vol=[f'{v:.1f}' for v in np.sqrt(252*g.bench.mean())*100])
    y=D.copy(); y['q']=pd.qcut(y.rv10,5,labels=False); gy=y.groupby('q')
    T5[n]['sameday_sf']=[round(v,1) for v in ((gy.rv1.mean()/gy.bench.mean())**.5-1)*100]
TB['T1']=T1; TB['T2']=T2; TB['T3']=dict(names=[l for _,l in EST],**T3); TB['T5']=T5
json.dump(TB,open('data/tables_meas.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(TB,ensure_ascii=False)[:3000])
