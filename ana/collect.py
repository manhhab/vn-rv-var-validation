import pickle, json, numpy as np, pandas as pd, sys
sys.path.insert(0,'ana'); from tests import dm, mcs
NM=['RV1','TSRV','RV5','RV10','RK','AC5','GARCH']
out={}
for n in ['VNINDEX','VN30']:
    R=pickle.load(open(f'data/R_Bforecast400_{n}.pkl','rb'))
    o={}
    for (m,mp,a),v in R.items():
        o[f'{m}|{mp}|{a}']={k:(float(v[k]) if k not in('basel','loss','hits') else (v[k] if k=='basel' else None)) for k in v if k not in('loss','hits')}
        o[f'{m}|{mp}|{a}']['nexc']=int(v['hits'].sum())
    for mp in ('par','fhs'):
        for a in (0.05,0.025,0.01):
            L=np.column_stack([R[(m,mp,a)]['loss'] for m in NM])
            el,pv,alive=mcs(L,NM)
            o[f'MCS|{mp}|{a}']=dict(elim=el,p={k:float(v) for k,v in pv.items()})
            o[f'DM|{mp}|{a}']={f'{x}-{y}':float(dm(R[(x,mp,a)]['loss'],R[(y,mp,a)]['loss'])) for x,y in [('RK','RV1'),('RV10','RV1'),('AC5','RV1'),('RK','GARCH'),('RV10','GARCH')]}
    out[n]=o
json.dump(out,open('data/results.json','w'),indent=1)
for n in out:
    print(n)
    for a in (0.05,0.025,0.01):
        for mp in ('par','fhs'):
            print(' ',mp,a,out[n][f'MCS|{mp}|{a}'],{k:round(v,2) for k,v in out[n][f'DM|{mp}|{a}'].items()})
