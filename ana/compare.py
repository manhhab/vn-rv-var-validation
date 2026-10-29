import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0,'ana'); from tests import dm, mcs
tag=sys.argv[1]
NM=['RV1','TSRV','RV5','RV10','RK','AC5','GARCH']
for n in ['VNINDEX','VN30']:
    R=pickle.load(open(f'data/R_{tag}_{n}.pkl','rb'))
    D=pd.read_pickle(f'data/{n}_daily.pkl'); oos=D.iloc[401:].copy()
    print('=====',n)
    for mp in ('par','fhs'):
        pooled={m:np.mean([R[(m,mp,a)][k]<0.05 for a in (0.05,0.025,0.01) for k in ['uc_p','ind_p','cc_p','dq_p','ues_p','ces_p','multi_p']])*100 for m in NM}
        print(mp,'pooled rej %', {k:round(v,1) for k,v in pooled.items()})
        print(mp,'IND rejections', sum(R[(m,mp,a)]['ind_p']<0.05 for m in NM for a in (0.05,0.025,0.01)), 'at 1%:', sum(R[(m,mp,0.01)]['ind_p']<0.05 for m in NM))
        for a in (0.05,0.025,0.01):
            L=np.column_stack([R[(m,mp,a)]['loss'] for m in NM])
            el,pv,alive=mcs(L,NM)
            d1=dm(R[('AC5',mp,a)]['loss'],R[('RV1',mp,a)]['loss']); dk=dm(R[('RK',mp,a)]['loss'],R[('RV1',mp,a)]['loss'])
            dg=dm(R[('RK',mp,a)]['loss'],R[('GARCH',mp,a)]['loss']); d10=dm(R[('RV10',mp,a)]['loss'],R[('RV1',mp,a)]['loss'])
            fz={m:round(R[(m,mp,a)]['fz0'],3) for m in NM}
            print(f'  {mp} {a}: elim {el} p {[round(pv[e],3) for e in el]} | DM RK-RV1 {dk:.2f} RV10-RV1 {d10:.2f} AC5-RV1 {d1:.2f} RK-GARCH {dg:.2f} | {fz}')
    # Table 10: share of exceptions in top quintile
    for key in ['rv10','lag10']:
        z = D.rv10 if key=='rv10' else D.rv10.shift(1)
        zo=z.iloc[401:]; top=(zo>=zo.quantile(0.8)).values
        row={}
        for m in NM:
            for a in (0.025,0.01):
                h=R[(m,'par',a)]['hits'].astype(bool)
                row[(m,a)]=round(top[h].mean()*100) if h.sum()>0 else np.nan
        print(' top-quintile share',key,row)
