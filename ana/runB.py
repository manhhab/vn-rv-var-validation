import pandas as pd, pickle, sys
sys.path.insert(0,'ana'); from evalrun import run_all
mode=sys.argv[1]; W=int(sys.argv[2]) if len(sys.argv)>2 else 400
M=[('RV1','rv1'),('TSRV','tsrv'),('RV5','rv5'),('RV10','rv10'),('RK','rk'),('AC5','ac5')]
for n in ['VNINDEX','VN30']:
    D=pd.read_pickle(f'data/{n}_daily.pkl').copy(); D['ret']=D.rct
    D['tsrv']=D.tsrv_adj.clip(lower=D.rv10*0.05)
    R=run_all(D,M,W=W,fhs_mode=mode,garch=True)
    pickle.dump(R,open(f'data/R_B{mode}{W}_{n}.pkl','wb'))
print('done')
