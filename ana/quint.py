import pandas as pd, numpy as np
for n in ['VNINDEX','VN30']:
    D=pd.read_pickle(f'data/{n}_daily.pkl').copy()
    D['lag10']=D.rv10.shift(1)
    D['har']=D.rv10.shift(1).rolling(5).mean()  # simple ex-ante proxy
    print('=====',n)
    for key in ['rv10','lag10','har']:
        x=D.dropna(subset=[key]).copy()
        x['q']=pd.qcut(x[key],5,labels=False)
        g=x.groupby('q')
        t=pd.DataFrame({'ac1':g.ac1day.mean().round(3),
            'sf_rv10':((g.rv1.mean()/g.rv10.mean())**.5-1).mul(100).round(1),
            'sf_bench':((g.rv1.mean()/g.bench.mean())**.5-1).mul(100).round(1),
            'vol':(np.sqrt(252*g.rv10.mean())*100).round(1),
            'volb':(np.sqrt(252*g.bench.mean())*100).round(1)})
        print(' sort',key); print(t.to_string())
