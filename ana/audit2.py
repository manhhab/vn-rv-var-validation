import pandas as pd, numpy as np
for n in ['VNINDEX','VN30']:
    df=pd.read_pickle(f'data/{n}.pkl'); df['t']=pd.to_datetime(df.TradingDate)
    df['d']=df.t.dt.normalize(); df['hm']=df.t.dt.hour*100+df.t.dt.minute
    c=df.groupby('d').size()
    print('=====',n)
    print(c['2025-01-20':'2025-06-10'].to_string())
    late=c['2025-05-15':]
    print('post May-2025 per-day counts unique:', late.value_counts().to_dict())
    early=c['2022-06-09':'2025-04-30']
    print('pre counts describe', early.describe().round(1).to_dict())
