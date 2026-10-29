import pandas as pd, numpy as np
for n in ['VNINDEX','VN30']:
    df=pd.read_pickle(f'data/{n}.pkl')
    df['t']=pd.to_datetime(df.TradingDate)
    print('=====',n, df.shape, df.t.min(), df.t.max())
    print('dup RecordId', df.RecordId.duplicated().sum(), 'dup ts', df.t.duplicated().sum())
    print('null', df.isna().sum()[df.isna().sum()>0].to_dict())
    print('sorted?', df.t.is_monotonic_increasing)
    df=df.sort_values('t')
    df['d']=df.t.dt.date; df['hm']=df.t.dt.hour*100+df.t.dt.minute
    print('seconds nonzero', (df.t.dt.second!=0).sum())
    m=df.groupby(df.t.dt.to_period('M')).size(); print(m.to_string())
    hm=df.hm.value_counts().sort_index()
    print('hm range', hm.index.min(), hm.index.max())
    print(hm[(hm.index<916)|(hm.index>1125)&(hm.index<1305)|(hm.index>1425)].to_string())
    print('closeindex<=0', (df.CloseIndex<=0).sum())
