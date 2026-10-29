import pandas as pd, numpy as np
for n in ['VNINDEX','VN30']:
    df=pd.read_pickle(f'data/{n}.pkl'); df['t']=pd.to_datetime(df.TradingDate)
    df['d']=df.t.dt.normalize(); df['hm']=df.t.dt.hour*100+df.t.dt.minute
    post=df[df.d>='2025-05-05']; pre=df[(df.d>='2022-06-09')&(df.d<'2025-05-05')]
    print('=====',n)
    for nm,x in [('pre',pre),('post',post)]:
        hm=x.hm.value_counts().sort_index()
        nd=x.d.nunique()
        print(nm,'days',nd,'first hm',hm.index.min(),'last',hm.index.max())
        full=hm[hm==nd].index
        print('  minutes present every day:',len(full), 'min',full.min(),'max',full.max())
        # gaps in continuous trading grid
        print('  hm present < all days in [915,1130]U[1300,1430]:', hm[((hm.index>=915)&(hm.index<=1130))|((hm.index>=1300)&(hm.index<=1430))].lt(nd).sum())
        # ATC
        print('  rows at >=1431:', (x.hm>1430).sum(), ' rows <915:', (x.hm<915).sum())
        # zero returns in continuous grid
        y=x[((x.hm>=915)&(x.hm<=1130))|((x.hm>=1300)&(x.hm<=1430))].copy()
        y['r']=np.log(y.CloseIndex).groupby(y.d).diff()
        print('  zero-return share (cont.)', round((y.r==0).mean()*100,2), ' Open==Close share', round((y.OpenIndex==y.CloseIndex).mean()*100,2))
    # check the post sample: index value at 14:30 vs official close? compare 1430 close & any 1445
    s=df[df.d=='2025-05-06']; print(s[['t','OpenIndex','CloseIndex','HighestIndex','LowestIndex','ReferenceIndex','TotalMatchVolume']].tail(4).to_string())
    s=df[df.d=='2025-04-29']; print(s[['t','OpenIndex','CloseIndex','HighestIndex','LowestIndex','ReferenceIndex','TotalMatchVolume']].tail(25).to_string())
