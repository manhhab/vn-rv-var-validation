import pickle, numpy as np, pandas as pd
for n in ['VNINDEX','VN30']:
    o=pickle.load(open(f'data/{n}_B.pkl','rb'))
    # pooled AC1 by 30-min block: sum r_i r_{i-1} / sum r_i^2 within block
    num={};den={}
    zero={'pre':[0,0],'post':[0,0]}
    for d,v in o.items():
        reg='post' if d>=pd.Timestamp('2025-05-05') else 'pre'
        for k,(p,start) in enumerate(zip(v['paths'],[9*60+15,13*60])):
            r=np.diff(p); mins=start+1+np.arange(len(r))
            zero[reg][0]+=np.sum(r==0); zero[reg][1]+=len(r)
            for i in range(1,len(r)):
                b=(mins[i]//30)*30
                num[b]=num.get(b,0)+r[i]*r[i-1]; den[b]=den.get(b,0)+r[i]**2
    print(n,{f'{b//60}:{b%60:02d}':round(num[b]/den[b],3) for b in sorted(num)})
    print('  zero share pre %.2f post %.2f'%(zero['pre'][0]/zero['pre'][1]*100,zero['post'][0]/zero['post'][1]*100))
    D=pd.read_pickle(f'data/{n}_daily.pkl')
    for reg,x in [('pre',D[D.index<'2025-05-05']),('post',D[D.index>='2025-05-05'])]:
        print('  ',reg,'AC1 %.3f pos %.1f bias rv1 %.1f rv10 %.1f rk %.1f'%(x.ac1day.mean(),(x.ac1day>0).mean()*100,(x.rv1.mean()/x.bench.mean()-1)*100,(x.rv10.mean()/x.bench.mean()-1)*100,(x.rk.mean()/x.bench.mean()-1)*100))
