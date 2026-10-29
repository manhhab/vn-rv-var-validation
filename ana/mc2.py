import numpy as np, sys
sys.path.insert(0,'ana')
exec(open('ana/mc.py').read().split('res={}')[0])
for u in (0.19,0.31):
    x=np.array([one(670,0.01,1-u) for _ in range(1500)]); print('power 1% understate',u,np.round(x.mean(0)*100,1))
