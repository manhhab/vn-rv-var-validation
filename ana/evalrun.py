import numpy as np, pandas as pd, pickle, sys
sys.path.insert(0,'ana')
from forecast import *; from tests import *
def evaluate(I_r, var, es, u, a):
    I=(I_r<=var).astype(int)
    out=dict(rate=I.mean()*100)
    out['uc_lr'],out['uc_p']=kupiec(I,a)
    _,out['ind_p'],_,out['cc_p']=christoffersen(I,a)
    _,out['dq_p']=dq(I,var,a)
    out['z1'],out['z2']=acerbi(I_r,I,var,es,a)
    out['ues'],out['ues_p'],_,out['ces_p']=du_escanciano(u,a)
    _,out['multi_p']=multinomial(u,a)
    out['fz0']=fz0(I_r,var,es,a).mean()
    out['nrej']=sum(out[k]<0.05 for k in ['uc_p','ind_p','cc_p','dq_p','ues_p','ces_p','multi_p'])
    out['basel']=basel(I)[0] if a==0.01 else ''
    out['loss']=fz0(I_r,var,es,a)
    out['hits']=I
    return out
def run_all(D, models, W=400, fhs_mode='forecast', garch=True):
    R={}
    for nm,col in models:
        res,y,xh=run_realized(D,col,W=W,fhs_mode=fhs_mode)
        for mp in ('par','fhs'):
            for a in ALPHAS:
                R[(nm,mp,a)]=evaluate(y,res[mp][a]['var'],res[mp][a]['es'],res[mp][a]['u'],a)
    if garch:
        g,y=run_garch(D,W=W)
        for mp in ('par','fhs'):
            for a in ALPHAS:
                R[('GARCH',mp,a)]=evaluate(y,g[a]['var'],g[a]['es'],g[a]['u'],a)
    return R
