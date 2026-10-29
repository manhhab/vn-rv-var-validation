"""Robustness: fat-tailed parametric mappings (Student t) for the HAR models."""
import numpy as np, pandas as pd, pickle, sys
from scipy import stats, optimize
sys.path.insert(0, 'ana')
from forecast import har_forecasts, ALPHAS
from evalrun import evaluate

def fit_t_unitvar(z):
    """unit-variance Student t, nu by ML (scale fixed by unit variance)."""
    def nll(nu):
        s = np.sqrt((nu - 2) / nu)
        return -np.sum(stats.t.logpdf(z / s, nu) - np.log(s))
    r = optimize.minimize_scalar(nll, bounds=(2.05, 200), method='bounded')
    nu = r.x
    return nu, np.sqrt((nu - 2) / nu)

def fit_t_free(z):
    """location fixed at 0, scale and nu by ML."""
    def nll(p):
        nu, ls = p
        if nu <= 2.05 or nu > 200: return 1e10
        return -np.sum(stats.t.logpdf(z, nu, scale=np.exp(ls)))
    r = optimize.minimize(nll, x0=[6.0, np.log(np.std(z) * 0.8)], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6})
    return r.x[0], np.exp(r.x[1])

def t_es(nu, q):
    # ES of standard t at quantile q (left tail), alpha = cdf(q)
    a = stats.t.cdf(q, nu)
    return -(nu + q ** 2) / (nu - 1) * stats.t.pdf(q, nu) / a

def run(D, col, W=400):
    r = D['ret'].values; x = D[col].values; T = len(r); start = W + 1
    fc = har_forecasts(x, W, start)
    out = {m: {a: dict(var=[], es=[], u=[]) for a in ALPHAS} for m in ('tfix', 'tfree')}
    nus = {'tfix': [], 'tfree': []}; scales = []
    for t in range(start, T):
        xt, idx, fit = fc[t]
        rw = r[idx]; mu = rw.mean(); sd = np.sqrt(xt)
        ok = ~np.isnan(fit); z = (rw[ok] - mu) / np.sqrt(fit[ok])
        nu1, s1 = fit_t_unitvar(z); nu2, s2 = fit_t_free(z)
        nus['tfix'].append(nu1); nus['tfree'].append(nu2); scales.append(s2)
        zt = (r[t] - mu) / sd
        for m, nu, s in (('tfix', nu1, s1), ('tfree', nu2, s2)):
            for a in ALPHAS:
                q = stats.t.ppf(a, nu)
                out[m][a]['var'].append(mu + sd * s * q)
                out[m][a]['es'].append(mu + sd * s * t_es(nu, q))
                out[m][a]['u'].append(stats.t.cdf(zt / s, nu))
    for m in out:
        for a in ALPHAS:
            for k in out[m][a]: out[m][a][k] = np.array(out[m][a][k])
    return out, r[start:], {k: np.array(v) for k, v in nus.items()}, np.array(scales)

if __name__ == '__main__':
    n = sys.argv[1]
    M = [('RV1', 'rv1'), ('TSRV', 'tsrv'), ('RV5', 'rv5'), ('RV10', 'rv10'), ('RK', 'rk'), ('AC5', 'ac5')]
    D = pd.read_pickle(f'data/{n}_daily.pkl').copy(); D['ret'] = D.rct
    D['tsrv'] = D.tsrv_adj.clip(lower=D.rv10 * 0.05)
    R = {}; info = {}
    for nm, col in M:
        res, y, nus, sc = run(D, col)
        info[nm] = dict(nu_fix=float(np.median(nus['tfix'])), nu_free=float(np.median(nus['tfree'])), scale=float(np.median(sc)))
        for mp in ('tfix', 'tfree'):
            for a in ALPHAS:
                R[(nm, mp, a)] = evaluate(y, res[mp][a]['var'], res[mp][a]['es'], res[mp][a]['u'], a)
    pickle.dump((R, info), open(f'data/R_t_{n}.pkl', 'wb'))
    print(n, 'done', info)
