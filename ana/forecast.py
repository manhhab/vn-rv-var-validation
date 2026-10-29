import numpy as np, pandas as pd, warnings
from scipy import stats, integrate
from arch import arch_model
warnings.filterwarnings('ignore')
ALPHAS = [0.05, 0.025, 0.01]


def har_design(x):
    l = np.log(x)
    s = pd.Series(l)
    d = s.shift(1); w = s.rolling(5).mean().shift(1); m = s.rolling(22).mean().shift(1)
    return l, np.column_stack([np.ones(len(l)), d, w, m])


def har_forecasts(x, W, start):
    """returns dict t -> (xhat_{t}, fitted xhat for window days s in window (one-step, in-sample))"""
    l, X = har_design(x)
    T = len(x)
    out = {}
    for t in range(start, T):
        idx = np.arange(t - W, t)
        ok = idx[~np.isnan(X[idx]).any(1)]
        b, *_ = np.linalg.lstsq(X[ok], l[ok], rcond=None)
        e = l[ok] - X[ok] @ b
        s2 = e.var(ddof=X.shape[1])
        xt = np.exp(X[t] @ b + s2 / 2)
        fit = np.full(len(idx), np.nan)
        fit[np.isin(idx, ok)] = np.exp(X[ok] @ b + s2 / 2)
        out[t] = (xt, idx, fit)
    return out


def gauss_var_es(mu, sd, a):
    q = stats.norm.ppf(a)
    return mu + sd * q, mu - sd * stats.norm.pdf(q) / a


def emp_var_es(z, a):
    z = np.sort(z)
    q = np.quantile(z, a)
    tail = z[z <= q]
    return q, tail.mean()


def run_realized(df, col, W=400, start=None, fhs_mode='forecast'):
    """df: daily with 'ret' and measure col. fhs_mode: 'forecast' (standardize by fitted one-step forecasts)
    or 'expost' (standardize by realized measure, as in draft)."""
    r = df['ret'].values; x = df[col].values
    T = len(r); start = W + 1 if start is None else start
    fc = har_forecasts(x, W, start)
    res = {m: {a: dict(var=[], es=[], u=[]) for a in ALPHAS} for m in ('par', 'fhs')}
    xhat = []
    for t in range(start, T):
        xt, idx, fit = fc[t]
        xhat.append(xt)
        rw = r[idx]; mu = rw.mean(); sd = np.sqrt(xt)
        if fhs_mode == 'forecast':
            ok = ~np.isnan(fit); z = (rw[ok] - mu) / np.sqrt(fit[ok])
        else:
            z = (rw - mu) / np.sqrt(x[idx])
        for a in ALPHAS:
            v, e = gauss_var_es(mu, sd, a)
            res['par'][a]['var'].append(v); res['par'][a]['es'].append(e)
            res['par'][a]['u'].append(stats.norm.cdf((r[t] - mu) / sd))
            q, tm = emp_var_es(z, a)
            res['fhs'][a]['var'].append(mu + sd * q); res['fhs'][a]['es'].append(mu + sd * tm)
            res['fhs'][a]['u'].append((np.sum(z <= (r[t] - mu) / sd) + 0.5) / (len(z) + 1))
    for m in res:
        for a in ALPHAS:
            for k in res[m][a]:
                res[m][a][k] = np.array(res[m][a][k])
    return res, r[start:], np.array(xhat)


def skewt_es(dist, params, a, q):
    f = lambda p: dist.ppf(p, params)
    val, _ = integrate.quad(lambda p: float(np.asarray(dist.ppf(np.array([p]), params))[0]), 1e-9, a, limit=200)
    return val / a


def run_garch(df, W=400, start=None, refit=10):
    r = df['ret'].values * 100
    T = len(r); start = W + 1 if start is None else start
    out = {a: dict(var=[], es=[], u=[]) for a in ALPHAS}
    params = None; res = None
    for k, t in enumerate(range(start, T)):
        idx = np.arange(t - W, t)
        if k % refit == 0:
            am = arch_model(r[idx], mean='Constant', vol='GARCH', p=1, q=1, dist='skewt')
            res = am.fit(disp='off', show_warning=False)
            params = res.params
            dist = am.distribution
            dparams = params[['eta', 'lambda']].values
            qs = {a: float(np.asarray(dist.ppf(np.array([a]), dparams))[0]) for a in ALPHAS}
            esz = {a: skewt_es(dist, dparams, a, qs[a]) for a in ALPHAS}
        # filter variance through current window with fixed params
        mu, om, al, be = params['mu'], params['omega'], params['alpha[1]'], params['beta[1]']
        e = r[t - W:t] - mu
        s2 = np.var(e)
        for ee in e:
            s2 = om + al * ee ** 2 + be * s2
        sd = np.sqrt(s2)
        zt = (r[t] - mu) / sd
        u = float(np.asarray(dist.cdf(np.array([zt]), dparams))[0])
        for a in ALPHAS:
            out[a]['var'].append((mu + sd * qs[a]) / 100); out[a]['es'].append((mu + sd * esz[a]) / 100)
            out[a]['u'].append(u)
    for a in ALPHAS:
        for kk in out[a]:
            out[a][kk] = np.array(out[a][kk])
    return out, df['ret'].values[start:]
