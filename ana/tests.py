import numpy as np
from scipy import stats


def kupiec(I, a):
    n = len(I); x = I.sum(); pi = x / n
    if x == 0:
        lr = -2 * (n * np.log(1 - a))
    elif x == n:
        lr = -2 * n * np.log(a)
    else:
        lr = -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return lr, stats.chi2.sf(lr, 1)


def christoffersen(I, a):
    I = I.astype(int)
    a0, a1 = I[:-1], I[1:]
    n00 = np.sum((a0 == 0) & (a1 == 0)); n01 = np.sum((a0 == 0) & (a1 == 1))
    n10 = np.sum((a0 == 1) & (a1 == 0)); n11 = np.sum((a0 == 1) & (a1 == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(k, q):
        return k * np.log(q) if k > 0 else 0.0
    l0 = ll(n00 + n10, 1 - p) + ll(n01 + n11, p)
    l1 = ll(n00, 1 - p01) + ll(n01, p01) + ll(n10, 1 - p11) + ll(n11, p11)
    lr_ind = max(-2 * (l0 - l1), 0.0)
    lr_uc, _ = kupiec(I, a)
    lr_cc = lr_uc + lr_ind
    return lr_ind, stats.chi2.sf(lr_ind, 1), lr_cc, stats.chi2.sf(lr_cc, 2)


def dq(I, var, a, lags=4):
    H = I - a
    T = len(H)
    X = [np.ones(T - lags)]
    for l in range(1, lags + 1):
        X.append(H[lags - l:T - l])
    X.append(var[lags:])
    X = np.column_stack(X); y = H[lags:]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    s = b @ X.T @ X @ b / (a * (1 - a))
    return s, stats.chi2.sf(s, X.shape[1])


def acerbi(r, I, var, es, a):
    n = len(r)
    nx = I.sum()
    # ES reported as a negative number: Z = 1 - mean(r/ES | exception), etc.
    z1 = (1 - np.sum(r * I / es) / nx) if nx > 0 else 0.0
    z2 = 1 - np.sum(r * I / (n * a * es))
    return z1, z2


def du_escanciano(u, a, lags=4):
    n = len(u)
    h = (a - u) * (u < a) / a
    uz = np.sqrt(n) * (h.mean() - a / 2) / np.sqrt(a * (1 / 3 - a / 4))
    p_u = 2 * stats.norm.sf(abs(uz))
    hc = h - a / 2
    g0 = np.mean(hc ** 2)
    rho = [np.mean(hc[j:] * hc[:-j]) / g0 for j in range(1, lags + 1)]
    ces = n * np.sum(np.square(rho))
    return uz, p_u, ces, stats.chi2.sf(ces, lags)


def multinomial(u, a, N=4):
    lv = [a * (N - j) / N for j in range(N)]  # a, .75a, .5a, .25a
    edges = [0] + sorted(lv) + [1.0]
    O = np.histogram(u, bins=edges)[0]
    probs = np.diff(edges)
    E = len(u) * probs
    m = O > 0
    lr = 2 * np.sum(O[m] * np.log(O[m] / E[m]))
    return lr, stats.chi2.sf(lr, N)


def basel(I_1pct):
    n = len(I_1pct); x = I_1pct.sum() * 250 / n
    x = round(x, 6)
    return 'Green' if x <= 4 else ('Yellow' if x <= 9 else 'Red'), x


def fz0(r, v, e, a):
    return -(1 / (a * e)) * (r <= v) * (v - r) + v / e + np.log(-e) - 1


def nw_var(d, L=None):
    T = len(d)
    if L is None:
        L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    dc = d - d.mean()
    v = np.dot(dc, dc) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        v += 2 * w * np.dot(dc[l:], dc[:-l]) / T
    return v


def dm(l1, l2):
    d = l1 - l2
    return d.mean() / np.sqrt(nw_var(d) / len(d))


def mcs(L, names, alpha=0.10, B=1200, block=5, seed=1):
    """Hansen-Lunde-Nason MCS, range statistic T_R. L: T x m loss matrix. returns eliminated list (order) and p-values."""
    rng = np.random.default_rng(seed)
    T, m = L.shape
    nb = int(np.ceil(T / block))
    idxs = []
    for _ in range(B):
        st = rng.integers(0, T - block + 1, nb)
        idxs.append(np.concatenate([np.arange(s, s + block) for s in st])[:T])
    idxs = np.array(idxs)
    alive = list(range(m)); elim = []; pv = {}; pmax = 0.0
    while len(alive) > 1:
        X = L[:, alive]
        dbar = X.mean(0)
        dij = dbar[:, None] - dbar[None, :]
        boot = np.array([X[i].mean(0) for i in idxs])  # B x k
        bd = boot[:, :, None] - boot[:, None, :]
        var = np.mean((bd - dij[None]) ** 2, axis=0)
        np.fill_diagonal(var, 1.0)
        t = dij / np.sqrt(var)
        TR = np.max(np.abs(t))
        tb = (bd - dij[None]) / np.sqrt(var)[None]
        TRb = np.max(np.abs(tb).reshape(B, -1), axis=1)
        p = np.mean(TRb >= TR)
        pmax = max(pmax, p)
        if p >= alpha:
            break
        # eliminate worst: max_j t_ij
        ti = np.max(t, axis=1)
        w = alive[int(np.argmax(ti))]
        elim.append(names[w]); pv[names[w]] = pmax
        alive.remove(w)
    return elim, pv, [names[i] for i in alive]
