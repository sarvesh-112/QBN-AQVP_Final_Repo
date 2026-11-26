"""Statistics: bootstrap CIs, paired tests, Holm-Bonferroni, rank-biserial effect sizes, cluster bootstrap."""
import numpy as np
from scipy.stats import wilcoxon, rankdata

B = 5000


def boot_ci(x, rng=None, b=B):
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def paired_rel_ci(a, b, rng=None, nb=B):
    """CI of 100*(mean(a)/mean(b) - 1) under paired resampling of units."""
    rng = rng or np.random.default_rng(1)
    idx = rng.integers(0, len(a), (nb, len(a)))
    r = 100 * (a[idx].mean(1) / b[idx].mean(1) - 1)
    return np.percentile(r, [2.5, 97.5])


def cluster_rel_ci(a, b, clusters, rng=None, nb=B):
    """two-stage cluster bootstrap (subgraphs, then windows within) of 100*(mean a / mean b - 1)."""
    rng = rng or np.random.default_rng(2)
    cl = np.unique(clusters)
    groups = [np.nonzero(clusters == c)[0] for c in cl]
    out = np.empty(nb)
    for i in range(nb):
        pick = rng.integers(0, len(cl), len(cl))
        ids = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in (groups[p] for p in pick)])
        out[i] = 100 * (a[ids].mean() / b[ids].mean() - 1)
    return np.percentile(out, [2.5, 97.5])


def wil(a, b):
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def rank_biserial(a, b):
    """matched-pairs rank-biserial correlation; positive when a > b (a worse for error metrics)."""
    d = a - b
    d = d[np.abs(d) > 1e-15]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    if p < 1e-4:
        return r"\ensuremath{<10^{-4}}"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return rf"\ensuremath{{{p / 10 ** e:.1f}\times10^{{{e}}}}}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
