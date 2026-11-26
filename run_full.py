"""Full-graph study (all 207 METR-LA sensors / all 325 PEMS-BAY sensors) made possible by Theorem 2:
the circuit's Z-basis statistics are sampled classically by ancestral sampling of the equivalent
Bayesian network in O(n + |E|) per sample, with Rao-Blackwellised scores and a likelihood-ratio
Jacobian estimated from the same samples."""
import pickle, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from exp import *
from exp import _init, run_one
import exp


class SampledBNScore(Score):
    def __init__(self, W, samples=1000, rng=None):
        self.n = self.P = n = W.shape[0]
        Beta = np.triu(np.pi * W / 2.0, k=1)
        self.pa = [np.nonzero(Beta[:, t])[0] for t in range(n)]
        self.beta = [Beta[self.pa[t], t] for t in range(n)]
        R = (Beta > 0).astype(np.int64)                          # ancestor mask by transitive closure
        anc = R.copy()
        while True:
            new = ((anc @ R) > 0).astype(np.int64) | anc
            if (new == anc).all():
                break
            anc = new
        self.anc = anc.T.astype(bool)                            # anc[k, m]: m is an ancestor of k
        self.S = samples
        self.rng = rng or np.random.default_rng(0)
        self.nedges = int((Beta > 0).sum())

    def eval(self, ctx, t):
        alpha = np.pi * ctx.F[t] + self.params
        n, S = self.n, self.S
        B = np.zeros((S, n))
        A = np.empty((S, n))
        U = self.rng.random((S, n))
        for k in range(n):
            a = alpha[k] + (B[:, self.pa[k]] @ self.beta[k] if len(self.pa[k]) else 0.0)
            A[:, k] = a
            B[:, k] = U[:, k] < np.sin(a / 2) ** 2
        p0 = np.cos(A / 2) ** 2                                  # Rao-Blackwellised P(b_k=0 | parents)
        Q = p0.mean(0)
        half = A / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(B > 0, 1 / np.tan(half), -np.tan(half))   # d log p(b_m | pa) / d alpha_m
        sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
        J = ((p0 - Q).T @ sc) / S                                # LR estimate, J[k, m]
        J = np.where(self.anc, J, 0.0)
        np.fill_diagonal(J, (-np.sin(A) / 2).mean(0))
        return 2 * Q - 1, 2 * J


def build_full(name, W, cfg, seed=0, noise=None):
    if name in ("M4", "M3", "M2"):
        sc = SampledBNScore(W, rng=np.random.default_rng(20_000 + seed))
        if name == "M2":
            return dict(score=sc, lam_rule="fixed", lam0=cfg["lam0"])
        if name == "M3":
            return dict(score=sc, lam_rule="signed", lam0=cfg["lam0"], eta_lam=cfg["eta_lam"])
        return dict(score=sc, lam_rule="signed", theta_rule="full", **cfg)
    return exp._orig_build(name, W, cfg, seed, noise)


def ridge_forecast_windows(X, A, cut, starts, L, lags=12, reg=1e-2, stride=5):
    n = X.shape[1]
    Wn = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    nbr = X @ Wn.T

    def feats(ts):
        F = [X[ts - l] for l in range(lags)] + [nbr[ts - l] for l in range(3)]
        tod = (ts % 288) / 288
        F += [np.repeat(np.sin(2 * np.pi * tod)[:, None], n, 1), np.repeat(np.cos(2 * np.pi * tod)[:, None], n, 1),
              np.ones((len(ts), n))]
        return np.stack(F, -1)

    tr = np.arange(lags, cut - 1, stride)
    Ft = feats(tr).reshape(-1, lags + 6); y = X[tr + 1].reshape(-1)
    w = np.linalg.solve(Ft.T @ Ft + reg * np.eye(Ft.shape[1]), Ft.T @ y)
    out = {}
    for st in starts:
        ts = np.arange(st, st + L)
        out[st] = np.clip(feats(np.maximum(ts, lags)) @ w, 0, 1)
    return out


def main(dataset="metr-la", out="results/full_metr.pkl", n_test="12"):
    t0 = time.time()
    exp._orig_build = exp.build
    exp.build = build_full
    X, A = load_dataset(dataset)
    gamma = 0.9; T = 288; extra = extra_steps(gamma)
    val, tst = draw_windows(len(X), n_val=3, n_test=int(n_test), T=T, extra=extra, seed=777)
    S = list(range(X.shape[1]))
    xh = ridge_forecast_windows(X, A, int(0.7 * len(X)), val + tst, T + extra)
    units, vk, tk = {}, [], []
    for j, st in enumerate(val):
        k = f"full_val{j}"; units[k] = make_unit(X, A, S, st, T, gamma, extra=extra); units[k]["xhat"] = xh[st]; vk.append(k)
    for j, st in enumerate(tst):
        k = f"full_test{j}"; units[k] = make_unit(X, A, S, st, T, gamma, extra=extra); units[k]["xhat"] = xh[st]; tk.append(k)
    grids = dict(GRIDS)
    small = [dict(lam0=0.05, eta_lam=a, eta_theta=b) for a in ETA_LAM for b in [0.0, 0.02, 0.1]]
    for m in ["M4", "C1", "C2", "C4"]:
        grids[m] = small
    grids["M3"] = [dict(lam0=0.05, eta_lam=a) for a in ETA_LAM]
    models = ["M3", "M4", "C1", "C2", "C4"]
    best = tune(models, vk, units, grids=grids)
    print(best, f"{time.time() - t0:.0f}s", flush=True)
    res = test(["M1"] + models + ["F1"], best, tk, units)
    _init(units)
    res["C3"] = {k: {s: run_one(("C3", {}, k, s, dict(offset=np.array(res["M4"][k][s]["qoffset"]))))[4]
                     for s in res["M4"][k]} for k in tk}
    nE = int((np.triu(A, 1) > 0).sum())
    pickle.dump(dict(best=best, res=res, test_keys=tk, n=X.shape[1], nedges=nE, dataset=dataset), open(out, "wb"))
    t = unit_table(res)
    print({m: round(float(t[m].mean()), 4) for m in t}, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
