"""Main controlled study (METR-LA or PEMS-BAY, 10-node subgraphs).
usage: python run_main.py --dataset metr-la --out results/main_metr.pkl [--gamma 0.9 --T 288] [--quick]"""
import argparse, pickle, time, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from exp import *
from exp import _init


def ridge_forecast(X, A, S, cut, lags=12, reg=1e-2):
    """pooled one-step-ahead ridge forecaster trained on [0, cut); returns xhat[t] ~ x(t+1)."""
    Xs = X[:, S]
    W = A[np.ix_(S, S)]
    Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    nbr = Xs @ Wn.T
    Tn, n = Xs.shape
    tod = (np.arange(Tn) % 288) / 288
    feats = []
    for l in range(lags):
        feats.append(np.roll(Xs, l, axis=0))
    for l in range(3):
        feats.append(np.roll(nbr, l, axis=0))
    feats.append(np.repeat(np.sin(2 * np.pi * tod)[:, None], n, 1))
    feats.append(np.repeat(np.cos(2 * np.pi * tod)[:, None], n, 1))
    feats.append(np.ones((Tn, n)))
    Fz = np.stack(feats, -1)                                     # (T, n, d)
    tr = np.arange(lags, cut - 1)
    Xtr = Fz[tr].reshape(-1, Fz.shape[-1]); ytr = Xs[tr + 1].reshape(-1)
    d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + reg * np.eye(d), Xtr.T @ ytr)
    xhat = np.clip(Fz @ w, 0, 1)
    xhat[:lags] = Xs[:lags]
    return xhat, w


def setup_units(X, A, T=288, gamma=0.9, n_sub=5, seed=0, feat=None, split=0.7, n_test=6):
    subs = greedy_subgraphs(A, k=n_sub, seed=seed)
    extra = extra_steps(gamma)
    units, val_keys, test_keys = {}, [], []
    cut = int(split * len(X))
    for i, S in enumerate(subs):
        val, tst = draw_windows(len(X), n_val=3, n_test=n_test, T=T, extra=extra, seed=100 + i)
        xh, _ = ridge_forecast(X, A, S, cut)
        for j, st in enumerate(val):
            k = f"sg{i}_val{j}"; units[k] = make_unit(X, A, S, st, T, gamma, feat=feat, extra=extra); val_keys.append(k)
            units[k]["xhat"] = xh[st:st + T + extra]
        for j, st in enumerate(tst):
            k = f"sg{i}_test{j}"; units[k] = make_unit(X, A, S, st, T, gamma, feat=feat, extra=extra); test_keys.append(k)
            units[k]["xhat"] = xh[st:st + T + extra]
    return subs, units, val_keys, test_keys


NOISE = [("Shots S=100", dict(shots=100)), ("Shots S=1000", dict(shots=1000)), ("Shots S=10000", dict(shots=10000)),
         ("Depol. p=0.001", dict(depol=0.001)), ("Depol. p=0.01", dict(depol=0.01)),
         ("S=1000, p=0.01", dict(shots=1000, depol=0.01))]


def ceiling(units, test_keys, res_m1):
    """hindsight in-sample R^2 of the classical error on instantaneous node features."""
    r2 = []
    for k in test_keys:
        u = units[k]; T = u["T"]
        ctx = Ctx(u["U"], u["F"], u["M"], u["gamma"])
        V = run_estimator(ctx, T)["V"]
        e = (u["Vs"][:T] - V)[12:]
        x = u["F"][:T]; W = u["W"]; Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
        nb = x @ Wn.T
        dx = np.vstack([np.zeros((1, x.shape[1])), np.diff(x, axis=0)])
        Z = np.stack([np.ones_like(x), x, nb, dx, x * dx, nb * dx], -1)[12:].reshape(-1, 6)
        y = e.reshape(-1)
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2.append(1 - ((y - Z @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr-la")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--T", type=int, default=288)
    ap.add_argument("--models", default="M1,M2,M3,M4,M5,C1,C2,C4,Lmono,Lleaky,Lsoft,Rdiag,V1,V1C")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--best_from", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    t0 = time.time()
    X, A = load_dataset(a.dataset)
    subs, units, val_keys, test_keys = setup_units(X, A, T=a.T, gamma=a.gamma)
    if a.quick:
        val_keys, test_keys = val_keys[:3], test_keys[:2]
    models = a.models.split(",")
    tuned = [m for m in models if m != "M1"]
    if a.best_from:
        best = pickle.load(open(a.best_from, "rb"))["best"]
    else:
        best = tune(tuned, val_keys, units, procs=a.procs)
    best["M1"] = dict(cfg={}, valmae=np.mean([run_one(("M1", {}, k, 0, None))[4]["mae"] for k in val_keys]) if a.procs == 1 else None)
    print({m: best[m] for m in best}, flush=True)
    extra = {"M4": dict(cos=True), "Rdiag": dict(cos=True)}
    seeds = tuple(int(z) for z in a.seeds.split(","))
    res = test(models, best, test_keys, units, procs=a.procs, extra=extra, seeds=seeds)
    # C3: matched constant offset from M4 in the same run
    _init(units)
    res["C3"] = {}
    for k in test_keys:
        res["C3"][k] = {}
        for s in res["M4"][k]:
            off = np.array(res["M4"][k][s]["qoffset"])
            res["C3"][k][s] = run_one(("C3", {}, k, s, dict(offset=off)))[4]
    res["F1"] = test(["F1"], {}, test_keys, units, procs=a.procs, seeds=seeds)["F1"]
    out = dict(subs=subs, best=best, res=res, test_keys=test_keys, val_keys=val_keys,
               nedges=[int((np.triu(A[np.ix_(S, S)], 1) > 0).sum()) for S in subs])
    # M1 validation MAE
    out["M1val"] = float(np.mean([run_one(("M1", {}, k, 0, None))[4]["mae"] for k in val_keys]))
    out["F1val"] = float(np.mean([run_one(("F1", {}, k, 0, None))[4]["mae"] for k in val_keys]))
    out["ceiling"] = ceiling(units, test_keys, res["M1"])
    if a.noise:
        nres = {}
        for lab, nz in NOISE:
            r = test(["M4"], best, test_keys, units, procs=a.procs, extra={"M4": dict(noise=nz)}, seeds=seeds)
            nres[lab] = r["M4"]
            print("noise", lab, time.time() - t0, flush=True)
        out["noise"] = nres
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(out, open(a.out, "wb"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
