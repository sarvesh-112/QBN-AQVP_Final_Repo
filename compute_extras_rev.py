"""Decision-gap analysis (M1, M2, M4) and full-network vs subgraph statistics for the revision.
usage: QBN_DATA=... python -I compute_extras_rev.py results/extras_rev.pkl"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from exp import *
import exp as EXP
from run_main import setup_units
X, A = load_dataset("metr-la"); subs, units, vk, tk = setup_units(X, A)
best = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "main_metr.pkl"), "rb"))["best"]
EXP._init(units)
out = {}
gaps = []
acc = {m: {b: [] for b in ("a", "b", "c")} for m in ("M1", "M2", "M4")}
for k in tk:
    u = units[k]; T = u["T"]; M = u["M"]; Vs = u["Vs"][12:T]
    Z = np.where(M[None], Vs[:, None, :], -np.inf); s = np.sort(Z, 2)
    g = s[:, :, -1] - s[:, :, -2]; g[~np.isfinite(g)] = np.inf
    gaps.append(g.ravel())
    bins = {"a": g < 0.02, "b": (g >= 0.02) & (g < 0.1), "c": g >= 0.1}
    for m in acc:
        oks = []
        for sd in (0, 1, 2):
            r = run_one((m, best[m]["cfg"] if m != "M1" else {}, k, sd, dict(keep=True)))[4]
            oks.append(argmax_nb(r["V"][12:], M) == argmax_nb(Vs, M))
        for b, msk in bins.items():
            acc[m][b].append(np.mean([o[msk].mean() for o in oks]) if msk.any() else np.nan)
g = np.concatenate(gaps); gf = g[np.isfinite(g)]
out["gap_med"] = float(np.median(gf)); out["gap_lt002"] = float((g < 0.02).mean()); out["gap_ge01"] = float((g >= 0.1).mean())
out["acc"] = {m: {b: float(np.nanmean(v)) for b, v in d.items()} for m, d in acc.items()}
# full network vs subgraphs
cut = int(0.7 * len(X)); ts = np.arange(cut, len(X) - 1, 5); sat = X[ts] >= 1 - 1e-12
def reach(M, h):
    R = M.astype(int); P = M.astype(int)
    for _ in range(h - 1):
        P = (P @ M.astype(int) > 0).astype(int); R = ((R + P) > 0).astype(int)
    return R > 0
Mf = nb_mask(A); Rf = reach(Mf, 3)
out["reach_full"] = float(np.mean([sat[:, Rf[i]].any(1).mean() for i in range(len(A))]))
fs = []
for S in subs:
    Rs = reach(nb_mask(A[np.ix_(S, S)]), 3); xs = sat[:, S]
    fs.append(np.mean([xs[:, Rs[i]].any(1).mean() for i in range(10)]))
out["reach_sub"] = float(np.mean(fs))
fm = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "full_metr.pkl"), "rb"))
vm = []
rng = np.random.default_rng(0)
for st in rng.integers(cut, len(X) - 400, 6):
    vm.append(vstar(X[st:st + 348], Mf, 0.9)[:288].mean())
out["vmean_full"] = float(np.mean(vm)); out["vmean_sub"] = float(np.mean([units[k]["Vs"][:288].mean() for k in tk]))
pickle.dump(out, open(sys.argv[1], "wb")); print(out)
