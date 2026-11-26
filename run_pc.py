"""Positive control: semi-synthetic METR-LA in which the score input is an exogenous congestion
advisory that carries information about future value.

advisory e_i(t) = clip(1/2 - c * g_i(t) + sigma * eps_i(t), 0, 1),
g_i(t) = gamma * ( max_{j in N+(i)} V*_j(t+1) - max_{j in N+(i)} V*_j(t-1) ),
i.e. a noisy, bounded encoding of the two-step change in optimal neighbourhood value that the
lagging classical estimator misses (an idealised incident / congestion-onset feed).  'shuffled'
replaces the signal by pure noise of the same spread (no information)."""
import pickle, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from exp import *
from exp import _init
from run_main import setup_units

C_ADV = 2.0
LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4, "shuffled"]
MODELS = ["M3", "M4", "C1", "C2"]


def attach_signal(units, gamma=0.9, seed=0):
    rng = np.random.default_rng(seed)
    for k in sorted(units):
        u = units[k]
        mx = maxnb(u["Vs"], u["M"])
        g = np.zeros_like(u["Vs"]); g[1:-1] = gamma * (mx[2:] - mx[:-2])
        u["G"] = g
        u["eps"] = rng.normal(size=g.shape)


def set_level(units, sig):
    for u in units.values():
        if sig == "shuffled":
            sd = (C_ADV * u["G"]).std()
            e = 0.5 + np.sqrt(sd ** 2 + 0.1 ** 2) * u["eps"]
        else:
            e = 0.5 - C_ADV * u["G"] + sig * u["eps"]
        u["F"] = np.clip(e, 0, 1)


def ceiling_adv(units, keys):
    r2 = []
    for k in keys:
        u = units[k]; T = u["T"]
        V = run_estimator(Ctx(u["U"], u["F"], u["M"], u["gamma"]), T)["V"]
        e = (u["Vs"][:T] - V)[12:].ravel()
        a = u["F"][:T][12:].ravel(); x = u["U"][:T][12:].ravel()
        Z = np.stack([np.ones_like(a), a, x, a * x], 1)
        b, *_ = np.linalg.lstsq(Z, e, rcond=None)
        r2.append(1 - ((e - Z @ b) ** 2).sum() / ((e - e.mean()) ** 2).sum())
    return np.array(r2)


def main(out="results/pc_metr.pkl"):
    t0 = time.time()
    X, A = load_dataset("metr-la")
    subs, units, vk, tk = setup_units(X, A)
    attach_signal(units)
    res = {}
    for sig in LEVELS:
        set_level(units, sig)
        best = tune(MODELS, vk, units)
        r = test(["M1"] + MODELS, best, tk, units)
        res[str(sig)] = dict(best=best, res=r, ceiling=ceiling_adv(units, tk))
        t = unit_table(r)
        print(sig, {m: round(float(t[m].mean()), 4) for m in t}, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(dict(levels=[str(s) for s in LEVELS], out=res, test_keys=tk, c=C_ADV), open(out, "wb"))


if __name__ == "__main__":
    main(*sys.argv[1:])
