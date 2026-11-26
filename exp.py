"""Experiment framework: units, model zoo, tuning on validation windows, paired testing."""
import itertools, os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from core import *

LAM0 = [0.05, 0.428]
ETA_LAM = [0.05, 0.2, 1.0]
ETA_TH = [0.0, 0.005, 0.02, 0.1]


def tod_of(start, L, steps_per_day=288):
    return ((start + np.arange(L)) % steps_per_day) / steps_per_day


def make_unit(X, A, S, start, T, gamma, feat=None, extra=None):
    extra = extra if extra is not None else extra_steps(gamma)
    W = A[np.ix_(S, S)]
    M = nb_mask(W)
    U = X[start:start + T + extra][:, S]
    F = U if feat is None else feat[start:start + T + extra][:, S]
    Vs = vstar(U, M, gamma)
    return dict(W=W, M=M, U=U, F=F, Vs=Vs, T=T, gamma=gamma, start=start, S=S,
                tod=tod_of(start, T + extra))


# --------------------------------------------------------------------------------------------
# model zoo: name -> (builder(W, cfg, seed) -> kwargs for run_estimator, grid)
# --------------------------------------------------------------------------------------------
def _grid(*axes):
    keys = ["lam0", "eta_lam", "eta_theta"][:len(axes)]
    return [dict(zip(keys, v)) for v in itertools.product(*axes)]


def build(name, W, cfg, seed=0, noise=None):
    c = dict(cfg)
    if name == "M1":
        return dict(score=None)
    if name == "M2":
        return dict(score=CircuitScore(W), lam_rule="fixed", lam0=c["lam0"])
    if name == "M3":
        return dict(score=CircuitScore(W), lam_rule="signed", theta_rule="none", lam0=c["lam0"], eta_lam=c["eta_lam"])
    if name == "M4":
        sc = CircuitScore(W)
        if noise:
            sc = CircuitScore(W, shots=noise.get("shots"), depol=noise.get("depol", 0.0),
                              rng=np.random.default_rng(10_000 + seed), affine=noise.get("affine"))
        return dict(score=sc, lam_rule="signed", theta_rule="full", **c)
    if name == "M5":
        return dict(score=ReadoutScore(W), lam_rule="signed", theta_rule="full", **c)
    if name == "C1":
        return dict(score=ProductScore(W), lam_rule="signed", theta_rule="full", **c)
    if name == "C2":
        return dict(score=ProductScore(W, mix=True), lam_rule="signed", theta_rule="full", **c)
    if name == "C4":
        return dict(score=MLPScore(W), lam_rule="signed", theta_rule="full", **c)
    if name in ("Lmono", "Lleaky", "Lsoft"):
        return dict(score=CircuitScore(W), lam_rule=name[1:], theta_rule="full", **c)
    if name == "Rdiag":
        return dict(score=CircuitScore(W), lam_rule="signed", theta_rule="diag", track=True, **c)
    if name == "V1":
        return dict(score=CircuitScore(W, centred=False), lam_rule="mono", theta_rule="diag", track=True, **c)
    if name == "V1C":
        return dict(score=ProductScore(W, centred=False), lam_rule="mono", theta_rule="diag", track=True, **c)
    raise ValueError(name)


GRIDS = {
    "M1": [dict()],
    "M2": [dict(lam0=l) for l in [0.01, 0.05, 0.1, 0.25, 0.5]],
    "M3": _grid(LAM0, ETA_LAM),
    "V1": [dict(lam0=0.428, eta_lam=a, eta_theta=b) for a in [0.01, 0.05, 0.2] for b in [0.005, 0.02, 0.1]],
}
for m in ["M4", "M5", "C1", "C2", "C4", "Lmono", "Lleaky", "Lsoft", "Rdiag"]:
    GRIDS[m] = _grid(LAM0, ETA_LAM, ETA_TH)
GRIDS["V1C"] = GRIDS["V1"]
GRIDS["Lsoft"] = _grid(LAM0, [0.05], ETA_TH)          # soft-target rule has no eta_lam


# --------------------------------------------------------------------------------------------
_UNITS = {}


def _init(units):
    global _UNITS
    _UNITS = units


def run_one(args):
    name, cfg, ukey, seed, extra = args
    u = _UNITS[ukey]
    ctx = Ctx(u["U"], u["F"], u["M"], u["gamma"], u["tod"])
    if name == "F1":
        out = forecast_estimator(ctx, u["T"], u["xhat"])
    elif name == "C3":
        out = run_estimator(ctx, u["T"], score=None, offset=extra["offset"])
    else:
        noise = dict(extra["noise"]) if extra and extra.get("noise") else None
        if noise and "affine_by_sg" in noise:
            noise["affine"] = noise.pop("affine_by_sg")[int(ukey.split("_")[0][2:])]
        kw = build(name, u["W"], cfg, seed, noise=noise)
        out = run_estimator(ctx, u["T"], seed=seed, record_grads=bool(extra and extra.get("cos")), **kw)
    m = metrics(out["V"], u["Vs"], u["M"], mtd=out["mtd"], lam=out["lam"])
    m["qoffset"] = out["qterm"].mean(0).tolist()
    if extra and extra.get("keep"):
        m["V"] = out["V"]; m["lamtraj"] = out["lam"]; m["mtdtraj"] = out["mtd"]
    if "cos" in out:
        m["cos"] = out["cos"]
    return (name, json.dumps(cfg, sort_keys=True), ukey, seed, m)


def pool_map(jobs, units, procs=2):
    if procs == 1:
        _init(units)
        return [run_one(j) for j in jobs]
    with Pool(procs, initializer=_init, initargs=(units,)) as p:
        return p.map(run_one, jobs, chunksize=max(1, len(jobs) // (procs * 8)))


def tune(models, val_keys, units, procs=2, grids=None, log=print):
    grids = grids or GRIDS
    jobs = [(m, cfg, k, 0, None) for m in models for cfg in grids[m] for k in val_keys]
    t0 = time.time()
    res = pool_map(jobs, units, procs)
    agg = {}
    for (m, cfgs, k, s, met) in res:
        agg.setdefault((m, cfgs), []).append(met["mae"])
    best = {}
    for m in models:
        cands = [(np.mean(v), json.loads(c)) for (mm, c), v in agg.items() if mm == m]
        cands.sort(key=lambda z: z[0])
        best[m] = dict(cfg=cands[0][1], valmae=cands[0][0])
    log(f"tuned {models} on {len(val_keys)} val units in {time.time() - t0:.0f}s")
    return best


def test(models, best, test_keys, units, seeds=(0, 1, 2), procs=2, extra=None):
    jobs = [(m, best[m]["cfg"] if m in best else {}, k, s, (extra or {}).get(m)) for m in models
            for k in test_keys for s in seeds]
    res = pool_map(jobs, units, procs)
    out = {}
    for (m, c, k, s, met) in res:
        out.setdefault(m, {}).setdefault(k, {})[s] = met
    return out


def unit_table(res, key="mae"):
    """model -> array over units (seed-averaged), units sorted."""
    tab = {}
    for m, d in res.items():
        ks = sorted(d.keys())
        tab[m] = np.array([np.mean([d[k][s][key] for s in d[k]]) for k in ks])
    return tab
