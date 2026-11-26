"""Revision experiments (METR-LA, 10-node subgraphs), answering the review:
  A  node-permutation sensitivity of M4 (+ forced feedback) and an orientation-averaged circuit (M4-sym)
  B  C4-off: the same MLP architecture trained offline on the first 70% of the timeline
  C  lag-augmented inputs for M4 / C1 / C2 (raw two-step lag and ridge-forecast change)
  D  decision accuracy with near-ties among neighbours removed
  E  why M1 is more accurate on full networks than on 10-node subgraphs
usage: QBN_DATA=... python -I run_revision.py results/revision.pkl
"""
import sys, os, pickle, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from exp import *
import exp as EXP
from run_main import setup_units, ridge_forecast

OUT = sys.argv[1]
GAMMA = 0.9
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

X, A = load_dataset("metr-la")
subs, units, vk, tk = setup_units(X, A)
P = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "main_metr.pkl"), "rb"))
best = P["best"]
R = dict()


# ------------------------------------------------------------------------------------------------
# extra score classes
# ------------------------------------------------------------------------------------------------
class SymCircuit(Score):
    """orientation-averaged circuit: mean of the circuit score over K node orderings."""
    def __init__(self, W, perms):
        self.n = self.P = W.shape[0]
        self.perms = perms
        self.bns = [BNCircuit(W[np.ix_(p, p)]) for p in perms]

    def eval(self, ctx, t):
        alpha = np.pi * ctx.F[t] + self.params
        Qs, Js = np.zeros(self.n), np.zeros((self.n, self.n))
        for p, bn in zip(self.perms, self.bns):
            Q, J = bn.QJ(alpha[p])
            inv = np.argsort(p)
            Qs += Q[inv]
            Js += J[np.ix_(inv, inv)]
        K = len(self.perms)
        return 2 * Qs / K - 1, 2 * Js / K


class FrozenMLP(MLPScore):
    """C4-off: MLP with offline-trained weights; no online parameter updates."""
    def __init__(self, W, params):
        super().__init__(W)
        self.fixed = np.array(params)

    def init(self, rng):
        self.params = self.fixed.copy()


def build_rev(name, W, cfg, extra):
    c = dict(cfg)
    if name == "M4sym":
        return dict(score=SymCircuit(W, extra["perms"]), lam_rule="signed", theta_rule="full", **c)
    if name == "C4off":
        rule = c.pop("rule")
        if rule == "fixed":
            return dict(score=FrozenMLP(W, extra["mlp"]), lam_rule="fixed", lam0=c["lam0"])
        return dict(score=FrozenMLP(W, extra["mlp"]), lam_rule="signed", theta_rule="none",
                    lam0=c["lam0"], eta_lam=c["eta_lam"])
    base = name.split("-")[0]
    return build(base, W, c)


def run_rev(args):
    name, cfg, ukey, seed, extra = args
    u = EXP._UNITS[ukey]
    ctx = Ctx(u["U"], u["F"] if not (extra and "Fkey" in extra) else u[extra["Fkey"]], u["M"], u["gamma"], u["tod"])
    if name == "F1":
        out = forecast_estimator(ctx, u["T"], u["xhat"])
    elif name == "C3":
        out = run_estimator(ctx, u["T"], score=None, offset=extra["offset"])
    else:
        kw = build_rev(name, u["W"], cfg, extra or {})
        out = run_estimator(ctx, u["T"], seed=seed, **kw)
    m = metrics(out["V"], u["Vs"], u["M"], mtd=out["mtd"], lam=out["lam"])
    m["qoffset"] = out["qterm"].mean(0).tolist()
    if extra and extra.get("keep"):
        m["V"] = out["V"]
    return (name, json.dumps(cfg, sort_keys=True), ukey, seed, m)


def pmap(jobs, U, procs=2):
    from multiprocessing import Pool
    with Pool(procs, initializer=EXP._init, initargs=(U,)) as p:
        return p.map(run_rev, jobs, chunksize=max(1, len(jobs) // 16))


def collect(res):
    out = {}
    for (m, c, k, s, met) in res:
        out.setdefault(m, {}).setdefault(k, {})[s] = met
    return out


def tune_rev(models, grids, U, extra=None):
    jobs = [(m, cfg, k, 0, (extra or {}).get(m)) for m in models for cfg in grids[m] for k in vk]
    agg = {}
    for (m, c, k, s, met) in pmap(jobs, U):
        agg.setdefault((m, c), []).append(met["mae"])
    b = {}
    for m in models:
        cands = sorted([(np.mean(v), json.loads(c)) for (mm, c), v in agg.items() if mm == m], key=lambda z: z[0])
        b[m] = dict(cfg=cands[0][1], valmae=cands[0][0])
    return b


def test_rev(models, b, U, extra=None, seeds=(0, 1, 2)):
    jobs = [(m, b[m]["cfg"] if m in b else {}, k, s, (extra or {}).get(m)) for m in models for k in tk for s in seeds]
    return collect(pmap(jobs, U))


# ------------------------------------------------------------------------------------------------
# A. node-permutation sensitivity
# ------------------------------------------------------------------------------------------------
NPERM = 5
prng = np.random.default_rng(2026)
perm_res = {}
for p in range(NPERM + 1):
    Up = {}
    # one permutation per subgraph and per p
    sgperm = {i: (np.arange(10) if p == 0 else prng.permutation(10)) for i in range(len(subs))}
    for k in vk + tk:
        u = units[k]
        i = int(k.split("_")[0][2:])
        pi = sgperm[i]
        S = list(np.array(u["S"])[pi])
        Up[k] = make_unit(X, A, S, u["start"], u["T"], GAMMA)
        Up[k]["xhat"] = u["xhat"][:, pi]
    cfg_ff = dict(best["M4"]["cfg"]); cfg_ff["eta_theta"] = 0.1
    bp = {"M1": dict(cfg={}), "M4": best["M4"], "M4-ff": dict(cfg=cfg_ff)}
    r = test_rev(["M1", "M4", "M4-ff"], bp, Up)
    # matched offset in permuted ordering
    EXP._init(Up)
    r["C3"] = {k: {s: run_rev(("C3", {}, k, s, dict(offset=np.array(r["M4"][k][s]["qoffset"]))))[4]
                   for s in r["M4"][k]} for k in tk}
    perm_res[p] = dict(res=r, perm={i: sgperm[i].tolist() for i in sgperm})
    log("perm", p, {m: np.mean(v) for m, v in unit_table(r).items()})
R["perm"] = perm_res

# orientation-averaged circuit (M4-sym), K=6 orderings (identity + 5 random), tuned
symperms = {i: [np.arange(10)] + [np.random.default_rng(7 + 31 * i + j).permutation(10) for j in range(5)]
            for i in range(len(subs))}
G4 = GRIDS["M4"]
# extra must be per-unit (perms depend on subgraph) -> pass via unit dict instead
for k in vk + tk:
    units[k]["symperms"] = symperms[int(k.split("_")[0][2:])]


# simple route: run M4sym with perms of each unit through a dedicated extra
def sym_jobs(cfgs, keys, seeds):
    return [("M4sym", c, k, s, dict(perms=units[k]["symperms"])) for c in cfgs for k in keys for s in seeds]


res = pmap(sym_jobs(G4, vk, (0,)), units)
agg = {}
for (m, c, k, s, met) in res:
    agg.setdefault(c, []).append(met["mae"])
cbest = min(agg.items(), key=lambda z: np.mean(z[1]))
symbest = json.loads(cbest[0])
R["M4sym"] = dict(cfg=symbest, res=collect(pmap(sym_jobs([symbest], tk, (0, 1, 2)), units))["M4sym"])
log("M4sym", symbest, unit_table({"x": R["M4sym"]["res"]})["x"].mean())

# ------------------------------------------------------------------------------------------------
# B. C4-off: offline supervised training of the C4 architecture on the first 70% of the timeline
# ------------------------------------------------------------------------------------------------
cut = int(0.7 * len(X))
valranges = [(units[k]["start"], units[k]["start"] + units[k]["T"] + extra_steps(GAMMA)) for k in vk]
mlp = MLPScore(np.zeros((10, 10)))
FE, TG = [], []
for i, S in enumerate(subs):
    W = A[np.ix_(S, S)]
    M = nb_mask(W)
    U = X[:cut][:, S]
    Vs = vstar(U, M, GAMMA)
    mx = maxnb(Vs, M)                                     # (T, n): max_{N+(i)} V*_j(t)
    q = np.zeros_like(U)
    q[1:-1] = GAMMA * (mx[2:] - mx[:-2])                  # ideal reward correction g_i(t)
    sc = MLPScore(W)
    ctx = Ctx(U, U, M, GAMMA, tod_of(0, cut))
    keep = np.ones(cut, bool)
    keep[:12] = False
    keep[-80:] = False
    for (a, b) in valranges:
        keep[max(a - 12, 0):b] = False
    ts = np.where(keep)[0]
    feats = np.stack([sc.features(ctx, t) for t in ts])   # (Tk, n, 7)
    FE.append(feats.reshape(-1, 7)); TG.append(q[ts].reshape(-1))
FE = np.concatenate(FE); TG = np.concatenate(TG)
log("C4-off training samples", FE.shape, "target std", TG.std())


def fit_mlp(Fx, y, h=8, epochs=30, bs=4096, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    d = Fx.shape[1]
    th = dict(W1=rng.normal(0, 1 / np.sqrt(d), (d, h)), b1=np.zeros(h), w2=rng.normal(0, 0.1, h), b2=np.zeros(1),
              ls=np.array([np.log(max(np.abs(y).max(), 1e-3))]))
    mom = {k: np.zeros_like(v) for k, v in th.items()}; vel = {k: np.zeros_like(v) for k, v in th.items()}
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(len(y))
        for s0 in range(0, len(y), bs):
            j = idx[s0:s0 + bs]
            x, t = Fx[j], y[j]
            H = np.tanh(x @ th["W1"] + th["b1"]); z = H @ th["w2"] + th["b2"][0]; s = np.tanh(z)
            L = np.exp(th["ls"][0]); pred = L * s; e = pred - t
            gp = 2 * e / len(t)
            gs = gp * L; gz = gs * (1 - s ** 2)
            g = dict(w2=H.T @ gz, b2=np.array([gz.sum()]), ls=np.array([(gp * pred).sum()]))
            gH = np.outer(gz, th["w2"]) * (1 - H ** 2)
            g["W1"] = x.T @ gH; g["b1"] = gH.sum(0)
            step += 1
            for k in th:
                mom[k] = 0.9 * mom[k] + 0.1 * g[k]; vel[k] = 0.999 * vel[k] + 0.001 * g[k] ** 2
                th[k] -= lr * (mom[k] / (1 - 0.9 ** step)) / (np.sqrt(vel[k] / (1 - 0.999 ** step)) + 1e-8)
    H = np.tanh(Fx @ th["W1"] + th["b1"]); pred = np.exp(th["ls"][0]) * np.tanh(H @ th["w2"] + th["b2"][0])
    r2 = 1 - ((pred - y) ** 2).mean() / y.var()
    params = np.concatenate([th["W1"].ravel(), th["b1"], th["w2"], th["b2"]])
    return params, float(np.exp(th["ls"][0])), float(r2)


mlp_params, lam_hat, r2_train = fit_mlp(FE, TG)
log("C4-off fitted: scale", lam_hat, "train R2", r2_train)
# out-of-sample R2 on test windows
ys, ps = [], []
for k in tk:
    u = units[k]; T = u["T"]
    mx = maxnb(u["Vs"], u["M"]); q = GAMMA * (mx[2:T + 2] - mx[0:T]); q = q[11:T - 1]
    sc = FrozenMLP(u["W"], mlp_params); sc.init(None)
    ctx = Ctx(u["U"], u["F"], u["M"], GAMMA, u["tod"])
    pr = np.stack([sc.eval(ctx, t)[0] for t in range(12, T)]) * lam_hat
    ys.append(q.reshape(-1)); ps.append(pr.reshape(-1))
ys = np.concatenate(ys); ps = np.concatenate(ps)
r2_test = float(1 - ((ys - ps) ** 2).mean() / ys.var())
log("C4-off test R2", r2_test)
grid_off = ([dict(rule="fixed", lam0=f * lam_hat) for f in (0.25, 0.5, 1.0, 1.5)] +
            [dict(rule="signed", lam0=l, eta_lam=e) for l in (0.05, lam_hat) for e in (0.05, 0.2, 1.0)])
ex = {"C4off": dict(mlp=mlp_params)}
b_off = tune_rev(["C4off"], {"C4off": grid_off}, units, extra=ex)
r_off = test_rev(["C4off"], b_off, units, extra=ex)
R["C4off"] = dict(cfg=b_off["C4off"], res=r_off["C4off"], lam_hat=lam_hat, r2_train=r2_train, r2_test=r2_test,
                  nsamples=int(len(TG)))
log("C4off", b_off, unit_table(r_off)["C4off"].mean())

# ------------------------------------------------------------------------------------------------
# C. lag-augmented score inputs
# ------------------------------------------------------------------------------------------------
CENC = 5.0
for k in vk + tk:
    u = units[k]; st = u["start"]; L = len(u["U"]); S = u["S"]
    x = X[st:st + L][:, S]; x2 = X[st - 2:st - 2 + L][:, S]
    u["F_lag"] = np.clip(0.5 + CENC * (x - x2), 0, 1)                   # two-step momentum from x(t), x(t-2)
    u["F_fc"] = np.clip(0.5 + CENC * (u["xhat"] - x), 0, 1)             # ridge forecast change (12 own + 3 nbr lags)
lag_models = []
for enc in ("lag", "fc"):
    for m in ("M4", "C1", "C2"):
        lag_models.append(f"{m}-{enc}")
ex = {m: dict(Fkey="F_" + m.split("-")[1]) for m in lag_models}
b_lag = tune_rev(lag_models, {m: GRIDS["M4"] for m in lag_models}, units, extra=ex)
r_lag = test_rev(lag_models, b_lag, units, extra=ex)
R["lag"] = dict(best=b_lag, res=r_lag, cenc=CENC)
log("lag", {m: (b_lag[m]["cfg"], unit_table(r_lag)[m].mean()) for m in lag_models})

# ------------------------------------------------------------------------------------------------
# D. ties: re-run main models keeping V
# ------------------------------------------------------------------------------------------------
bD = dict(best); bD["M1"] = dict(cfg={})
mainm = ["M1", "M2", "M4", "C1", "C2", "C4"]
exk = {m: dict(keep=True) for m in mainm + ["F1"]}
rD = test_rev(mainm + ["F1"], bD, units, extra=exk, seeds=(0,))
EXP._init(units)
rD["C3"] = {k: {0: run_rev(("C3", {}, k, 0, dict(offset=np.array(rD["M4"][k][0]["qoffset"]), keep=True)))[4]} for k in tk}
D = {}
for eps in (1e-6, 1e-3, 1e-2):
    for m in rD:
        acc_all, acc_nt, acc_tol, frac_tie = [], [], [], []
        for k in tk:
            u = units[k]; T = u["T"]; M = u["M"]
            V = rD[m][k][0]["V"][12:]; Vs = u["Vs"][12:T]
            Zs = np.where(M[None], Vs[:, None, :], -np.inf)
            srt = np.sort(Zs, axis=2)
            gap = srt[:, :, -1] - srt[:, :, -2]
            gap[~np.isfinite(gap)] = np.inf
            ah = argmax_nb(V, M); ast = argmax_nb(Vs, M)
            ok = ah == ast
            chosen = np.take_along_axis(Zs, ah[:, :, None], 2)[:, :, 0]
            tol_ok = chosen >= srt[:, :, -1] - eps
            nt = gap > eps
            acc_all.append(ok.mean()); acc_nt.append(ok[nt].mean()); acc_tol.append(tol_ok.mean()); frac_tie.append(1 - nt.mean())
        D[(m, eps)] = dict(acc=np.mean(acc_all), acc_nt=np.mean(acc_nt), acc_tol=np.mean(acc_tol), tie=np.mean(frac_tie),
                           acc_nt_u=np.array(acc_nt), acc_tol_u=np.array(acc_tol))
R["ties"] = D
# fraction of exact ties in M1's own estimate and fraction of saturated speeds
sat = np.mean([(units[k]["U"][12:units[k]["T"]] >= 1 - 1e-12).mean() for k in tk])
R["sat_speed_sub"] = float(sat)
log("ties", {f"{m}@{e}": (round(v['acc'], 3), round(v['acc_nt'], 3), round(v['acc_tol'], 3), round(v['tie'], 3)) for (m, e), v in D.items() if e == 1e-3})

# ------------------------------------------------------------------------------------------------
# E. full network vs subgraph: neighbourhood size and saturation of the neighbourhood maximum
# ------------------------------------------------------------------------------------------------
Mf = nb_mask(A)
tsamp = np.arange(cut, len(X) - 1, 7)
xf = X[tsamp]
mxf = maxnb(xf, Mf)
stats_full = dict(deg=float(Mf.sum(1).mean() - 1), satmax=float((mxf >= 1 - 1e-12).mean()),
                  dmax=float(np.abs(np.diff(maxnb(X[cut:cut + 2016], Mf), axis=0)).mean()),
                  dx=float(np.abs(np.diff(X[cut:cut + 2016], axis=0)).mean()))
sm, dm, dg = [], [], []
for S in subs:
    W = A[np.ix_(S, S)]; M = nb_mask(W)
    xs = X[tsamp][:, S]
    sm.append((maxnb(xs, M) >= 1 - 1e-12).mean())
    dm.append(np.abs(np.diff(maxnb(X[cut:cut + 2016][:, S], M), axis=0)).mean())
    dg.append(M.sum(1).mean() - 1)
stats_sub = dict(deg=float(np.mean(dg)), satmax=float(np.mean(sm)), dmax=float(np.mean(dm)))
R["scale_stats"] = dict(full=stats_full, sub=stats_sub)
log("scale", R["scale_stats"])

pickle.dump(R, open(OUT, "wb"))
log("done")
