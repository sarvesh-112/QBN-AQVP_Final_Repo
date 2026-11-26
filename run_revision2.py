"""Revision part 2: offline-trained circuit (M4-off) and classical scores (C1-off, C2-off),
fitted on the first 70% of the timeline exactly like C4-off (target: ideal reward correction).
usage: QBN_DATA=... python -I run_revision2.py results/revision2.pkl"""
import sys, os, pickle, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from exp import *
import exp as EXP
from run_main import setup_units

OUT = sys.argv[1]
GAMMA = 0.9
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
X, A = load_dataset("metr-la")
subs, units, vk, tk = setup_units(X, A)
cut = int(0.7 * len(X))
valranges = [(units[k]["start"], units[k]["start"] + units[k]["T"] + extra_steps(GAMMA)) for k in vk]


def training_set(S):
    W = A[np.ix_(S, S)]; M = nb_mask(W)
    U = X[:cut][:, S]
    Vs = vstar(U, M, GAMMA); mx = maxnb(Vs, M)
    q = np.zeros_like(U); q[1:-1] = GAMMA * (mx[2:] - mx[:-2])
    keep = np.ones(cut, bool); keep[:12] = False; keep[-80:] = False
    for (a, b) in valranges:
        keep[max(a - 12, 0):b] = False
    ts = np.where(keep)[0]
    return W, U[ts], q[ts]


def adam_fit(fun_grad, p0, steps, lr, seed=0):
    p = p0.copy(); m = np.zeros_like(p); v = np.zeros_like(p)
    for s in range(1, steps + 1):
        g = fun_grad(p, s)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        p -= lr * (m / (1 - 0.9 ** s)) / (np.sqrt(v / (1 - 0.999 ** s)) + 1e-8)
    return p


fitted = {}
for i, S in enumerate(subs):
    W, F, q = training_set(S)
    n = W.shape[0]
    rng = np.random.default_rng(i)
    bn = BNCircuit(W)
    Wn = W
    mixF = (F + F @ Wn.T) / (1 + Wn.sum(1))[None]
    for name in ("M4", "C1", "C2"):
        Fin = mixF if name == "C2" else F

        def score_and_jac(theta, idx):
            a = np.pi * Fin[idx] + theta[None]
            if name == "M4":
                K = len(idx)
                As = np.concatenate([a] + [a + (np.pi / 2) * np.eye(n)[m][None] for m in range(n)]
                                    + [a - (np.pi / 2) * np.eye(n)[m][None] for m in range(n)])
                Qs = bn.Q(As).reshape(2 * n + 1, K, n)
                s = 2 * Qs[0] - 1
                J = (Qs[1:n + 1] - Qs[n + 1:])            # (n_param, K, n) = 2*0.5*(Q+ - Q-)
                return s, J
            s = np.cos(a)
            J = np.zeros((n, len(idx), n)); J[np.arange(n), :, np.arange(n)] = -np.sin(a).T
            return s, J

        def fg(p, step):
            idx = rng.integers(0, len(q), 128 if name == 'M4' else 1024)
            theta, ls = p[:n], p[n]
            s, J = score_and_jac(theta, idx)
            L = np.exp(ls); e = L * s - q[idx]
            g_theta = np.einsum('pkn,kn->p', J, 2 * e * L) / e.size
            g_ls = (2 * e * L * s).sum() / e.size
            return np.concatenate([g_theta, [g_ls]])

        p = adam_fit(fg, np.concatenate([rng.uniform(-0.1, 0.1, n), [np.log(0.05)]]), 600 if name == "M4" else 1500, 0.03)
        # in-sample R2 on all training rows (chunked)
        ss, ys = [], []
        for c0 in range(0, len(q), 512):
            idx = np.arange(c0, min(c0 + 512, len(q)))
            a = np.pi * Fin[idx] + p[:n][None]
            s = 2 * bn.Q(a) - 1 if name == "M4" else np.cos(a)
            ss.append(np.exp(p[n]) * s); ys.append(q[idx])
        ss = np.concatenate(ss); ys = np.concatenate(ys)
        r2 = 1 - ((ss - ys) ** 2).mean() / ys.var()
        fitted[(name, i)] = dict(theta=p[:n], scale=float(np.exp(p[n])), r2=float(r2))
        log("fit", name, i, "scale", round(float(np.exp(p[n])), 4), "R2", round(float(r2), 4))


class Frozen(Score):
    def __init__(self, base, theta):
        self.base, self.theta = base, theta
        self.n = base.n; self.P = base.P

    def init(self, rng):
        self.params = self.theta.copy(); self.base.params = self.params

    def eval(self, ctx, t):
        self.base.params = self.params
        return self.base.eval(ctx, t)


def run_rev(args):
    name, cfg, ukey, seed, extra = args
    u = EXP._UNITS[ukey]
    ctx = Ctx(u["U"], u["F"], u["M"], u["gamma"], u["tod"])
    sg = int(ukey.split("_")[0][2:])
    base_name = name.split("-")[0]
    fit = fitted[(base_name, sg)]
    base = {"M4": CircuitScore(u["W"]), "C1": ProductScore(u["W"]), "C2": ProductScore(u["W"], mix=True)}[base_name]
    sc = Frozen(base, fit["theta"])
    c = dict(cfg)
    lam0 = c["mult"] * fit["scale"] if c["rule"] == "fixed" else (fit["scale"] if c["lam0"] == "fit" else c["lam0"])
    if c["rule"] == "fixed":
        out = run_estimator(ctx, u["T"], score=sc, lam_rule="fixed", lam0=lam0, seed=seed)
    else:
        out = run_estimator(ctx, u["T"], score=sc, lam_rule="signed", theta_rule="none", lam0=lam0,
                            eta_lam=c["eta_lam"], seed=seed)
    m = metrics(out["V"], u["Vs"], u["M"], mtd=out["mtd"], lam=out["lam"])
    return (name, json.dumps(cfg, sort_keys=True), ukey, seed, m)


def pmap(jobs):
    from multiprocessing import Pool
    with Pool(2, initializer=EXP._init, initargs=(units,)) as p:
        return p.map(run_rev, jobs, chunksize=max(1, len(jobs) // 16))


grid = ([dict(rule="fixed", mult=f) for f in (0.25, 0.5, 1.0, 1.5)] +
        [dict(rule="signed", lam0=l, eta_lam=e) for l in (0.05, "fit") for e in (0.05, 0.2, 1.0)])
models = ["M4-off", "C1-off", "C2-off"]
agg = {}
for (m, c, k, s, met) in pmap([(m, g, k, 0, None) for m in models for g in grid for k in vk]):
    agg.setdefault((m, c), []).append(met["mae"])
best = {}
for m in models:
    cands = sorted([(np.mean(v), json.loads(c)) for (mm, c), v in agg.items() if mm == m], key=lambda z: z[0])
    best[m] = dict(cfg=cands[0][1], valmae=cands[0][0])
log("best", best)
res = {}
for (m, c, k, s, met) in pmap([(m, best[m]["cfg"], k, s, None) for m in models for k in tk for s in (0, 1, 2)]):
    res.setdefault(m, {}).setdefault(k, {})[s] = met
log({m: unit_table(res)[m].mean() for m in models})
pickle.dump(dict(best=best, res=res, fitted={f"{k[0]}@{k[1]}": v for k, v in fitted.items()}), open(OUT, "wb"))
log("done")
