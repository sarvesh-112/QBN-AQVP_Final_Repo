"""Extras: theory verification, forced-feedback comparison (full vs diagonal rule with eta_theta=0.1),
device-calibrated noise inside the M4 loop, sampled-estimator accuracy, illustrative dynamics."""
import pickle, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from exp import *
from exp import _init, run_one
from run_main import setup_units
from run_full import SampledBNScore


def main(out="results/extras.pkl"):
    X, A = load_dataset("metr-la")
    subs, units, vk, tk = setup_units(X, A)
    main_res = pickle.load(open("results/main_metr.pkl", "rb"))
    best = main_res["best"]
    E = {}
    # ---------------- theory checks -------------------------------------------------------
    rng = np.random.default_rng(123)
    bnerr, upper, offmass, rodev = [], [], [], []
    for r in range(50):
        S = subs[r % 5]; W = A[np.ix_(S, S)]
        bn, sv = BNCircuit(W), StateVector(W)
        t = int(rng.integers(len(X)))
        alpha = np.pi * X[t, S] + rng.uniform(-0.5, 0.5, 10)
        bnerr.append(np.abs((2 * bn.Q(alpha[None]) - 1)[0] - sv.expZ(alpha[None])[0]).max())
        Q, J = bn.QJ(alpha)
        upper.append(np.abs(np.triu(J, 1)).max())
        offmass.append(np.abs(np.tril(J, -1)).sum() / np.abs(J).sum())
        phi = rng.uniform(-0.5, 0.5, 10)
        rodev.append(np.abs(sv.expZ(alpha[None], phi[None])[0] - (2 * bn.Q((alpha + phi)[None]) - 1)[0]).mean())
    E["theory"] = dict(bnerr=max(bnerr), upper=max(upper), offmass=float(np.mean(offmass)),
                       rodevmean=float(np.mean(rodev)), rodevmax=float(np.max(rodev)), ninst=50)
    # sampled estimator accuracy
    S = subs[2]; W = A[np.ix_(S, S)]
    sc = SampledBNScore(W, samples=200000, rng=np.random.default_rng(5)); sc.params = rng.uniform(-.1, .1, 10)
    class C: pass
    ctx = C(); ctx.F = X[1000:1001][:, S]
    s, Js = sc.eval(ctx, 0); Q, Je = BNCircuit(W).QJ(np.pi * ctx.F[0] + sc.params)
    E["samp"] = dict(serr=float(np.abs(s - (2 * Q - 1)).max()), jerr=float(np.abs(Js - 2 * Je).max()))
    print(E, flush=True)
    # ---------------- forced feedback: full vs diagonal rule ------------------------------
    cfgF = dict(best["M4"]["cfg"]); cfgF["eta_theta"] = 0.1
    cfgD = dict(best["Rdiag"]["cfg"]); cfgD["eta_theta"] = 0.1
    bf = {"M4": dict(cfg=cfgF), "Rdiag": dict(cfg=cfgD), "M3": best["M3"]}
    E["forced"] = test(["M4", "Rdiag"], bf, tk, units)
    E["forced_cfg"] = (cfgF, cfgD)
    # ---------------- device-calibrated noise ---------------------------------------------
    qk = pickle.load(open("results/qiskit.pkl", "rb"))
    aff = {}
    for si in range(5):
        Zi, Zn = qk["noise"][si]["Zi"], qk["noise"][si]["Zn"]
        c = np.empty(10); d = np.empty(10)
        for k in range(10):
            Z = np.stack([Zi[:, k], np.ones(len(Zi))], 1)
            (c[k], d[k]), *_ = np.linalg.lstsq(Z, Zn[:, k], rcond=None)
        aff[si] = (c, d)
    E["affine"] = aff
    E["devnoise"] = test(["M4"], best, tk, units, extra={"M4": dict(noise=dict(affine_by_sg=aff, shots=1000))})["M4"]
    # forced feedback under device noise as well
    E["devnoise_fb"] = test(["M4"], bf, tk, units, extra={"M4": dict(noise=dict(affine_by_sg=aff, shots=1000))})["M4"]
    pickle.dump(E, open(out, "wb"))
    print("done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
