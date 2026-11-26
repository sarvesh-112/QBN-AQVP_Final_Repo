"""Forced-feedback check for the learned classical score C4 (eta_theta forced to 0.1, other settings as tuned)."""
import pickle, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from exp import *
from run_main import setup_units

if __name__ == "__main__":
    t0 = time.time()
    X, A = load_dataset("metr-la")
    subs, units, vk, tk = setup_units(X, A)
    best = pickle.load(open("results/main_metr.pkl", "rb"))["best"]
    cfg = dict(best["C4"]["cfg"]); cfg["eta_theta"] = 0.1
    bf = {"C4": dict(cfg=cfg), "M3": best["M3"], "M4": best["M4"]}
    R = test(["C4"], bf, tk, units)
    pickle.dump(dict(res=R, cfg=cfg), open("results/c4fb.pkl", "wb"))
    print("done", time.time() - t0, flush=True)
