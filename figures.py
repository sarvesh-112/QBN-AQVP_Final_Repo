"""All data figures of the manuscript (PDF, print)."""
import pickle, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from exp import unit_table
from stats import boot_ci
from run_main import setup_units

OUT = os.environ.get("QBN_FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_out"))
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
R = os.path.join(os.path.dirname(__file__), "results")


def fig_ablation():
    mm = pickle.load(open(f"{R}/main_metr.pkl", "rb"))
    res = mm["res"]
    order = ["M1", "M2", "M3", "M4", "M5", "C1", "C2", "C3", "C4", "F1", "V1C", "V1"]
    lab = {"V1C": "V1-C"}
    col = {m: BLUE for m in ["M2", "M3", "M4", "M5"]}
    col.update({m: ORANGE for m in ["C1", "C2", "C3", "C4", "F1"]}); col.update({"V1": MAGENTA, "V1C": MAGENTA, "M1": MUTED})
    mae, pol = unit_table(res, "mae"), unit_table(res, "policy_acc")
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for ax, tab, yl, cap in [(axs[0], mae, "MAE vs. $V^*$", 0.4), (axs[1], pol, "Decision accuracy", None)]:
        x = np.arange(len(order))
        for i, m in enumerate(order):
            v = tab[m]; lo, hi = boot_ci(v); mu = v.mean()
            top = min(mu, cap) if cap else mu
            ax.bar(i, top, width=0.68, color=col[m], edgecolor="white", linewidth=1)
            if cap and mu > cap:
                ax.text(i, cap * 0.97, f"{mu:.2f}↑", ha="center", va="top", fontsize=6.5, color="white", rotation=90)
            else:
                ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color=INK, lw=0.8, capsize=2)
        ax.axhline(tab["M1"].mean(), color=INK, ls=":", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels([lab.get(m, m) for m in order], rotation=0, fontsize=6.6)
        ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
        if cap: ax.set_ylim(0, cap)
        else: ax.set_ylim(0.3, 1.0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=MUTED, label="classical Bellman (M1)"), Patch(color=BLUE, label="circuit ladder"),
                        Patch(color=ORANGE, label="classical competitors"), Patch(color=MAGENTA, label="v1 design")],
               loc="upper center", fontsize=6.5, ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(f"{OUT}/ablation.pdf"); fig.savefig(f"{OUT}/ablation.png", dpi=110)


def fig_dynamics():
    X, A = load_dataset("metr-la")
    subs, units, vk, tk = setup_units(X, A)
    mm = pickle.load(open(f"{R}/main_metr.pkl", "rb")); best = mm["best"]
    u = units[tk[0]]; T = u["T"]; ctx = Ctx(u["U"], u["F"], u["M"], u["gamma"], u["tod"])
    from exp import build
    outs = {}
    for m in ["M1", "M4", "Lmono", "Lleaky", "Lsoft", "V1"]:
        kw = build(m, u["W"], best[m]["cfg"] if m != "M1" else {}, 0)
        outs[m] = run_estimator(ctx, T, seed=0, **kw)
    N = 100
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.1))
    sty = {"Lmono": (MAGENTA, "monotone (v1)"), "Lsoft": (YELLOW, "soft-target"), "Lleaky": (AQUA, "leaky"), "M4": (BLUE, "signed (M4)")}
    for m, (c, l) in sty.items():
        axs[0].plot(outs[m]["lam"][:N], color=c, lw=1.4, label=l)
    axs[0].set_title("(a) coefficient $\\lambda_t$", fontsize=7.5, loc="left"); axs[0].set_xlabel("step"); axs[0].legend(fontsize=6.2)
    for m, (c, l) in sty.items():
        axs[1].plot(outs[m]["mtd"][:N], color=c, lw=1.2, label=l)
    axs[1].plot(outs["M1"]["mtd"][:N], color=INK, lw=1.0, ls=":", label="M1")
    axs[1].set_yscale("log"); axs[1].set_title("(b) mean $|\\delta_i(t)|$", fontsize=7.5, loc="left"); axs[1].set_xlabel("step")
    k = int(np.argmax(u["Vs"][:T].mean(0)))
    axs[2].plot(u["Vs"][:N, k], color=INK, lw=1.6, label="$V^*$")
    axs[2].plot(outs["M1"]["V"][:N, k], color=MUTED, lw=1.2, ls="--", label="M1")
    axs[2].plot(outs["M4"]["V"][:N, k], color=BLUE, lw=1.2, ls=":", label="M4")
    axs[2].plot(outs["V1"]["V"][:N, k], color=MAGENTA, lw=1.4, label="V1")
    axs[2].set_title("(c) value at top node", fontsize=7.5, loc="left"); axs[2].set_xlabel("step"); axs[2].legend(fontsize=6.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/dynamics.pdf"); fig.savefig(f"{OUT}/dynamics.png", dpi=110)
    sat = np.argmax(outs["Lmono"]["lam"] >= 1) if (outs["Lmono"]["lam"] >= 1).any() else -1
    return dict(v1_sat=int(np.argmax(outs["V1"]["lam"] >= 1)), mono_sat=int(sat), soft_final=float(outs["Lsoft"]["lam"][N]),
                v1_end=float(outs["V1"]["V"][N - 1, k]), star_end=float(u["Vs"][N - 1, k]), cl_end=float(outs["M1"]["V"][N - 1, k]))


def fig_jacobian():
    X, A = load_dataset("metr-la")
    S = greedy_subgraphs(A, seed=0)[0]; W = A[np.ix_(S, S)]
    mm = pickle.load(open(f"{R}/main_metr.pkl", "rb"))
    Q, J = BNCircuit(W).QJ(np.pi * X[25000, S])
    cos = np.concatenate([[mm["res"][m][k][s]["cos"] for k in mm["res"][m] for s in mm["res"][m][k]] for m in ["M4", "Rdiag"]])
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw=dict(width_ratios=[1, 1.15]))
    im = axs[0].imshow(np.abs(J), cmap="Blues", vmin=0); axs[0].grid(False)
    axs[0].set_xlabel("parameter $m$"); axs[0].set_ylabel("score $k$"); axs[0].set_title("(a) $|\\partial Q_k/\\partial\\theta_m|$", fontsize=7, loc="left")
    axs[0].set_xticks([0, 9]); axs[0].set_yticks([0, 9])
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[1].hist(cos[np.isfinite(cos)], bins=20, color=BLUE, edgecolor="white", linewidth=0.8)
    axs[1].set_xlabel("cosine similarity"); axs[1].set_title("(b) per-run mean", fontsize=7, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/jacobian.pdf"); fig.savefig(f"{OUT}/jacobian.png", dpi=110)


def fig_pc():
    pc = pickle.load(open(f"{R}/pc_metr.pkl", "rb"))
    lv = pc["levels"]
    xs = list(range(len(lv))); xl = [("$\\sigma$=" + l) if l != "shuffled" else "shuffled" for l in lv]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for m, c, mk in [("C1", ORANGE, "o"), ("C2", AQUA, "s"), ("M4", BLUE, "D")]:
        rel, lam = [], []
        for l in lv:
            r = pc["out"][l]["res"]; t = unit_table(r); rel.append(100 * (t[m].mean() / t["M1"].mean() - 1))
            lam.append(unit_table(r, "lammean")[m].mean())
        axs[0].plot(xs, rel, color=c, marker=mk, ms=4, lw=1.4, label=m)
        axs[1].plot(xs, lam, color=c, marker=mk, ms=4, lw=1.4, label=m)
    axs[0].axhline(0, color=INK, ls=":", lw=0.8)
    axs[0].set_ylabel("$\\Delta$MAE vs. M1 (%)"); axs[1].set_ylabel("mean $\\lambda_t$")
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels([l.replace("$\\sigma$=", "") for l in xl], rotation=45, fontsize=6)
        ax.set_xlabel("advisory noise $\\sigma$")
    axs[1].legend(fontsize=6.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/positive.pdf"); fig.savefig(f"{OUT}/positive.png", dpi=110)


if __name__ == "__main__":
    what = sys.argv[1:] or ["ablation", "dynamics", "jacobian"]
    for w in what:
        r = globals()["fig_" + w]()
        if r:
            print(w, r)
            import json; json.dump(r, open(f"{R}/dyn.json", "w"))
