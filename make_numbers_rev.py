"""Writes the macros for the revision experiments (run_revision.py, run_revision2.py) to numbers_rev.tex.
usage: QBN_DATA=... python -I make_numbers_rev.py OUT.tex"""
import sys, os, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
from exp import unit_table
import re

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = sys.argv[1]
M = {}
_MINUS = re.compile(r"(?<![\w{^])-(?=\d)")


def put(k, v):
    v = _MINUS.sub(lambda _m: "\\textminus{}", v) if isinstance(v, str) else v
    M["rv@" + k] = v


def eqp(p):
    s = fmt_p(p)
    return s.replace("\\ensuremath{<", "\\ensuremath{{}<") if "<" in s else "\\ensuremath{{}=" + s + "}"


f3 = lambda x: f"{x:.3f}"
s1 = lambda x: f"{x:+.1f}"
s2 = lambda x: f"{x:+.2f}"

main = pickle.load(open(os.path.join(R, "main_metr.pkl"), "rb"))
rev = pickle.load(open(os.path.join(R, "revision.pkl"), "rb"))
rev2 = pickle.load(open(os.path.join(R, "revision2.pkl"), "rb")) if os.path.exists(os.path.join(R, "revision2.pkl")) else None
m1 = unit_table(main["res"])["M1"]
m1pol = unit_table(main["res"], "policy_acc")["M1"]

# ---------------------------------------------------------------- new models vs M1 (family R)
new = {}
new["C5"] = rev["C4off"]["res"]
for m in ("M4-lag", "C1-lag", "C2-lag", "M4-fc", "C1-fc", "C2-fc"):
    new[m] = rev["lag"]["res"][m]
new["M4sym"] = rev["M4sym"]["res"]
if rev2:
    for m in ("M4-off", "C1-off", "C2-off"):
        new[m] = rev2["res"][m]
tab = {m: unit_table({m: r})[m] for m, r in new.items()}
pol = {m: unit_table({m: r}, "policy_acc")[m] for m, r in new.items()}
lam = {m: unit_table({m: r}, "lammean")[m] for m, r in new.items()}
names = list(new)
praw = [wil(tab[m], m1) for m in names]
ph = holm(praw)
put("nfamR", str(len(names)))
ID = lambda m: m.replace("-", "").replace("4", "Four").replace("1", "One").replace("2", "Two").replace("5", "Five")
rows = []
for m, p, q in zip(names, praw, ph):
    i = ID(m)
    a = tab[m]
    lo, hi = boot_ci(a)
    rl, rh = paired_rel_ci(a, m1)
    rel = 100 * (a.mean() / m1.mean() - 1)
    put(f"{i}@mae", f3(a.mean())); put(f"{i}@maelo", f3(lo)); put(f"{i}@maehi", f3(hi))
    put(f"{i}@rel", s1(rel)); put(f"{i}@relabs", f"{abs(rel):.1f}"); put(f"{i}@rello", s1(rl)); put(f"{i}@relhi", s1(rh))
    put(f"{i}@p", fmt_p(p)); put(f"{i}@ph", fmt_p(q)); put(f"{i}@phe", eqp(q)); put(f"{i}@r", s2(rank_biserial(a, m1)))
    put(f"{i}@pol", f3(pol[m].mean())); put(f"{i}@lam", f3(lam[m].mean()))
    put(f"{i}@wins", str(int((a < m1).sum())))
    print(m, round(a.mean(), 4), round(rel, 2), fmt_p(q), round(pol[m].mean(), 3))
# C5 vs F1 and C5 vs M4
f1 = unit_table(main["res"])["F1"]; m4 = unit_table(main["res"])["M4"]
for tag, b in (("FOne", f1), ("MFour", m4)):
    a = tab["C5"]
    put(f"cmp@CFive@{tag}@rel", s1(100 * (a.mean() / b.mean() - 1))); put(f"cmp@CFive@{tag}@p", fmt_p(wil(a, b)))
    put(f"cmp@CFive@{tag}@pe", eqp(wil(a, b)))
# cluster CI for C5
cl = np.array([int(k.split("_")[0][2:]) for k in sorted(new["C5"])])
lo, hi = cluster_rel_ci(tab["C5"], m1, cl)
put("CFive@clo", s1(lo)); put("CFive@chi", s1(hi))
loo = [100 * (tab["C5"][cl != c].mean() / m1[cl != c].mean() - 1) for c in range(5)]
put("CFive@loomin", s1(min(loo))); put("CFive@loomax", s1(max(loo)))
put("CFive@lamhat", f"{rev['C4off']['lam_hat']:.2f}")
put("CFive@rtrain", f"{100 * rev['C4off']['r2_train']:.1f}"); put("CFive@rtest", f"{100 * rev['C4off']['r2_test']:.1f}")
put("CFive@nsamp", f"{rev['C4off']['nsamples'] / 1e5:.1f}\\times10^{{5}}")
cfg = rev["C4off"]["cfg"]["cfg"]
put("CFive@rule", cfg["rule"]); put("CFive@lamz", f"{cfg['lam0']:.2f}"); put("CFive@etal", f"{cfg.get('eta_lam', 0):g}")
put("cenc", f"{rev['lag']['cenc']:g}")
for m in ("M4-lag", "C1-lag", "C2-lag", "M4-fc", "C1-fc", "C2-fc"):
    c = rev["lag"]["best"][m]["cfg"]
    put(f"{ID(m)}@etat", f"{c['eta_theta']:g}")
if rev2:
    for m in ("M4-off", "C1-off", "C2-off"):
        put(f"{ID(m)}@r2", f"{100 * np.mean([v['r2'] for k, v in rev2['fitted'].items() if k.startswith(m.split('-')[0] + '@')]):.1f}")
        put(f"{ID(m)}@cfg", json.dumps(rev2["best"][m]["cfg"]))
    print({m: rev2["best"][m]["cfg"] for m in rev2["best"]})

# ---------------------------------------------------------------- permutation study
P = rev["perm"]
mae_p = np.array([unit_table(P[p]["res"])["M4"] for p in range(6)])       # (6, units)
ff_p = np.array([unit_table(P[p]["res"])["M4-ff"] for p in range(6)])
c3_p = np.array([unit_table(P[p]["res"])["C3"] for p in range(6)])
pol_p = np.array([unit_table(P[p]["res"], "policy_acc")["M4"] for p in range(6)])
m1p = unit_table(P[0]["res"])["M1"]
assert np.allclose(m1p, m1)
rels = 100 * (mae_p.mean(1) / m1.mean() - 1)
pp = [wil(mae_p[p], m1) for p in range(6)]
put("perm@n", "5")
put("perm@relmin", s1(rels.min())); put("perm@relmax", s1(rels.max())); put("perm@relmean", s1(rels.mean()))
put("perm@maemin", f3(mae_p.mean(1).min())); put("perm@maemax", f3(mae_p.mean(1).max()))
put("perm@nworse", str(int(sum(1 for p in range(6) if mae_p[p].mean() > m1.mean()))))
put("perm@nsig", str(int(sum(1 for p in range(6) if pp[p] < 0.05 and mae_p[p].mean() > m1.mean()))))
put("perm@pmax", fmt_p(max(pp))); put("perm@pmin", fmt_p(min(pp)))
sd_unit = mae_p.std(0, ddof=1)
put("perm@sdunit", f3(sd_unit.mean())); put("perm@sdunitmax", f3(sd_unit.max()))
put("perm@cvunit", f"{100 * np.mean(sd_unit / mae_p.mean(0)):.1f}")
put("perm@spreadrel", f"{100 * np.mean((mae_p.max(0) - mae_p.min(0)) / m1):.1f}")
put("perm@polmin", f3(pol_p.mean(1).min())); put("perm@polmax", f3(pol_p.mean(1).max()))
relsff = 100 * (ff_p.mean(1) / m1.mean() - 1)
put("perm@ffmin", s1(relsff.min())); put("perm@ffmax", s1(relsff.max()))
relc3 = 100 * (c3_p.mean(1) / m1.mean() - 1)
put("perm@cthreemin", s1(relc3.min())); put("perm@cthreemax", s1(relc3.max()))
dm = [100 * (mae_p[p].mean() / c3_p[p].mean() - 1) for p in range(6)]
pm = [wil(mae_p[p], c3_p[p]) for p in range(6)]
put("perm@vscthreemin", s1(min(dm))); put("perm@vscthreemax", s1(max(dm)))
put("perm@vscthreensig", str(int(sum(1 for q in holm(pm) if q < 0.05))))
# relative between-ordering variability vs between-model gap
put("perm@range", f"{100 * (mae_p.mean(1).max() / mae_p.mean(1).min() - 1):.1f}")
sym = unit_table({"x": rev["M4sym"]["res"]})["x"]
put("sym@mae", f3(sym.mean())); put("sym@rel", s1(100 * (sym.mean() / m1.mean() - 1)))
put("sym@ph", fmt_p(wil(sym, m1))); put("sym@pe", eqp(wil(sym, m1)))
for p in range(6):
    put(f"perm@{p}@rel", s1(rels[p])); put(f"perm@{p}@p", fmt_p(pp[p])); put(f"perm@{p}@pol", f3(pol_p[p].mean()))
    put(f"perm@{p}@cthree", s1(relc3[p])); put(f"perm@{p}@ff", s1(relsff[p]))
print("perm rels", rels.round(2), "p", np.round(pp, 4), "sym", sym.mean())

# ---------------------------------------------------------------- ties (from run_revision, eps=1e-3)
D = rev["ties"]
for m in ("M1", "M2", "M4", "C1", "C2", "C4", "F1"):
    i = {"M1": "MOne", "M2": "MTwo", "M4": "MFour", "C1": "COne", "C2": "CTwo", "C4": "CFour", "F1": "FOne"}[m]
    put(f"tie@{i}@nt", f3(D[(m, 1e-3)]["acc_nt"])); put(f"tie@{i}@tol", f3(D[(m, 1e-3)]["acc_tol"]))
put("tie@frac", f"{100 * D[('M1', 1e-3)]['tie']:.1f}")

# ---------------------------------------------------------------- scale stats
S = rev["scale_stats"]
put("deg@full", f"{S['full']['deg']:.1f}"); put("deg@sub", f"{S['sub']['deg']:.1f}")

# ---------------------------------------------------------------- decision gaps and scale statistics
EX = pickle.load(open(os.path.join(R, "extras_rev.pkl"), "rb"))
put("gap@med", f"{EX['gap_med']:.3f}"); put("gap@lt", f"{100 * EX['gap_lt002']:.0f}")
for m, i in (("M1", "MOne"), ("M2", "MTwo"), ("M4", "MFour")):
    for b in ("a", "b", "c"):
        put(f"gap@{i}@{b}", f3(EX["acc"][m][b]))
put("reach@full", f"{100 * EX['reach_full']:.0f}"); put("reach@sub", f"{100 * EX['reach_sub']:.1f}")
put("vmean@full", f"{EX['vmean_full']:.2f}"); put("vmean@sub", f"{EX['vmean_sub']:.2f}")

# ---------------------------------------------------------------- Table rows (tabrow_rev.tex)
desc = {"C5": ("MLP of C4", "rich features", "offline"), "M4-off": ("entangled circuit", "$x_i(t)$", "offline"),
        "C1-off": ("product state", "$x_i(t)$", "offline"), "C2-off": ("graph-mixing", "$x_i(t)$", "offline"),
        "M4-lag": ("entangled circuit", "lag", "online"), "C1-lag": ("product state", "lag", "online"),
        "C2-lag": ("graph-mixing", "lag", "online"), "M4-fc": ("entangled circuit", "forecast", "online"),
        "C1-fc": ("product state", "forecast", "online"), "C2-fc": ("graph-mixing", "forecast", "online"),
        "M4sym": ("circuit, mean of 6 orderings", "$x_i(t)$", "online")}
label = {"C5": "C5", "M4-off": "M4-off", "C1-off": "C1-off", "C2-off": "C2-off", "M4-lag": "M4-lag", "C1-lag": "C1-lag",
         "C2-lag": "C2-lag", "M4-fc": "M4-fc", "C1-fc": "C1-fc", "C2-fc": "C2-fc", "M4sym": "M4-sym"}
groups = [["C5", "M4-off", "C1-off", "C2-off"], ["M4-lag", "C1-lag", "C2-lag", "M4-fc", "C1-fc", "C2-fc"], ["M4sym"]]
lines = []
for gi, g in enumerate(groups):
    g = [m for m in g if m in new]
    if gi and g:
        lines.append("\\midrule")
    for m in g:
        i = ID(m); a = tab[m]
        lo, hi = boot_ci(a); rl, rh = paired_rel_ci(a, m1)
        rel = 100 * (a.mean() / m1.mean() - 1); q = ph[names.index(m)]
        relstr = f"{rel:+.1f}".replace("-", "\\textminus{}")
        if rel < 0 and q < 0.05:
            relstr = "\\textbf{" + relstr + "}"
        row = (f"{label[m]} & {desc[m][0]} ({desc[m][2]}) & {desc[m][1]} & {a.mean():.3f} [{lo:.3f}, {hi:.3f}] & "
               f"{relstr} [{rl:+.1f}, {rh:+.1f}] & {fmt_p(q)} & {pol[m].mean():.3f} & {lam[m].mean():.3f} \\\\")
        lines.append(_MINUS.sub(lambda _m: "\\textminus{}", row))
open(os.path.join(os.path.dirname(OUT), "tabrow_rev.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- sentence on offline-trained circuit / product scores
if rev2:
    def rr(m):
        a = tab[m]; return 100 * (a.mean() / m1.mean() - 1), ph[names.index(m)]
    r4, q4 = rr("M4-off"); r1, q1 = rr("C1-off"); r2, q2 = rr("C2-off")
    R2 = {m: 100 * np.mean([v["r2"] for k, v in rev2["fitted"].items() if k.startswith(m + "@")]) for m in ("M4", "C1", "C2")}
    sgn = lambda x: f"{x:+.1f}".replace("-", "\\textminus{}")
    def pe(q):
        return eqp(q)
    w4 = int((tab["M4-off"] < m1).sum()); w1 = int((tab["C1-off"] < m1).sum())
    p41 = wil(tab["M4-off"], tab["C1-off"])
    sent = (f"Fitting the circuit's own parameters and output scale to the same target (M4-off) gives a different picture. "
            f"The fit explains only {R2['M4']:.1f}\\% of the ideal correction in training (product state: {R2['C1']:.1f}\\%; graph-mixing: {R2['C2']:.1f}\\%), "
            f"yet mean MAE falls by {abs(r4):.1f}\\%, and by {abs(r1):.1f}\\% for the product-state score fitted in the same way (C1-off). "
            f"Neither change is consistent: M4-off is better than M1 on only {w4} and C1-off on {w1} of the \\Q{{nunits}} units, " +
            (f"neither is significant ($p_{{\\mathrm{{H}}}}={fmt_p(q4)}$ for both), " if fmt_p(q4) == fmt_p(q1) else f"neither is significant ($p_{{\\mathrm{{H}}}}={fmt_p(q4)}$ and ${fmt_p(q1)}$), ") +
            f"the two are indistinguishable from each other ($p{pe(p41)}$), and both lower decision accuracy. "
            f"The circuit therefore gains nothing from offline training that a product-state score of the same form does not also gain.")
    put("offline@sentence", sent)
else:
    put("offline@sentence", "")

with open(OUT, "w") as fh:
    fh.write("% generated by make_numbers_rev.py -- do not edit\n\\makeatletter\n")
    for k, v in M.items():
        fh.write(f"\\expandafter\\def\\csname qbn@{k}\\endcsname{{{v}}}\n")
    fh.write("\\makeatother\n")
print("wrote", len(M), "macros")
