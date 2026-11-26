"""Writes every number used in the manuscript to paper/numbers.tex (macro \\Q{key})."""
import pickle, sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
from exp import unit_table

R = os.path.join(os.path.dirname(__file__), "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/paper/numbers.tex"
ID = {"M1": "MOne", "M2": "MTwo", "M3": "MThree", "M4": "MFour", "M5": "MFive", "C1": "COne", "C2": "CTwo",
      "C3": "CThree", "C4": "CFour", "F1": "FOne", "Lmono": "LMono", "Lleaky": "LLeaky", "Lsoft": "LSoft",
      "Rdiag": "RDiag", "V1": "VOne", "V1C": "VOneC"}
M = {}


import re
_MINUS = re.compile(r"(?<![\w{^])-(?=\d)")


def put(k, v):
    M[k] = _MINUS.sub(lambda _m: "\\textminus{}", v) if isinstance(v, str) else v
    if (k.endswith("@p") or k.endswith("@ph")) and isinstance(v, str):
        M[k + "e"] = v.replace("\\ensuremath{<", "\\ensuremath{{}<") if "<" in v else "\\ensuremath{{}=" + v + "}"


def f3(x):
    return f"{x:.3f}"


def sgn1(x):
    return f"{x:+.1f}"


def ld(name):
    p = os.path.join(R, name)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def per_run(res, m, key):
    return np.array([res[m][k][s][key] for k in sorted(res[m]) for s in res[m][k]])


def clusters_of(keys):
    return np.array([int(k.split("_")[0][2:]) if k.startswith("sg") else 0 for k in sorted(keys)])


def family_A(pre, res, models, best=None, base="M1"):
    """per-model macros incl. Holm across all models vs base."""
    mae = unit_table(res, "mae"); pol = unit_table(res, "policy_acc")
    other = [m for m in models if m != base and m in res]
    praw = [wil(mae[m], mae[base]) for m in other]
    ph = holm(praw)
    for m in models:
        if m not in res:
            continue
        i = ID.get(m, m)
        a = mae[m]
        lo, hi = boot_ci(a)
        put(f"{pre}{i}@mae", f3(a.mean())); put(f"{pre}{i}@maelo", f3(lo)); put(f"{pre}{i}@maehi", f3(hi))
        put(f"{pre}{i}@maesix", f"{a.mean():.6f}")
        put(f"{pre}{i}@rmse", f3(unit_table(res, "rmse")[m].mean()))
        put(f"{pre}{i}@mtd", f3(unit_table(res, "mtd")[m].mean()))
        put(f"{pre}{i}@pol", f3(pol[m].mean()))
        put(f"{pre}{i}@bias", f"{unit_table(res, 'bias')[m].mean():.3f}")
        put(f"{pre}{i}@lam", f3(unit_table(res, "lammean")[m].mean()))
        put(f"{pre}{i}@lamfinal", f3(unit_table(res, "lamfinal")[m].mean()))
        put(f"{pre}{i}@sat", f"{100 * per_run(res, m, 'sat').mean():.0f}")
        if m == base:
            continue
        b = mae[base]
        rel = 100 * (a.mean() / b.mean() - 1)
        rl, rh = paired_rel_ci(a, b)
        put(f"{pre}{i}@rel", sgn1(rel)); put(f"{pre}{i}@relabs", f"{abs(rel):.1f}")
        put(f"{pre}{i}@rello", sgn1(rl)); put(f"{pre}{i}@relhi", sgn1(rh))
        j = other.index(m)
        put(f"{pre}{i}@p", fmt_p(praw[j])); put(f"{pre}{i}@ph", fmt_p(ph[j]))
        put(f"{pre}{i}@r", f"{rank_biserial(a, b):+.2f}")
        put(f"{pre}{i}@wins", str(int((a < b - 1e-12).sum())))
        put(f"{pre}{i}@polrel", sgn1(100 * (pol[m].mean() / pol[base].mean() - 1)))
    if best:
        for m in models:
            if m in best and best[m].get("cfg"):
                c = best[m]["cfg"]; i = ID.get(m, m)
                if "lam0" in c: put(f"{pre}{i}@lamz", f"{c['lam0']:g}")
                if "eta_lam" in c: put(f"{pre}{i}@etal", f"{c['eta_lam']:g}")
                if "eta_theta" in c: put(f"{pre}{i}@etat", f"{c['eta_theta']:g}")
                if best[m].get("valmae") is not None: put(f"{pre}{i}@valmae", f3(best[m]["valmae"]))
    put(f"{pre}nunits", str(len(mae[base])))
    return mae, pol


def family_B(pre, res, pairs):
    """pairs: list of (A, B, metric); Holm across all."""
    tabs = {"mae": unit_table(res, "mae"), "pol": unit_table(res, "policy_acc")}
    praw = [wil(tabs[me][a], tabs[me][b]) for a, b, me in pairs]
    ph = holm(praw)
    for (a, b, me), p, q in zip(pairs, praw, ph):
        x, y = tabs[me][a], tabs[me][b]
        k = f"{pre}cmp@{a}@{b}@{me}"
        put(k + "@rel", sgn1(100 * (x.mean() / y.mean() - 1)))
        put(k + "@p", fmt_p(p)); put(k + "@ph", fmt_p(q)); put(k + "@r", f"{rank_biserial(x, y):+.2f}")
        put(k + "@wins", str(int((x < y - 1e-12).sum())) if me == "mae" else str(int((x > y + 1e-12).sum())))
        lo, hi = paired_rel_ci(x, y)
        put(k + "@rello", sgn1(lo)); put(k + "@relhi", sgn1(hi))
    return praw, ph


def main():
    mm = ld("main_metr.pkl")
    res = mm["res"]
    models = ["M1", "M2", "M3", "M4", "M5", "C1", "C2", "C3", "C4", "F1", "Lmono", "Lleaky", "Lsoft", "Rdiag", "V1", "V1C"]
    mae, pol = family_A("", res, models, mm["best"])
    pairs = [("M3", "M2", "mae"), ("M4", "M3", "mae"), ("M5", "M4", "mae"), ("M4", "C1", "mae"), ("M4", "C2", "mae"),
             ("M4", "C3", "mae"), ("M4", "C4", "mae"), ("M4", "F1", "mae"), ("M4", "Rdiag", "mae"),
             ("M4", "Lmono", "mae"), ("M4", "Lleaky", "mae"), ("M4", "Lsoft", "mae"), ("V1", "V1C", "mae"),
             ("M4", "C1", "pol"), ("M4", "C2", "pol"), ("M4", "C3", "pol"), ("M4", "C4", "pol"), ("M4", "F1", "pol"),
             ("M4", "Lleaky", "pol")]
    family_B("", res, pairs)
    put("nfamA", str(len(models) - 1))
    names = {"M2": "M2 Fixed weight", "M3": "M3 + adaptive $\\lambda$", "M4": "M4 + circuit feedback", "M5": "M5 + readout layer",
             "C1": "C1 Product state", "C2": "C2 Graph-mixing", "C3": "C3 Matched offset", "C4": "C4 Learned MLP",
             "F1": "F1 Forecast-augmented", "Lmono": "L-monotone", "Lleaky": "L-leaky", "Lsoft": "L-soft",
             "Rdiag": "R-diag", "V1": "V1 Original design", "V1C": "V1-C"}
    groups = [["M2", "M3", "M4", "M5"], ["C1", "C2", "C3", "C4", "F1"], ["Lmono", "Lleaky", "Lsoft", "Rdiag"], ["V1", "V1C"]]
    lines = []
    for gi, g in enumerate(groups):
        for m in g:
            i = ID[m]; q = lambda k: M[f"{i}@{k}"]
            lines.append(f"{names[m]} & {q('mae')} [{q('maelo')}, {q('maehi')}] & {q('rmse')} & {q('mtd')} & {q('pol')} & {q('lam')} & "
                         f"{q('rel')} [{q('rello')}, {q('relhi')}] & {q('p')} & {q('ph')} & {q('r')} \\\\")
        if gi < len(groups) - 1:
            lines.append("\\midrule")
    open(os.path.join(os.path.dirname(OUT), "tabrow_main.tex"), "w").write("\n".join(lines) + "\n")
    put("nunits", str(len(mae["M1"]))); put("nruns", str(3 * len(mae["M1"])))
    put("nedges", ", ".join(map(str, mm["nedges"])))
    put("nedgesmin", str(min(mm["nedges"]))); put("nedgesmax", str(max(mm["nedges"]))); put("nedgesfirst", str(mm["nedges"][0]))
    put("MOne@valmae", f3(mm["M1val"])); put("FOne@valmae", f3(mm["F1val"]))
    put("rtwo", f"{100 * np.mean(mm['ceiling']):.1f}"); put("rtwomax", f"{100 * np.max(mm['ceiling']):.1f}")
    cos = np.concatenate([per_run(res, "M4", "cos"), per_run(res, "Rdiag", "cos")])
    put("cosmean", f"{np.nanmean(cos):.2f}"); put("cosn", str(int(np.isfinite(cos).sum())))
    put("mlpP", "73")
    # dependence: cluster bootstrap + leave-one-subgraph-out
    cl = clusters_of(res["M1"].keys())
    for m in ["M4", "C1", "C2", "C4", "F1"]:
        lo, hi = cluster_rel_ci(mae[m], mae["M1"], cl)
        put(f"dep@{ID[m]}@clo", sgn1(lo)); put(f"dep@{ID[m]}@chi", sgn1(hi))
        rels, ps = [], []
        for g in range(5):
            keep = cl != g
            rels.append(100 * (mae[m][keep].mean() / mae["M1"][keep].mean() - 1)); ps.append(wil(mae[m][keep], mae["M1"][keep]))
        put(f"dep@{ID[m]}@loomin", sgn1(min(rels))); put(f"dep@{ID[m]}@loomax", sgn1(max(rels)))
        put(f"dep@{ID[m]}@loopmax", fmt_p(max(ps))); put(f"dep@{ID[m]}@loopmin", fmt_p(min(ps)))
        put(f"dep@{ID[m]}@loosame", str(sum(1 for r_ in rels if np.sign(r_) == np.sign(100 * (mae[m].mean() / mae['M1'].mean() - 1)))))
    for m in ["C1", "C2", "C3"]:
        lo, hi = cluster_rel_ci(mae["M4"], mae[m], cl)
        put(f"dep@MFourvs{ID[m]}@clo", sgn1(lo)); put(f"dep@MFourvs{ID[m]}@chi", sgn1(hi))
    # noise
    if "noise" in mm:
        m4 = mae["M4"]; m4p = pol["M4"]
        labs = list(mm["noise"].keys()); praw = []
        rows = []
        for i, lab in enumerate(labs):
            r = {"M4": mm["noise"][lab]}
            a = unit_table(r)["M4"]; pp = unit_table(r, "policy_acc")["M4"]
            lo, hi = boot_ci(a)
            praw.append(wil(a, m4))
            put(f"noise@{i}@label", lab); put(f"noise@{i}@mae", f3(a.mean())); put(f"noise@{i}@lo", f3(lo)); put(f"noise@{i}@hi", f3(hi))
            put(f"noise@{i}@rel", f"{100 * (a.mean() / m4.mean() - 1):+.2f}"); put(f"noise@{i}@pol", f3(pp.mean()))
            put(f"noise@{i}@lam", f3(unit_table(r, 'lammean')['M4'].mean()))
            put(f"noise@{i}@relMOne", sgn1(100 * (a.mean() / mae['M1'].mean() - 1)))
        ph = holm(praw)
        for i in range(len(labs)):
            put(f"noise@{i}@p", fmt_p(praw[i])); put(f"noise@{i}@ph", fmt_p(ph[i]))
    # extras
    ex = ld("extras.pkl")
    if ex:
        th = ex["theory"]
        e = int(np.floor(np.log10(th["bnerr"]))); put("bnerr", rf"{th['bnerr'] / 10 ** e:.1f}\times10^{{{e}}}")
        put("offdiagmass", f"{100 * th['offmass']:.0f}"); put("rodevmean", f"{th['rodevmean']:.2f}")
        put("rodevmax", f"{th['rodevmax']:.2f}"); put("ninst", str(th["ninst"]))
        e = int(np.floor(np.log10(ex["samp"]["serr"]))); put("samp@serr", rf"\ensuremath{{{ex['samp']['serr'] / 10 ** e:.0f}\times10^{{{e}}}}}")
        e = int(np.floor(np.log10(ex["samp"]["jerr"]))); put("samp@jerr", rf"\ensuremath{{{ex['samp']['jerr'] / 10 ** e:.0f}\times10^{{{e}}}}}")
        fr = ex["forced"]
        a, b = unit_table(fr)["M4"], unit_table(fr)["Rdiag"]
        put("fb@MFour@mae", f3(a.mean())); put("fb@RDiag@mae", f3(b.mean()))
        put("fb@rel", sgn1(100 * (a.mean() / b.mean() - 1))); put("fb@p", fmt_p(wil(a, b)))
        put("fb@MFour@rel", sgn1(100 * (a.mean() / mae["M1"].mean() - 1))); put("fb@MFour@relMFour", f"{100 * (a.mean() / mae['M4'].mean() - 1):+.2f}")
        put("fb@MFour@p", fmt_p(wil(a, mae["M4"])))
        put("fb@MFour@lam", f3(unit_table(fr, "lammean")["M4"].mean()))
        dv = unit_table({"M4": ex["devnoise"]})["M4"]; lo, hi = boot_ci(dv)
        put("dev@mae", f3(dv.mean())); put("dev@lo", f3(lo)); put("dev@hi", f3(hi))
        put("dev@rel", f"{100 * (dv.mean() / mae['M4'].mean() - 1):+.2f}"); put("dev@p", fmt_p(wil(dv, mae["M4"])))
        put("dev@pol", f3(unit_table({'M4': ex['devnoise']}, 'policy_acc')['M4'].mean()))
        put("dev@relMOne", sgn1(100 * (dv.mean() / mae['M1'].mean() - 1)))
        dvf = unit_table({"M4": ex["devnoise_fb"]})["M4"]
        put("devfb@mae", f3(dvf.mean())); put("devfb@rel", f"{100 * (dvf.mean() / a.mean() - 1):+.2f}"); put("devfb@p", fmt_p(wil(dvf, a)))
        cs = np.concatenate([v[0] for v in ex["affine"].values()]); ds = np.concatenate([v[1] for v in ex["affine"].values()])
        put("aff@cmin", f"{cs.min():.2f}"); put("aff@cmax", f"{cs.max():.2f}"); put("aff@cmean", f"{cs.mean():.2f}")
        put("aff@dmean", f"{np.abs(ds).mean():.3f}"); put("aff@dmax", f"{np.abs(ds).max():.3f}")
    c4 = ld("c4fb.pkl")
    if c4:
        f = unit_table(c4["res"])["C4"]
        put("fbc@CFour@mae", f3(f.mean())); put("fbc@CFour@rel", sgn1(100 * (f.mean() / mae["M1"].mean() - 1)))
        put("fbc@CFour@relC", sgn1(100 * (f.mean() / mae["C4"].mean() - 1))); put("fbc@CFour@p", fmt_p(wil(f, mae["C4"])))
        put("fbc@CFour@lam", f3(unit_table(c4["res"], "lammean")["C4"].mean()))
    qk = ld("qiskit.pkl")
    if qk:
        put("qk@ninputs", str(qk["noise"][0]["Zi"].shape[0])); put("qk@shots", "4000")
        for b in ["sherbrooke", "torino"]:
            rows = [qk["transpile"][(b, s)] for s in range(5)]
            put(f"qk@{b}@twoq", f"{np.median([np.median(r['routed_2q']) for r in rows]):.0f}")
            put(f"qk@{b}@twoqmin", str(min(min(r['routed_2q']) for r in rows))); put(f"qk@{b}@twoqmax", str(max(max(r['routed_2q']) for r in rows)))
            put(f"qk@{b}@depmin", str(min(min(r['routed_depth']) for r in rows))); put(f"qk@{b}@depmax", str(max(max(r['routed_depth']) for r in rows)))
            for s in range(5):
                r = rows[s]
                put(f"qk@{b}@{s}@twoq", f"{np.median(r['routed_2q']):.0f}"); put(f"qk@{b}@{s}@dep", f"{np.median(r['routed_depth']):.0f}")
                put(f"qk@{s}@E", str(r["nedges"])); put(f"qk@{s}@lcx", str(r["logical_cx"])); put(f"qk@{s}@ldep", str(r["logical_depth"]))
        ratio = np.median([np.median(qk['transpile'][('sherbrooke', s)]['routed_2q']) / qk['transpile'][('sherbrooke', s)]['nedges'] for s in range(5)])
        put("qk@ratio", f"{ratio:.1f}")
        nz = qk["noise"]
        put("qk@mad", f"{np.mean([nz[s]['mad'] for s in range(5)]):.3f}"); put("qk@maxdev", f"{np.max([nz[s]['maxdev'] for s in range(5)]):.2f}")
        put("qk@slope", f"{np.mean([nz[s]['slope'] for s in range(5)]):.2f}")
        put("qk@slopemin", f"{np.min([nz[s]['slope'] for s in range(5)]):.2f}"); put("qk@slopemax", f"{np.max([nz[s]['slope'] for s in range(5)]):.2f}")
        put("qk@dp001", f"{np.mean([nz[s]['depol_pred'][0.001] for s in range(5)]):.2f}"); put("qk@dp01", f"{np.mean([nz[s]['depol_pred'][0.01] for s in range(5)]):.2f}")
        # equivalent global depolarizing strength per logical gate: (1-p)^G = slope
        peq = [1 - nz[s]['slope'] ** (1 / nz[s]['G']) for s in range(5)]
        put("qk@peq", f"{np.mean(peq):.4f}")
        put("qk@secpercirc", f"{np.mean([nz[s]['sec_per_circ'] for s in range(5)]):.1f}")
    # positive control
    pc = ld("pc_metr.pkl")
    if pc:
        for lv in pc["levels"]:
            o = pc["out"][lv]; tag = {"0.0": "s0", "0.05": "s005", "0.1": "s01", "0.2": "s02", "0.4": "s04", "shuffled": "shuf"}[lv]
            family_A(f"pc@{tag}@", o["res"], ["M1", "M3", "M4", "C1", "C2"], o["best"])
            family_B(f"pc@{tag}@", o["res"], [("M4", "C1", "mae"), ("M4", "C2", "mae"), ("M4", "M3", "mae"), ("C1", "C2", "mae")])
            put(f"pc@{tag}@ceil", f"{100 * np.mean(o['ceiling']):.1f}")
        # main level alias
        for k in list(M.keys()):
            if k.startswith("pc@s01@"):
                M["pc@main@" + k[len("pc@s01@"):]] = M[k]
        put("pc@c", f"{pc['c']:g}"); put("pc@lvl@main", "0.1")
        rows = []
        for lv, tag in [("0.0", "s0"), ("0.05", "s005"), ("0.1", "s01"), ("0.2", "s02"), ("0.4", "s04"), ("shuffled", "shuf")]:
            def cell(i):
                v = M[f"pc@{tag}@{i}@rel"]; ph = M[f"pc@{tag}@{i}@ph"]
                sig = ("<" in ph) or (float(ph.replace("\\ensuremath{", "").split("\\times")[0]) * (10 ** -4 if "times10^{-4}" in ph else 1) < 0.05 if "times" in ph else float(ph) < 0.05)
                return f"\\textbf{{{v}}}" if sig and (v.startswith("-") or v.startswith("\\textminus")) else v
            name = f"$\\sigma={lv}$" if lv != "shuffled" else "shuffled"
            rows.append(f"{name} & {M[f'pc@{tag}@ceil']} & {cell('MFour')} & {cell('COne')} & {cell('CTwo')} & "
                        f"{M[f'pc@{tag}@MFour@lam']} & {M[f'pc@{tag}@COne@lam']} & {M[f'pc@{tag}@CTwo@lam']} \\\\")
        open(os.path.join(os.path.dirname(OUT), "tabrow_pc.tex"), "w").write("\n".join(rows) + "\n")
    # full graph and PEMS-BAY
    for tag, fn in [("fm", "full_metr.pkl"), ("fp", "full_pems.pkl")]:
        fr = ld(fn)
        if fr:
            family_A(f"{tag}@", fr["res"], ["M1", "M3", "M4", "C1", "C2", "C3", "C4", "F1"], fr["best"])
            family_B(f"{tag}@", fr["res"], [("M4", "C1", "mae"), ("M4", "C2", "mae"), ("M4", "C3", "mae"), ("M4", "C4", "mae"), ("M4", "F1", "mae")])
            put(f"{tag}@n", str(fr["n"])); put(f"{tag}@E", str(fr["nedges"]))
    pb = ld("main_pems.pkl")
    if pb:
        family_A("pb@", pb["res"], ["M1", "M2", "M3", "M4", "C1", "C2", "C3", "C4", "F1"], pb["best"])
        family_B("pb@", pb["res"], [("M4", "C1", "mae"), ("M4", "C2", "mae"), ("M4", "C3", "mae"), ("M4", "C4", "mae"), ("M4", "F1", "mae")])
        put("pb@nedges", ", ".join(map(str, pb["nedges"]))); put("pb@rtwo", f"{100 * np.mean(pb['ceiling']):.1f}")
    for tag, fn in [("gA", "sens_g080.pkl"), ("gB", "sens_g095.pkl"), ("TA", "sens_T864.pkl"), ("TB", "sens_T2016.pkl")]:
        sr = ld(fn)
        if sr:
            family_A(f"sens@{tag}@", sr["res"], ["M1", "M4", "C1", "C2", "C3", "F1"], sr["best"])
    if ld("full_metr.pkl"):
        put("fm@nunits", str(len(ld("full_metr.pkl")["test_keys"])))
    if os.path.exists(f"{R}/dyn.json"):
        d = json.load(open(f"{R}/dyn.json"))
        put("dyn@v1sat", str(d["v1_sat"])); put("dyn@monosat", str(d["mono_sat"])); put("dyn@soft", f"{d['soft_final']:.2f}")
        put("dyn@vone", f"{d['v1_end']:.2f}"); put("dyn@star", f"{d['star_end']:.2f}"); put("dyn@cl", f"{d['cl_end']:.2f}")
    put("fullsec", "22")
    if "fm@E" in M: put("fm@deg", f"{2 * int(M['fm@E']) / int(M['fm@n']):.0f}")
    lams = [float(M[k]) for k in ("fm@MFour@lam", "pb@MFour@lam", "fp@MFour@lam") if k in M]
    if lams: put("scale@lammax", f"{max(lams):.3f}")
    def cellrel(pre, i):
        if f"{pre}{i}@rel" not in M: return "--"
        ph = M[f"{pre}{i}@ph"]
        return f"{M[f'{pre}{i}@rel']} ({ph})"
    rows = []
    cols = ["fm@", "pb@", "fp@"]
    rows.append("M1 MAE & " + " & ".join(M.get(f"{c}MOne@mae", "--") for c in cols) + " \\\\")
    rows.append("M1 dec.\\ acc. & " + " & ".join(M.get(f"{c}MOne@pol", "--") for c in cols) + " \\\\")
    rows.append("\\midrule")
    for i, nm in [("MFour", "M4"), ("COne", "C1"), ("CTwo", "C2"), ("CThree", "C3"), ("CFour", "C4"), ("FOne", "F1")]:
        rows.append(f"{nm} & " + " & ".join(cellrel(c, i) for c in cols) + " \\\\")
    rows.append("\\midrule")
    rows.append("M4 dec.\\ acc. & " + " & ".join(M.get(f"{c}MFour@pol", "--") for c in cols) + " \\\\")
    rows.append("M4 $\\bar\\lambda$ & " + " & ".join(M.get(f"{c}MFour@lam", "--") for c in cols) + " \\\\")
    open(os.path.join(os.path.dirname(OUT), "tabrow_scale.tex"), "w").write("\n".join(rows) + "\n")
    cols = ["sens@gA@", "", "sens@gB@", "sens@TA@", "sens@TB@"]
    rows = ["M1 MAE & " + " & ".join(M.get(f"{c}MOne@mae", "--") for c in cols) + " \\\\", "\\midrule"]
    for i, nm in [("MFour", "M4"), ("COne", "C1"), ("CTwo", "C2"), ("CThree", "C3"), ("FOne", "F1")]:
        rows.append(f"{nm} & " + " & ".join(cellrel(c, i) for c in cols) + " \\\\")
    rows.append("\\midrule")
    rows.append("M4 $\\bar\\lambda$ & " + " & ".join(M.get(f"{c}MFour@lam", "--") for c in cols) + " \\\\")
    open(os.path.join(os.path.dirname(OUT), "tabrow_sens.tex"), "w").write("\n".join(rows) + "\n")
    put("coderepo", r"\textcolor{red}{[GitHub URL and Zenodo DOI to be inserted]}")
    put("runtime", "3")
    with open(OUT, "w") as f:
        f.write("% generated by make_numbers.py -- do not edit\n\\makeatletter\n")
        for k, v in M.items():
            f.write(f"\\expandafter\\def\\csname qbn@{k}\\endcsname{{{v}}}\n")
        f.write("\\makeatother\n\\newcommand{\\Q}[1]{\\ifcsname qbn@#1\\endcsname\\csname qbn@#1\\endcsname\\else\\textbf{??#1}\\fi}\n")
    print(len(M), "macros written")


if __name__ == "__main__":
    main()
