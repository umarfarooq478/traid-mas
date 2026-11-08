"""Generate LaTeX tables + numbers.tex for the paper from results/*.jsonl (no LLM calls).

Every comparison is paired on the intersection of trajectory ids that all compared runs share.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as E  # noqa: E402
from traid import metrics as M  # noqa: E402

MAIN = "ministral-14b-latest"
TAB = ROOT / "paper" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
NUM = {}
LOG = {}


def load(model, stem):
    f = ROOT / "results" / model / f"{stem}.jsonl"
    if not f.exists():
        return {}
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["tid"]: r for r in reversed(rows)}


def predictor(model, variant, subset, method, gt=1, obs="L0"):
    """Return {tid: (row, (agent, step, ranked))} for a method name incl. derived DCS predictors."""
    derived = method in ("dcs_top1", "dcs_earliest", "dcs_origin", "carr_temporal", "carr")
    stem = f"{variant}_{subset}_{'dcs' if derived else method}_gt{gt}_{obs}"
    rows = load(model, stem)
    out = {}
    for tid, r in rows.items():
        if derived:
            out[tid] = (r, E.derive(r)[method])
        else:
            out[tid] = (r, (r.get("agent"), r.get("step"), []))
    return out


def stats(P, ids):
    res = {k: [] for k in ("agent", "step", "lenient", "tol1", "tol2", "top3", "rtr", "wre", "dr", "calls", "tok", "lat")}
    for tid in ids:
        r, (a, s, ranked) = P[tid]
        g = r["gold_step"]
        res["agent"].append(M.agent_correct(a, r["gold_agent"]))
        res["step"].append(M.step_correct(s, g))
        res["lenient"].append(M.step_correct(s, g, "lenient"))
        res["tol1"].append(M.step_correct(s, g, tol=1)); res["tol2"].append(M.step_correct(s, g, tol=2))
        res["top3"].append(M.topk_hit(ranked or ([s] if s is not None else []), g, 3))
        res["rtr"].append(s is not None and s <= g)
        if s is not None and s <= g:
            res["wre"].append((g - s) / r["T"])
        if r.get("extra_error_steps"):
            res["dr"].append(s in r["extra_error_steps"])
        res["calls"].append(r["calls"]); res["tok"].append(r["in_tok"] + r["out_tok"]); res["lat"].append(r["latency_s"])
    pct = lambda v: 100 * sum(v) / len(v) if v else float("nan")
    s = {k: pct(res[k]) for k in ("agent", "step", "lenient", "tol1", "tol2", "top3", "rtr", "dr")}
    s["wre"] = 100 * st.mean(res["wre"]) if res["wre"] else float("nan")
    s["calls"] = st.mean(res["calls"]); s["tok"] = st.mean(res["tok"]); s["lat"] = st.mean(res["lat"])
    s["step_ci"] = [100 * x for x in M.bootstrap_ci([float(x) for x in res["step"]])]
    s["n"] = len(ids); s["_step"] = res["step"]; s["_agent"] = res["agent"]
    return s


def f1(x):
    return "--" if x != x else f"{x:.1f}"


NAMES = {"aao": "All-at-Once~\\cite{zhang2025whowhen}", "aao_idx": "All-at-Once + step numbers", "sbs": "Step-by-Step~\\cite{zhang2025whowhen}",
         "bs": "Binary Search~\\cite{zhang2025whowhen}", "dcs_top1": "DCS top-1 (ours)", "dcs_earliest": "DCS earliest cand.",
         "dcs_origin": "DCS earliest `origin' tag", "carr_temporal": "DCS + CARR, temporal graph", "carr": "\\textbf{DCS + CARR (ours)}"}


def main_table(model, variant, subset, methods, ids_from, label, caption, extra_cols=True):
    P = {m: predictor(model, variant, subset, m) for m in methods}
    P = {m: p for m, p in P.items() if p}
    ids = None
    for m in ids_from:
        if m in P:
            ids = set(P[m]) if ids is None else ids & set(P[m])
    if not ids:
        return None
    for m in P:
        ids &= set(P[m]) if m in ids_from else ids
    ids = sorted(ids)
    S = {m: stats(P[m], ids) for m in P}
    ref = S.get("carr")
    lines = [f"\\begin{{table*}}[t]\n\\centering\n\\caption{{{caption} ($n={len(ids)}$).}}\n\\label{{{label}}}\n\\footnotesize",
             "\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule",
             "Method & Agent & Step & 95\\% CI & Acc@$\\pm$1 & Top-3 & RTR & WRE$\\downarrow$ & Calls & Tokens & $p$ vs CARR \\\\\n\\midrule"]
    for m, s in S.items():
        p = "--"
        if ref is not None and m != "carr":
            b01, b10, pv = M.mcnemar_exact(s["_step"], ref["_step"])
            p = f"{pv:.3f}"
            LOG[f"{label}:{m}"] = (b01, b10, pv)
        lines.append(f"{NAMES[m]} & {f1(s['agent'])} & {f1(s['step'])} & [{s['step_ci'][0]:.1f}, {s['step_ci'][1]:.1f}] & {f1(s['tol1'])} & "
                     f"{f1(s['top3'])} & {f1(s['rtr'])} & {f1(s['wre'])} & {s['calls']:.1f} & {s['tok'] / 1000:.1f}k & {p} \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")
    (TAB / f"{label.split(':')[1]}.tex").write_text("\n".join(lines), encoding="utf-8")
    return S, ids


def macro(name, val):
    NUM[name] = val


def main():
    out = {}
    # ---------- reproduction (alg, subset where available) ----------
    reported = {"aao": (54.33, 12.50), "sbs": (35.20, 25.51), "bs": (44.13, 23.98)}
    rep = []
    for subset in ("alg", "hand"):
        for m in ("aao", "sbs", "bs"):
            P = predictor(MAIN, "orig", subset, m)
            if not P:
                continue
            s = stats(P, sorted(P))
            out[f"repro_{subset}_{m}"] = {k: v for k, v in s.items() if not k.startswith("_")}
            rep.append((subset, m, s))
    rep_hand_reported = {"aao": (55.17, 5.26), "sbs": (34.48, 7.02), "bs": (51.72, 6.90)}
    lines = ["\\begin{table}[t]\n\\centering\n\\caption{Reproduction of the Who\\&When baselines (with ground truth). Reported: GPT-4o, Table~1 of~\\cite{zhang2025whowhen}. Reproduced: \\texttt{ministral-14b}; strict / lenient (official substring) scoring.}\n\\label{tab:repro}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n & & \\multicolumn{2}{c}{Reported} & \\multicolumn{3}{c}{Reproduced} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\nSubset & Method & Agent & Step & $n$ & Agent & Step (strict / len.) \\\\\n\\midrule"]
    for subset, m, s in rep:
        ra, rs = (reported if subset == "alg" else rep_hand_reported)[m]
        lines.append(f"{'Alg.' if subset == 'alg' else 'Hand'} & {m.upper()} & {ra:.1f} & {rs:.1f} & {s['n']} & {f1(s['agent'])} & {f1(s['step'])} / {f1(s['lenient'])} \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    (TAB / "repro.tex").write_text("\n".join(lines), encoding="utf-8")

    # ---------- main comparisons ----------
    full = main_table(MAIN, "orig", "alg", ["aao", "dcs_top1", "carr"], ["aao", "carr"], "tab:main_full",
                      "Algorithm-generated subset, all trajectories: baseline vs. proposed (L0, with ground truth)")
    sub = main_table(MAIN, "orig", "alg", ["aao", "sbs", "bs", "dcs_top1", "carr"], ["sbs", "bs"], "tab:main_sub",
                     "Paired comparison of all methods on the fixed 60-trajectory algorithm-generated subset")
    hand = main_table(MAIN, "orig", "hand", ["aao", "dcs_top1", "carr"], ["aao", "carr"], "tab:main_hand",
                      "Hand-crafted (Magentic-One) subset, logs within the API token budget")
    abl = main_table(MAIN, "orig", "alg", ["dcs_top1", "dcs_earliest", "dcs_origin", "carr_temporal", "carr"], ["carr"], "tab:ablation",
                     "Ablation of the resolution step on the same DCS candidates (algorithm-generated)")
    for key, res in (("full", full), ("sub", sub), ("hand", hand), ("abl", abl)):
        if res:
            out[key] = {m: {k: v for k, v in s.items() if not k.startswith("_")} for m, s in res[0].items()}

    # ---------- cascade (paired with the same ids on the original logs) ----------
    cas_rows = []
    for model in (MAIN, "open-mistral-nemo"):
        for m in ("aao", "bs", "dcs_top1", "carr_temporal", "carr"):
            Pc = predictor(model, "cascade", "alg", m)
            Po = predictor(model, "orig", "alg", m)
            if not Pc or not Po:
                continue
            ids = sorted(t for t in Pc if t.replace("-cas/", "/") in Po)
            Po2 = {t: Po[t.replace("-cas/", "/")] for t in ids}
            sc, so = stats(Pc, ids), stats(Po2, ids)
            cas_rows.append((model, m, so, sc))
            out[f"cascade_{model}_{m}"] = {"orig": {k: v for k, v in so.items() if not k.startswith("_")},
                                           "cas": {k: v for k, v in sc.items() if not k.startswith("_")}}
    lines = ["\\begin{table}[t]\n\\centering\n\\caption{TRAID-Cascade: step accuracy on the original logs vs.\\ the same logs with one injected downstream error, and distraction rate (DR, share of predictions blaming the injected step).}\n\\label{tab:cascade}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\nJudge & Method & $n$ & Orig. & Cascade & $\\Delta$ & DR$\\downarrow$ \\\\\n\\midrule"]
    for model, m, so, sc in cas_rows:
        lines.append(f"{'14B' if model == MAIN else 'Nemo'} & {NAMES[m].replace('~\\cite{zhang2025whowhen}', '')} & {sc['n']} & {f1(so['step'])} & {f1(sc['step'])} & {sc['step'] - so['step']:+.1f} & {f1(sc['dr'])} \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    (TAB / "cascade.tex").write_text("\n".join(lines), encoding="utf-8")

    # ---------- conditions (O1): gt, obs, numbering ----------
    cond = []
    for label, m, gt, obs in (("AAO, L0, $g{=}1$", "aao", 1, "L0"), ("AAO, L0, $g{=}0$", "aao", 0, "L0"), ("AAO + numbering, L0, $g{=}1$", "aao_idx", 1, "L0"),
                              ("CARR, L0, $g{=}1$", "carr", 1, "L0"), ("CARR, L0, $g{=}0$", "carr", 0, "L0"), ("CARR, L1, $g{=}1$", "carr", 1, "L1")):
        P = predictor(MAIN, "orig", "alg", m, gt, obs)
        if P:
            s = stats(P, sorted(P)); cond.append((label, s))
            out[f"cond_{m}_{gt}_{obs}"] = {k: v for k, v in s.items() if not k.startswith("_")}
    lines = ["\\begin{table}[t]\n\\centering\n\\caption{Effect of evaluation conditions on the algorithm-generated subset (one factor changed at a time).}\n\\label{tab:conditions}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nCondition & $n$ & Agent & Step & Step (len.) & RTR \\\\\n\\midrule"]
    for label, s in cond:
        lines.append(f"{label} & {s['n']} & {f1(s['agent'])} & {f1(s['step'])} & {f1(s['lenient'])} & {f1(s['rtr'])} \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    (TAB / "conditions.tex").write_text("\n".join(lines), encoding="utf-8")

    # ---------- backbone transfer ----------
    bb = main_table("open-mistral-nemo", "orig", "alg", ["aao", "dcs_top1", "carr"], ["aao", "carr"], "tab:backbone",
                    "Second backbone (\\texttt{open-mistral-nemo}), algorithm-generated subset")
    if bb:
        out["backbone"] = {m: {k: v for k, v in s.items() if not k.startswith("_")} for m, s in bb[0].items()}

    cas_path = ROOT / "data" / "cascade" / "alg_seed0.json"
    if cas_path.exists():
        macro("NcasAlg", str(len(json.loads(cas_path.read_text(encoding="utf-8")))))
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in NUM.items()) +
                                                ("\n" if NUM else "") + ("" if "NcasAlg" in NUM else "\\newcommand{\\NcasAlg}{\\TBD{n}}\n"), encoding="utf-8")
    out["mcnemar"] = {k: list(v) for k, v in LOG.items()}
    (ROOT / "results" / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv['step'], 1) if isinstance(vv, dict) and 'step' in vv else vv) for kk, vv in v.items()}) for k, v in out.items()}, indent=0, default=str)[:4000])


if __name__ == "__main__":
    main()
