"""Figures for the paper from results/analysis.json and results/sensitivity.json (run make_tables.py and sensitivity.py first)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
A = json.loads((ROOT / "results" / "analysis.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
LBL = {"aao": "AAO", "sbs": "SBS", "bs": "BS", "dcs_top1": "DCS", "carr": "DCS+CARR", "dcs_earliest": "earliest", "dcs_origin": "origin tag", "carr_temporal": "CARR-temp."}
C_BASE, C_OURS = "#8a8f98", "#1f5fa8"

# Fig 2: accuracy and rollback-trigger rate on the paired subset (or full set)
src = A.get("sub") or A.get("full")
if src:
    ms = list(src)
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.7))
    for k, (key, title) in enumerate((("step", "Step accuracy (%)"), ("rtr", "Rollback-trigger rate (%)"))):
        vals = [src[m][key] for m in ms]
        cols = [C_OURS if m in ("carr", "dcs_top1") else C_BASE for m in ms]
        ax[k].bar(range(len(ms)), vals, color=cols, width=0.7)
        if key == "step":
            lo = [src[m][key] - src[m]["step_ci"][0] for m in ms]; hi = [src[m]["step_ci"][1] - src[m][key] for m in ms]
            ax[k].errorbar(range(len(ms)), vals, yerr=[lo, hi], fmt="none", ecolor="black", lw=0.6, capsize=1.5)
        ax[k].set_xticks(range(len(ms))); ax[k].set_xticklabels([LBL[m] for m in ms], rotation=45, ha="right")
        ax[k].set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "main_bars.pdf", bbox_inches="tight")

# Fig 3: cost vs step accuracy
if src:
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for m in src:
        ax.scatter(src[m]["tok"] / 1000, src[m]["step"], color=C_OURS if m in ("carr", "dcs_top1") else C_BASE, s=18)
        ax.annotate(LBL[m], (src[m]["tok"] / 1000, src[m]["step"]), textcoords="offset points", xytext=(3, 2), fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Tokens per trajectory (thousands, log)"); ax.set_ylabel("Step accuracy (%)")
    fig.tight_layout(); fig.savefig(FIG / "cost.pdf", bbox_inches="tight")

# Fig 4: cascade degradation and distraction
cas = [(k.split("_", 2)[1:], v) for k, v in A.items() if k.startswith("cascade_ministral")]
if cas:
    ms = [k[1].replace("latest_", "") for k, _ in cas]
    ms = [k.split("latest_")[-1] if "latest_" in k else k for k in [c[0][1] for c in cas]]
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    x = range(len(cas)); w = 0.27
    ax.bar([i - w for i in x], [v["orig"]["step"] for _, v in cas], w, color="#c9ced6", label="original")
    ax.bar(list(x), [v["cas"]["step"] for _, v in cas], w, color=C_OURS, label="cascade")
    ax.bar([i + w for i in x], [v["cas"]["dr"] for _, v in cas], w, color="#c0392b", label="distraction")
    ax.set_xticks(list(x)); ax.set_xticklabels([LBL.get(m, m) for m in ms], rotation=30, ha="right"); ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(); fig.savefig(FIG / "cascade.pdf", bbox_inches="tight")

# Fig 5: CARR sensitivity heatmap (main model, original alg)
sp = ROOT / "results" / "sensitivity.json"
if sp.exists():
    S = json.loads(sp.read_text(encoding="utf-8"))
    key = next((k for k in S if k.startswith("ministral-14b-latest/orig_alg_dcs_gt1_L0")), None)
    if key:
        g = sorted({float(k.split("|")[0]) for k in S[key]}); b = sorted({float(k.split("|")[1]) for k in S[key]})
        Z = [[S[key][f"{gi}|{bi}"]["step"] for bi in b] for gi in g]
        fig, ax = plt.subplots(figsize=(2.6, 2.0))
        im = ax.imshow(Z, cmap="Blues", origin="lower", aspect="auto")
        for i in range(len(g)):
            for j in range(len(b)):
                ax.text(j, i, f"{Z[i][j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if Z[i][j] > (max(map(max, Z)) + min(map(min, Z))) / 2 else "black")
        ax.set_xticks(range(len(b))); ax.set_xticklabels(b); ax.set_yticks(range(len(g))); ax.set_yticklabels(g)
        ax.set_xlabel(r"$\beta$ (explanation bonus)"); ax.set_ylabel(r"$\gamma$ (symptom discount)")
        fig.colorbar(im, ax=ax, fraction=0.05, label="step acc. (%)")
        fig.tight_layout(); fig.savefig(FIG / "sensitivity.pdf", bbox_inches="tight")
print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
