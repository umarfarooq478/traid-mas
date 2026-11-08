"""CARR gamma/beta sensitivity on stored DCS candidates (zero LLM calls). Writes results/sensitivity.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as E  # noqa: E402

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
out = {}
for f in sorted((ROOT / "results").glob("*/*_dcs_*.jsonl")):
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    grid = {}
    for g in GAMMAS:
        for b in BETAS:
            acc = [E.derive(r, g, b)["carr"][1] == r["gold_step"] for r in rows if r.get("candidates")]
            dis = [E.derive(r, g, b)["carr"][1] in r["extra_error_steps"] for r in rows if r.get("candidates") and r["extra_error_steps"]]
            grid[f"{g}|{b}"] = {"step": round(100 * sum(acc) / max(1, len(rows)), 2),
                                "distract": round(100 * sum(dis) / len(dis), 2) if dis else None}
    out[f"{f.parent.name}/{f.stem}"] = grid
    print(f.stem, {k: v["step"] for k, v in grid.items()})
(ROOT / "results" / "sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
