"""Exploratory (post hoc, zero LLM calls): does CARR act once DCS probabilities are flattened?
p_k <- p_k^(1/tau) / Z. tau = 1 is the pre-registered CARR. Reported as a full curve; no tau is selected."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as E
from traid import carr, metrics as M
out = {}
for model in ("ministral-14b-latest", "open-mistral-nemo"):
    for var in ("orig", "cascade"):
        f = ROOT / "results" / model / f"{var}_alg_dcs_gt1_L0.jsonl"
        rows = list({(r := json.loads(l))["tid"]: r for l in f.read_text().splitlines()}.values())
        base = None
        for tau in (1, 2, 3, 5, 10, 1000):
            ok, dr, top1 = [], [], []
            for r in rows:
                c = [dict(x) for x in r.get("candidates") or []]
                if not c:
                    ok.append(False); top1.append(False); continue
                z = sum(x["p"] ** (1 / tau) for x in c) or 1
                for x in c:
                    x["p"] = x["p"] ** (1 / tau) / z
                P = carr.path_strength(carr.provenance_graph(E.traj(r["tid"])))
                s = carr.carr(c, P, 1.0, 0.5)[0]["step"]
                ok.append(s == r["gold_step"]); top1.append(c[0]["step"] == r["gold_step"])
                if r.get("extra_error_steps"):
                    dr.append(s in r["extra_error_steps"])
            if base is None:
                base = top1
            n01, n10, p = M.mcnemar_exact(base, ok)
            key = f"{model}/{var}/tau={tau}"
            out[key] = {"n": len(rows), "acc": round(100 * sum(ok) / len(ok), 1), "dr": round(100 * sum(dr) / len(dr), 1) if dr else None,
                        "vs_dcs_top1": [n01, n10, round(p, 3)]}
            print(key, out[key])
(ROOT / "results" / "flatten_carr.json").write_text(json.dumps(out, indent=1))
