"""Score every results file and write results/summary.json + results/summary.csv.

Derived (zero-extra-call) predictors from one DCS run:
  dcs_top1        highest-probability candidate (DCS alone)
  dcs_earliest    earliest listed candidate (positional heuristic, ablation)
  dcs_origin      earliest candidate the LLM itself labelled 'origin' (LLM-only root resolution, ablation)
  carr_temporal   CARR with a complete temporal graph instead of the provenance graph (ablation)
  carr            full method: DCS + CARR over the Artifact Provenance Graph
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traid import carr, data, metrics as M  # noqa: E402

_TRAJ = {}


def traj(tid):
    if not _TRAJ:
        for s in ("alg", "hand"):
            for t in data.load(s):
                _TRAJ[t.tid] = t
        for p in (ROOT / "data" / "cascade").glob("*.json"):
            for r in json.loads(p.read_text(encoding="utf-8")):
                _TRAJ[r["tid"]] = data.Trajectory(**r)
    return _TRAJ[tid]


def derive(row, gamma=1.0, beta=0.5):
    """Return {predictor: (agent, step, ranked_steps)} for a DCS row."""
    c = row.get("candidates") or []
    out = {}
    if not c:
        return {k: (None, None, []) for k in ("dcs_top1", "dcs_earliest", "dcs_origin", "carr_temporal", "carr")}
    tr = traj(row["tid"])
    out["dcs_top1"] = (c[0]["agent"], c[0]["step"], [x["step"] for x in c])
    e = min(c, key=lambda x: x["step"])
    out["dcs_earliest"] = (e["agent"], e["step"], sorted(x["step"] for x in c))
    o = [x for x in c if x.get("kind", "").startswith("origin")]
    o = min(o, key=lambda x: x["step"]) if o else c[0]
    out["dcs_origin"] = (o["agent"], o["step"], [x["step"] for x in c])
    W = carr.provenance_graph(tr)
    for name, mode in (("carr_temporal", "temporal"), ("carr", "provenance")):
        r = carr.carr(c, carr.path_strength(W, mode), gamma, beta)
        out[name] = (r[0]["agent"], r[0]["step"], [x["step"] for x in r])
    return out


def score(rows, getter):
    res = {k: [] for k in ("agent_s", "agent_l", "step_s", "step_l", "tol1", "tol2", "tol3", "top3", "rtr", "late",
                           "distract", "reexec")}
    for r in rows:
        agent, step, ranked = getter(r)
        g_a, g_s, T = r["gold_agent"], r["gold_step"], r["T"]
        res["agent_s"].append(M.agent_correct(agent, g_a, "strict"))
        res["agent_l"].append(M.agent_correct(agent, g_a, "lenient"))
        res["step_s"].append(M.step_correct(step, g_s, "strict"))
        res["step_l"].append(M.step_correct(step, g_s, "lenient"))
        for t in (1, 2, 3):
            res[f"tol{t}"].append(M.step_correct(step, g_s, "strict", tol=t))
        res["top3"].append(M.topk_hit(ranked or ([step] if step is not None else []), g_s, 3))
        res["rtr"].append(step is not None and step <= g_s)            # rollback at p re-generates t*
        res["late"].append(step is None or step > g_s)
        if r.get("extra_error_steps"):
            res["distract"].append(step in r["extra_error_steps"])
        if step is not None and step <= g_s:
            res["reexec"].append((g_s - step) / max(1, T))                # wasted re-execution before t*
    return res


def summarise(name, rows, res):
    n = len(rows)
    s = {"run": name, "n": n}
    for k, v in res.items():
        if v:
            s[k] = round(100 * sum(v) / len(v), 2) if k != "reexec" else round(100 * st.mean(v), 2)
    lo, hi = M.bootstrap_ci([float(x) for x in res["step_s"]])
    s["step_s_ci"] = [round(100 * lo, 1), round(100 * hi, 1)]
    lo, hi = M.bootstrap_ci([float(x) for x in res["agent_s"]])
    s["agent_s_ci"] = [round(100 * lo, 1), round(100 * hi, 1)]
    for k in ("calls", "in_tok", "out_tok", "latency_s"):
        if rows and k in rows[0]:
            s[k] = round(st.mean(r[k] for r in rows), 2)
    return s


def main():
    summary, per_item = [], {}
    for f in sorted((ROOT / "results").glob("*/*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = list({r["tid"]: r for r in reversed(rows)}.values())  # de-duplicate (keep first)
        rows.sort(key=lambda r: r["tid"])
        base = f"{f.parent.name}/{f.stem}"
        if "_dcs_" in f.stem:
            D = [derive(r) for r in rows]
            for pred in ("dcs_top1", "dcs_earliest", "dcs_origin", "carr_temporal", "carr"):
                name = base.replace("_dcs_", f"_{pred}_")
                res = score(rows, lambda r, _d=iter(D), _p=pred: next(_d)[_p])
                summary.append(summarise(name, rows, res)); per_item[name] = (rows, res)
        else:
            res = score(rows, lambda r: (r.get("agent"), r.get("step"), []))
            summary.append(summarise(base, rows, res)); per_item[base] = (rows, res)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    keys = sorted({k for s in summary for k in s}, key=lambda k: (k != "run", k != "n", k))
    with (ROOT / "results" / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(summary)
    for s in summary:
        print(f"{s['run']:<60} n={s['n']:<4} agent={s.get('agent_s')}/{s.get('agent_l')} step={s.get('step_s')}/{s.get('step_l')} "
              f"tol1={s.get('tol1')} top3={s.get('top3')} rtr={s.get('rtr')} distr={s.get('distract', '-')} calls={s.get('calls')} tok={s.get('in_tok')}")
    return per_item


if __name__ == "__main__":
    main()
