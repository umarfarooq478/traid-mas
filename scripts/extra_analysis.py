"""Zero-LLM-call analyses requested in the revision (writes results/extra_analysis.json).

  offsets      signed error t_hat - t* distribution per method (is AAO off by one in a fixed direction?)
  paired       paired bootstrap CI of the accuracy difference DCS - AAO
  position     step accuracy split by gold position (t* = 0 vs t* > 0) and by trajectory length
  calibration  is the DCS top-1 probability calibrated? (ECE, Brier, reliability bins)
  ceiling      top-K coverage of the DCS list = upper bound for any re-ranker of that list
  cascade      manipulation check (is the injected step in the DCS list?), pre-injection rate of
               predicting the same step e on the ORIGINAL log, and chance DR = E[1/T]
  schedule     rollback-schedule proxy: try pi_1, pi_2, ... until pi_k <= t*; count re-executed steps
  agreement    agreement of the two judges' DCS top-1
  aao_consist  how often AAO's named agent equals the agent who spoke at its predicted step
"""
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traid import data, metrics as M  # noqa: E402

R = ROOT / "results"


def load(model, stem):
    f = R / model / f"{stem}.jsonl"
    if not f.exists():
        return {}
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for r in rows:
        out.setdefault(r["tid"], r)  # keep first
    return out


def acc(rows, ids, key="step"):
    return [r[key] is not None and int(r[key]) == int(r["gold_step"]) for r in (rows[i] for i in ids)]


def paired_boot(a, b, n=5000, seed=0):
    rng = random.Random(seed); m = len(a)
    d = sorted(sum(b[j] - a[j] for j in (rng.randrange(m) for _ in range(m))) / m for _ in range(n))
    return round(100 * (sum(b) - sum(a)) / m, 1), [round(100 * d[int(.025 * n)], 1), round(100 * d[int(.975 * n) - 1], 1)]


def offsets(rows):
    c = Counter()
    for r in rows.values():
        c["none" if r["step"] is None else max(-3, min(3, int(r["step"]) - r["gold_step"]))] += 1
    n = sum(c.values())
    return {str(k): round(100 * v / n, 1) for k, v in sorted(c.items(), key=lambda x: (isinstance(x[0], str), x[0]))}


def calibration(rows, bins=(0, .5, .7, .8, .9, 1.01)):
    ps, ys = [], []
    for r in rows.values():
        c = r.get("candidates") or []
        if not c:
            continue
        ps.append(c[0]["p"]); ys.append(int(c[0]["step"] == r["gold_step"]))
    out, ece = [], 0.0
    for lo, hi in zip(bins, bins[1:]):
        idx = [i for i, p in enumerate(ps) if lo <= p < hi]
        if idx:
            mp = st.mean(ps[i] for i in idx); ma = st.mean(ys[i] for i in idx)
            ece += len(idx) / len(ps) * abs(mp - ma)
            out.append({"bin": [lo, min(hi, 1.0)], "n": len(idx), "mean_p": round(mp, 3), "acc": round(ma, 3)})
    brier = st.mean((p - y) ** 2 for p, y in zip(ps, ys))
    return {"n": len(ps), "mean_top_p": round(st.mean(ps), 3), "top1_acc": round(st.mean(ys), 3),
            "ECE": round(ece, 3), "Brier": round(brier, 3), "bins": out}


def ceiling(rows):
    out = {}
    for k in (1, 2, 3, 5):
        out[f"top{k}"] = round(100 * st.mean(r["gold_step"] in [c["step"] for c in (r.get("candidates") or [])[:k]]
                                             for r in rows.values()), 1)
    out["mean_list_len"] = round(st.mean(len(r.get("candidates") or []) for r in rows.values()), 2)
    return out


def position(rows, ids):
    g = {"t*=0": [], "t*>0": [], "T<=6": [], "T>6": []}
    for i in ids:
        r = rows[i]; ok = r["step"] is not None and int(r["step"]) == r["gold_step"]
        g["t*=0" if r["gold_step"] == 0 else "t*>0"].append(ok)
        g["T<=6" if r["T"] <= 6 else "T>6"].append(ok)
    return {k: {"n": len(v), "acc": round(100 * st.mean(v), 1) if v else None} for k, v in g.items()}


def schedule(rows, ranked_fn):
    """Rollback schedule proxy. Attempts until a rollback point <= t* is tried; if the list is exhausted,
    fall back to step 0. Cost = re-executed steps summed over attempts, as a share of T."""
    att, cost = [], []
    for r in rows.values():
        T, g = r["T"], r["gold_step"]
        sched = [s for s in ranked_fn(r) if s is not None]
        n, c = 0, 0
        for s in sched + [0]:
            n += 1; c += T - s
            if s <= g:
                break
        att.append(n); cost.append(c / T)
    return {"mean_attempts": round(st.mean(att), 2), "first_try_success": round(100 * st.mean(a == 1 for a in att), 1),
            "mean_reexec_T": round(st.mean(cost), 2)}


def cascade(model, orig_dcs, cas_dcs, cas_aao, orig_aao):
    out = {}
    ids = sorted(cas_dcs)
    if not ids:
        return out
    e_of = {i: cas_dcs[i]["extra_error_steps"][0] for i in ids}
    oid = lambda i: i.replace("-cas/", "/")
    out["n"] = len(ids)
    out["chance_DR"] = round(100 * st.mean(1 / cas_dcs[i]["T"] for i in ids), 1)
    out["inj_in_list"] = round(100 * st.mean(e_of[i] in [c["step"] for c in cas_dcs[i]["candidates"]] for i in ids), 1)
    pin = [next((c["p"] for c in cas_dcs[i]["candidates"] if c["step"] == e_of[i]), 0.0) for i in ids]
    pout = [next((c["p"] for c in orig_dcs[oid(i)]["candidates"] if c["step"] == e_of[i]), 0.0) for i in ids if oid(i) in orig_dcs]
    out["mean_p_on_e_cascade"] = round(st.mean(pin), 3)
    out["mean_p_on_e_original"] = round(st.mean(pout), 3)
    out["inj_in_list_original"] = round(100 * st.mean(e_of[i] in [c["step"] for c in orig_dcs[oid(i)]["candidates"]]
                                                       for i in ids if oid(i) in orig_dcs), 1)
    for name, cas, orig in (("dcs", cas_dcs, orig_dcs), ("aao", cas_aao, orig_aao)):
        common = [i for i in ids if i in cas and oid(i) in orig]
        if not common:
            continue
        pre = [orig[oid(i)]["step"] == e_of[i] for i in common]
        post = [cas[i]["step"] == e_of[i] for i in common]
        a = [orig[oid(i)]["step"] == orig[oid(i)]["gold_step"] for i in common]
        b = [cas[i]["step"] is not None and cas[i]["step"] == cas[i]["gold_step"] for i in common]
        n01, n10, p = M.mcnemar_exact(a, b)
        out[name] = {"n": len(common), "DR_pre_injection": round(100 * st.mean(pre), 1),
                     "DR_post_injection": round(100 * st.mean(post), 1),
                     "acc_orig": round(100 * st.mean(a), 1), "acc_cascade": round(100 * st.mean(b), 1),
                     "mcnemar_orig_vs_cascade": [n01, n10, round(p, 4)]}
    return out


def main():
    out = {}
    for model in ("ministral-14b-latest", "open-mistral-nemo"):
        aao, dcs = load(model, "orig_alg_aao_gt1_L0"), load(model, "orig_alg_dcs_gt1_L0")
        ids = sorted(set(aao) & set(dcs))
        res = {"n": len(ids)}
        res["offsets_aao"] = offsets({i: aao[i] for i in ids})
        res["offsets_dcs"] = offsets({i: dcs[i] for i in ids})
        res["paired_diff_dcs_minus_aao"] = paired_boot(acc(aao, ids), acc(dcs, ids))
        res["position_aao"] = position(aao, ids); res["position_dcs"] = position(dcs, ids)
        res["calibration_dcs"] = calibration(dcs)
        res["ceiling_dcs"] = ceiling(dcs)
        res["schedule_dcs"] = schedule(dcs, lambda r: [c["step"] for c in r["candidates"]])
        res["schedule_aao"] = schedule(aao, lambda r: [r["step"]])
        res["aao_agent_matches_its_step"] = round(100 * st.mean(
            aao[i]["step"] is not None and aao[i]["step"] < aao[i]["T"] and aao[i]["agent"] is not None
            and M.norm_agent(aao[i]["agent"]) == M.norm_agent(data_steps(i)[aao[i]["step"]][0]) for i in ids), 1)
        res["aao_agent_ok_step_wrong"] = round(100 * st.mean(
            M.agent_correct(aao[i]["agent"], aao[i]["gold_agent"]) and aao[i]["step"] != aao[i]["gold_step"] for i in ids), 1)
        cas_dcs, cas_aao = load(model, "cascade_alg_dcs_gt1_L0"), load(model, "cascade_alg_aao_gt1_L0")
        res["cascade"] = cascade(model, dcs, cas_dcs, cas_aao, aao)
        cal = load(model, "cascade_alg_dcs_gt1_L0")
        if cal:
            res["calibration_dcs_cascade"] = calibration(cal)
        out[model] = res
    a, b = load("ministral-14b-latest", "orig_alg_dcs_gt1_L0"), load("open-mistral-nemo", "orig_alg_dcs_gt1_L0")
    ids = sorted(set(a) & set(b))
    out["judge_agreement_dcs_top1"] = round(100 * st.mean(a[i]["step"] == b[i]["step"] for i in ids), 1)
    out["both_correct"] = round(100 * st.mean(a[i]["step"] == b[i]["step"] == a[i]["gold_step"] for i in ids), 1)
    out["either_correct"] = round(100 * st.mean(a[i]["step"] == a[i]["gold_step"] or b[i]["step"] == b[i]["gold_step"]
                                                for i in ids), 1)
    (R / "extra_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


_D = {}


def data_steps(tid):
    if not _D:
        for t in data.load("alg"):
            _D[t.tid] = t.steps
    return _D[tid]


if __name__ == "__main__":
    main()
