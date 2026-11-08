"""Run one attribution method over one Who&When subset under one fixed condition.

Examples
  python scripts/run.py --method aao --subset alg --gt 1
  python scripts/run.py --method dcs --subset hand --gt 0
  python scripts/run.py --method dcs --subset alg --gt 1 --variant cascade
Results are appended to results/<model>/<variant>_<subset>_<method>_gt<g>_<obs>.jsonl (resumable).
"""
import argparse
import dataclasses
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traid import baselines, carr, cascade, data  # noqa: E402
from traid.llm import make_llm  # noqa: E402


def load_cascade(subset, llm, seed, limit=None):
    path = ROOT / "data" / "cascade" / f"{subset}_seed{seed}.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [data.Trajectory(**r) for r in rows]
    base = data.load(subset)[:limit]
    out, lock = [], threading.Lock()

    def job(tr):
        for attempt in range(6):
            try:
                tr2, _ = cascade.make_cascade(llm, tr, seed)
                return tr2
            except RuntimeError:
                time.sleep(60)
        raise RuntimeError(f"injection failed for {tr.tid}")

    with ThreadPoolExecutor(2) as ex:
        for f in as_completed([ex.submit(job, t) for t in base]):
            t2 = f.result()
            if t2 is not None:
                with lock:
                    out.append(t2)
    out.sort(key=lambda t: int(t.tid.split("/")[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dataclasses.asdict(t) for t in out]), encoding="utf-8")
    return out


def run_one(method, llm, tr, gt, obs, seed):
    if method == "aao":
        return baselines.all_at_once(llm, tr, with_gt=gt, obs=obs)
    if method == "aao_idx":
        return baselines.all_at_once(llm, tr, with_gt=gt, numbered=True, obs=obs)
    if method == "sbs":
        return baselines.step_by_step(llm, tr, with_gt=gt, obs=obs)
    if method == "bs":
        return baselines.binary_search(llm, tr, with_gt=gt, seed=seed, obs=obs)
    if method == "dcs":
        cands, raw, cost = carr.dense_candidates(llm, tr, with_gt=gt, obs=obs)
        top = cands[0] if cands else None
        return {"agent": top["agent"] if top else None, "step": top["step"] if top else None,
                "candidates": cands, "raw": raw[:3000], **cost.as_dict()}
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["aao", "aao_idx", "sbs", "bs", "dcs"])
    ap.add_argument("--subset", required=True, choices=["alg", "hand"])
    ap.add_argument("--gt", type=int, default=1)
    ap.add_argument("--obs", default="L0", choices=["L0", "L1"])
    ap.add_argument("--variant", default="orig", choices=["orig", "cascade"])
    ap.add_argument("--model", default="mistral-medium-latest")
    ap.add_argument("--inject_model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None, help="evaluate a fixed random subset of this size")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min_interval", type=float, default=0.0)
    a = ap.parse_args()

    llm = make_llm(a.model, max_tokens=2048 if a.method == "dcs" else 1024, min_interval_s=a.min_interval)
    if a.variant == "cascade":
        inj = make_llm(a.inject_model or a.model, max_tokens=4096, temperature=0.7)
        trajs = load_cascade(a.subset, inj, a.seed)
    else:
        trajs = data.load(a.subset)
    trajs = trajs[: a.limit] if a.limit else trajs
    if a.sample:  # fixed random subset (same ids for every method -> paired comparison)
        import random
        keep = set(random.Random(a.sample_seed).sample(sorted(t.tid for t in data.load(a.subset)), a.sample))
        trajs = [t for t in trajs if t.tid.replace("-cas/", "/") in keep]

    out = ROOT / "results" / a.model / f"{a.variant}_{a.subset}_{a.method}_gt{a.gt}_{a.obs}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["tid"])
    todo = [t for t in trajs if t.tid not in done]
    print(f"{out.name}: {len(done)} done, {len(todo)} to run", flush=True)
    lock = threading.Lock()
    t0 = time.time()

    def job(tr):
        try:
            pred = run_one(a.method, llm, tr, bool(a.gt), a.obs, a.seed)
        except RuntimeError as e:
            if "quota" in str(e).lower() or "401" in str(e) or "403" in str(e):
                raise
            with lock, out.with_suffix(".skipped").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"tid": tr.tid, "error": str(e)[:300]}) + "\n")
            print(f"  skipped {tr.tid}: {str(e)[:120]}", flush=True)
            return tr.tid
        row = {"tid": tr.tid, "T": tr.T, "gold_agent": tr.mistake_agent, "gold_step": tr.mistake_step,
               "extra_error_steps": tr.extra_error_steps, **pred}
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return tr.tid

    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, t) for t in todo]
        for f in as_completed(futs):
            try:
                f.result(); n += 1
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
            except RuntimeError as e:
                print("STOP:", e, flush=True)
                ex.shutdown(cancel_futures=True)
                sys.exit(2)
    print(f"finished {n} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
