"""TRAID-MAS attribution: Dense Candidate Scoring (DCS) + Cascade-Aware Root Resolution (CARR).

DCS  : one LLM call returns a *set* of candidate erroneous steps with confidences,
       instead of a single (agent, step) pair.
APG  : an Artifact Provenance Graph built deterministically (no LLM) from the trace:
       an edge j -> i exists when step i reuses a concrete artifact (number, URL,
       file name, identifier, quoted string) that was first introduced at step j.
CARR : re-ranks candidates so that a candidate whose likely-erroneous ancestors can
       explain it (a propagated symptom) is discounted, and a candidate that explains
       likely-erroneous descendants (a root) is promoted.

    r_i = p_i * prod_{j in C, j ~> i} (1 - gamma * p_j * w(j ~> i))
              * (1 + beta * sum_{k in C, i ~> k} p_k * w(i ~> k))
"""
from __future__ import annotations

import json
import re

from .baselines import Cost
from .data import Trajectory, render, role_block

# ---------------------------------------------------------------- DCS (LLM)

DCS_SYSTEM = "You are an expert debugger of LLM multi-agent systems. You answer with strict JSON only."


def dcs_prompt(tr: Trajectory, with_gt: bool, obs: str, k: int) -> str:
    gt = f"The correct final answer is: {tr.ground_truth}\n" if with_gt else ""
    return (
        "A team of LLM agents tried to solve the problem below and FAILED. Every message in the log is a numbered step.\n"
        f"Problem: {tr.question}\n{gt}"
        + (role_block(tr) if obs == "L1" else "") +
        "Failure log:\n" + render(tr, numbered=True) + "\n\n"
        f"List up to {k} candidate steps that contain a genuine error (wrong fact, wrong reasoning, wrong tool use, "
        "ignoring instructions, premature termination, failure to verify). Errors may cascade: a later step can look wrong "
        "only because it reused an earlier wrong result. For each candidate give the probability (0-1) that it is the "
        "DECISIVE error, i.e. the earliest error that, if corrected, would most likely have led to success; probabilities "
        "should sum to at most 1. Mark each candidate as 'origin' (introduces a new error) or 'propagated' (inherits an earlier one).\n"
        "Return ONLY JSON of the form:\n"
        '{"candidates": [{"step": <int>, "agent": "<agent name exactly as in the log>", "p": <float>, '
        '"kind": "origin|propagated", "reason": "<short>"}]}'
    )


def parse_dcs(text: str, tr: Trajectory):
    m = re.search(r"\{.*\}", text, re.S)
    cands = []
    if m:
        try:
            obj = json.loads(m.group(0))
            for c in obj.get("candidates", []):
                s = int(c.get("step"))
                if 0 <= s < tr.T:
                    cands.append({"step": s, "agent": tr.steps[s][0], "p": max(0.0, float(c.get("p", 0))),
                                  "kind": str(c.get("kind", "")), "llm_agent": str(c.get("agent", ""))})
        except Exception:
            pass
    # merge duplicates
    merged = {}
    for c in cands:
        if c["step"] in merged:
            merged[c["step"]]["p"] += c["p"]
        else:
            merged[c["step"]] = c
    cands = sorted(merged.values(), key=lambda c: -c["p"])
    z = sum(c["p"] for c in cands)
    if z > 1:
        for c in cands:
            c["p"] /= z
    return cands


def dense_candidates(llm, tr: Trajectory, with_gt=True, obs="L0", k=5):
    cost = Cost()
    r = llm(dcs_prompt(tr, with_gt, obs, k), system=DCS_SYSTEM)
    cost.add(r)
    return parse_dcs(r.text, tr), r.text, cost


# ---------------------------------------------------------------- APG (no LLM)

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*(?![\w])")
_URL = re.compile(r"https?://[^\s'\"<>)\]]+")
_FILE = re.compile(r"\b[\w\-]+\.(?:py|csv|xlsx|xls|json|txt|pdf|png|jpg|mp3|pptx|docx|md|html|zip)\b", re.I)
_IDENT = re.compile(r"\b[a-zA-Z]+(?:_[a-zA-Z0-9]+)+\b|\b[a-z]+[A-Z][a-zA-Z0-9]+\b")
_QUOTE = re.compile(r"[\"']([^\"'\n]{4,60})[\"']")
_STOP_NUM = {str(i) for i in range(0, 13)} | {"100", "1000", "2023", "2024", "2025"}


def artifacts(text: str) -> set:
    a = set()
    for n in _NUM.findall(text):
        n = n.replace(",", "").rstrip(".")
        if n and n not in _STOP_NUM and len(n.strip("-")) >= 2:
            a.add("n:" + n)
    a |= {"u:" + u.rstrip(".,") for u in _URL.findall(text)}
    a |= {"f:" + f.lower() for f in _FILE.findall(text)}
    a |= {"i:" + i for i in _IDENT.findall(text) if len(i) >= 6}
    a |= {"q:" + q.strip().lower() for q in _QUOTE.findall(text)}
    return a


def provenance_graph(tr: Trajectory, question_artifacts: bool = True):
    """Return W[j][i] in [0,1]: share of step i's reused artifacts that originate at step j (j < i)."""
    T = tr.T
    origin = {}
    if question_artifacts:
        for x in artifacts(tr.question + " " + tr.ground_truth):
            origin[x] = -1          # artifacts given in the task are nobody's fault
    A = [artifacts(c) for _, c in tr.steps]
    W = [[0.0] * T for _ in range(T)]
    for i in range(T):
        reused = [x for x in A[i] if x in origin and origin[x] >= 0]
        if reused:
            for x in reused:
                W[origin[x]][i] += 1.0 / len(reused)
        for x in A[i]:
            origin.setdefault(x, i)
    return W


def path_strength(W, mode: str = "provenance"):
    """P[j][i] = strongest (max-product) dependency path j ~> i, j < i. 'temporal' = 1 for all j < i."""
    T = len(W)
    if mode == "temporal":
        return [[1.0 if j < i else 0.0 for i in range(T)] for j in range(T)]
    P = [[0.0] * T for _ in range(T)]
    for i in range(T):
        for j in range(i - 1, -1, -1):
            best = W[j][i]
            for m in range(j + 1, i):
                if P[j][m] and W[m][i]:
                    best = max(best, P[j][m] * W[m][i])
            P[j][i] = min(1.0, best)
    return P


def carr(cands, P, gamma=1.0, beta=0.5):
    out = []
    for c in cands:
        i, p = c["step"], c["p"]
        disc, bonus = 1.0, 0.0
        for d in cands:
            j = d["step"]
            if j < i:
                disc *= 1.0 - gamma * d["p"] * P[j][i]
            elif j > i:
                bonus += d["p"] * P[i][j]
        out.append({**c, "r": p * disc * (1.0 + beta * bonus)})
    return sorted(out, key=lambda c: (-c["r"], c["step"]))
