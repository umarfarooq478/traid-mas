"""Evaluation metrics and statistics for failure attribution."""
from __future__ import annotations

import math
import random
import re


def norm_agent(a) -> str:
    if a is None:
        return ""
    a = re.sub(r"\(.*?\)", "", str(a)).strip().strip("*").strip().lower()
    return a


def agent_correct(pred, gold, protocol="strict") -> bool:
    if pred is None:
        return False
    if protocol == "lenient":           # original evaluate.py: substring test
        return str(gold) in str(pred)
    return norm_agent(pred) == norm_agent(gold)


def step_correct(pred, gold, protocol="strict", tol=0) -> bool:
    if pred is None:
        return False
    if protocol == "lenient":           # original evaluate.py: str(gold) in str(pred)
        return str(gold) in str(pred)
    return abs(int(pred) - int(gold)) <= tol


def topk_hit(ranked_steps, gold, k) -> bool:
    return int(gold) in [int(s) for s in ranked_steps[:k]]


def bootstrap_ci(xs, n=2000, seed=0, alpha=0.05):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(xs)
    means = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired binary outcomes a, b. Returns (b01, b10, p)."""
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def kendall_tau(r1, r2):
    """Kendall tau-b between two score lists over the same items."""
    n = len(r1); c = d = t1 = t2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (r1[i] > r1[j]) - (r1[i] < r1[j]); b = (r2[i] > r2[j]) - (r2[i] < r2[j])
            if a == 0 and b == 0:
                continue
            if a == 0:
                t1 += 1
            elif b == 0:
                t2 += 1
            elif a == b:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + t1) * (c + d + t2))
    return (c - d) / den if den else float("nan")
