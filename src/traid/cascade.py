"""TRAID-Cascade: multi-error trajectories built from Who&When by downstream error injection.

For a failed trajectory with annotated decisive step t*, we choose a later step e > t*
written by a reasoning agent (not a tool/terminal/human turn) and ask an LLM to rewrite
it so that it contains one additional, independent, plausible error while keeping the
agent's style and length. The decisive step is unchanged (it is still the earliest error
that caused the failure), but the trajectory now contains a second, later error that an
attribution method may confuse with the root cause. Ground truth: decisive = t*,
error set = {t*, e}.
"""
from __future__ import annotations

import copy
import random

from .data import Trajectory

NON_AGENT = ("computer_terminal", "human", "user", "terminal")

INJECT_SYSTEM = "You rewrite one message of a multi-agent log for a controlled robustness study. Output only the rewritten message."


def inject_prompt(tr: Trajectory, e: int) -> str:
    a, c = tr.steps[e]
    return (
        f"Task the agents were solving: {tr.question}\n\n"
        f"Original message by agent '{a}':\n<<<\n{c[:6000]}\n>>>\n\n"
        "Rewrite this message so that it contains exactly ONE new, plausible, self-contained error of one of these kinds: "
        "a wrong number or calculation, a misread fact, a wrong assumption about the task, or an unjustified conclusion. "
        "Keep the same voice, format, code style and approximately the same length. Do not mention that an error was "
        "introduced. Output only the rewritten message."
    )


def choose_step(tr: Trajectory, rng: random.Random):
    t = tr.mistake_step
    pool = [i for i in range(t + 1, tr.T)
            if not any(k in tr.steps[i][0].lower() for k in NON_AGENT) and len(tr.steps[i][1]) >= 80]
    return rng.choice(pool) if pool else None


def make_cascade(llm, tr: Trajectory, seed: int = 0):
    rng = random.Random(f"{seed}-{tr.tid}")
    e = choose_step(tr, rng)
    if e is None:
        return None, None
    r = llm(inject_prompt(tr, e), system=INJECT_SYSTEM)
    new = r.text.strip()
    old = tr.steps[e][1]
    if not new or new == old.strip() or not (0.4 * len(old[:6000]) <= len(new) <= 2.5 * len(old[:6000]) + 200):
        return None, r
    tr2 = copy.deepcopy(tr)
    tr2.steps[e] = (tr.steps[e][0], new + (old[6000:] if len(old) > 6000 else ""))
    tr2.tid = tr.tid.replace("/", "-cas/")
    tr2.extra_error_steps = [e]
    return tr2, r
