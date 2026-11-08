"""Loading Who&When trajectories and rendering them at fixed observability levels."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(os.environ.get("WHOWHEN_DIR", ROOT.parent / "ext" / "Agents_Failure_Attribution" / "Who&When"))
SUBSETS = {"alg": "Algorithm-Generated", "hand": "Hand-Crafted"}


@dataclass
class Trajectory:
    tid: str                 # e.g. "alg/12"
    subset: str              # alg | hand
    question: str
    ground_truth: str
    steps: list              # list of (agent, content)
    mistake_agent: str
    mistake_step: int
    system_prompts: dict = field(default_factory=dict)
    extra_error_steps: list = field(default_factory=list)  # injected (cascade) errors

    @property
    def T(self) -> int:
        return len(self.steps)


def load(subset: str, data_dir: Path = DEFAULT_DATA) -> list[Trajectory]:
    d = Path(data_dir) / SUBSETS[subset]
    files = sorted((f for f in os.listdir(d) if f.endswith(".json")), key=lambda x: int("".join(c for c in x if c.isdigit())))
    out = []
    key = "role" if subset == "hand" else "name"
    for f in files:
        x = json.loads((d / f).read_text(encoding="utf-8"))
        steps = [(str(e.get(key, "Unknown Agent")), str(e.get("content", ""))) for e in x["history"]]
        sp = x.get("system_prompt") or {}
        if isinstance(sp, str):
            try:
                import ast
                sp = ast.literal_eval(sp)
            except Exception:
                sp = {"_raw": sp}
        out.append(Trajectory(tid=f"{subset}/{f[:-5]}", subset=subset, question=x.get("question", ""),
                              ground_truth=str(x.get("ground_truth", "")), steps=steps,
                              mistake_agent=str(x["mistake_agent"]), mistake_step=int(x["mistake_step"]),
                              system_prompts=sp if isinstance(sp, dict) else {}))
    return out


def render(tr: Trajectory, numbered: bool = False, start: int = 0, end: int | None = None) -> str:
    """Render steps [start, end]. numbered=False reproduces the Who&When 'agent: content' format."""
    end = tr.T - 1 if end is None else end
    lines = []
    for i in range(start, end + 1):
        a, c = tr.steps[i]
        lines.append(f"Step {i} - {a}: {c}" if numbered else f"{a}: {c}")
    return "\n".join(lines)


def role_block(tr: Trajectory, max_chars: int = 1500) -> str:
    """Observability level L1: the agents' role specifications (system prompts)."""
    if not tr.system_prompts:
        return ""
    parts = [f"[{k}]\n{str(v)[:max_chars]}" for k, v in tr.system_prompts.items()]
    return "Agent role specifications (system prompts) available to the debugger:\n" + "\n\n".join(parts) + "\n\n"
