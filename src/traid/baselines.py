"""Re-implementation of the three Who&When attribution baselines (Zhang et al., ICML 2025).

Prompts are copied from the official repository (Automated_FA/Lib/utils.py, MIT licence);
the only change is that the ground-truth line is omitted in the 'without ground truth'
setting, matching Section 4.2 of the paper. Each function returns a structured prediction
plus cost accounting instead of printing to a log file.
"""
from __future__ import annotations

import random
import re

from .data import Trajectory, render, role_block


def _gt(tr: Trajectory, with_gt: bool) -> str:
    return f"The Answer for the problem is: {tr.ground_truth}\n" if with_gt else ""


class Cost:
    def __init__(self):
        self.calls = 0; self.in_tok = 0; self.out_tok = 0; self.latency = 0.0

    def add(self, r):
        self.calls += 1; self.in_tok += r.in_tokens; self.out_tok += r.out_tokens; self.latency += r.latency_s

    def as_dict(self):
        return {"calls": self.calls, "in_tok": self.in_tok, "out_tok": self.out_tok, "latency_s": round(self.latency, 3)}


def parse_agent_step(text: str):
    a = re.search(r"Agent Name:\s*\**\s*([\w\-\(\) ]+?)\s*\**\s*(?:\n|$)", text, re.I)
    s = re.search(r"Step Number:\s*\**\s*(\d+)", text, re.I)
    return (a.group(1).strip() if a else None), (int(s.group(1)) if s else None)


def all_at_once(llm, tr: Trajectory, with_gt=True, numbered=False, obs="L0"):
    cost = Cost()
    prefix = role_block(tr) if obs == "L1" else ""
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {tr.question}\n" + _gt(tr, with_gt) + prefix +
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + render(tr, numbered=numbered) +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )
    r = llm(prompt, system="You are a helpful assistant skilled in analyzing conversations.")
    cost.add(r)
    agent, step = parse_agent_step(r.text)
    return {"agent": agent, "step": step, "raw": r.text[:2000], **cost.as_dict()}


def step_by_step(llm, tr: Trajectory, with_gt=True, obs="L0"):
    cost = Cost()
    hist = role_block(tr) if obs == "L1" else ""
    detect_at = None
    for idx, (agent, content) in enumerate(tr.steps):
        hist += f"Step {idx} - {agent}: {content}\n"
        prompt = (
            f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {tr.question}. "
            + _gt(tr, with_gt) +
            f"Here is the conversation history up to the current step:\n{hist}\n"
            f"The most recent step ({idx}) was by '{agent}'.\n"
            "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
            "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
            "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
            "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
        )
        r = llm(prompt, system="You are a precise step-by-step conversation evaluator.")
        cost.add(r)
        if r.text.lower().strip().startswith("1. yes"):
            return {"agent": agent, "step": idx, "raw": r.text[:1000], **cost.as_dict()}
    return {"agent": None, "step": None, "raw": "no error flagged", **cost.as_dict()}


def binary_search(llm, tr: Trajectory, with_gt=True, seed=0, obs="L0"):
    cost = Cost()
    rng = random.Random(f"{seed}-{tr.tid}")
    start, end = 0, tr.T - 1
    prefix = role_block(tr) if obs == "L1" else ""
    while start < end:
        mid = start + (end - start) // 2
        seg = "\n".join(f"{a}: {c}" for a, c in tr.steps[start:end + 1])
        prompt = (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
            f"The problem to address is as follows: {tr.question}\n" + _gt(tr, with_gt) + prefix +
            f"Review the following conversation segment from step {start} to step {end}:\n\n{seg}\n\n"
            f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half (from step {start} to step {mid}) or the lower half (from step {mid + 1} to step {end}) of this segment.\n"
            "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
        )
        r = llm(prompt, system="You are an AI assistant specializing in localizing errors in conversation segments.")
        cost.add(r)
        t = r.text.lower()
        if "upper half" in t:
            end = mid
        elif "lower half" in t:
            start = min(mid + 1, end)
        elif rng.randint(0, 1) == 0:
            end = mid
        else:
            start = min(mid + 1, end)
    return {"agent": tr.steps[start][0], "step": start, "raw": "", **cost.as_dict()}
