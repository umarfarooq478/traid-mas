"""Sanitise injected messages: keep only the text between echoed <<< >>> delimiters, strip trailing '---'
blocks, and DROP trajectories whose injected message still reveals the injection (e.g. '(Error: ...)')."""
import json, re, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8")); keep = []; dropped = []
LEAK = re.compile(r"error introduced|\(error|\*\(error|rewritten message|introduced (an|one|the) error|injected|\bnew error\b", re.I)
for r in d:
    e = r["extra_error_steps"][0]; a, c = r["steps"][e]
    if "<<<" in c:
        c = c.split("<<<", 1)[1]
    if ">>>" in c:
        c = c.split(">>>", 1)[0]
    c = re.sub(r"(\n\s*-{3,}\s*)+$", "", c).strip()
    if LEAK.search(c) or len(c) < 40:
        dropped.append(r["tid"]); continue
    r["steps"][e] = [a, c]; keep.append(r)
json.dump(keep, open(p, "w", encoding="utf-8"))
print("kept", len(keep), "dropped", dropped)
