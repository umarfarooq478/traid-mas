"""Offline statistics of the Artifact Provenance Graph (no LLM calls)."""
import sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from traid import data, carr
for s in ["alg", "hand"]:
    E=[]; dens=[]; gold_out=0; gold_reach_last=0; n=0
    for tr in data.load(s):
        W = carr.provenance_graph(tr); P = carr.path_strength(W)
        T = tr.T; e = sum(1 for j in range(T) for i in range(T) if W[j][i] > 0)
        E.append(e); dens.append(e / max(1, T*(T-1)/2)); n += 1
        g = tr.mistake_step
        gold_out += any(W[g][i] > 0 for i in range(T))
        gold_reach_last += any(P[g][i] > 0 for i in range(g+1, T))
    print(s, "n", n, "edges med", st.median(E), "density mean", round(st.mean(dens),3), "gold step has out-edge", round(gold_out/n,3), "gold reaches later step", round(gold_reach_last/n,3))
