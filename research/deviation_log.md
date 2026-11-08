# Deviation log (pre-registration vs. what was run)

| Planned | Actual | Reason |
|---|---|---|
| Base paper backbone GPT-4o | `ministral-14b-latest` (judge), `open-mistral-nemo` (2nd backbone) | No paid API available; Mistral free tier |
| AAO, SBS, BS on both subsets | AAO on all 126 algorithm-generated logs; SBS/BS and hand-crafted subset not completed | Free tier = 2 requests/min; SBS/BS need up to T and ~log2 T calls per log |
| Cascade: all methods, both judges | 14B: DCS 111/111, AAO 58/111; Nemo: DCS 48/111 | Time (deadline); all comparisons paired on completed ids |
| Conditions g=0, L1, numbered AAO | Implemented, not run | Time |
| Official substring step scoring | Strict integer match primary, substring reported for reproduction | Substring match counts "12" as correct for gold "1" |
| Official random fallback on unparsable answers | Counted as wrong | Determinism; can only lower our scores |

Pre-registered values (K=5, gamma=1, beta=0.5, cascade seed 0, injector ministral-8b @ T=0.7) were not changed after seeing results.
Finding contrary to the hypothesis: CARR does not change DCS top-1 (14B: identical on all 126; Nemo: 2 flips, both correct->wrong).
