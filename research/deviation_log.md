# Deviation log (pre-registration vs. what was run)

| Planned | Actual | Reason |
|---|---|---|
| Base paper backbone GPT-4o | `ministral-14b-latest` (judge), `open-mistral-nemo` (2nd backbone) | No paid API available; Mistral free tier |
| AAO, SBS, BS on both subsets | AAO on all 126 alg logs and 57/58 hand logs (both judges); BS on a fixed 60-log alg sample (14B); SBS not run | Free tier = 2 requests/min; SBS needs up to T calls per log; hand/30 failed after retries |
| Cascade: all methods, both judges | AAO, numbered AAO and DCS on all 111 cascade logs, both judges | Completed in the revision round |
| Conditions g=0, L1 | Implemented, not run | API budget |
| (post-hoc) numbered AAO, DCS K=1 controls; probability flattening | Run on all 126 alg logs (both judges); flattening offline | Added after the first results to decompose the gain; labelled post-hoc in the paper |
| Official substring step scoring | Strict integer match primary, substring reported for reproduction | Substring match counts "12" as correct for gold "1" |
| Official random fallback on unparsable answers | Counted as wrong | Determinism; can only lower our scores |

Pre-registered values (K=5, gamma=1, beta=0.5, cascade seed 0, injector ministral-8b @ T=0.7) were not changed after seeing results.
Finding contrary to the hypothesis: CARR does not change DCS top-1 (14B: identical on all 126; Nemo: 2 flips, both correct->wrong).

Revision round: the pre-registration file was committed together with the first results, so the git history alone cannot prove the order; the paper states this.
