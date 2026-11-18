# Reviewer-style critique of the CARMA draft (v1, PDF of 2026-11) and revision plan

Reviewed as a strict program-committee member would. Evidence for each point is either from the draft,
from the logged predictions in `results/` (re-analysed by `scripts/extra_analysis.py`, zero LLM calls),
or from Table 1 of the base paper (arXiv:2505.00212, read from the PDF).

## Verdict on v1

Honest and unusually careful for a course paper (strict scoring, pre-registration, McNemar, negative result
reported). The weaknesses are in **what the headline comparison can be attributed to**, **which baseline it
is compared with**, and **an evaluation that stops at the easy subset and at partial cascade runs**.
As written, a reviewer would accept the gain over the reproduced All-at-Once but not the explanation of it.

## Major issues

| # | Issue | Evidence | Fix in v2 |
|---|---|---|---|
| M0 | **Novelty threat from the closest prior work.** A2P (arXiv:2509.10401, Table 3, 5 runs, same 126 alg logs) already shows that adding "Step {idx} -" prefixes alone moves its step accuracy from 17.78% to 47.46% (+29.68 pts). A2P's own baselines in its Table 1: AAO 16.67%, BS 28.57%. The draft cites A2P only for its headline number and DoVer for "judges miscount", so a reviewer will read DCS's +26 pts as a re-discovery of a published effect. | A2P PDF lines 401-410, 555-590. | Cite the A2P numbering ablation explicitly; make the numbered-AAO control (M1) the centre of the analysis; claim for DCS only what survives that control (list/ranking quality, rollback schedule, cascade robustness), not the numbering gain. |
| M1 | **The headline gain is confounded.** DCS differs from AAO in three ways at once: numbered steps, a different prompt (definition of the decisive error, cascade warning, JSON), and a K=5 list. The numbered-AAO control has n=7. | Draft Sec. IV-D "Critical interpretation". | Run numbered AAO (`aao_idx`) and **DCS with K=1** (`dcs1`, same prompt, one candidate) on all 126 logs with both judges. Decompose 14.3 → 40.5 into numbering / prompt / list. |
| M2 | **The baseline is the weakest one in the base paper, and its protocol has an indexing artefact.** Base paper Table 1 (GPT-4o, alg, with GT): AAO 12.50, **SBS 25.51**, **BS 23.98**, random 19.06 step accuracy. The draft omits SBS/BS numbers and says DCS "nearly triples" accuracy. Reproduced AAO **never predicts step 0** (0/126, both judges) while t*=0 in 20/126 logs; the agent it names is the agent speaking at its predicted step in only 26.2% (14B) / 23.8% (Nemo) of logs; 14B predicts "step 1" in 43/126 logs. | `results/extra_analysis.json`; base-paper PDF Table 1. | Add reported SBS/BS rows to the reproduction table; run BS (and SBS if time) on a fixed 60-log sample; frame the gain against the strongest base-paper method, not only AAO; report the indexing artefact as a G1 finding. |
| M3 | **CARR is a named component of the method with exactly zero effect** with the main judge, yet the title, abstract and Fig. 1 present it as part of the method. | Tables VI, IX. | Re-centre the contribution on DCS + protocol + CARMA-Cascade; keep CARR as an analysed negative result. Add the **calibration analysis** that explains it (ECE 0.38, mean top-p 0.79 vs. 41% top-1 accuracy) and the **re-ranking ceiling** (gold in DCS list: top-3 71.4%, top-5 84.9%), which shows the room a better re-ranker would have. |
| M4 | **Cascade results are partial and lack controls.** AAO 58/111, Nemo-DCS 48/111. Distraction rate is reported without (a) chance level, (b) how often the method already predicted that step *before* injection, (c) a manipulation check that the injected error is detectable at all. | Table VIII. | Complete both runs. Add pre-injection DR (14B DCS 1.8% → 5.4%), chance DR (E[1/T]=12.1%), and listing rate of the injected step (64.0% → 85.6% of DCS lists; mean probability mass on it 0.045 → 0.104). Paired McNemar original vs. cascade per method. |
| M5 | **Only the short algorithm-generated subset.** Hand-crafted logs are 5x longer and are where the base paper reports <8% step accuracy. | Sec. IV-A. | Run AAO and DCS on the 51 hand-crafted logs that fit the token budget, with both judges. Note that the agent label differs from the speaker at the gold step in 21/58 hand-crafted logs, so DCS's "agent read from the step" rule has a ceiling there. |
| M6 | **Single run per condition; possible API drift.** Temperature-0 API runs are not bit-reproducible; new runs are compared with older logs. | Sec. IV-A. | Re-run AAO on a fixed 20-log sample and report agreement with the logged run (drift check). Report a **paired bootstrap CI of the difference** (14B: +26.2 pts [16.7, 35.7]; Nemo: +27.8 [17.5, 38.1]) instead of "non-overlapping CIs". |
| M7 | **RTR rewards early predictions.** DCS predicts ≥3 steps before t* in 32.5% of logs (14B). RTR = 90.5% is therefore partly earliness, not localisation. | Offset histogram. | Report the signed-offset distribution; keep RTR only with WRE; add a rollback-**schedule** proxy that uses the ranked list (attempts until a rollback point ≤ t*, re-executed steps). |

## Minor issues

- m1 Step 0 of every algorithm-generated log is the task brief + manager plan + code carried over from a previous round, attributed to the first expert. The `alg/1` example ("the Excel_Expert writes code") should say this; it also explains why t*=0 is common and why "always step 0" is a strong trivial baseline.
- m2 Intro quotes 53.5% / 14.2% (base-paper abstract, best over settings) while Table V uses 54.3 / 12.5 (AAO, alg, with GT). Say which is which.
- m3 Random-step reference: the draft uses E[1/T] = 12.0%, the base paper reports 19.06% for the same subset. Report both and say the definitions differ.
- m4 Table I "Better than single-pass judges", "Beats larger judges", "Gains over judges" are not results. Use numbers where verifiable or say "not comparable".
- m5 DCS agent accuracy is derived from the step; do not count it as an independent improvement (draft already says so; keep it out of the abstract).
- m6 The paragraph "This work starts from an earlier, purely evaluative design" reads as course history; move to a footnote.
- m7 Repository: `.gitignore` excludes `*.tex`, so the paper source was never committed; `research/reference_check.md` cited in the README does not exist; README still uses the old project names (TRAID). Fix.
- m8 Fig. 2 should print values; Table VI "p vs CARR" column is uninformative once CARR == DCS; replace with p vs AAO.
- m9 References: every entry re-verified (see `reference_check.md`).

## What new evidence already says (offline, from logged runs)

1. AAO's errors are not "off by one" in a symmetric way: offsets +1: 17.5%, −1: 7.1%, ≥+3: 19.8%, ≤−3: 21.4% (14B). Shifting AAO predictions by −1 only raises it to 17.5%, so 1-based counting is part of the story, not all of it.
2. DCS is strongly over-confident but its confidence ranks well: accuracy is 82% when top-p ≥ 0.9 and 15–21% when top-p < 0.8 (14B). A calibrated or abstaining variant is a natural next step and directly explains CARR's null effect.
3. The two judges agree on DCS top-1 in 61.9% of logs; at least one is correct in 51.6% (oracle of two judges), both in 30.2%.
4. Injected errors are noticed (listed more often, more probability mass) but rarely promoted to top-1 by the 14B judge; the Nemo judge is pulled to them (DR 0% → 17.3% on the same logs, accuracy 48.1 → 30.8, McNemar p = 0.012, n = 52 at time of writing).

## Revision plan (status tracked in the task list)

1. LLM runs (scripts/queue2.sh, two lanes): drift re-run, numbered AAO, DCS K=1, cascade completion, hand-crafted AAO/DCS, BS sample, DCS without GT, SBS sample.
2. Offline analyses (scripts/extra_analysis.py) — done, re-run when lanes finish.
3. Reference verification (Crossref / arXiv / DBLP) → `research/reference_check.md`.
4. Rebuild `paper/main.tex` (IEEEtran), reframe contributions, new tables, compile.
