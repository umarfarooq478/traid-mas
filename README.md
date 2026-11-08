# TRAID-MAS: Trace-Guided Attribution, Intervention and Diagnosis in LLM Multi-Agent Systems

Code for the paper *"Cascade-Aware Root Resolution for Failure Attribution in LLM Multi-Agent Systems under Controlled Observability"*
(M. U. Farooq, FAST-NUCES Islamabad, Research Methodology course, 2026).

It contains

1. **A re-implementation of the base paper** — the three LLM-judge attribution methods (All-at-Once, Step-by-Step, Binary Search) of
   Zhang et al., *Which Agent Causes Task Failures and When?* (ICML 2025), with the official prompts, on the Who&When benchmark (`src/traid/baselines.py`).
2. **The proposed component** — Dense Candidate Scoring (DCS) + an Artifact Provenance Graph (APG) + Cascade-Aware Root Resolution (CARR) (`src/traid/carr.py`).
3. **TRAID-Cascade** — a multi-error variant of Who&When built by downstream error injection (`src/traid/cascade.py`).
4. **A controlled evaluation protocol** — strict vs. the original substring scoring, fixed observability levels (L0 outputs only, L1 + agent role specs),
   ground-truth on/off, top-k, distraction rate, rollback-trigger metrics, cost accounting, bootstrap CIs and McNemar tests (`src/traid/metrics.py`, `scripts/evaluate.py`).

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/mingyin1/Agents_Failure_Attribution.git ../ext/Agents_Failure_Attribution   # Who&When data (MIT)
echo MISTRAL_API_KEY=... > .env        # or GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY
```

Set `WHOWHEN_DIR` if the dataset lives elsewhere.

## Reproduce every table

```bash
bash scripts/queue.sh                 # all LLM runs, priority-ordered, resumable (results/<model>/*.jsonl)
python scripts/evaluate.py            # -> results/summary.json, results/summary.csv
python scripts/make_tables.py         # -> paper/tables/*.tex
python scripts/make_figures.py        # -> paper/figures/*.pdf
python scripts/sensitivity.py         # CARR gamma/beta sweep (no LLM calls)
```

Every LLM response is cached under `cache/llm/` keyed by a hash of (model, prompts, generation config); re-running
any script with the cache present makes zero API calls and reproduces the numbers exactly. The cascade set is stored in `data/cascade/`.

Settings were fixed before any result was observed: see `research/preregistration.md`.

## Layout

```
src/traid/   llm.py (clients+cache) data.py baselines.py carr.py cascade.py metrics.py
scripts/     run.py queue.sh build_cascade.py evaluate.py make_tables.py make_figures.py sensitivity.py data_stats.py apg_stats.py
paper/       main.tex refs.bib workflow.tex figures/ tables/
research/    preregistration.md reference_check.md deviation_log.md
results/     raw per-trajectory predictions (jsonl) + summaries
```

Baseline prompts are taken from the official Who&When repository (MIT licence, © the original authors).

## Results (algorithm-generated subset, L0, ground truth given, strict step accuracy)

| Judge | All-at-Once (reproduced) | DCS (ours) | DCS + CARR (ours) |
|---|---|---|---|
| ministral-14b | 14.3 | 40.5 | 40.5 |
| open-mistral-nemo | 13.5 | 41.3 | 39.7 |

TRAID-Cascade (14b): AAO 6.9 (n=58) vs DCS 41.4 on the same logs; DCS over all 111: 45.9, distraction rate 5.4%.
See `paper/main.pdf` and `research/deviation_log.md` for what was and was not run.
