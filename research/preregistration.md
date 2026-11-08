# Pre-registered settings (fixed 2026-11-08, before any LLM result was seen)

- Backbone (revised 2026-11-08 before any result, because the Gemini key had no usable tier): ministral-14b-latest
  (Mistral free tier: 2 req/min, 40k tok/min, 262k context), temperature 0, max output 1024 (baselines) / 2048 (DCS).
- DCS: K = 5 candidates.
- CARR: gamma = 1.0 (symptom discount), beta = 0.5 (explanation bonus). Not tuned on Who&When;
  a sensitivity sweep is reported separately and is NOT used to pick the headline setting.
- APG artifact types: numbers (>=2 chars, excluding 0-12 and common years), URLs, file names,
  identifiers with '_' or camelCase (>=6 chars), quoted strings (4-60 chars). Task/answer artifacts excluded.
- Primary metric: strict step-level accuracy; agent-level accuracy strict. Lenient (substring, as in the
  official evaluate.py) reported for reproduction only.
- Cascade injection: seed 0, one injected error per trajectory at a random later reasoning-agent step,
  injection model ministral-8b-latest (a different model from the judge) at temperature 0.7.
- Statistics: bootstrap 95% CI (2000 resamples) over trajectories; exact two-sided McNemar for paired comparisons.
