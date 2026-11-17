#!/usr/bin/env bash
# After Binary Search finishes in queue2 (14b lane), pre-empt the remaining low-priority runs (g=0, SBS)
# and run the cascade control for numbered AAO first, then resume g=0 DCS.
cd "$(dirname "$0")/.."
export MISTRAL_API_KEY=${MISTRAL_API_KEY:-proxy-injected}
F=results/ministral-14b-latest/orig_alg_bs_gt1_L0.jsonl
until [ -f "$F" ] && [ $(wc -l < "$F") -ge 60 ]; do sleep 30; done
pkill -f "queue2.sh 14b"; sleep 2; pkill -f "run.py --model ministral-14b-latest"; sleep 5
R="python3 -P scripts/run.py --model ministral-14b-latest --workers 1 --min_interval 31"
$R --method aao_idx --subset alg --gt 1 --variant cascade --inject_model ministral-8b-latest
$R --method dcs --subset alg --gt 1 --variant cascade --tag "" >/dev/null 2>&1   # no-op (complete)
$R --method dcs --subset alg --gt 0
$R --method sbs --subset alg --gt 1 --sample 60
echo QUEUE3_DONE
