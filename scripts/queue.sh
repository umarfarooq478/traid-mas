#!/usr/bin/env bash
# Priority-ordered experiment queue for the Mistral free tier (2 req/min per model).
# SBS and BS run on a fixed random subset of 60 algorithm-generated trajectories (seed 0) to fit the time budget.
cd "$(dirname "$0")/.."
M=ministral-14b-latest; R="python -P scripts/run.py --model $M --workers 1 --min_interval 30"
INJ="--inject_model ministral-8b-latest"
$R --method aao --subset alg --gt 1
$R --method dcs --subset alg --gt 1
$R --method dcs --subset alg --gt 1 --variant cascade $INJ
$R --method aao --subset alg --gt 1 --variant cascade $INJ
$R --method bs  --subset alg --gt 1 --sample 60
$R --method sbs --subset alg --gt 1 --sample 60
$R --method aao --subset hand --gt 1
$R --method dcs --subset hand --gt 1
$R --method dcs --subset alg --gt 0
$R --method aao_idx --subset alg --gt 1
$R --method dcs --subset alg --gt 1 --obs L1
$R --method bs  --subset alg --gt 1 --sample 60 --variant cascade $INJ
$R --method aao --subset alg --gt 0
echo QUEUE_DONE
