#!/usr/bin/env bash
# Revision-round queue: the controls a reviewer needs. One lane per model (rate limits are per model).
# usage: bash scripts/queue2.sh 14b|nemo
cd "$(dirname "$0")/.."
export MISTRAL_API_KEY=${MISTRAL_API_KEY:-proxy-injected}
INJ="--inject_model ministral-8b-latest"
if [ "$1" = "14b" ]; then
  R="python3 -P scripts/run.py --model ministral-14b-latest --workers 1 --min_interval 31"
  $R --method aao --subset alg --gt 1 --sample 20 --sample_seed 1 --tag _rerun   # API drift check vs logged run
  $R --method aao_idx --subset alg --gt 1                       # numbering control
  $R --method dcs1    --subset alg --gt 1                       # K=1 control (list vs prompt)
  $R --method aao     --subset alg --gt 1 --variant cascade $INJ  # finish cascade AAO (58 -> 111)
  $R --method dcs     --subset hand --gt 1                      # hand-crafted subset
  $R --method aao     --subset hand --gt 1
  $R --method bs      --subset alg --gt 1 --sample 60
  $R --method dcs     --subset alg --gt 0
  $R --method sbs     --subset alg --gt 1 --sample 60
else
  R="python3 -P scripts/run.py --model open-mistral-nemo --workers 1 --min_interval 31"
  $R --method dcs     --subset alg --gt 1 --variant cascade $INJ  # finish cascade DCS (48 -> 111)
  $R --method aao     --subset alg --gt 1 --variant cascade $INJ
  $R --method aao_idx --subset alg --gt 1
  $R --method dcs1    --subset alg --gt 1
  $R --method dcs     --subset hand --gt 1
  $R --method aao     --subset hand --gt 1
fi
echo QUEUE_DONE
