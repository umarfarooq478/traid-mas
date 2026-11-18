#!/usr/bin/env bash
# Resume after a container restart (runs skip trajectories already in the output file).
# usage: bash scripts/queue4.sh 14b|nemo
cd "$(dirname "$0")/.."
export MISTRAL_API_KEY=${MISTRAL_API_KEY:-proxy-injected}
INJ="--inject_model ministral-8b-latest"
if [ "$1" = "14b" ]; then
  R="python3 -P scripts/run.py --model ministral-14b-latest --workers 1 --min_interval 31"
  $R --method bs      --subset alg --gt 1 --sample 60
  $R --method aao_idx --subset alg --gt 1 --variant cascade $INJ   # earliness control on CARMA-Cascade
  $R --method dcs     --subset alg --gt 0
  $R --method sbs     --subset alg --gt 1 --sample 60
else
  R="python3 -P scripts/run.py --model open-mistral-nemo --workers 1 --min_interval 31"
  $R --method aao     --subset hand --gt 1
fi
echo QUEUE_DONE
# nemo2: earliness control on CARMA-Cascade with the second judge (added after the nemo lane finished)
if [ "$1" = "nemo2" ]; then
  python3 -P scripts/run.py --model open-mistral-nemo --workers 1 --min_interval 31 --method aao_idx --subset alg --gt 1 --variant cascade --inject_model ministral-8b-latest
  echo QUEUE_DONE
fi
