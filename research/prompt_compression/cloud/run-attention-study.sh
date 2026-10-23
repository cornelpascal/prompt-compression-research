#!/usr/bin/env bash
set -euo pipefail
config="${1:?Supply an attention locked config}"
data="${2:?Supply the prepared dataset directory}"
supervision="${3:?Supply completed deletion supervision from the existing study}"
run="${4:?Supply a new or matching run directory}"
: "${PROMPTCOMP_QWEN_URL:?Start the pinned Qwen receiver server first}"
: "${PROMPTCOMP_MISTRAL_URL:?Start the pinned Mistral receiver server first}"
promptcomp preflight --config "$config" --require-cuda
seeds=$(python -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["seeds"])))' "$config")
for seed in $seeds; do
  bash cloud/train.sh --config "$config" --data "$data" --method attention --stage sft --seed "$seed" \
    --supervision "$supervision" --output "$run/attention/seed-$seed/sft"
  promptcomp register --registry "$run/registry.json" --name "attention_sft:$seed" \
    --checkpoint "$run/attention/seed-$seed/sft"
done
promptcomp evaluate --config "$config" --data "$data" --split validation --registry "$run/registry.json" \
  --output "$run/validation" --cache "$run/cache"
promptcomp calibrate-attention --validation "$run/validation" --threshold 0.98 --output "$run/calibration.json"
promptcomp report --run "$run/validation" --output "$run/validation-report"
profile=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$config")
if [[ "$profile" != "full" ]]; then
  exit 0
fi
: "${PROMPTCOMP_PHI_URL:?Start the held-out receiver server for final evaluation}"
promptcomp evaluate --config "$config" --data "$data" --split test --registry "$run/registry.json" \
  --calibration "$run/calibration.json" --output "$run/test" --cache "$run/cache"
promptcomp report --run "$run/test" --output "$run/test-report"
