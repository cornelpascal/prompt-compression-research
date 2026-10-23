#!/usr/bin/env bash
set -euo pipefail
config="${1:?Supply a locked config}"
data="${2:?Supply prepared dataset directory}"
run="${3:?Supply a new or matching run directory}"
: "${PROMPTCOMP_TEACHER_URL:?Start the pinned teacher server first}"
: "${PROMPTCOMP_QWEN_URL:?Start the pinned Qwen target server first}"
: "${PROMPTCOMP_MISTRAL_URL:?Start the pinned Mistral target server first}"
promptcomp preflight --config "$config" --require-cuda
promptcomp supervise --config "$config" --data "$data" --output "$run/supervision" --cache "$run/cache"
seeds=$(python -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["seeds"])))' "$config")
for seed in $seeds; do
  for method in rewrite delete; do
    bash cloud/train.sh --config "$config" --data "$data" --method "$method" --stage sft --seed "$seed" \
      --supervision "$run/supervision" --output "$run/$method/seed-$seed/sft" --cache "$run/cache"
    promptcomp register --registry "$run/registry.json" --name "${method}_sft:$seed" --checkpoint "$run/$method/seed-$seed/sft"
    bash cloud/train.sh --config "$config" --data "$data" --method "$method" --stage rl --seed "$seed" \
      --initial-checkpoint "$run/$method/seed-$seed/sft" --output "$run/$method/seed-$seed/rl" --cache "$run/cache"
    promptcomp register --registry "$run/registry.json" --name "$method:$seed" --checkpoint "$run/$method/seed-$seed/rl"
  done
done
promptcomp evaluate --config "$config" --data "$data" --split validation --registry "$run/registry.json" \
  --methods original truncate rewrite_sft rewrite delete_sft delete --output "$run/validation" --cache "$run/cache"
promptcomp select --validation "$run/validation" --registry "$run/registry.json" --output "$run/selection.json"
promptcomp report --run "$run/validation" --output "$run/validation-report"
profile=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$config")
if [[ "$profile" != "full" ]]; then
  exit 0
fi
: "${PROMPTCOMP_PHI_URL:?Start the held-out target server for final evaluation}"
promptcomp evaluate --config "$config" --data "$data" --split test --registry "$run/registry.json" --selection "$run/selection.json" \
  --methods original truncate llmlingua2 untrained_rewrite rewrite_sft rewrite delete_sft delete rewrite_selected delete_selected question_only direct_answer \
  --output "$run/test" --cache "$run/cache"
promptcomp report --run "$run/test" --output "$run/report"
for variant in paraphrase entity; do
  promptcomp evaluate --config "$config" --data "$data" --split test --registry "$run/registry.json" --selection "$run/selection.json" \
    --methods original rewrite_selected delete_selected question_only --variant "$variant" --output "$run/$variant" --cache "$run/cache"
  promptcomp report --run "$run/$variant" --output "$run/report-$variant"
done
