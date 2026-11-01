#!/usr/bin/env bash
set -euo pipefail
# GPU ids are selected externally with CUDA_VISIBLE_DEVICES / container --gpus.
processes="${NPROC_PER_NODE:-1}"
nodes="${NNODES:-1}"
if [[ "${PROMPTCOMP_DEVICE:-}" == "xla" ]]; then
  if [[ "$processes" != "1" || "$nodes" != "1" ]]; then
    echo "TPU mode currently supports one device per independent run" >&2
    exit 2
  fi
  exec python -m promptcomp train "$@"
fi
if [[ "$nodes" == "1" ]]; then
  exec torchrun --standalone --nnodes=1 --nproc-per-node="$processes" -m promptcomp train "$@"
else
  : "${NODE_RANK:?Set NODE_RANK for each node}"
  : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-zero host}"
  exec torchrun --nnodes="$nodes" --nproc-per-node="$processes" --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29500}" -m promptcomp train "$@"
fi

