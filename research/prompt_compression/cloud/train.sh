#!/usr/bin/env bash
set -euo pipefail
# GPU ids are selected externally with CUDA_VISIBLE_DEVICES / container --gpus.
processes="${NPROC_PER_NODE:-1}"
nodes="${NNODES:-1}"
if [[ "$nodes" == "1" ]]; then
  exec torchrun --standalone --nnodes=1 --nproc-per-node="$processes" -m promptcomp train "$@"
else
  : "${NODE_RANK:?Set NODE_RANK for each node}"
  : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-zero host}"
  exec torchrun --nnodes="$nodes" --nproc-per-node="$processes" --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29500}" -m promptcomp train "$@"
fi

