# Cloud execution

For the experimental single-device PyTorch/XLA path, see [TPU execution](../../TPU.md).
This CUDA image and its vLLM server recipe are separate from TPU installation.

Run all commands from the `research/prompt_compression` directory. Use separate GPUs for training and frozen model servers; the 32B teacher is needed only during supervision. No cloud jobs are submitted automatically.

## Build the training image

```sh
docker build --platform linux/amd64 -f cloud/Dockerfile -t promptcomp:0.1 .
```

The image installs the complete hash-pinned Linux dependency set. The NVIDIA driver is provided by the GPU host and must support the CUDA runtime used by PyTorch. Model weights use immutable Hugging Face revisions from the locked config. Preserve model caches between runs.

To intentionally refresh the Python lock, use uv 0.8.22:

```sh
uv pip compile cloud/requirements.in --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --generate-hashes \
  --only-binary=:all: --output-file cloud/requirements.lock
```

## Serve frozen models

Use the separate `vllm/vllm-openai:v0.10.2` image. Mount this project at `/work` and the Hugging Face cache at `/root/.cache/huggingface`. Override the image entrypoint to `python`, then run:

```sh
python /work/cloud/serve.py --config /work/configs/full.locked.json \
  --alias qwen --port 8000 --tensor-parallel-size 1
```

Repeat for `mistral` and `teacher`, using distinct GPUs/ports and tensor-parallel sizes appropriate to the available memory. The server launcher enforces exact model/tokenizer revisions and exposes the revision in the served model name. Its context limit is 8,192 tokens. Start `phi` only for final evaluation. Optionally serve `rewriter` to accelerate the untrained baseline; trainable policies always run inside the trainer process.

The trainer expects endpoints including `/v1`:

```sh
export PROMPTCOMP_QWEN_URL=http://qwen-host:8000/v1
export PROMPTCOMP_MISTRAL_URL=http://mistral-host:8000/v1
export PROMPTCOMP_TEACHER_URL=http://teacher-host:8000/v1
```

The client supports `PROMPTCOMP_SERVER_TOKEN` for an authenticated service. These values are read at runtime and are not written into run manifests. Without an endpoint, CLI model calls load locally; for the full study the supplied launcher requires separate frozen-model endpoints to avoid placing all large models on a training GPU.

## Hardware smoke and pilot

First run the CPU suite inside the built image, then GPU smoke jobs:

```sh
promptcomp preflight --config configs/smoke.locked.json --require-cuda
NPROC_PER_NODE=1 bash cloud/run-study.sh configs/smoke.locked.json data/squad-v1 runs/smoke-one-gpu
NPROC_PER_NODE=2 bash cloud/run-study.sh configs/smoke.locked.json data/squad-v1 runs/smoke-two-gpu
NPROC_PER_NODE=2 bash cloud/run-study.sh configs/pilot.locked.json data/squad-v1 runs/pilot
promptcomp estimate --config configs/full.locked.json --data data/squad-v1 --pilot-dir runs/pilot
```

Smoke and pilot stop after validation and checkpoint selection. Inspect model-loading memory, training throughput, completion, budget failures, and teacher acceptance before scaling. `estimate` reports teacher/solver call counts and extrapolates measured training wall time for the same topology. Training GPU hours mean allocated training devices × elapsed wall time; they exclude separate model-serving GPUs and are not GPU utilization measurements.

For Docker execution, mount data, configs, caches, and run output at stable paths, pass the endpoint environment variables, expose selected GPUs, and override the training image entrypoint with `bash` to invoke `cloud/run-study.sh`. The image's default entrypoint is `promptcomp` for individual stages.

For multi-node training, `cloud/train.sh` supports `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `NPROC_PER_NODE`. Launch that training stage on every node with identical arguments and a shared run filesystem. The all-stages `run-study.sh` is a single-node orchestrator; do not run it concurrently on each node. Supervision and evaluation can be resumed separately through their CLI stages.

## Full study

After both GPU smoke tests and the pilot pass, freeze the configuration and run:

```sh
export PROMPTCOMP_PHI_URL=http://phi-host:8000/v1
NPROC_PER_NODE=2 bash cloud/run-study.sh configs/full.locked.json data/squad-v1 runs/full
```

The script executes supervision, both SFT and RL tracks for all three seeds, validation, selection, canonical final evaluation, and separate diagnostic evaluations. It is safe to rerun with unchanged inputs; completed cases and checkpoints are reused. Do not change model pins, code, dataset, topology, or optimizer settings within an existing run directory.

The final transfer model never supplies teacher labels, rewards, or checkpoint-selection scores. Keep final results sealed until configuration choices are fixed. SQuAD's official development set is the study's held-out set, not an unseen-pretraining-data guarantee.
# Attention-only experiment

The separate [attention runbook](../ATTENTION.md) documents frozen and trained first-layer scorers, calibration, and uncached benchmarking. Use `cloud/run-attention-study.sh` with an attention configuration and completed deletion supervision; the original study launcher remains available.

