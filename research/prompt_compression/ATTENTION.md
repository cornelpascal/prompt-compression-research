# Attention-only compression experiment

This experiment compares frozen first-layer Qwen3 attention with trained query/key projections. It reads every prompt token but runs no value projection, attention output projection, feed-forward block, later layer, language-model head, or generation step. Model loading reads only the required safetensors tensors; downloaded shards may also contain unrelated tensors. The frozen embedding table still occupies memory. Computation and parameters use float32 in both variants.

The scorer appends a copy of the question's token IDs solely for scoring. These appended query positions can attend to every original prompt token. Scores reproduce the corresponding first-layer causal attention, including input normalization, query/key normalization, rotary positions, grouped heads, and normalization over visible appended tokens. A two-pass block calculation uses global softmax normalization; training recomputes block intermediates during backward. There is no full prompt-by-prompt attention matrix.

Scores are averaged across heads and question positions, then across subtokens within each source word. Words with the lowest scores are removed until the actual serialized text fits the reference-token budget. Remaining words retain their wording and order. All prompt words, including instructions and the question, are eligible. The scoring-only question copy is never sent to the receiver. An already-fitting prompt is returned byte-for-byte unchanged. Empty selections are explicit failures; there is no fallback.

## Training

Only `q_proj` and `k_proj` parameters are trainable. Embeddings and all normalization parameters stay frozen. Accepted deletion teacher masks from the existing training split are normalized into word distributions, then averaged across available budgets per prompt. Cross-entropy trains the attention-derived word distribution toward that target. Teacher masks remain offline supervision; the runtime scorer receives only prompt text, a source focus range, and a budget.

Defaults are one epoch, learning rate `2e-5`, and seeds 11, 23, 47. The training command uses the existing optimizer, distributed gradient averaging, checkpoint integrity checks, and deterministic recovery. Attention RL and warm-starting from a different checkpoint are rejected. Use a new run directory after code/configuration changes.

## Local checks

```sh
.venv/bin/python -m pytest -q --tb=short
.venv/bin/python -m promptcomp.attention_smoke --output runs/attention-engineering
PROMPTCOMP_DISTRIBUTED_TEST=1 .venv/bin/python -m pytest -q --tb=short tests/test_distributed.py
```

The attention smoke creates tiny random Qwen3 weights, generates fixture supervision, trains Q/K, evaluates both variants, calibrates, evaluates locked test settings, and writes reports. Its solvers and data are fixtures. Its quality numbers establish no scientific result.

## Cloud study

Use the existing CUDA image and pinned receiver servers described in [the cloud runbook](cloud/README.md). New `configs/attention.{smoke,pilot,full}.locked.json` files reuse existing model revisions. They do not alter the original experiments.

Reuse completed supervision from the corresponding original study. If none exists, generate it using that study's original configuration and the teacher/Qwen/Mistral servers before this run. Only accepted deletion records are used; the attention study does not require rewrite training.

```sh
bash cloud/run-attention-study.sh configs/attention.smoke.locked.json \
  data/squad-v1 runs/smoke/supervision runs/attention-smoke

bash cloud/run-attention-study.sh configs/attention.pilot.locked.json \
  data/squad-v1 runs/pilot/supervision runs/attention-pilot

bash cloud/run-attention-study.sh configs/attention.full.locked.json \
  data/squad-v1 runs/full/supervision runs/attention-full
```

Smoke and pilot stop after validation/calibration. Full evaluation adds Phi only after calibration is locked. `NPROC_PER_NODE` controls the existing training launcher. Model-serving processes should run on separate GPUs from the scorer when measuring scorer memory.

## Calibration and acceptance

Validation evaluates retention fractions `0.125, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.98`, against the original prompt. Each variant/seed selects the smallest measured reference-token footprint that achieves at least **0.98 times the original F1 on both training receivers**. Invalid-budget cases count as zero. Every candidate is evaluated; quality is not assumed monotonic. Equal footprints favor the smaller requested ratio. A zero-F1 original baseline is rejected because a relative quality claim would be uninformative.

`calibrate-attention` accepts canonical validation results only. Artifacts bind the scorer code, model revision, checkpoint, precision, block settings, reference tokenizer, receiver configuration, and profile. Test examples and Phi do not participate in selecting the ratio. Calibrated final evaluation requires all three receivers and never adjusts the ratio. The report's `attention_threshold` section gives each receiver's held-out F1 ratio and an article-bootstrap interval for `compressed_F1 - 0.98 * original_F1`.

The acceptance rule uses observed dataset means; intervals communicate sampling uncertainty. This is not a per-prompt confidence score or a guarantee on future data. Keeping 98% of attention mass does not imply 98% F1. A variant with no qualifying compressed setting records `no_qualifying_compression`; test evaluation reports that status and does not substitute a different prompt.

## Inference

The existing API accepts an optional `focus_span=(start, end)`, using Python character offsets and an exclusive end. Attention methods require a nonempty span within the prompt. SQuAD adapters supply the exact question suffix without parsing ambiguous internal delimiters. Existing callers without focus spans remain compatible for other methods.

```python
from promptcomp import compress
result = compress(prompt, 128, attention_compressor,
                  focus_span=(question_start, len(prompt)))
```

For CLI inference, provide `--focus-start` and `--focus-end` together and choose exactly one of `--budget` or `--calibration`:

```sh
promptcomp compress --config configs/attention.full.locked.json \
  --registry runs/attention-full/registry.json --method attention_sft --seed 11 \
  --input prompt.txt --focus-start 500 --focus-end 550 \
  --calibration runs/attention-full/calibration.json
```

The example offsets must be replaced with the actual question range. Calibration derives a budget from the validated retention ratio; it does not verify the new prompt's answer accuracy. Explicit budgets are experimental and carry no calibration claim.

## Latency measurements

Run timing separately from quality evaluation. This command bypasses the local answer cache, warms the models, alternates original/compressed receiver request order, and records median/p95 total latency plus peak scorer CUDA allocation:

```sh
promptcomp benchmark-attention --config configs/attention.full.locked.json \
  --data data/squad-v1 --registry runs/attention-full/registry.json \
  --calibration runs/attention-full/calibration.json --method attention_sft --seed 11 \
  --samples 16 --repeats 3 --warmup 2 --output runs/attention-full/timing-sft-11
```

Use `--method attention_frozen` in a separate output directory for the frozen variant. Disable server-side prefix caching for comparable prefill measurements. Reported total latency adds warmed compression time to an uncached receiver request; loading time is excluded, and remote timings include network/queueing. The scorer is released before loading local receivers. CPU memory values of zero mean unmeasured. These measurements, and the full GPU quality experiment, must pass before claiming useful compression or speedup.
