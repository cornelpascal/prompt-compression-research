# Whole-prompt compression research pipeline

Train two compressors that turn an entire instruction + passage + question into shorter ordinary text. The receiver sees only that compressed prompt in its native chat template. The question is not secretly supplied again.

- **Rewrite:** Qwen3-4B-Instruct-2507 with LoRA; supervised warm-start followed by group-relative policy optimization (GRPO).
- **Delete:** ModernBERT-base with a word-level keep/drop head; supervised warm-start followed by Bernoulli policy gradients with a leave-one-out baseline.
- **Teacher:** Qwen3-32B generates four candidates per example, budget, and method. Candidates are ranked by the weaker frozen solver's F1; shorter candidates break ties. Extraction candidates must align to source words.
- **Frozen solvers:** Qwen3-8B and Mistral-Small-3.1-24B-Instruct-2503. Phi-4-mini-instruct is reserved for final transfer evaluation.

This is independent of PaperKB and the controlled-leakage package. No database or private-attribute annotations are required. See [cloud instructions](cloud/README.md) and [validation record](VALIDATION.md).

An additional [attention-only experiment](ATTENTION.md) compares frozen first-layer Q/K scores with trained query/key projections. It includes validation calibration for retaining 98% of baseline answer F1, locked transfer evaluation, and uncached latency measurements.

## Local setup and tests

Python 3.12 is required. A fresh installation:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[train,report,baseline,test]'
source .venv/bin/activate
.venv/bin/python -m pytest -q --tb=short
.venv/bin/python -m promptcomp.smoke --output runs/cpu-engineering
```

The existing local `.venv` reuses the parent workspace's PyTorch installation while keeping compatible Transformers/PEFT and plotting dependencies isolated. It is a local test convenience, not the cloud environment. Cloud dependencies are pinned with distribution hashes for Linux x86-64, Python 3.12, and PyTorch's CUDA 12.8 dependencies in `cloud/requirements.lock`.

The CPU smoke uses randomly initialized tiny Llama and ModernBERT models, a fixture teacher, and a deliberately artificial reward/solver. It executes real supervised and policy-gradient updates, reloads checkpoints, evaluates, and renders plots. Its scores are **not evidence of useful compression**. The fixture plots are labelled accordingly.

An explicit two-process CPU integration test exercises synchronization and exact interrupted-run recovery:

```sh
PROMPTCOMP_DISTRIBUTED_TEST=1 .venv/bin/python -m pytest -q --tb=short tests/test_distributed.py
```

## Data and model provenance

```sh
promptcomp prepare --download-dir data/raw --output data/squad-v1
promptcomp lock-models --config configs/full.json --output configs/full.locked.json
python scripts/make_profiles.py
```

Locked configurations are supplied. Deliberately refresh model pins only for a new study. The package records model revisions, dataset and source hashes, code hashes, decoding settings, and package versions. It rejects incompatible resumes. SQuAD JSON source snapshots are canonically serialized on download; their recorded hashes identify those saved snapshots.

SQuAD 1.1's official training articles are deterministically split 90/10 by article. With seed 11, this yields 79,100 training and 8,499 validation examples. The 10,570 official development examples are held out as this study's final test set; this is not the hidden official leaderboard test set. Answer offsets and article/passage separation are checked before writing the immutable dataset. Data attribution: [Stanford SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), CC BY-SA 4.0.

Dataset adapters define example parsing, grouping metadata, prompt construction, and scoring. `squad_v1` is the first registered adapter. Additional QA datasets can use the same pipeline. Coding-agent tasks will require their own behavioral scorers and diagnostic adapters; SQuAD results do not establish coding-agent instruction fidelity.

## Training and inference contract

All prompt text is eligible for compression. Requested budgets use the pinned rewriting model's tokenizer for both methods. Each receiver's achieved input length is independently counted, including its native chat template. The same compressed string is reused across receivers.

Teacher and compressor input types contain only the public prompt and budget. Gold answers are used exclusively by the reward/scoring layer. Teacher supervision is generated only from the training split. Empty, over-budget, truncated-generation, or unalignable teacher candidates are rejected; missing supervision is recorded rather than fabricated.

Supervised training uses response-only likelihood for rewriting and binary cross-entropy for deletion. The shared downstream reward is:

```text
min(F1_qwen, F1_mistral) - 2 * max(0, output_tokens / budget_tokens - 1)
```

Empty or exhausted rewrite generations receive -1. Rewriting uses token-averaged clipped GRPO, group-normalized advantages, and a sampled KL penalty against the frozen SFT checkpoint. Deletion samples independent word masks and uses exact mask log likelihood with a leave-one-out baseline and analytic Bernoulli KL. Both use four samples, one optimizer update per group, dropout disabled, and gradient clipping. At inference, deletion starts with nonnegative keep logits and removes the lowest-scoring retained words until the actual serialized output fits. Rewriting does not silently truncate or fall back to the original.

`compress(prompt, budget_tokens, method)` takes an initialized compressor and returns text, token counts, elapsed time, budget status, and source spans for deletion. The CLI equivalent:

```sh
promptcomp compress --config configs/full.locked.json \
  --registry runs/full/registry.json --selection runs/full/selection.json \
  --method rewrite_selected --seed 11 --input prompt.txt --budget 128
```

## Evaluation and artifacts

Smoke and pilot profiles cap training and validation at 32 and 512 examples respectively and use one seed. They do not evaluate the final test set or transfer model. The full profile uses all examples and three independently trained seeds (11, 23, 47). Budgets retain 75%, 50%, 25%, and 12.5% of reference-token length.

Baselines include the original prompt, simple prefix truncation, actual pretrained LLMLingua-2, and the unfine-tuned rewriting backbone. SFT-only checkpoints remain available. Validation selects SFT versus RL separately for each method and seed by average budget-compliant F1 across all budgets and both training solvers; exact ties favor SFT. Selection verifies checkpoint identities against evaluated artifacts.

Reports contain raw and budget-compliant F1, exact match, achieved input lengths, overflow rate, per-seed scores, article-clustered confidence intervals and paired deltas, correct-to-incorrect cases, timing, training compute, CSV/JSON outputs, and PNG/SVG tradeoff plots. Invalid-budget cases remain in results; the budget-compliant score counts them as zero. The plots show raw quality at achieved length, so consult compliance alongside them.

Question-only and direct-answer controls probe memorization and answer forwarding. Instruction paraphrases and controlled answer-string substitutions are evaluated separately from canonical SQuAD. Lexical flags identify lost negations, new numbers, and answer-only outputs for inspection; they do not establish semantic fidelity or detect every invented fact. The direct-answer control scores the trained rewriting model itself and is kept separate from receiver curves.

Stage directories contain manifests, per-case records, checksummed indexes, checkpoints, and reports. Atomic writes and process locks prevent duplicate concurrent work. Solver caches include exact prompt/model/revision/decoding identities. Service failures stop the stage and remain retryable. Training saves every 50 optimizer steps and at stage completion; an interrupted unsaved tail is replayed deterministically on the same topology. Different world sizes require a new run. Old checkpoints are retained.

The remaining scientific questions are empirical: whether either method improves the tradeoff, whether rewrites preserve evidence rather than merely forward answers, and whether any benefit transfers beyond SQuAD and the tested model families.
