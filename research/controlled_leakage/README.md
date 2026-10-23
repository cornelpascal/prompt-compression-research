# Controlled leakage experiments

An isolated research package for prompt compression, task utility, and private-information reconstruction. It implements a CPU engineering pilot, a finite-source information-theoretic solver, and neural experiment/training paths. **The CPU pilot is not evidence that a neural compressor improves privacy.**

See [implementation validation](docs/VALIDATION.md) for executed checks, generated results and remaining GPU/external-data work.

## Run the verified CPU workflow

From the repository root, using the existing environment:

```sh
PYTHONPATH=research/controlled_leakage .venv/bin/python -m leaklab generate --output research/controlled_leakage/datasets/pilot --size 24
PYTHONPATH=research/controlled_leakage .venv/bin/python -m leaklab run --config research/controlled_leakage/configs/pilot.json --data research/controlled_leakage/datasets/pilot --output research/controlled_leakage/runs/pilot
PYTHONPATH=research/controlled_leakage .venv/bin/python -m leaklab theory --output research/controlled_leakage/runs/frontier.json
PYTHONPATH=research/controlled_leakage .venv/bin/python -m leaklab compare --run research/controlled_leakage/runs/pilot --method redact_compress --baseline original --output research/controlled_leakage/runs/pilot/comparison.json
```

The pilot uses a deterministic structured-record task checker and **UTF-8 bytes**, explicitly labelled in every result. The original and redaction controls can exceed the requested budget; they are not silently truncated. `redact_compress` prioritizes task fields and enforces the budget. Neural runs measure context length with the receiving model's tokenizer and count the complete templated request separately.

`--size` is the number of base records **per split**. Each has two private-code counterfactuals. Training, validation, test, and correlation-shift splits have disjoint entities and template identifiers. The six domain/regime combinations cycle evenly when size is divisible by six. Changing only the private code preserves nonessential-task answers. The group attribute is independent, 90% correlated with task outcome, or unrelated to a task-essential private code. Shift data reverses the correlated-group relationship to 10%.

## Install on a GPU machine

```sh
cd research/controlled_leakage
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[gpu,llmlingua,test]'
```

The package has no database dependency. Three simultaneous models during alternating training are intended for an approximately 80 GB CUDA device; this is a planning estimate, not a measured memory requirement. No automatic cluster submission or background training service is installed.

Use the supplied immutable model revisions in `configs/gpu.locked.json` and `configs/training.locked.json`. Refresh deliberately with `leaklab lock-models --config configs/gpu.json --output configs/gpu.locked.json`; this resolves public metadata and does not download weights. Runs record package versions, model revisions, source-code hashes, data hashes, and adapter hashes. Install and retain an environment lock on the actual GPU host before the main study.

## Neural workflow

Commands below assume the package is installed and `leaklab` is on PATH.

```sh
leaklab generate --output datasets/main --size 1200 --seed 11
leaklab train-compressor --config configs/training.locked.json --data datasets/main/train.jsonl --output checkpoints/compressor
leaklab tune-anonymizer --config configs/gpu.locked.json --data datasets/main --output checkpoints/anonymizer
leaklab configure-evaluation --base configs/gpu.locked.json --adapters checkpoints/compressor/compressor-adapters.json --policy checkpoints/anonymizer/policy.json --split test --output configs/evaluation.json
leaklab run --config configs/evaluation.json --data datasets/main --output runs/main-test
leaklab export-attacks --run runs/main-test --output runs/main-test/attack-training.jsonl
leaklab list-scopes --export runs/main-test/attack-training.jsonl
```

Dataset size 1200 is an executable starting configuration, not a statistical-power claim. Choose and freeze the main sample size using pilot variance before inspecting main-test results.

The compressor warm-starts from query-aware, redacted structured-record targets. Alternating rounds retrain an attacker using **training observations only**, then update the compressor using a two-sample leave-one-out policy gradient. The reward combines exact task correctness, a smooth protected-value likelihood surrogate, and raw budget overflow. Evaluation reports actual recovery rather than treating this surrogate as a privacy guarantee. The policy gradient scores the exact sampled token IDs, including a sampled EOS, instead of retokenizing decoded text.

The adaptive anonymizer performs validation-selected instruction search with reflective proposals and per-candidate retrained attribute attacks. This is an independently implemented adaptive baseline, **not a reproduction of the published GEPA algorithm**. The fixed summarization baseline uses the original model; learned adapters do not contaminate it.

### Train and evaluate reconstruction attacks

`list-scopes` identifies each seed/model/compressor/budget/access combination. Train a separate inverter for each scope. For `context`, `text`, and `public`, use:

```sh
leaklab train-attacker --export runs/main-test/attack-training.jsonl --scope SCOPE_ID --output checkpoints/attack --model Qwen/Qwen2.5-7B-Instruct --revision PINNED_COMMIT --epochs 3
```

Train the corresponding `public` scope independently. For top-5, top-20, and full probability access, use the pinned `google-t5/t5-base` revision from the model lock:

```sh
leaklab train-inverter --export runs/main-test/attack-training.jsonl --run runs/main-test --scope SCOPE_ID --output checkpoints/inverter --model google-t5/t5-base --revision PINNED_COMMIT --epochs 3
leaklab train-inverter --export runs/main-test/attack-training.jsonl --run runs/main-test --scope SCOPE_ID --output checkpoints/inverter-public --model google-t5/t5-base --revision PINNED_COMMIT --epochs 3 --public-only
leaklab evaluate-attacker --run runs/main-test --data datasets/main/test.jsonl --checkpoint checkpoints/inverter --public-checkpoint checkpoints/inverter-public --output runs/main-test/inverter-evaluation
```

The public-only sequence control receives query, public auxiliary information, and policy names; it receives neither generated text, probabilities, nor trace length. Repeat probability training with `--target compressed_text` in a **separate checkpoint** to measure compressed-input reconstruction. Those scores are saved separately from private-fact recovery.

For an independent final attacker family, train Mistral-7B-Instruct-v0.3 text attackers and their public controls using the same exported training scopes. These checkpoints must not participate in compressor optimization. Probability inverters are trained independently of the compressor's online attacker.

Top-k views preserve actual vocabulary IDs and original log probabilities; omitted probabilities are never filled in or renormalized. Full views use selected additive-log-ratio coordinates, following the idea in [PILS](https://github.com/dill-lab/PILS). Each target model gets an output-head rank/conditioning diagnostic. A failed rank check means those coordinates are an information-reducing feature map. Even with full rank, finite-precision effects remain. This is a PILS-style adaptation, not a reproduced PILS result.

### Ablations and transfer

The three locked training ablations change only `privacy_weight=0`, `no_attacker_retraining=true`, or `no_query=true`. Train each into its own directory and configure evaluation with its emitted seed-to-adapter registry. For the query ablation also set `no_query=true` in the evaluation config. Task models and attackers continue to see the query. Reuse the same datasets and evaluation cases.

Use `configure-evaluation --split shift` for the correlation-shift experiment. The main GPU configuration already evaluates both frozen target-model families. Do not substitute a different compressor checkpoint while reusing an old run directory.

## External PII-Bench evaluation

[PII-Bench](https://aclanthology.org/2026.acl-long.227/) annotates query-related PII and masking. Those annotations must not be silently used as gold answers to the underlying user questions. The importer requires explicit task-answer and protected-value annotations, with field mapping in `configs/external-mapping.json`:

```sh
leaklab import-external --source annotated-pii-bench.jsonl --mapping configs/external-mapping.json --source-url https://aclanthology.org/2026.acl-long.227/ --output datasets/external/pii-bench.jsonl
leaklab configure-evaluation --base configs/evaluation.json --split external --evaluation-file datasets/external/pii-bench.jsonl --output configs/external-evaluation.json
leaklab run --config configs/external-evaluation.json --data datasets/main --output runs/external
```

The package does not bundle or fabricate PII-Bench records or their missing task-answer labels. Imported data is evaluation-only, with explicit source and checksum provenance. Its correlation regime is marked `unspecified`.

## Outputs, resumption, and interpretation

- `manifest.json`: frozen inputs, source hashes, environment, completion state and limitations.
- `cases/`: durable per-case compression and task-output checkpoints. Evaluator labels have a separate section; inference adapters receive only whitelisted observations.
- `probabilities/`: checksum-verified numeric observation artifacts with pickle disabled.
- `results.json`, `summary.json`, `summary.csv`: task accuracy, lexical required-fact recall, per-attribute recovery, public-only recovery, paired excess recovery, achieved lengths, budget compliance, and per-model request/rewrite token counts.
- `comparison.json`: paired differences with length and utility checks. Comparing methods at the same requested ratio does not establish matched achieved lengths.

Task/rewrite tokens use different model tokenizers and are kept in separate columns. Token counts are not dollar costs or measured FLOPs. One task call is reused for all observers; do not sum duplicated cost fields across attack rows. Lexical fact recall can undercount faithful paraphrases. Pair-cluster bootstrap intervals keep counterfactuals and repeated seeds together; seed means are also reported. They do not by themselves establish generalization over training randomness or correct for multiple comparisons.

Runs reject changed inputs or code on resume and use a single-writer lock. Completed case checkpoints are reused, including independently trained attacker evaluations. Those evaluations also freeze checkpoint-file hashes and verify the test-label file against the original run. SFT/inverter training resumes at complete epochs; compressor optimization resumes at complete rounds and deliberately resets its optimizer at each alternating round. An interrupted epoch/round is replayed. No research process continues when its command is stopped.

See [theory](docs/THEORY.md), [experimental protocol](docs/PROTOCOL.md), and [related work](docs/RELATED_WORK.md). The proposed method's scientific advantage remains an experimental question.

## Tests

From the repository root:

```sh
.venv/bin/python -m pytest -q --tb=short research/controlled_leakage/tests
.venv/bin/python -m pytest -q --tb=short
```

The first command needs no database. Tiny random models exercise probability inversion, complete inversion training/evaluation/resumption, and exact sampled-token likelihood without downloading pretrained weights. The real LoRA checkpoint test runs when `peft` is installed. The second command also runs PaperKB's existing integration tests against disposable schemas in `paperkb_test`.
