# Implementation validation — 2026-10-17

The isolated package runs without PaperKB or PostgreSQL. Its scientific hypothesis has **not** been established: the completed runs validate software and finite-source controls, not a trained compressor's privacy advantage.

## Executed checks

| Check | Result |
|---|---|
| Package tests | 29 passed, including a real LoRA update/checkpoint reload using a tiny locally initialized Llama |
| Neural inversion integration | Tiny local causal model → real probability traces → training export → T5 inverter/public-only prior → held-out evaluation → resume |
| CPU pilot | 9,216 evaluation rows, 384 summary groups, 4,608 case checkpoints |
| Pilot configuration | Three seeds, four requested ratios, four methods, context/text observers, copying/attribute attacks |
| Training export | 6,912 rows; no evaluation IDs; labels separated from inference observations |
| Resume | Every completed pilot case retained its original modification time |
| Budget enforcement | Every empty-context and redaction-plus-compression row met its budget |
| Paired comparisons | 96 model/budget/observer/attacker/domain/regime cells |
| Finite optimization | 48 grid points: 33 numerically certified feasible points, 15 infeasible points |
| Largest finite-solver gap | 7.214 × 10⁻⁸ bits; floating-point certificate only |
| Packaging | Wheel built; all Python modules compiled |

The LoRA test used temporary `peft` and `accelerate` dependencies; it skips when those optional dependencies are absent. No pretrained model weights were downloaded for these tests.

At expected length 0.5 and zero task error, the binary solver returned 0 bits of leakage for independent secrets, 0.531004406 bits for 90%-correlated secrets, and 1 bit for task-essential secrets. These reproduce the finite controls described in [THEORY.md](THEORY.md), not a new general privacy guarantee.

The CPU pilot uses UTF-8 byte counts and a deterministic rule task. Its repeated baseline seeds do not represent independently trained models. Original-context and redaction-only controls are deliberately unconstrained and sometimes exceed the requested budget; their rows must not be treated as matched-length comparisons. At the smallest retained budget, utility drops, as expected.

## Local artifacts

- [Machine-readable validation and checksums](validation.json)
- [Pilot summary](../runs/pilot/summary.csv)
- [Paired comparison cells](../runs/pilot/comparison.json)
- [Finite-source frontier](../runs/frontier.json)

Generated datasets, checkpoints and full run artifacts are ignored by Git. Commands in the package README reproduce them. Earlier engineering runs remain in the `runs/pilot-before-*` directories; the current verified run is `runs/pilot`, with literal secret-recovery scoring.

## Work requiring research infrastructure

Large-model training and comparisons were not run: this host has no CUDA device. The alternating compressor trainer, pinned model configurations, independent attacker trainers, ablations and evaluation commands are implemented, but require a GPU smoke run before expensive experiments. Memory estimates have not been measured.

PII-Bench evaluation requires separately annotated task answers and protected values. The importer and provenance checks are implemented; no external evaluation results are claimed. Adaptive anonymization and PILS components are independently implemented adaptations, not verified reproductions of their published experiments. Structured redaction is applicable to the synthetic records; it is not a general natural-language PII detector.

The required repository-wide test command was attempted. Existing PaperKB integration tests encountered 137 setup errors because PostgreSQL refused the connection. No database schema, source PDF, review status or production graph state was changed. Those tests need a running disposable `paperkb_test` database before their status can be verified.
