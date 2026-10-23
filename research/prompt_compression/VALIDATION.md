# Implementation validation

Validated locally on 2026-10-22. This records engineering checks, not evidence of useful compression or a new scientific result.

| Check | Result |
|---|---|
| Package test suite | 20 passed; the explicit distributed test is skipped in the ordinary run |
| Two-process CPU integration | 1 passed: synchronized weights and bit-identical interrupted versus uninterrupted recovery |
| Offline end-to-end smoke | Completed supervision, SFT and policy updates for both methods, checkpoint reload, evaluation, and plot export |
| Real parameter updates | Confirmed for tiny Llama rewriting, tiny ModernBERT deletion, and a real PEFT LoRA adapter |
| Package build | Wheel and source distribution built successfully |
| Linux dependency lock | Complete transitive resolution with distribution hashes for Python 3.12 / Linux x86-64 |
| Model revisions | Seven public model revisions resolved and pinned |
| SQuAD 1.1 | Downloaded and validated; 79,100 train / 8,499 validation / 10,570 held out |
| Cloud shell scripts | Shell syntax checked |

The final CPU smoke artifacts are in `runs/cpu-final/`, including `validation.json` and `report/`. Earlier smoke runs are retained. The teacher, reward, and solvers in this smoke are artificial fixtures; the neural models are randomly initialized. Fixture scores must not be presented as benchmark performance. The ordinary tests also cover official-score normalization, label isolation, extraction provenance, overflow accounting, integrity checks, service-failure recovery, transfer-model exclusion, and checkpoint-selection identity.

The distributed check initially timed out with automatic hostname rendezvous on macOS. An explicit `127.0.0.1` rendezvous and loopback interface passed. The test creates and cleans up only its own process group. This validates the synchronization/recovery logic on CPU; it does not validate CUDA/NCCL or a multi-node GPU cluster.

Local environment: Python 3.12.14, PyTorch 2.14.0 (CPU), Transformers 4.56.2, PEFT 0.17.1, Accelerate 1.10.1. The Linux cloud lock uses PyTorch 2.8.0 and its CUDA 12.8 dependencies; it must receive a GPU smoke test before the full study. The local isolated environment intentionally preserves the parent workspace's packages.

## Attention extension validation

The extended package suite passes **30 tests**, with two distributed cases skipped in the ordinary run. Both explicit two-process cases pass: the existing rewriting path and the new attention path synchronize weights and reproduce uninterrupted training exactly after recovery. CPU checks establish first-layer score equivalence, blockwise gradient equivalence, selective weight loading, frozen non-Q/K parameters, Unicode source extraction, budget handling, training/test isolation, non-monotonic calibration, explicit calibration failure, identity checks, held-out transfer failure, and answer-cache bypass for benchmarking.

The complete attention engineering workflow is saved in `runs/attention-cpu-ready/`. Only query/key weights changed. Both fixture variants produced calibration artifacts and completed locked final evaluation with reports. These are random tiny models and artificial solvers, not evidence of 98% quality retention on real SQuAD. CLI training, checkpoint registration, and budgeted compression were also exercised using local fixtures. The machine-readable extension record is `attention-validation.json`; the earlier validation record remains unchanged.

No pretrained attention-compression quality, GPU memory, CUDA/NCCL recovery, or latency improvement has been established. The [attention runbook](ATTENTION.md) includes the cloud workflow and a separate warmed, uncached benchmark command.

## Remaining GPU validation

The existing repository-wide suite was also attempted outside the sandbox: 55 tests passed, 1 skipped, and 137 setup errors occurred because PostgreSQL on `127.0.0.1:55432` refused connections. Those PaperKB integration tests were not validated. The new compression package has no database dependency; its own checks passed independently.

- Build and run the Linux image on the GPU infrastructure.
- Confirm revision-named vLLM endpoints, target chat templates, memory use, and throughput.
- Run single-GPU and two-GPU smoke profiles, followed by the 512-example pilot.
- Measure workload and teacher acceptance before launching full three-seed training.
- Keep the final SQuAD set and Phi transfer model out of pilot tuning.

No large model weights were downloaded for these engineering checks. No cloud job was submitted. The package contains a custom small GRPO implementation; it is not presented as an exact reproduction of a published training recipe.
