# Running experiments from the notebook

## Public CPU workflows

The repository owner can connect a fine-grained GitHub token scoped to `cornelpascal/prompt-compression-research` with **Actions read/write**. GitHub also provides repository metadata read access. The interface checks the authenticated username and repository write permission, dispatches only `research.yml` on `main`, and allows cancellation of selected runs. Credentials remain in JavaScript memory until disconnect or reload. They are never saved in localStorage, sessionStorage, URLs or exports. Do not paste a token into an issue, command output or research artifact.

The workflow independently checks the actor and branch. It validates task and request identifiers before executing a fixed Python argument list. User inputs are passed via environment variables, not interpolated into shell source. Visitors may read run metadata and published results without authentication, subject to GitHub API rate limits.

| Preset | Workload | Published evidence |
| --- | --- | --- |
| `attention-demo` | Tiny first-layer Q/K SFT, validation calibration, held-out fixture test | F1, retention, selected settings, engineering label |
| `compression-demo` | Tiny rewrite/deletion supervision, training, recovery, evaluation | Method/receiver curves and engineering label |
| `privacy-pilot` | 12 base synthetic records per split, three seeds, four controls | Utility/recovery summaries with declared byte units |
| `theory` | 48-point finite-source grid | Feasibility, leakage, residuals and numerical gaps |
| `tests` | PaperKB, compression, leakage, and explicit two-process recovery | Per-suite JUnit counts with failures/errors/skips separate |

Jobs run on GitHub-hosted Ubuntu with Python 3.12. Neural presets install Torch 2.8 CPU and pinned dependencies. The test preset uses a temporary PostgreSQL 18/pgvector service at `paperkb_test`; each database test receives its own disposable schema. The service is destroyed with the job.

No pretrained weights, paid model API calls or GPU jobs are required for these presets. The workflow timeout is 30 minutes. Cancellation is explicit. A cancelled workflow does not publish an incomplete measurement as a result; its cancelled status remains visible in GitHub activity.

## Publication and failures

The runner writes `verification/current/public/outcome.json` and optional JUnit files. Aggregated numerical rows, source SHA and status are uploaded as a GitHub artifact. Raw local execution logs, raw case records, model weights and fixture checkpoints are not uploaded. The publisher verifies the outcome schema, excludes private-looking values, adds a permanent JSON record, refreshes hashes and explicitly calls the Pages deployment workflow. Failed experiments can publish their failure record; the overall workflow still reports failure.

If dependency installation or artifact creation fails before a usable outcome exists, the workflow logs are the source of truth and no result record is invented. If a push conflicts with an unrelated change, rerun the publication job against the latest `main`.

## Private GPU execution repository

The public repository intentionally has **no active self-hosted workflow**. Put [gpu-study.yml](runner-templates/gpu-study.yml) in a separate private execution repository only after connecting a trusted runner. Keep pull-request code away from that runner. Configure the `research-gpu` environment, labels `self-hosted, linux, x64, research-gpu`, a reviewed immutable public source revision, locked model configurations, prepared datasets, persistent output directories and receiver URLs. Install the research package using its cloud lock/Dockerfile before running. Enable `ENABLE_GPU_RESEARCH=true` only when these requirements are met.

The template supports the existing compression and attention study scripts plus controlled-leakage training/evaluation. It provides no compute provisioning and incurs no GPU work until explicitly enabled and dispatched in the private repository. Real-model training, calibration, validation/test separation, transfer tests, and uncached latency benchmarking remain necessary before publishing quality or speed claims.

Connecting private execution controls to the public UI is a future configuration change: the current browser session is scoped to this public repository. Do not silently broaden it or dispatch into an arbitrary repository.

## Production PaperKB snapshots

[paperkb-snapshot.yml](runner-templates/paperkb-snapshot.yml) is a disabled private-runner template. It requires a database account restricted to read-only access, the existing private workspace and saved research artifacts. It exports a raw snapshot only to private runner storage, then creates an allowlisted public bundle. It does not alter paper states, run extraction/review workers or publish graph parents. Review the bundle and commit it through the public repository's normal publication flow.

The original evidence workflow remains governed by [WORKER_PROTOCOL.md](../WORKER_PROTOCOL.md). This website does not create an unattended extraction or review service.
