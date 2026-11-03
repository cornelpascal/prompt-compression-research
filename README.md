# Prompt Compression Research

Research software and public notebook by **Ganea Corneliu**.

**[Open the research interface](https://cornelpascal.github.io/prompt-compression-research/)** · [Run workflows](https://cornelpascal.github.io/prompt-compression-research/#/run) · [Results and analysis](https://cornelpascal.github.io/prompt-compression-research/#/results)

This workspace connects evidence-linked literature, prompt compression, and controlled information leakage. The interface includes method documentation, dataset protocols, interactive graphs, case examples, test evidence, provenance, downloads, and owner-operated CPU workflows.

| Section | Implementation | Evidence available |
| --- | --- | --- |
| PaperKB | [Source](src/paperkb), [operations guide](PAPERKB.md), [worker protocol](WORKER_PROTOCOL.md) | Dated snapshot of 1,565 papers and 3,241 current independently reviewed ideas; no production ancestry links |
| Prompt compression | [Package](research/prompt_compression), [attention method](research/prompt_compression/ATTENTION.md) | Rewrite, deletion and first-layer Q/K engineering fixtures; calibration and held-out evaluation infrastructure |
| Controlled leakage | [Package](research/controlled_leakage), [protocol](research/controlled_leakage/docs/PROTOCOL.md) | Synthetic CPU controls and finite-source optimization, with explicit attacker access |
| Research interface | [React application](web), [interface guide](docs/INTERFACE.md) | Curated, checksummed public snapshots and newly published workflow outcomes |
| Execution | [Runner guide](docs/RUNNERS.md), [CPU workflow](.github/workflows/research.yml) | Bounded CPU jobs; private GPU/database templates are inactive |

The 98% relative-F1 target is an acceptance criterion, **not an established real-model result**. Tiny randomly initialized models and artificial receivers verify software behavior. Receiver aliases such as Qwen and Mistral in those fixtures do not represent measured pretrained models. The attention scorer reads every candidate token through a shallow embedding/Q/K computation; it avoids the full generator forward pass, not reading the prompt. Attention weights alone do not certify downstream answer accuracy.

## Local interface

Requires Node 24 and npm. The public snapshot is already included; no database or model download is required.

```sh
cd web
npm ci
npm run dev
```

Open the URL printed by Vite. Production builds use the `/prompt-compression-research/` base path and hash routes, so direct section links work on GitHub Pages.

```sh
cd web
npm test
npm run build
# Browser integration suite, after installing Chromium in your test environment:
npx playwright install chromium
npm run test:e2e
```

## Reproduce CPU research

For experimental single-device TPU execution and its limits, see
[research/TPU.md](research/TPU.md).

For a small real-receiver Colab T4 pilot, install the prompt-compression training
extra and `bitsandbytes`, then run the standalone
[`colab_receiver_pilot.py`](research/prompt_compression/colab_receiver_pilot.py):

```sh
python -m pip install -e 'research/prompt_compression[train]' bitsandbytes
python research/prompt_compression/colab_receiver_pilot.py --output /content/qwen14b-pilot --count 1
python research/prompt_compression/colab_receiver_pilot.py --output /content/qwen14b-study --count 24
```

The default is Qwen2.5-14B-Instruct with 4-bit NF4 weights. The one-example run
checks model loading and memory before the 24-example comparison. Each example
is answered with full context, prefix truncation, random sentence selection,
and query-aware BM25 sentence selection. The result JSON records model revision,
dataset hash, question IDs, per-case accuracy, token counts, latency, and peak GPU
memory. This is an exploratory SQuAD training-split pilot, not a held-out result.
For a smaller FP16 receiver, pass `--model Qwen/Qwen2.5-3B-Instruct
--quantization none` with a new output directory.

For an API receiver that needs no Colab GPU, use OpenAI's `gpt-6-sol` with
`reasoning.effort=medium`. Create its encrypted local config in a private terminal:

```sh
python -m pip install -e 'research/prompt_compression[api]'
python research/prompt_compression/secure_openai_config.py init
python research/prompt_compression/api_receiver_pilot.py --output research/prompt_compression/runs/api-one --count 1
python research/prompt_compression/api_receiver_pilot.py --output research/prompt_compression/runs/api-24 --count 24
```

The setup prompts for the API key and a passphrase without echoing either. It
encrypts the key with AES-256-GCM and a scrypt-derived key in
`research/prompt_compression/.secrets/openai.enc.json`; that directory and the
run outputs are Git-ignored. Each run prompts for the passphrase, decrypts only
in memory, and sends the public SQuAD passage and question to the Responses API.
The one-example run makes four API requests. The 24-example run makes 96.
`max_output_tokens` defaults to 4096 per request, including reasoning tokens;
actual usage is recorded in the result. Review the one-example usage before
starting the larger run. The API model is a remote receiver, so it does not
measure local GPU memory or latency.

## Prompt inversion

The [Prompt Inversion section](https://cornelpascal.github.io/prompt-compression-research/#/prompt-inversion)
defines a bounded black-box probe of GPT-6 Luna's hidden instructions. It pairs
24 fixed user prompts with a synthetic known-prompt control, for 48 requests
at most. The control measures exact recovery of its random marker and complete
prompt. Luna's own hidden prompt is not available as ground truth, so any live
claim to reveal it is reported as unverified. Only visible output is observed.

With the encrypted API config initialized as above, run the two-request check
before the full experiment:

```sh
python research/prompt_compression/prompt_inversion.py --output research/prompt_compression/runs/inversion-smoke --limit 2
python research/prompt_compression/prompt_inversion.py --output research/prompt_compression/runs/inversion-full --limit 48
```

Use Python 3.12 in a fresh virtual environment. Linux CPU installation:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ci.txt
python -m pip install --no-deps -e . -e research/prompt_compression -e research/controlled_leakage
python scripts/run_research.py --task attention-demo --run-id local --request-id local-attention --output verification/attention
python scripts/run_research.py --task theory --run-id local --request-id local-theory --output verification/theory
```

On macOS, install Torch 2.8 from the default PyPI index instead of the Linux CPU index. The `tests` preset also requires a disposable `paperkb_test` PostgreSQL database with pgvector. The GitHub workflow provisions this automatically. It never connects to the production PaperKB database.

```sh
python3 scripts/validate_site_data.py
python3 -m unittest discover -s scripts/tests -v
gh workflow run research.yml -f task=theory -f request_id=manual-theory-001
```

The CPU presets are `attention-demo`, `compression-demo`, `privacy-pilot`, `theory`, and `tests`. Each records status, source commit, elapsed time and curated results. Failed experiments remain failures. Successful publication explicitly redeploys Pages; a bot commit alone does not trigger a second workflow.

## Evidence and publication

`site-data/manifest.json` hashes every public JSON artifact. Exporting a fresh original workspace requires its saved research artifacts and accepted PaperKB snapshot:

```sh
python3 scripts/export_site.py --workspace /path/to/private-workspace
python3 scripts/validate_site_data.py
```

The export uses field allowlists and excludes raw worker records, database connection details, local paths, PDFs, weights, and full datasets. It preserves the original observation dates. A newly built website does not make historical data current. The public snapshot is sufficient to reproduce displayed graphs; reproducing the underlying studies requires the corresponding data and environment.

Referenced papers, datasets and models retain their authorship and licenses. This repository does not redistribute their PDFs or pretrained weights. Cite the workspace as: Ganea Corneliu, *Prompt Compression Research: research software and evidence notebook*, 2026.
