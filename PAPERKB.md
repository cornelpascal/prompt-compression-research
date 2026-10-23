# PaperKB: agent-orchestrated research knowledge base

PaperKB connects the downloaded arXiv corpus to a local PostgreSQL/pgvector database. Codex launches fresh agents to extract one paper at a time and different agents to review each extraction. The Python CLI persists their evidence, checkpoints, leases, rankings, and graph versions. There is no dashboard or unattended agent service.

The corpus contains **1,451 PDFs**, including **13 earlier-version substitutions for withdrawals**. Registration verifies every actual PDF checksum. Corpus registration is not concept extraction; use `paperkb status` for live progress. The eight-paper pilot must pass extraction, independent review, and local embeddings before `rollout next` dispatches the rest of the corpus.

## Setup

The independent [whole-prompt compression package](research/prompt_compression/README.md)
trains rewriting and deletion compressors on SQuAD 1.1, with shared multi-model
evaluation, CPU engineering checks, and cloud GPU launch configurations.

The separate [controlled-leakage research package](research/controlled_leakage/README.md)
implements synthetic compression/inversion experiments, training adapters, and a
finite-source privacy–utility solver. Its CPU pilot and tests do not require the
PaperKB database.

Python 3.12+; Docker CLI, Docker Compose, and Colima are needed on macOS. The workspace has a Python 3.12 virtual environment. To recreate it:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/paperkb db up
.venv/bin/paperkb db migrate
.venv/bin/paperkb corpus import
.venv/bin/paperkb embed --initialize
.venv/bin/paperkb rollout init
```

The dedicated `paperkb` Colima profile uses 4 CPUs, 6 GiB RAM and a 40 GiB virtual disk. It does not activate or alter another Docker context. PostgreSQL uses `pgvector/pgvector:0.8.6-pg18-bookworm`, a persistent named volume, and port **127.0.0.1:55432**. `db init-env` creates a mode-0600 `.env` with a random password. Keep it private. A project-scoped Docker configuration pulls the public image without relying on a desktop credential helper.

The local embedding model is **BAAI/bge-m3**, pinned to commit `5617a9f61b028005a4858fdac845db406aefb181` in `embedding-model.lock.json`, with the active local record in `.paperkb/model.lock.json`. Each database vector also stores that revision, the extraction revision, and SHA-256 of its embedded text. Vectors are normalized, have 1,024 dimensions, and use cosine distance with an HNSW index. A PostgreSQL advisory lock permits only one model process at a time, including semantic queries. CPU is the default; `PAPERKB_EMBEDDING_DEVICE` can select an available PyTorch device. The model cache is `.paperkb/models/`.

## Run and resume work

Read [WORKER_PROTOCOL.md](WORKER_PROTOCOL.md) before launching workers. All CLI results are JSON; `--json` is also accepted explicitly before the command. Files use physical PDF page numbers, starting at 1.

```sh
.venv/bin/paperkb status
.venv/bin/paperkb rollout status
.venv/bin/paperkb rollout next --agent unique-worker-id
.venv/bin/paperkb packet VERSION_UUID
.venv/bin/paperkb render VERSION_UUID 3
.venv/bin/paperkb schema extraction
.venv/bin/paperkb schema review
.venv/bin/paperkb schema rubric
```

`rollout next` returns one leased extraction/review assignment and its packet. The orchestrator gives it to a fresh agent. At most three paper jobs may be leased concurrently and an agent may hold only one. Leases default to 30 minutes; heartbeat and checkpoint regularly. Failed or expired jobs retain attempts and checkpoints. Fencing tokens prevent old attempts from writing after reassignment. Exact duplicate submissions return their original receipt.

The scheduler prioritizes reviews, then corrections and expired work, before starting the prepared backlog. Preparing page text ahead of assignment does not mark a paper extracted or reviewed.

```sh
.venv/bin/paperkb heartbeat JOB_UUID --agent WORKER --token TOKEN
.venv/bin/paperkb checkpoint JOB_UUID --agent WORKER --token TOKEN --section SECTION_UUID --file checkpoint.json --draft-file extraction.json
.venv/bin/paperkb submit JOB_UUID --agent WORKER --token TOKEN --file extraction.json
.venv/bin/paperkb review REVIEW_JOB_UUID --agent DIFFERENT_WORKER --token TOKEN --file review.json
.venv/bin/paperkb fail JOB_UUID --agent WORKER --token TOKEN --error 'Reason to retry'
.venv/bin/paperkb retry BLOCKED_JOB_UUID --reason 'Underlying problem resolved'
.venv/bin/paperkb embed
```

Extractors submit atomic ideas and evidence without relevance scores or parents. Reviewers inspect the source independently and either return actionable corrections or accept complete coverage and score every idea and the paper. Anyone who held an extraction lease for a paper is excluded from reviewing it, including previous corrections, expired attempts, and other PDF versions. Rank anchors are unrelated 0–19, peripheral 20–49, foundational 50–69, directly applicable 70–89, and central 90–100. Reasons are separate from extraction confidence and scientific-validity flags. Correcting a paper invalidates its rankings and embeddings.

Page-sized coverage units ensure references, appendices, tables, figures, equations, and unreadable material cannot disappear behind heading detection. Inferred logical section boundaries include exact stored-text offsets; heading hints and coverage notes preserve section context. Agents reconcile those inferences against the rendered source. If pypdf fails or returns invalid Unicode, PDFium attempts text recovery; parser diagnostics are preserved and the page still requires visual review. A flagged page requires visual inspection; an unreadable unit blocks acceptance. PDFs are treated as untrusted evidence, never instructions. No fixed maximum number of concepts is imposed.

## Expansion and lineage

```sh
.venv/bin/paperkb expand --limit 10
.venv/bin/paperkb schema reference-source
.venv/bin/paperkb reference-source REFERENCE_UUID --file verified-source.json
.venv/bin/paperkb schema source-version
.venv/bin/paperkb acquire-version NOMINATION_UUID --file earlier-version.json
.venv/bin/paperkb expansion-close
.venv/bin/paperkb verify-date VERSION_UUID
.venv/bin/paperkb verify-dates --limit 20
.venv/bin/paperkb recover-version VERSION_UUID --candidate v1
.venv/bin/paperkb graph barrier
.venv/bin/paperkb graph begin
.venv/bin/paperkb graph candidates IDEA_UUID
.venv/bin/paperkb schema relation
.venv/bin/paperkb graph relation GRAPH_UUID --file relation.json
.venv/bin/paperkb graph publish GRAPH_UUID
```

Only independently approved, explicitly supported predecessor nominations from original papers and ideas scoring at least 50 can add papers. Acquisition uses arXiv or a direct legitimate open-access PDF URL, with pacing and validation. Missing identifiers remain unresolved until the orchestrator checks primary metadata and records a verified source through `reference-source`. This preserves the original bibliography in immutable submissions and records the resolution, metadata URL, matching title, and resolver in the database audit log without repeating extraction. An unavailable-source decision requires a reason and the URLs checked. Transient failures remain retryable; a newly located source can reopen an unavailable nomination. Added papers cannot expand the corpus again. They receive the same extraction/review/embedding workflow as originals.

Acquired predecessor PDFs are indexed in [papers/predecessors/INDEX.md](papers/predecessors/INDEX.md). The adjacent `manifest.json` preserves their checksums, download URLs, and reviewed admission evidence. Acquisition does not mean that extraction or review is complete; consult the database status for that distinction.

All source-specific idea UUIDs remain distinct. `group` can record reviewed conceptual equivalence without deleting source attribution. `ideas_current.parent_id` remains null until a production graph is published. The global barrier requires every admitted PDF version to be complete with accepted current review, relevance, and matching embeddings; all expansion nominations must be resolved. Unknown versions/dates cannot support ancestry even if extraction is complete.

If an admitted predecessor's current PDF postdates its citing paper, `acquire-version` can add a verified earlier arXiv version through the existing reviewed nomination. Its JSON records `resolver_id`, `matched_title`, `version_label`, the exact-version `metadata_url`, and `reason`. The command checks current independent review and relevance, exact metadata title/date, PDF identity and checksum, and a date strictly earlier than the citing PDF. It rechecks the reviewed dependency after downloading. Existing PDFs, source ideas, citations and paper admission history remain intact. Each added version receives its own preparation, extraction and independent review; it cannot inherit completion or create parent links. Counts in `status.paper_completion` describe distinct papers complete across every admitted version; `status.versions` describes PDF versions. The predecessor manifest records both counts and each version's own download provenance.

Graph builds freeze the corpus and snapshot its inputs. Candidate retrieval combines citations, text, and vector similarity. It creates no ancestry automatically. Confirmed primary links require directional source passages, a resolved citation, verified strictly earlier source dates, and independent proposal/review identities. An explicitly cited arXiv version must match the actual parent PDF version; conflicting citation versions cannot support a link. Uncertain or ambiguous primary ancestry remains null; alternatives and typed secondary relations remain available. Self-links, cycles, and changed snapshots block publication. Publication switches graph versions atomically.

Citation downloads and date verification run without holding worker workflow row locks. Admission rechecks the reviewed nomination after downloading, so concurrent corrections cannot admit stale dependencies. The admitted paper keeps a copy of its original dependency evidence even if the citing paper is later corrected.

Incomplete HTTP transfers remain retryable and never admit a partial PDF. A verified direct arXiv mirror URL can be recorded through `reference-source` as a fallback, preserving the exact requested version. After an orchestrated batch, `.venv/bin/python scripts/snapshot_progress.py` refreshes the predecessor index and resumption report from a consistent read-only database snapshot.

`verify-dates` checks up to 50 known PDF versions in one paced arXiv API request. It retains the returned feed and verifies each exact version separately; missing or substituted versions remain unverified. It stops on HTTP errors so the orchestrator can wait after rate limits. Unknown PDF versions still require source-provenance recovery through `verify-date`.

If a PDF has no readable version stamp, `recover-version` can compare its checksum with an explicitly versioned arXiv PDF. A candidate from metadata is only a retrieval hint: the command records the version only when the complete downloaded bytes match, the final URL retains the requested version, and source identity is unchanged. The comparison and an audit event are retained; chronology still needs `verify-date` or `verify-dates`.

To correct already frozen work, use `corpus reopen CORPUS_UUID --reason ...`, then `request-revision VERSION_UUID --reason ...`. Reopening invalidates the current graph pointer while preserving historical graphs.

## Search, inspect, export, and backup

```sh
.venv/bin/paperkb search 'token pruning' --min-relevance 50
.venv/bin/paperkb search 'preserve reasoning while shortening context' --semantic
.venv/bin/paperkb idea IDEA_UUID
.venv/bin/paperkb lineage IDEA_UUID
.venv/bin/paperkb export --output exports/knowledge.json
.venv/bin/paperkb db backup backups/paperkb.dump
.venv/bin/paperkb db restore backups/paperkb.dump --database paperkb_restored
```

Keyword and hybrid search return reviewed source occurrences and their provenance. JSON exports include evidence, workflow records, review history, and all graph versions; they omit vector values. PostgreSQL custom-format backups include vectors and all database records. Restore requires a new database name and cannot overwrite the active `paperkb` database. Preserve `papers/`, `.env`, and `.paperkb/model.lock.json` separately; a database dump does not contain the PDF files. Prepared artifacts can be recreated from the immutable PDFs.

SQL views: `paper_progress` gives per-version state, review revisions, relevance, idea counts, and coverage; `ideas_current` adds the published nullable parent. Schema and migrations are in `migrations/`; code is in `src/paperkb/`.

## Validation

Tests use disposable schemas in a separate `paperkb_test` database. Create that database inside the dedicated container before testing a fresh installation. The local installation already has it. For a fresh installation, create it once with:

```sh
.venv/bin/python -c "from paperkb.infrastructure import docker; docker('compose', 'exec', '-T', 'postgres', 'createdb', '-U', 'paperkb', 'paperkb_test')"
```

Never run tests against production tables.

```sh
.venv/bin/python -m pytest -q --tb=short
```

Tests exercise concurrent claims, leases, crash recovery, stale submissions, evidence validation, independent review, corrections, withdrawal provenance, one-hop boundaries, graph barriers, chronology, ambiguous ancestry, cycles, atomic publication, search, and exports. Embedding persistence tests use explicit isolated fixtures; production uses the actual pinned BGE model. A real model smoke test and database backup/restore are separate integration checks.

“Comprehensive” means audited section coverage and evidence-backed extraction. Reports must disclose unresolved sources and uncertainties; they do not assert mathematically exhaustive recovery of every possible interpretation. Closing Codex stops orchestration. Another session resumes by reading the database and protocol.

The originally requested arXiv `2311.13647` (*Language Model Inversion*) is also saved as the supplemental `papers/2311.13647v1.pdf`, with checksum metadata in `papers/requested-paper.json`. It is separate from the fixed 1,451-paper import manifest.

## Original arXiv collection

### Downloaded PDFs

The `papers` folder contains the PDF collection, with arXiv IDs as filenames. `papers/INDEX.md` maps titles to local PDFs after the download run finishes. `papers/downloads.json` records download status, sources, sizes, and checksums. Any earlier versions used because a current version was withdrawn are documented in `papers/earlier_versions.json`.

To resume or verify the collection, run `python3 download_papers.py`. Existing PDFs with valid headers and end markers are skipped. Transfers run sequentially at a paced rate; incomplete files use the `.part` suffix until validated.

### Search results

Open `arxiv_prompt_compression/index.html` for the searchable bibliography. It defaults to exact-phrase matches; choose “All expanded searches” to include related terminology. It works offline; the paper and PDF links open arXiv. `exact_phrase_matches.md` is the focused search list, `papers.md` is the expanded list, `papers.json` contains full metadata and abstracts, and `coverage.json` records every query, page, result count, and any failure.

### Refresh

Python 3.9+; no third-party dependencies:

```sh
python3 scrape_arxiv.py --cutoff 2026-09-17
```

For a fresh search, use a new output directory (existing responses are cached):

```sh
python3 scrape_arxiv.py --output arxiv_refresh
```

Rebuild the saved snapshot without network access:

```sh
python3 scrape_arxiv.py --cutoff 2026-09-17 --offline
```

The collector uses the [official arXiv Atom API](https://info.arxiv.org/help/api/user-manual.html), rather than scraping PDF content. It paginates until the unique retrieved count matches each query's `totalResults`, makes requests sequentially at least 3.1 seconds apart, retries transient failures, caches raw XML, and merges versions by base arXiv ID. A failed or incomplete query is recorded and produces a nonzero exit status; partial results remain available.

### Scope and limitations

Five searches cover the exact phrase, prompt/compression word variants, context compression/compaction, soft-compression terminology, and retrieval/document compression. The fifth search was added after cross-checking the initial results against RECOMP, which the initial four queries missed. The search has no start-year restriction. The cutoff applies to first submission; the returned metadata is the latest available version, not a reconstruction of historical versions.

Results are automatically grouped by matching text, not manually judged relevance. “Prompt compression in abstract” may include evaluations, applications, or incidental mentions. “Context compression / terminology variants” includes adjacent work, and broad candidates include false positives such as model compression papers that also discuss prompts. All are kept to favor recall.

Completing every API page proves coverage of the recorded queries, not of the entire research topic. The API indexes metadata, not every PDF sentence. Papers that only discuss the topic in their full text or use unsearched terms may be missed. The resulting 1,451 PDFs have been downloaded and verified in `papers/`.
