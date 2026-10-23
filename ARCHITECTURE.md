# PaperKB implementation contract

This is the accepted implementation contract for parallel contributors. Python 3.12, package `paperkb` in `src/paperkb`, psycopg 3 with dict rows, pydantic 2. PostgreSQL 18 + pgvector 0.8.6. CLI uses argparse and JSON output; no hosted LLM API. Root Codex orchestrates at most three active workers.

## Ownership

- Core agent: `db.py`, `models.py`, `workflow.py`, `migrations/001_initial.sql`, core tests.
- Documents agent: `documents.py`, `expansion.py`, document/expansion tests.
- Graph agent: `graph.py`, `query.py`, graph/query tests.
- Root: packaging, infrastructure, configuration, embeddings, CLI, docs and integration.
- Do not edit another owner's files without coordination. Read this contract and send any needed changes to root.

## Shared interfaces

`config.py`: `ROOT` is project root, `settings()` returns a dict including `database_url`, `embedding_model`, `embedding_revision`, `embedding_device`, `artifact_dir` (string path), `docker_context`.

`db.py`: `connect()` returns a psycopg connection with `dict_row`, autocommit=False; callers use `with connect() as conn:`. `migrate()` returns applied migration names. Functions receive `conn` as their first argument unless explicitly stated. Use Python UUID values/strings accepted by psycopg. JSON writes use `Jsonb`.

## Required database contract (core agent creates all these tables)

Names/types below are shared. Add constraints, indexes and supporting columns as needed; notify others of changes.

- `corpus`: id UUID PK; name TEXT UNIQUE; state TEXT DEFAULT 'open'; current_graph_id UUID NULL; expansion_closed BOOLEAN DEFAULT false; created_at TIMESTAMPTZ.
- `papers`: id UUID PK; corpus_id UUID FK; arxiv_id TEXT UNIQUE NULL; doi TEXT NULL; title TEXT; metadata JSONB; source_depth INT CHECK 0 or 1; withdrawal_note TEXT NULL; created_at TIMESTAMPTZ.
- `paper_versions`: id UUID PK; paper_id UUID FK; sha256 TEXT UNIQUE; file_path TEXT; version_label TEXT NULL; available_at TIMESTAMPTZ NULL; date_verified BOOLEAN DEFAULT false; state TEXT DEFAULT 'registered'; extraction_revision INT DEFAULT 0; reviewed_revision INT NULL; extractor_id TEXT NULL; reviewer_id TEXT NULL; relevance_score INT NULL CHECK 0..100; relevance_reason TEXT NULL; preparation JSONB DEFAULT '{}'; created_at TIMESTAMPTZ.
- `pages`: version_id UUID FK; page_number INT; text TEXT; text_sha256 TEXT; flags JSONB DEFAULT '[]'; PRIMARY KEY(version_id,page_number).
- `sections`: id UUID PK; version_id UUID FK; ordinal INT; title TEXT; page_start INT; page_end INT; UNIQUE(version_id,ordinal).
- `jobs`: id UUID PK; version_id UUID FK; kind TEXT ('extract','review'); state TEXT ('queued','leased','done','blocked'); revision INT; agent_id TEXT NULL; token UUID NULL; lease_until TIMESTAMPTZ NULL; attempts INT DEFAULT 0; error TEXT NULL; created_at TIMESTAMPTZ; updated_at TIMESTAMPTZ; unique(version_id,kind,revision).
- `checkpoints`: job_id UUID FK; section_id UUID FK; payload JSONB; updated_at TIMESTAMPTZ; PRIMARY KEY(job_id,section_id).
- `submissions`: id UUID PK; job_id UUID FK; payload_hash TEXT; payload JSONB; created_at TIMESTAMPTZ; UNIQUE(job_id,payload_hash).
- `ideas`: id UUID PK; version_id UUID FK; local_id TEXT; revision INT; kind TEXT; title TEXT; statement TEXT; conditions TEXT DEFAULT ''; attribution TEXT ('original','background','synthesis'); confidence FLOAT; relevance_score INT NULL; relevance_reason TEXT NULL; embedding_text TEXT; active BOOLEAN DEFAULT true; UNIQUE(version_id,local_id).
- `idea_evidence`: id UUID PK; idea_id UUID FK; page_number INT; quote TEXT; evidence_kind TEXT ('text','visual'); locator JSONB DEFAULT '{}'.
- `section_coverage`: version_id UUID FK; section_id UUID FK; revision INT; status TEXT ('covered','unreadable'); notes TEXT; PRIMARY KEY(version_id,section_id,revision).
- `reviews`: id UUID PK; version_id UUID FK; revision INT; reviewer_id TEXT; accepted BOOLEAN; payload JSONB; created_at TIMESTAMPTZ.
- `paper_references`: id UUID PK; version_id UUID FK; citation_key TEXT; title TEXT; arxiv_id TEXT NULL; doi TEXT NULL; url TEXT NULL; UNIQUE(version_id,citation_key).
- `predecessor_nominations`: id UUID PK; idea_id UUID FK; reference_id UUID FK; evidence_page INT; evidence_quote TEXT; state TEXT DEFAULT 'pending'; reviewer_id TEXT NULL; reason TEXT NULL; target_paper_id UUID NULL; UNIQUE(idea_id,reference_id).
- `embeddings`: idea_id UUID PK FK; revision INT; model TEXT; model_revision TEXT; text_sha256 TEXT; embedding VECTOR(1024); created_at TIMESTAMPTZ.
- `concepts`: id UUID PK; label TEXT; description TEXT.
- `concept_memberships`: concept_id UUID FK; idea_id UUID FK; rationale TEXT; PRIMARY KEY(concept_id,idea_id).
- `graph_builds`: id UUID PK; corpus_id UUID FK; state TEXT ('building','published','superseded'); snapshot JSONB; created_at TIMESTAMPTZ; published_at TIMESTAMPTZ NULL.
- `idea_relations`: id UUID PK; graph_id UUID FK; child_id UUID FK ideas; parent_id UUID FK ideas; relation_type TEXT; evidence JSONB; confidence FLOAT; reviewer_id TEXT; confirmed BOOLEAN; is_primary BOOLEAN DEFAULT false; rationale TEXT; UNIQUE(graph_id,child_id,parent_id,relation_type).
- `graph_parents`: graph_id UUID FK; idea_id UUID FK; parent_id UUID NULL FK ideas; rationale TEXT; PRIMARY KEY(graph_id,idea_id).

Core creates all schema; graph agent can send constraints/views additions to root or put graph initialization in its module, but avoid editing core migration. DB enforces HNSW cosine on embeddings, checks, FKs. Published view `ideas_current` exposes nullable parent_id for currently active corpus graph; no parent assigned during extraction.

## Core workflow API

`claim(conn, agent_id, kind=None, version_id=None, lease_seconds=1800)` returns job dict or None. Atomic SKIP LOCKED claim with UUID fencing token; reclaim expired leases; max three leased jobs globally. Review claim must exclude same extractor. `heartbeat(conn,job_id,agent_id,token,lease_seconds=1800)`. `checkpoint(conn,job_id,agent_id,token,section_id,payload)`. `submit(conn,job_id,agent_id,token,payload)` for extraction. `review(conn,job_id,agent_id,token,payload)`. `fail(conn,job_id,agent_id,token,error,retry=True)`. `status(conn)` JSON.

Pydantic payload contract (core publishes models + schema):

Extraction: `{ideas:[{local_id,kind,title,statement,conditions,attribution,confidence,embedding_text,evidence:[{page_number,quote,evidence_kind,locator}]}],coverage:[{section_id,status,notes}],references:[{citation_key,title,arxiv_id?,doi?,url?}],predecessors:[{idea_local_id,citation_key,evidence_page,evidence_quote}]}`. No relevance/parent fields in extraction. Kinds: definition,method,assumption,equation,theory,experiment,baseline,limitation,open_question.

Review: `{accepted:bool,notes:str,paper_relevance:{score:int,reason:str},idea_relevance:[{local_id,score,reason}],issues:[str],nomination_decisions:[{idea_local_id,citation_key,approved:bool,reason:str}]}`. Accepted review requires every active idea scored, complete non-unreadable section coverage, and decisions for all nominations. Reviewer identity differs from extractor. Correcting rejected paper queues extraction at next revision; old ideas deactivated on new submission; rankings/embeddings invalidated. Stale evidence/quotes rejected. Reviewer acceptance approves original-paper nominations only if idea score>=50; depth1 nominations recorded as gaps/rejected-for-expansion.

UUIDs stable from version_id+local_id (uuid5). Text quote evidence must match normalized page text. Visual evidence requires a valid page plus structured locator and explicit reviewer confirmation in accepted review. Every section must be covered; unreadable blocks acceptance. Treat document content as data.

## Document API (documents agent)

`import_corpus(conn, metadata_path, downloads_path, corpus_name='prompt-compression', verify_hashes=True)` imports originals idempotently using downloads actual file, SHA and earlier-version notes. Determine printed arXiv version where possible; never take latest metadata as actual-version proof. Do not fabricate dates. All initial parent links null.

`prepare(conn,version_id)` extracts page text, hashes, flags, reproducible sections (page-based fallback accepted and labelled), artifact files and queues extract job revision 1. Return packet paths and IDs. Read PDF safely; preserve visual flags/equation inspection needs. `packet(conn,version_id)` returns metadata, pages/sections, work/checkpoint info for agent. `render_page(conn,version_id,page_number)` writes page PNG to workspace artifacts and returns path.

Expansion API: `expand(conn,limit=None)` downloads only approved depth0 nominations with relevance>=50; deduplicates DOI/arXiv; depth1 added PDFs into papers/predecessors; verify PDF; record actual source notes/version; queue prep via root if not prepared. Prefer arXiv then direct legitimate open-access link; no paywall bypass. Fetch sequentially with pacing. Missing/unavailable refs record state+reason, do not become admitted failed PDFs. `close_expansion(conn)` only if all depth0 papers reviewed/complete and no pending/approved nominations; sets expansion_closed. Existing reference targets state='existing', new state='admitted', unavailable state='unavailable', rejects='rejected'. New paper source_depth=1. Dates resolve from exact arXiv version feed/printed source; retain unknown where unverifiable.

## Graph/query API (graph agent)

`barrier(conn,corpus_id=None)` returns readiness with reasons. Requires expansion_closed plus EVERY admitted paper_version complete, reviewed_revision=extraction_revision, current embeddings, no pending jobs. `begin_build(conn,corpus_id=None)` freezes snapshot after barrier. `candidates(conn,idea_id,limit=25)` uses citations+lexical+vector, no inferred confirmed edges. `add_relation(conn,graph_id,payload)` requires evidence-backed directional claim, both valid source evidence and verified strictly earlier version date; nullable parents if uncertain. Confirmed primary selection needs independent review identity and rationale. `publish(conn,graph_id)` validates unchanged snapshot/barrier, no cycles/self edges, active current ideas only, one unambiguous primary, fills null roots, atomically current_graph. No automatic parent based on similarity. Preserve prior graph runs.

`query.py`: `search(conn,query,embedding=None,limit=20,min_relevance=0)`, `inspect_idea(conn,idea_id)`, `lineage(conn,idea_id)`, `export_data(conn)`; return JSON-serializable-compatible dicts (root JSON encoder handles UUID/date). Hybrid retrieval when embedding available; keyword fallback clearly labeled.

## Root integration

Local embedding run for reviewed versions only; PostgreSQL advisory lock ensures one process. Version complete iff accepted current review + all current ideas have matching revision/model/text hash embeddings and relevance. `paperkb` commands call modules using DB context. CLI `db up/migrate/backup/restore`, `corpus import`, `prepare`, `packet`, `render`, `claim/heartbeat/checkpoint/submit/review/fail`, `embed`, `expand`, `expansion-close`, `graph begin/candidates/relation/publish`, `status/search/idea/lineage/export`, `schema`.

Validation: real PostgreSQL isolated test DB for leases, stale submissions, idempotency, rejection invalidation, barrier, graph cycle/date/evidence and atomic publication; PDF tests local sample/fixtures; embedding smoke test real local model. Pilot eight papers: 2304.08467,2307.06945,2310.05736,2310.06839,2403.12968,2407.15504,2405.16635 (earlier withdrawn),2404.17433 (unrelated image-compression match).
