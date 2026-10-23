CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE corpus (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text NOT NULL UNIQUE,
    state text NOT NULL DEFAULT 'open', current_graph_id uuid,
    expansion_closed boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE papers (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), corpus_id uuid NOT NULL REFERENCES corpus(id),
    arxiv_id text UNIQUE, doi text, title text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}', source_depth integer NOT NULL CHECK(source_depth IN (0,1)),
    withdrawal_note text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE corpus_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), corpus_id uuid NOT NULL REFERENCES corpus(id),
    action text NOT NULL, reason text NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX papers_corpus_idx ON papers(corpus_id);
CREATE INDEX papers_doi_idx ON papers(lower(doi)) WHERE doi IS NOT NULL;
CREATE TABLE paper_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), paper_id uuid NOT NULL REFERENCES papers(id),
    sha256 text NOT NULL UNIQUE CHECK(sha256 ~ '^[a-f0-9]{64}$'), file_path text NOT NULL,
    version_label text, available_at timestamptz, date_verified boolean NOT NULL DEFAULT false,
    state text NOT NULL DEFAULT 'registered', extraction_revision integer NOT NULL DEFAULT 0 CHECK(extraction_revision >= 0),
    reviewed_revision integer, extractor_id text, reviewer_id text,
    relevance_score integer CHECK(relevance_score BETWEEN 0 AND 100), relevance_reason text,
    preparation jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(),
    CHECK(NOT date_verified OR available_at IS NOT NULL),
    CHECK(reviewer_id IS NULL OR extractor_id IS NULL OR reviewer_id <> extractor_id)
);
CREATE INDEX paper_versions_paper_idx ON paper_versions(paper_id);
CREATE TABLE pages (
    version_id uuid NOT NULL REFERENCES paper_versions(id), page_number integer NOT NULL CHECK(page_number > 0),
    text text NOT NULL, text_sha256 text NOT NULL CHECK(text_sha256 ~ '^[a-f0-9]{64}$'),
    flags jsonb NOT NULL DEFAULT '[]', PRIMARY KEY(version_id,page_number)
);
CREATE TABLE sections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), version_id uuid NOT NULL REFERENCES paper_versions(id),
    ordinal integer NOT NULL CHECK(ordinal >= 0), title text NOT NULL,
    page_start integer NOT NULL CHECK(page_start > 0), page_end integer NOT NULL,
    UNIQUE(version_id,ordinal), UNIQUE(version_id,id), CHECK(page_end >= page_start)
);
CREATE TABLE jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), version_id uuid NOT NULL REFERENCES paper_versions(id),
    kind text NOT NULL CHECK(kind IN ('extract','review')),
    state text NOT NULL DEFAULT 'queued' CHECK(state IN ('queued','leased','done','blocked')),
    revision integer NOT NULL CHECK(revision > 0), agent_id text, token uuid,
    lease_until timestamptz, attempts integer NOT NULL DEFAULT 0 CHECK(attempts >= 0), error text,
    created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(version_id,kind,revision),
    CHECK(state <> 'leased' OR (agent_id IS NOT NULL AND token IS NOT NULL AND lease_until IS NOT NULL))
);
CREATE INDEX jobs_queue_idx ON jobs(kind,state,created_at);
CREATE INDEX jobs_lease_idx ON jobs(lease_until) WHERE state='leased';
CREATE TABLE checkpoints (
    job_id uuid NOT NULL REFERENCES jobs(id), section_id uuid NOT NULL REFERENCES sections(id),
    payload jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(job_id,section_id)
);
CREATE TABLE submissions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), job_id uuid NOT NULL REFERENCES jobs(id),
    payload_hash text NOT NULL, payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(job_id,payload_hash)
);
CREATE TABLE ideas (
    id uuid PRIMARY KEY, version_id uuid NOT NULL REFERENCES paper_versions(id), local_id text NOT NULL,
    revision integer NOT NULL CHECK(revision > 0),
    kind text NOT NULL CHECK(kind IN ('definition','method','assumption','equation','theory','experiment','baseline','limitation','open_question')),
    title text NOT NULL, statement text NOT NULL, conditions text NOT NULL DEFAULT '',
    attribution text NOT NULL CHECK(attribution IN ('original','background','synthesis')),
    confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    relevance_score integer CHECK(relevance_score BETWEEN 0 AND 100), relevance_reason text,
    embedding_text text NOT NULL, validity_flags jsonb NOT NULL DEFAULT '[]', active boolean NOT NULL DEFAULT true,
    UNIQUE(version_id,local_id)
);
CREATE INDEX ideas_version_active_idx ON ideas(version_id) WHERE active;
CREATE INDEX ideas_full_text_idx ON ideas USING gin(to_tsvector('english', title || ' ' || statement || ' ' || conditions));
CREATE TABLE idea_evidence (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), idea_id uuid NOT NULL REFERENCES ideas(id),
    page_number integer NOT NULL CHECK(page_number > 0), quote text NOT NULL,
    evidence_kind text NOT NULL CHECK(evidence_kind IN ('text','visual')), locator jsonb NOT NULL DEFAULT '{}',
    CHECK(evidence_kind <> 'visual' OR locator <> '{}'::jsonb)
);
CREATE INDEX idea_evidence_idea_idx ON idea_evidence(idea_id);
CREATE TABLE section_coverage (
    version_id uuid NOT NULL REFERENCES paper_versions(id), section_id uuid NOT NULL,
    revision integer NOT NULL CHECK(revision > 0), status text NOT NULL CHECK(status IN ('covered','unreadable')),
    notes text NOT NULL, PRIMARY KEY(version_id,section_id,revision),
    FOREIGN KEY(version_id,section_id) REFERENCES sections(version_id,id)
);
CREATE TABLE reviews (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), version_id uuid NOT NULL REFERENCES paper_versions(id),
    revision integer NOT NULL CHECK(revision > 0), reviewer_id text NOT NULL, accepted boolean NOT NULL,
    payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(version_id,revision)
);
CREATE TABLE paper_references (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), version_id uuid NOT NULL REFERENCES paper_versions(id),
    citation_key text NOT NULL, title text NOT NULL, arxiv_id text, doi text, url text,
    UNIQUE(version_id,citation_key)
);
CREATE TABLE predecessor_nominations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), idea_id uuid NOT NULL REFERENCES ideas(id),
    reference_id uuid NOT NULL REFERENCES paper_references(id), evidence_page integer NOT NULL CHECK(evidence_page > 0),
    evidence_quote text NOT NULL, state text NOT NULL DEFAULT 'pending', reviewer_id text,
    reason text, target_paper_id uuid REFERENCES papers(id), UNIQUE(idea_id,reference_id),
    CHECK(state IN ('pending','approved','existing','admitted','unavailable','rejected'))
);
CREATE TABLE embeddings (
    idea_id uuid PRIMARY KEY REFERENCES ideas(id), revision integer NOT NULL CHECK(revision > 0),
    model text NOT NULL, model_revision text NOT NULL, text_sha256 text NOT NULL,
    embedding vector(1024) NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX embeddings_cosine_idx ON embeddings USING hnsw(embedding vector_cosine_ops);
CREATE TABLE concepts (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), label text NOT NULL, description text NOT NULL);
CREATE TABLE concept_memberships (
    concept_id uuid NOT NULL REFERENCES concepts(id), idea_id uuid NOT NULL REFERENCES ideas(id),
    rationale text NOT NULL, PRIMARY KEY(concept_id,idea_id)
);
CREATE TABLE graph_builds (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), corpus_id uuid NOT NULL REFERENCES corpus(id),
    state text NOT NULL DEFAULT 'building' CHECK(state IN ('building','published','superseded')),
    snapshot jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), published_at timestamptz
);
ALTER TABLE corpus ADD CONSTRAINT corpus_current_graph_fk FOREIGN KEY(current_graph_id) REFERENCES graph_builds(id);
CREATE TABLE idea_relations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(), graph_id uuid NOT NULL REFERENCES graph_builds(id),
    child_id uuid NOT NULL REFERENCES ideas(id), parent_id uuid NOT NULL REFERENCES ideas(id),
    relation_type text NOT NULL, evidence jsonb NOT NULL, confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    reviewer_id text NOT NULL, confirmed boolean NOT NULL, is_primary boolean NOT NULL DEFAULT false,
    rationale text NOT NULL, UNIQUE(graph_id,child_id,parent_id,relation_type), CHECK(child_id <> parent_id),
    CHECK(NOT is_primary OR confirmed)
);
CREATE INDEX idea_relations_child_idx ON idea_relations(graph_id,child_id);
CREATE TABLE graph_parents (
    graph_id uuid NOT NULL REFERENCES graph_builds(id), idea_id uuid NOT NULL REFERENCES ideas(id),
    parent_id uuid REFERENCES ideas(id), rationale text NOT NULL,
    PRIMARY KEY(graph_id,idea_id), CHECK(idea_id <> parent_id)
);
CREATE VIEW ideas_current AS
    SELECT i.*, gp.parent_id FROM ideas i
    JOIN paper_versions v ON v.id=i.version_id JOIN papers p ON p.id=v.paper_id
    JOIN corpus c ON c.id=p.corpus_id
    LEFT JOIN graph_parents gp ON gp.graph_id=c.current_graph_id AND gp.idea_id=i.id
    WHERE i.active;
CREATE VIEW paper_progress AS
    SELECT p.id AS paper_id,p.corpus_id,p.title,p.source_depth,v.id AS version_id,v.state,
        v.extraction_revision,v.reviewed_revision,v.relevance_score,
        (SELECT count(*) FROM ideas i WHERE i.version_id=v.id AND i.active) AS idea_count,
        (SELECT count(*) FROM sections s WHERE s.version_id=v.id) AS section_count,
        (SELECT count(*) FROM section_coverage sc WHERE sc.version_id=v.id AND sc.revision=v.extraction_revision AND sc.status='covered') AS covered_sections
    FROM papers p JOIN paper_versions v ON v.paper_id=p.id;
