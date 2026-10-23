"""Evidence-backed, snapshot-versioned intellectual lineage.

Retrieval only suggests candidates. A reviewed directional claim is required to
publish a parent, and an ambiguous claim leaves the child without a parent.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from pydantic import Field, model_validator

from .config import settings
from .documents import normalize_arxiv, normalize_doi
from .models import Payload, normalized_text


class GraphError(ValueError):
    """A graph operation violates a corpus or evidence invariant."""


class SourceQuote(Payload):
    page_number: int = Field(ge=1, strict=True)
    quote: str = Field(min_length=1)


class RelationEvidence(Payload):
    child: SourceQuote
    parent: SourceQuote
    reference_id: UUID | None = None
    directional_claim: str = Field(min_length=1)


class RelationPayload(Payload):
    child_id: UUID
    parent_id: UUID
    relation_type: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z_]*$")
    evidence: RelationEvidence
    confidence: float = Field(ge=0, le=1)
    confirmed: bool = Field(strict=True)
    is_primary: bool = Field(default=False, strict=True)
    rationale: str = Field(min_length=1)
    proposer_id: str = Field(min_length=1)
    reviewer_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def confirmed_direction(self):
        if self.child_id == self.parent_id:
            raise ValueError("An idea cannot be its own parent")
        if self.confirmed and (not self.reviewer_id or self.reviewer_id == self.proposer_id):
            raise ValueError("Confirmation requires a reviewer distinct from the proposer")
        if self.is_primary and (not self.confirmed or self.relation_type != "intellectual_predecessor"):
            raise ValueError("A primary relation must be a confirmed intellectual_predecessor")
        if self.relation_type == "intellectual_predecessor" and not self.evidence.reference_id:
            raise ValueError("An intellectual predecessor requires a resolved source citation")
        return self


def _corpus(conn, corpus_id=None, *, lock=False):
    suffix = " FOR UPDATE" if lock else ""
    if corpus_id is None:
        rows = conn.execute("SELECT * FROM corpus ORDER BY id" + suffix).fetchall()
        if len(rows) != 1:
            raise GraphError("Specify corpus_id when there is not exactly one corpus")
        return rows[0]
    row = conn.execute("SELECT * FROM corpus WHERE id=%s" + suffix, (corpus_id,)).fetchone()
    if not row:
        raise GraphError("Unknown corpus")
    return row


def _lock_inputs(conn, corpus_id):
    # The corpus row prevents new papers entering through its FK while these
    # locks protect versions, accepted reviews, ideas, and their embeddings.
    _corpus(conn, corpus_id, lock=True)
    conn.execute("SELECT id FROM papers WHERE corpus_id=%s ORDER BY id FOR UPDATE", (corpus_id,)).fetchall()
    conn.execute("""SELECT v.id FROM paper_versions v JOIN papers p ON p.id=v.paper_id
                    WHERE p.corpus_id=%s ORDER BY v.id FOR UPDATE OF v""", (corpus_id,)).fetchall()
    conn.execute("""SELECT i.id FROM ideas i JOIN paper_versions v ON v.id=i.version_id
                    JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s
                    ORDER BY i.id FOR UPDATE OF i""", (corpus_id,)).fetchall()


def barrier(conn, corpus_id=None):
    corpus = _corpus(conn, corpus_id)
    reasons = []
    if not corpus["expansion_closed"]:
        reasons.append("One-hop expansion is not closed")
    unresolved = conn.execute("""SELECT count(*) AS n FROM predecessor_nominations n
        JOIN ideas i ON i.id=n.idea_id JOIN paper_versions v ON v.id=i.version_id
        JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s AND i.active
        AND n.state IN ('pending','approved')""", (corpus["id"],)).fetchone()["n"]
    if unresolved:
        reasons.append("Unresolved predecessor nominations remain")
    versions = conn.execute("""SELECT v.*, p.title FROM papers p
        LEFT JOIN paper_versions v ON v.paper_id=p.id WHERE p.corpus_id=%s
        ORDER BY p.id,v.id""", (corpus["id"],)).fetchall()
    if not versions:
        reasons.append("Corpus contains no admitted papers")
    cfg = settings()
    idea_count = 0
    for version in versions:
        label = version["title"]
        if version["id"] is None:
            reasons.append(f"{label}: admitted paper has no source version")
            continue
        vid, rev = version["id"], version["extraction_revision"]
        if version["state"] != "complete":
            reasons.append(f"{label}: state is {version['state']}, not complete")
        if not rev or version["reviewed_revision"] != rev:
            reasons.append(f"{label}: current extraction has not been reviewed")
        if not version["reviewer_id"] or version["reviewer_id"] == version["extractor_id"]:
            reasons.append(f"{label}: independent paper review is missing")
        if version["relevance_score"] is None:
            reasons.append(f"{label}: paper relevance is missing")
        accepted = conn.execute("""SELECT 1 FROM reviews WHERE version_id=%s
            AND revision=%s AND accepted AND reviewer_id=%s LIMIT 1""",
            (vid, rev, version["reviewer_id"])).fetchone()
        if not accepted:
            reasons.append(f"{label}: accepted current review record is missing")
        pending = conn.execute("""SELECT count(*) AS n FROM jobs WHERE version_id=%s
            AND ((revision=%s AND state<>'done') OR state IN ('queued','leased'))""", (vid, rev)).fetchone()["n"]
        if pending:
            reasons.append(f"{label}: unfinished jobs remain")
        coverage = conn.execute("""SELECT count(*) AS total,
            count(*) FILTER (WHERE sc.status='covered') AS covered FROM sections s
            LEFT JOIN section_coverage sc ON sc.version_id=s.version_id AND sc.section_id=s.id
            AND sc.revision=%s WHERE s.version_id=%s""", (rev, vid)).fetchone()
        if not coverage["total"] or coverage["total"] != coverage["covered"]:
            reasons.append(f"{label}: current section coverage is incomplete or unreadable")
        ideas = conn.execute("""SELECT i.id,i.revision,i.relevance_score,i.embedding_text,
            e.revision AS embedding_revision,e.model,e.model_revision,e.text_sha256,
            e.embedding IS NOT NULL AS has_embedding
            FROM ideas i LEFT JOIN embeddings e ON e.idea_id=i.id
            WHERE i.version_id=%s AND i.active""", (vid,)).fetchall()
        idea_count += len(ideas)
        if not ideas:
            reasons.append(f"{label}: no active extracted ideas")
        for idea in ideas:
            if idea["revision"] != rev or idea["relevance_score"] is None:
                reasons.append(f"{label}: idea {idea['id']} is stale or unranked")
            digest = hashlib.sha256(idea["embedding_text"].encode()).hexdigest()
            if (not idea["has_embedding"] or idea["embedding_revision"] != rev
                    or idea["model"] != cfg["embedding_model"]
                    or idea["model_revision"] != cfg["embedding_revision"]
                    or idea["text_sha256"] != digest):
                reasons.append(f"{label}: idea {idea['id']} has no current matching embedding")
    return {"ready": not reasons, "corpus_id": corpus["id"], "reasons": reasons,
            "version_count": sum(v["id"] is not None for v in versions), "idea_count": idea_count}


def _json_default(value):
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _snapshot(conn, corpus_id):
    digest = hashlib.sha256()
    queries = {
        "versions": """SELECT v.id,v.paper_id,v.sha256,v.version_label,v.available_at,v.date_verified,
            v.extraction_revision,v.reviewed_revision,v.reviewer_id,v.extractor_id,
            v.relevance_score,v.relevance_reason,p.arxiv_id,p.doi,p.source_depth,p.withdrawal_note
            FROM paper_versions v JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s ORDER BY v.id""",
        "ideas": """SELECT i.* FROM ideas i JOIN paper_versions v ON v.id=i.version_id
            JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s AND i.active ORDER BY i.id""",
        "evidence": """SELECT e.* FROM idea_evidence e JOIN ideas i ON i.id=e.idea_id
            JOIN paper_versions v ON v.id=i.version_id JOIN papers p ON p.id=v.paper_id
            WHERE p.corpus_id=%s AND i.active ORDER BY e.id""",
        "pages": """SELECT t.version_id,t.page_number,t.text_sha256,t.flags FROM pages t
            JOIN paper_versions v ON v.id=t.version_id JOIN papers p ON p.id=v.paper_id
            WHERE p.corpus_id=%s ORDER BY t.version_id,t.page_number""",
        "coverage": """SELECT sc.* FROM section_coverage sc JOIN paper_versions v ON v.id=sc.version_id
            JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s AND sc.revision=v.extraction_revision
            ORDER BY sc.version_id,sc.section_id""",
        "reviews": """SELECT r.* FROM reviews r JOIN paper_versions v ON v.id=r.version_id
            JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s AND r.revision=v.extraction_revision
            ORDER BY r.id""",
        "references": """SELECT r.* FROM paper_references r JOIN paper_versions v ON v.id=r.version_id
            JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s ORDER BY r.id""",
        "nominations": """SELECT n.* FROM predecessor_nominations n JOIN ideas i ON i.id=n.idea_id
            JOIN paper_versions v ON v.id=i.version_id JOIN papers p ON p.id=v.paper_id
            WHERE p.corpus_id=%s AND i.active ORDER BY n.id""",
        "embeddings": """SELECT e.idea_id,e.revision,e.model,e.model_revision,e.text_sha256
            FROM embeddings e JOIN ideas i ON i.id=e.idea_id JOIN paper_versions v ON v.id=i.version_id
            JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s AND i.active ORDER BY e.idea_id""",
    }
    counts, versions, embedding_models = {}, [], []
    for name, query in queries.items():
        digest.update(name.encode())
        rows = conn.execute(query, (corpus_id,)).fetchall()
        counts[name] = len(rows)
        if name == 'embeddings':
            embedding_models = [{'model':model,'revision':revision} for model,revision in
                                sorted({(r['model'],r['model_revision']) for r in rows})]
        if name == "versions":
            # Old graph runs must identify which immutable source and accepted
            # extraction revision they used, even after current ideas change.
            versions = [json.loads(json.dumps(row, default=_json_default)) for row in rows]
        for row in rows:
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default).encode())
            digest.update(b"\n")
    return {"schema_version": 1, "corpus_id": str(corpus_id), "digest": digest.hexdigest(),
            "counts": counts, "versions": versions, "embedding_models": embedding_models}


def begin_build(conn, corpus_id=None):
    with conn.transaction():
        corpus = _corpus(conn, corpus_id, lock=True)
        _lock_inputs(conn, corpus["id"])
        readiness = barrier(conn, corpus["id"])
        if not readiness["ready"]:
            raise GraphError("Graph barrier is not ready: " + "; ".join(readiness["reasons"]))
        snapshot = _snapshot(conn, corpus["id"])
        conn.execute("UPDATE corpus SET state='frozen' WHERE id=%s", (corpus["id"],))
        existing = conn.execute("""SELECT * FROM graph_builds WHERE corpus_id=%s
            AND state='building' AND snapshot=%s ORDER BY created_at LIMIT 1""",
            (corpus["id"], Jsonb(snapshot))).fetchone()
        if existing:
            return existing
        return conn.execute("""INSERT INTO graph_builds(id,corpus_id,state,snapshot)
            VALUES(%s,%s,'building',%s) RETURNING *""", (uuid4(), corpus["id"], Jsonb(snapshot))).fetchone()


def _idea(conn, idea_id):
    row = conn.execute("""SELECT i.*,v.paper_id,v.version_label,v.available_at,v.date_verified,v.extractor_id,
        v.extraction_revision,v.reviewed_revision,p.corpus_id,p.arxiv_id,p.doi,p.title AS paper_title
        FROM ideas i JOIN paper_versions v ON v.id=i.version_id JOIN papers p ON p.id=v.paper_id
        WHERE i.id=%s""", (idea_id,)).fetchone()
    if not row:
        raise GraphError("Unknown idea")
    return row


def _require_current(idea):
    if (not idea["active"] or idea["revision"] != idea["extraction_revision"]
            or idea["reviewed_revision"] != idea["extraction_revision"]):
        raise GraphError("Relations require active ideas from current reviewed extractions")


def _source_quote(conn, idea, evidence):
    page = conn.execute("SELECT text FROM pages WHERE version_id=%s AND page_number=%s",
                        (idea["version_id"], evidence.page_number)).fetchone()
    if not page or normalized_text(evidence.quote) not in normalized_text(page["text"]):
        raise GraphError("Relation quotation does not match the cited source page")


def _arxiv_base(value):
    return normalize_arxiv(value)[0]


def _doi(value):
    return normalize_doi(value)


def _validate_relation(conn, build, payload):
    child, parent = _idea(conn, payload.child_id), _idea(conn, payload.parent_id)
    for idea in (child, parent):
        _require_current(idea)
        if idea["corpus_id"] != build["corpus_id"]:
            raise GraphError("Both ideas must belong to this graph's corpus")
    if (not child["date_verified"] or not parent["date_verified"]
            or not child["available_at"] or not parent["available_at"]
            or parent["available_at"] >= child["available_at"]):
        raise GraphError("Parent requires a verified strictly earlier source-version date")
    if payload.confirmed and payload.reviewer_id == child["extractor_id"]:
        raise GraphError("Relation reviewer must differ from the child's extractor")
    _source_quote(conn, child, payload.evidence.child)
    _source_quote(conn, parent, payload.evidence.parent)
    if payload.evidence.reference_id:
        reference = conn.execute("SELECT * FROM paper_references WHERE id=%s AND version_id=%s",
            (payload.evidence.reference_id, child["version_id"])).fetchone()
        if not reference:
            raise GraphError("Relation citation must belong to the child's source version")
        resolved = conn.execute("""SELECT 1 FROM predecessor_nominations WHERE idea_id=%s
            AND reference_id=%s AND target_paper_id=%s AND state IN ('existing','admitted') LIMIT 1""",
            (child["id"], reference["id"], parent["paper_id"])).fetchone()
        field_arxiv, field_version = normalize_arxiv(reference["arxiv_id"])
        url_arxiv, url_version = normalize_arxiv(reference["url"])
        field_doi, url_doi = _doi(reference["doi"]), _doi(reference["url"])
        if ((field_arxiv and url_arxiv and field_arxiv != url_arxiv)
                or (field_version and url_version and field_version != url_version)
                or (field_doi and url_doi and field_doi != url_doi)):
            raise GraphError("Source citation contains conflicting identifiers between its fields and URL")
        ref_arxiv, parent_arxiv = field_arxiv or url_arxiv, _arxiv_base(parent["arxiv_id"])
        ref_doi, parent_doi = field_doi or url_doi, _doi(parent["doi"])
        if ((ref_arxiv and parent_arxiv and ref_arxiv != parent_arxiv)
                or (ref_doi and parent_doi and ref_doi != parent_doi)):
            raise GraphError("Source citation contains conflicting parent-paper identifiers")
        identifier_match = ((ref_arxiv and ref_arxiv == parent_arxiv) or (ref_doi and ref_doi == parent_doi))
        if not identifier_match and not resolved:
            raise GraphError("Source citation has not been resolved to the proposed parent paper")
        cited_version = field_version or url_version
        if cited_version and cited_version != parent["version_label"]:
            raise GraphError("Explicitly cited arXiv version does not match the parent source version")
    return child, parent


def add_relation(conn, graph_id, payload):
    payload = RelationPayload.model_validate(payload)
    with conn.transaction():
        build = conn.execute("SELECT * FROM graph_builds WHERE id=%s FOR UPDATE", (graph_id,)).fetchone()
        if not build or build["state"] != "building":
            raise GraphError("Relations can only be changed on a building graph")
        _validate_relation(conn, build, payload)
        evidence = payload.evidence.model_dump(mode="json")
        evidence["proposer_id"] = payload.proposer_id
        evidence["primary_requested"] = payload.is_primary
        return conn.execute("""INSERT INTO idea_relations
            (id,graph_id,child_id,parent_id,relation_type,evidence,confidence,reviewer_id,confirmed,is_primary,rationale)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(graph_id,child_id,parent_id,relation_type) DO UPDATE SET
            evidence=EXCLUDED.evidence,confidence=EXCLUDED.confidence,reviewer_id=EXCLUDED.reviewer_id,
            confirmed=EXCLUDED.confirmed,is_primary=EXCLUDED.is_primary,rationale=EXCLUDED.rationale
            RETURNING *""", (uuid4(), graph_id, payload.child_id, payload.parent_id, payload.relation_type,
                Jsonb(evidence), payload.confidence, payload.reviewer_id, payload.confirmed,
                payload.is_primary, payload.rationale)).fetchone()


def _assert_acyclic(parents):
    finished = set()
    for start in parents:
        path, current = set(), start
        while current is not None and current not in finished:
            if current in path:
                raise GraphError("Primary parent relations contain a cycle")
            path.add(current)
            current = parents.get(current)
        finished.update(path)


def publish(conn, graph_id):
    with conn.transaction():
        # Lock corpus first: input writers and other publications share this order.
        initial = conn.execute("SELECT * FROM graph_builds WHERE id=%s", (graph_id,)).fetchone()
        if not initial:
            raise GraphError("Unknown graph build")
        corpus = _corpus(conn, initial["corpus_id"], lock=True)
        build = conn.execute("SELECT * FROM graph_builds WHERE id=%s FOR UPDATE", (graph_id,)).fetchone()
        if build["state"] == "published" and corpus["current_graph_id"] == build["id"]:
            return {"graph_id": build["id"], "state": "published", "already_published": True}
        if build["state"] != "building":
            raise GraphError("Only a building graph can be published")
        _lock_inputs(conn, corpus["id"])
        readiness = barrier(conn, corpus["id"])
        if not readiness["ready"]:
            raise GraphError("Graph barrier is not ready: " + "; ".join(readiness["reasons"]))
        if _snapshot(conn, corpus["id"]) != build["snapshot"]:
            raise GraphError("Corpus inputs changed after this graph snapshot; begin a new build")
        ideas = conn.execute("""SELECT i.id FROM ideas i JOIN paper_versions v ON v.id=i.version_id
            JOIN papers p ON p.id=v.paper_id WHERE p.corpus_id=%s AND i.active
            AND i.revision=v.extraction_revision ORDER BY i.id""", (corpus["id"],)).fetchall()
        proposed = defaultdict(list)
        relations = conn.execute("SELECT * FROM idea_relations WHERE graph_id=%s ORDER BY id", (graph_id,)).fetchall()
        for relation in relations:
            evidence = dict(relation["evidence"])
            proposer = evidence.pop("proposer_id", None)
            evidence.pop("primary_requested", None)
            payload = RelationPayload.model_validate({
                key: relation[key] for key in ("child_id", "parent_id", "relation_type", "confidence",
                                               "confirmed", "is_primary", "rationale", "reviewer_id")
            } | {"evidence": evidence, "proposer_id": proposer})
            _validate_relation(conn, build, payload)
            if payload.is_primary:
                proposed[payload.child_id].append(payload)
        parents, rationales, ambiguous = {}, {}, []
        for idea in ideas:
            choices = proposed[idea["id"]]
            if len(choices) == 1:
                parents[idea["id"]] = choices[0].parent_id
                rationales[idea["id"]] = choices[0].rationale
            else:
                parents[idea["id"]] = None
                rationales[idea["id"]] = ("Ambiguous: multiple independently confirmed primary predecessors"
                    if choices else "No independently confirmed, chronologically supported primary predecessor")
                if choices:
                    ambiguous.append(idea["id"])
        _assert_acyclic(parents)
        conn.execute("DELETE FROM graph_parents WHERE graph_id=%s", (graph_id,))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO graph_parents(graph_id,idea_id,parent_id,rationale) VALUES(%s,%s,%s,%s)",
                            [(graph_id, idea, parent, rationales[idea]) for idea, parent in parents.items()])
        if ambiguous:
            conn.execute("UPDATE idea_relations SET is_primary=false WHERE graph_id=%s AND child_id=ANY(%s)",
                         (graph_id, ambiguous))
        conn.execute("UPDATE graph_builds SET state='superseded' WHERE corpus_id=%s AND state='published'", (corpus["id"],))
        conn.execute("UPDATE graph_builds SET state='published',published_at=now() WHERE id=%s", (graph_id,))
        conn.execute("UPDATE corpus SET current_graph_id=%s,state='published' WHERE id=%s", (graph_id, corpus["id"]))
        return {"graph_id": graph_id, "state": "published", "idea_count": len(parents),
                "parent_count": sum(p is not None for p in parents.values()), "ambiguous_count": len(ambiguous)}


def candidates(conn, idea_id, limit=25):
    from .query import _merge_rankings, _idea_select
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise GraphError("limit must be an integer from 1 to 100")
    child = _idea(conn, idea_id)
    _require_current(child)
    pool = max(50, limit * 4)
    common = """ i.id<>%(child)s AND i.active AND i.revision=v.extraction_revision
        AND v.reviewed_revision=v.extraction_revision AND p.corpus_id=%(corpus)s """
    params = {"child": child["id"], "corpus": child["corpus_id"], "version": child["version_id"],
              "query": child["title"] + " " + child["statement"][:1000], "pool": pool}
    citation = conn.execute(_idea_select() + " WHERE " + common + """ AND EXISTS (
        SELECT 1 FROM paper_references r LEFT JOIN predecessor_nominations n
        ON n.reference_id=r.id AND n.idea_id=%(child)s WHERE r.version_id=%(version)s
        AND ((r.arxiv_id IS NOT NULL AND regexp_replace(r.arxiv_id,'v[0-9]+$','')=p.arxiv_id)
          OR (r.doi IS NOT NULL AND lower(r.doi)=lower(p.doi)) OR n.target_paper_id=p.id))
        ORDER BY i.relevance_score DESC NULLS LAST,i.id LIMIT %(pool)s""", params).fetchall()
    # OR tokens preserve recall for long statements; plainto_tsquery would require
    # every word in the source idea to be present in the candidate.
    words = list(dict.fromkeys(re.findall(r"[\w-]+", params["query"], flags=re.UNICODE)))[:40]
    params["terms"] = " OR ".join(words)
    lexical = conn.execute(_idea_select() + " WHERE " + common + """ AND
        to_tsvector('english',i.title||' '||i.statement||' '||i.conditions) @@ websearch_to_tsquery('english',%(terms)s)
        ORDER BY ts_rank_cd(to_tsvector('english',i.title||' '||i.statement||' '||i.conditions),
        websearch_to_tsquery('english',%(terms)s)) DESC,i.id LIMIT %(pool)s""", params).fetchall()
    cfg = settings()
    params.update(model=cfg["embedding_model"], model_revision=cfg["embedding_revision"])
    vector = conn.execute(_idea_select(extra="e.embedding <=> source.embedding AS distance") + """
        JOIN embeddings e ON e.idea_id=i.id JOIN embeddings source ON source.idea_id=%(child)s
        WHERE """ + common + """ AND e.revision=i.revision AND source.revision=%(revision)s
        AND e.model=%(model)s AND e.model_revision=%(model_revision)s
        AND source.model=e.model AND source.model_revision=e.model_revision
        ORDER BY e.embedding <=> source.embedding,i.id LIMIT %(pool)s""",
        params | {"revision": child["revision"]}).fetchall()
    results = _merge_rankings({"citation": citation, "lexical": lexical, "vector": vector}, limit)
    for result in results:
        result["confirmed"] = False
        result["chronologically_eligible"] = bool(child["date_verified"] and child["available_at"]
            and result["date_verified"] and result["available_at"] and result["available_at"] < child["available_at"])
    return {"idea_id": child["id"], "purpose": "retrieval_only", "results": results}


def canonical_group(conn, label, description, idea_ids, rationale):
    """Explicit equivalence grouping; never changes idea identity or ancestry."""
    if not all(isinstance(v, str) and v.strip() for v in (label, description, rationale)):
        raise GraphError("Canonical grouping requires a label, description, and rationale")
    ids = list(dict.fromkeys(UUID(str(value)) for value in idea_ids))
    if len(ids) < 2:
        raise GraphError("Canonical grouping requires at least two distinct source ideas")
    with conn.transaction():
        ideas = [_idea(conn, value) for value in ids]
        for idea in ideas:
            _require_current(idea)
        if len({idea["corpus_id"] for idea in ideas}) != 1:
            raise GraphError("A canonical group cannot span corpora")
        # Repeating the same explicit grouping reuses its identity.
        existing = conn.execute("""SELECT c.id FROM concepts c JOIN concept_memberships m ON m.concept_id=c.id
            WHERE c.label=%s AND c.description=%s GROUP BY c.id
            HAVING array_agg(m.idea_id ORDER BY m.idea_id)=%s::uuid[] LIMIT 1""",
            (label, description, sorted(ids))).fetchone()
        cid = existing["id"] if existing else uuid4()
        if not existing:
            conn.execute("INSERT INTO concepts(id,label,description) VALUES(%s,%s,%s)", (cid,label,description))
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO concept_memberships(concept_id,idea_id,rationale) VALUES(%s,%s,%s)",
                                [(cid, iid, rationale) for iid in ids])
        return {"concept_id": cid, "label": label, "idea_ids": ids}
