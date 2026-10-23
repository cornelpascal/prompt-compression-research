"""Auditable keyword/hybrid retrieval and source-preserving graph inspection."""
from __future__ import annotations

import math
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from psycopg.pq import TransactionStatus

from .config import settings


def _idea_select(extra=""):
    extra_sql = "," + extra if extra else ""
    return """SELECT i.id,i.version_id,i.local_id,i.kind,i.title,i.statement,i.conditions,
        i.attribution,i.confidence,i.relevance_score,i.relevance_reason,i.validity_flags,
        v.paper_id,v.sha256,v.version_label,v.available_at,v.date_verified,
        p.title AS paper_title,p.arxiv_id,p.doi,p.withdrawal_note,p.corpus_id""" + extra_sql + """
        FROM ideas i JOIN paper_versions v ON v.id=i.version_id JOIN papers p ON p.id=v.paper_id """


def _merge_rankings(rankings, limit):
    merged, scores = {}, defaultdict(float)
    for source, rows in rankings.items():
        seen = set()
        for rank, row in enumerate(rows, 1):
            key = row["id"]
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(key, dict(row) | {"retrieval_sources": [], "retrieval_ranks": {}})
            item["retrieval_sources"].append(source)
            item["retrieval_ranks"][source] = rank
            if "distance" in row:
                item["distance"] = row["distance"]
            scores[key] += 1.0 / (60 + rank)
    result = sorted(merged.values(), key=lambda row: (-scores[row["id"]], str(row["id"])))[:limit]
    for row in result:
        row["retrieval_score"] = scores[row["id"]]
    return result


def _vector_literal(embedding):
    if isinstance(embedding, (str, bytes)):
        raise ValueError("embedding must contain 1024 finite numbers")
    try:
        values = list(embedding)
        if any(isinstance(value, bool) for value in values):
            raise ValueError("Boolean values are not valid vector components")
        values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding must contain 1024 finite numbers") from exc
    if len(values) != 1024 or not all(math.isfinite(value) for value in values) or not any(values):
        raise ValueError("embedding must contain 1024 finite numbers and have nonzero norm")
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


def search(conn, query, embedding=None, limit=20, min_relevance=0):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty text")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if not isinstance(min_relevance, int) or isinstance(min_relevance, bool) or not 0 <= min_relevance <= 100:
        raise ValueError("min_relevance must be an integer from 0 to 100")
    common = """i.active AND i.revision=v.extraction_revision
        AND v.reviewed_revision=v.extraction_revision AND i.relevance_score>=%(relevance)s"""
    params = {"query": query, "relevance": min_relevance, "pool": max(100, limit * 5)}
    lexical = conn.execute(_idea_select(extra="""ts_rank_cd(to_tsvector('english',i.title||' '||i.statement||' '||i.conditions),
        websearch_to_tsquery('english',%(query)s)) AS lexical_score""") + " WHERE " + common + """
        AND to_tsvector('english',i.title||' '||i.statement||' '||i.conditions) @@ websearch_to_tsquery('english',%(query)s)
        ORDER BY lexical_score DESC,i.relevance_score DESC,i.id LIMIT %(pool)s""", params).fetchall()
    rankings = {"keyword": lexical}
    if embedding is not None:
        cfg = settings()
        params.update(vector=_vector_literal(embedding), model=cfg["embedding_model"], model_revision=cfg["embedding_revision"])
        rankings["vector"] = conn.execute(_idea_select(extra="e.embedding <=> %(vector)s::vector AS distance") + """
            JOIN embeddings e ON e.idea_id=i.id WHERE """ + common + """
            AND e.revision=i.revision AND e.model=%(model)s AND e.model_revision=%(model_revision)s
            ORDER BY e.embedding <=> %(vector)s::vector,i.id LIMIT %(pool)s""", params).fetchall()
    return {"query": query, "mode": "hybrid" if embedding is not None else "keyword",
            "ranking": "reciprocal_rank_fusion" if embedding is not None else "keyword_rank",
            "results": _merge_rankings(rankings, limit)}


def inspect_idea(conn, idea_id):
    idea = conn.execute(_idea_select(extra="i.active,i.revision,i.embedding_text") + " WHERE i.id=%s", (idea_id,)).fetchone()
    if not idea:
        raise ValueError("Unknown idea")
    idea["evidence"] = conn.execute("SELECT * FROM idea_evidence WHERE idea_id=%s ORDER BY page_number,id", (idea_id,)).fetchall()
    idea["concepts"] = conn.execute("""SELECT c.*,m.rationale FROM concepts c
        JOIN concept_memberships m ON m.concept_id=c.id WHERE m.idea_id=%s ORDER BY c.label,c.id""", (idea_id,)).fetchall()
    current = conn.execute("""SELECT gp.*,g.published_at FROM corpus c JOIN graph_builds g ON g.id=c.current_graph_id
        JOIN graph_parents gp ON gp.graph_id=g.id WHERE c.id=%s AND gp.idea_id=%s AND g.state='published'""",
        (idea["corpus_id"], idea_id)).fetchone()
    idea["parent_id"] = current["parent_id"] if current else None
    idea["graph"] = current
    idea["relations"] = conn.execute("""SELECT r.* FROM idea_relations r JOIN corpus c ON c.current_graph_id=r.graph_id
        WHERE r.child_id=%s OR r.parent_id=%s ORDER BY r.child_id,r.parent_id,r.relation_type""", (idea_id, idea_id)).fetchall()
    return idea


def lineage(conn, idea_id):
    first = conn.execute(_idea_select() + " WHERE i.id=%s", (idea_id,)).fetchone()
    if not first:
        raise ValueError("Unknown idea")
    graph = conn.execute("""SELECT g.id FROM corpus c JOIN graph_builds g ON g.id=c.current_graph_id
        WHERE c.id=%s AND g.state='published'""", (first["corpus_id"],)).fetchone()
    if not graph:
        return {"idea_id": first["id"], "graph_id": None, "state": "unpublished", "chain": [], "complete": False}
    chain, seen, current = [], set(), first["id"]
    while current is not None:
        if current in seen:
            raise ValueError("Published graph contains a cycle")
        seen.add(current)
        row = conn.execute(_idea_select(extra="gp.parent_id,gp.rationale AS parent_rationale") + """
            JOIN graph_parents gp ON gp.idea_id=i.id WHERE gp.graph_id=%s AND i.id=%s""", (graph["id"], current)).fetchone()
        if not row:
            if not chain:
                return {"idea_id": first["id"], "graph_id": graph["id"], "state": "not_in_current_graph", "chain": [], "complete": False}
            raise ValueError("Published graph has a missing parent node")
        chain.append(row)
        current = row["parent_id"]
    return {"idea_id": first["id"], "graph_id": graph["id"], "state": "published", "chain": chain, "complete": True}


EXPORT_TABLES = (
    "corpus", "papers", "paper_versions", "pages", "sections", "ideas", "idea_evidence",
    "section_coverage", "reviews", "paper_references", "predecessor_nominations",
    "concepts", "concept_memberships", "graph_builds", "idea_relations", "graph_parents",
    "jobs", "job_attempts", "checkpoints", "submissions", "corpus_events",
)
EMBEDDING_EXPORT_COLUMNS = "idea_id,revision,model,model_revision,text_sha256,created_at"


def export_data(conn):
    """Return a small corpus in memory; CLI exports use stream_export instead."""
    # A single SQL statement provides one MVCC snapshot even if processing is
    # active in another connection. JSON aggregation also normalizes UUID/dates.
    pieces = [f"'{table}',(SELECT COALESCE(jsonb_agg(to_jsonb(t)), '[]'::jsonb) FROM {table} t)" for table in EXPORT_TABLES]
    pieces.append(f"""'embeddings',(SELECT COALESCE(jsonb_agg(to_jsonb(t)), '[]'::jsonb) FROM
        (SELECT {EMBEDDING_EXPORT_COLUMNS} FROM embeddings) t)""")
    row = conn.execute("SELECT jsonb_build_object(" + ",".join(pieces) + ") AS data").fetchone()
    return {"schema_version": 1, "format": "paperkb-evidence-and-graph", "vector_values_included": False, "data": row["data"]}


def stream_export(conn, output, batch_size=128):
    """Write one JSON document using bounded cursors and one read-only snapshot.

    The connection must be idle: this function never commits or rolls back a
    caller's existing work to acquire its own repeatable-read transaction.
    Memory is bounded by a fetch batch, including its largest individual row.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 10000:
        raise ValueError("Export batch_size must be an integer from 1 to 10000")
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError("Streaming export requires an idle connection; finish the existing transaction first")
    counts = {}
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        # Establish the snapshot before the first byte is written. Later named
        # cursors see this same state while workers continue committing changes.
        conn.execute("SELECT pg_current_snapshot()").fetchone()
        output.write('{"schema_version":1,"format":"paperkb-evidence-and-graph",'
                     '"vector_values_included":false,"data":{')
        for table_number, table in enumerate((*EXPORT_TABLES, "embeddings")):
            if table_number:
                output.write(",")
            output.write(json.dumps(table) + ":[")
            source = (f"(SELECT {EMBEDDING_EXPORT_COLUMNS} FROM embeddings)" if table == "embeddings" else table)
            # Only constant schema identifiers enter this query. JSON conversion
            # occurs one row at a time, never as a corpus-sized JSONB aggregate.
            sql = f"SELECT to_jsonb(t)::text AS row_json FROM {source} t"
            count = 0
            with conn.cursor(name="paperkb_export_" + uuid4().hex) as cursor:
                cursor.execute(sql)
                while batch := cursor.fetchmany(batch_size):
                    for row in batch:
                        if count:
                            output.write(",")
                        output.write(row["row_json"])
                        count += 1
            counts[table] = count
            output.write("]")
        output.write("}}\n")
    return {"row_counts": counts, "rows": sum(counts.values())}


def export_to_file(conn, path, batch_size=128):
    """Publish a complete export atomically, without overwriting existing files."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Export destination already exists: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            stream_export(conn, output, batch_size=batch_size)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path)}
