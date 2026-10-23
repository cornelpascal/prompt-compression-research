"""Durable, fenced one-paper jobs and evidence-checked agent ingestion.

Identity is a trusted orchestrator parameter. Worker output is always data; no
output field is ever interpreted as SQL, a tool instruction, or executable code.
"""
from __future__ import annotations

import hashlib
import json
from uuid import UUID, uuid4, uuid5

from psycopg.types.json import Jsonb

from .models import ExtractionPayload, ReviewPayload, normalized_text


class WorkflowError(ValueError):
    pass


def _payload(model, payload):
    parsed = model.model_validate(payload)
    value = parsed.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return parsed, value, digest


def _identity(agent_id, lease_seconds=None):
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise WorkflowError("A nonempty trusted orchestrator agent identity is required")
    if lease_seconds is not None and (isinstance(lease_seconds,bool) or not isinstance(lease_seconds,int) or not 1 <= lease_seconds <= 86400):
        raise WorkflowError("lease_seconds must be an integer from 1 to 86400")


def _lock_corpus(conn, version_id, require_open=True):
    row = conn.execute("""SELECT c.* FROM corpus c JOIN papers p ON p.corpus_id=c.id
        JOIN paper_versions v ON v.paper_id=p.id WHERE v.id=%s FOR UPDATE OF c""", (version_id,)).fetchone()
    if row is None:
        raise WorkflowError("Unknown paper version")
    if require_open and row["state"] != "open":
        raise WorkflowError("Corpus is frozen; explicitly reopen it before changing scientific inputs")
    return row


def _invalidate_graph(conn, corpus_id):
    conn.execute("UPDATE graph_builds SET state='superseded' WHERE corpus_id=%s AND state IN ('building','published')", (corpus_id,))
    conn.execute("UPDATE corpus SET current_graph_id=NULL WHERE id=%s", (corpus_id,))


def _job(conn, job_id, agent_id, token, kind=None, allow_done=False):
    _identity(agent_id)
    initial = conn.execute("SELECT version_id FROM jobs WHERE id=%s", (job_id,)).fetchone()
    if initial is None:
        raise WorkflowError("Unknown job")
    corpus = _lock_corpus(conn, initial["version_id"], require_open=False)
    row = conn.execute("SELECT *,lease_until > clock_timestamp() AS lease_valid FROM jobs WHERE id=%s FOR UPDATE", (job_id,)).fetchone()
    if row["agent_id"] != agent_id or str(row["token"]) != str(token):
        raise WorkflowError("Stale or foreign job fencing token")
    if kind and row["kind"] != kind:
        raise WorkflowError(f"Expected {kind} job")
    if allow_done and row["state"] == "done":
        return row, corpus
    if corpus["state"] != "open":
        raise WorkflowError("Corpus is frozen; explicitly reopen before mutation")
    if row["state"] != "leased" or not row["lease_valid"]:
        raise WorkflowError("Job is not currently leased by this worker, or its lease expired")
    return row, corpus


def _replay(conn, job, digest):
    if job["state"] != "done":
        return None
    previous = conn.execute("SELECT id FROM submissions WHERE job_id=%s AND payload_hash=%s", (job["id"],digest)).fetchone()
    if previous is None:
        raise WorkflowError("Completed job cannot accept a different submission")
    return {"job_id": job["id"], "version_id": job["version_id"], "revision": job["revision"], "submission_id": previous["id"], "idempotent": True}


def _record_submission(conn, job, digest, payload):
    record = conn.execute("INSERT INTO submissions(job_id,payload_hash,payload) VALUES(%s,%s,%s) RETURNING id", (job["id"],digest,Jsonb(payload))).fetchone()
    conn.execute("UPDATE jobs SET state='done',lease_until=NULL,error=NULL,updated_at=clock_timestamp() WHERE id=%s", (job["id"],))
    conn.execute("UPDATE job_attempts SET state='done',finished_at=clock_timestamp() WHERE token=%s", (job["token"],))
    return {"job_id": job["id"], "version_id": job["version_id"], "revision": job["revision"], "submission_id": record["id"], "idempotent": False}


def _extraction_participant(conn, version_id, agent_id):
    """Independence applies to the paper's whole extraction history.

    An expired or replaced extractor may have contributed the resumed draft.
    Conservatively exclude every extraction lease holder across PDF versions,
    rather than relying only on the most recent submission's extractor ID.
    """
    return bool(conn.execute("""SELECT 1 FROM paper_versions target
        JOIN paper_versions source ON source.paper_id=target.paper_id
        WHERE target.id=%s AND (source.extractor_id=%s OR EXISTS (
            SELECT 1 FROM jobs j JOIN job_attempts a ON a.job_id=j.id
            WHERE j.version_id=source.id AND j.kind='extract' AND a.agent_id=%s)) LIMIT 1""",
        (version_id,agent_id,agent_id)).fetchone())


def claim(conn, agent_id, kind=None, version_id=None, lease_seconds=1800):
    _identity(agent_id,lease_seconds)
    if kind not in (None,"extract","review"):
        raise WorkflowError("kind must be extract or review")
    with conn.transaction():
        # Serialize only the short scheduler transaction, including the global cap.
        conn.execute("SELECT pg_advisory_xact_lock(731991, 2)")
        active = conn.execute("SELECT count(*) AS n FROM jobs WHERE state='leased' AND lease_until>clock_timestamp()").fetchone()["n"]
        if active >= 3:
            return None
        if conn.execute("SELECT 1 FROM jobs WHERE agent_id=%s AND state='leased' AND lease_until>clock_timestamp()", (agent_id,)).fetchone():
            raise WorkflowError("A worker may hold only one paper job at a time")
        filters = ["c.state='open'", "(j.state='queued' OR (j.state='leased' AND j.lease_until<=clock_timestamp()))",
                   "(j.kind<>'review' OR v.extractor_id IS DISTINCT FROM %s)"]
        args = [agent_id]
        if kind:
            filters.append("j.kind=%s")
            args.append(kind)
        if version_id:
            filters.append("j.version_id=%s")
            args.append(version_id)
        sql = """SELECT j.id,j.version_id FROM jobs j JOIN paper_versions v ON v.id=j.version_id
                 JOIN papers p ON p.id=v.paper_id JOIN corpus c ON c.id=p.corpus_id WHERE """ + " AND ".join(filters)
        # Lock corpus before job, consistently with graph publication and ingestion.
        # Resume unfinished work before starting the prepared backlog. Reviews
        # still come first; corrections and expired leases must not wait behind
        # every newly prepared paper.
        candidates = conn.execute(sql + " ORDER BY (j.kind='review') DESC,"
                                  "(v.state='revision_required') DESC,(j.state='leased') DESC,j.created_at,j.id", args).fetchall()
        for candidate in candidates:
            corpus = _lock_corpus(conn,candidate["version_id"],require_open=False)
            if corpus["state"] != "open":
                continue
            current = conn.execute("SELECT * FROM jobs WHERE id=%s AND (state='queued' OR (state='leased' AND lease_until<=clock_timestamp())) FOR UPDATE SKIP LOCKED", (candidate["id"],)).fetchone()
            if not current:
                continue
            if current["kind"] == "review" and _extraction_participant(conn,current["version_id"],agent_id):
                continue
            conn.execute("UPDATE job_attempts SET state='expired',finished_at=clock_timestamp() WHERE job_id=%s AND state='leased'", (current["id"],))
            claimed = conn.execute("""UPDATE jobs SET state='leased',agent_id=%s,token=%s,
                lease_until=clock_timestamp()+make_interval(secs => %s),attempts=attempts+1,error=NULL,
                updated_at=clock_timestamp() WHERE id=%s RETURNING *""", (agent_id,uuid4(),lease_seconds,current["id"])).fetchone()
            conn.execute("""INSERT INTO job_attempts(token,job_id,attempt_number,agent_id,state,lease_until)
                VALUES(%s,%s,%s,%s,'leased',%s)""", (claimed["token"],claimed["id"],claimed["attempts"],agent_id,claimed["lease_until"]))
            return claimed
        return None


def heartbeat(conn, job_id, agent_id, token, lease_seconds=1800):
    _identity(agent_id,lease_seconds)
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(731991, 2)")
        _job(conn,job_id,agent_id,token)
        renewed = conn.execute("UPDATE jobs SET lease_until=clock_timestamp()+make_interval(secs => %s),updated_at=clock_timestamp() WHERE id=%s RETURNING id,lease_until", (lease_seconds,job_id)).fetchone()
        conn.execute("UPDATE job_attempts SET last_heartbeat_at=clock_timestamp(),lease_until=%s WHERE token=%s", (renewed["lease_until"],token))
        return renewed


def checkpoint(conn, job_id, agent_id, token, section_id, payload):
    if not isinstance(payload,dict):
        raise WorkflowError("Checkpoint must be a JSON object")
    json.dumps(payload, allow_nan=False)
    with conn.transaction():
        job,_ = _job(conn,job_id,agent_id,token)
        if not conn.execute("SELECT 1 FROM sections WHERE id=%s AND version_id=%s", (section_id,job["version_id"])).fetchone():
            raise WorkflowError("Checkpoint section does not belong to the job's paper")
        return conn.execute("""INSERT INTO checkpoints(job_id,section_id,payload) VALUES(%s,%s,%s)
            ON CONFLICT(job_id,section_id) DO UPDATE SET payload=EXCLUDED.payload,updated_at=clock_timestamp()
            RETURNING job_id,section_id,updated_at""", (job_id,section_id,Jsonb(payload))).fetchone()


def _quote(pages, page_number, quote, label="Evidence"):
    if page_number not in pages:
        raise WorkflowError(f"{label} references nonexistent page {page_number}")
    normalized = normalized_text(quote)
    if not normalized or normalized not in normalized_text(pages[page_number]["text"]):
        raise WorkflowError(f"{label} quote is not present on page {page_number}")


def submit(conn, job_id, agent_id, token, payload):
    parsed,value,digest = _payload(ExtractionPayload,payload)
    with conn.transaction():
        job,corpus = _job(conn,job_id,agent_id,token,kind="extract",allow_done=True)
        replay = _replay(conn,job,digest)
        if replay:
            return replay
        version_id = job["version_id"]
        version = conn.execute("SELECT * FROM paper_versions WHERE id=%s FOR UPDATE", (version_id,)).fetchone()
        if job["revision"] <= version["extraction_revision"]:
            raise WorkflowError("Extraction job revision is stale")
        pages = {row["page_number"]:row for row in conn.execute("SELECT * FROM pages WHERE version_id=%s", (version_id,))}
        sections = {row["id"]:row for row in conn.execute("SELECT * FROM sections WHERE version_id=%s", (version_id,))}
        if set(sections) != {item.section_id for item in parsed.coverage}:
            raise WorkflowError("Coverage must account for every prepared section exactly once")
        for item in parsed.ideas:
            for evidence in item.evidence:
                if evidence.evidence_kind == "text":
                    _quote(pages,evidence.page_number,evidence.quote)
                elif evidence.page_number not in pages:
                    raise WorkflowError("Visual evidence references nonexistent page")
        for nomination in parsed.predecessors:
            _quote(pages,nomination.evidence_page,nomination.evidence_quote,"Predecessor")
        _invalidate_graph(conn,corpus["id"])
        conn.execute("DELETE FROM embeddings WHERE idea_id IN (SELECT id FROM ideas WHERE version_id=%s)", (version_id,))
        conn.execute("DELETE FROM predecessor_nominations WHERE idea_id IN (SELECT id FROM ideas WHERE version_id=%s)", (version_id,))
        conn.execute("DELETE FROM idea_evidence WHERE idea_id IN (SELECT id FROM ideas WHERE version_id=%s)", (version_id,))
        conn.execute("UPDATE ideas SET active=false,relevance_score=NULL,relevance_reason=NULL WHERE version_id=%s", (version_id,))
        idea_ids = {}
        for item in parsed.ideas:
            idea_id = uuid5(UUID(str(version_id)),item.local_id)
            idea_ids[item.local_id] = idea_id
            conn.execute("""INSERT INTO ideas(id,version_id,local_id,revision,kind,title,statement,conditions,attribution,
                confidence,embedding_text,validity_flags) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(version_id,local_id) DO UPDATE SET revision=EXCLUDED.revision,kind=EXCLUDED.kind,
                title=EXCLUDED.title,statement=EXCLUDED.statement,conditions=EXCLUDED.conditions,
                attribution=EXCLUDED.attribution,confidence=EXCLUDED.confidence,embedding_text=EXCLUDED.embedding_text,
                validity_flags=EXCLUDED.validity_flags,active=true,relevance_score=NULL,relevance_reason=NULL""",
                (idea_id,version_id,item.local_id,job["revision"],item.kind,item.title,item.statement,item.conditions,
                 item.attribution,item.confidence,item.embedding_text,Jsonb(item.validity_flags)))
            for evidence in item.evidence:
                conn.execute("INSERT INTO idea_evidence(idea_id,page_number,quote,evidence_kind,locator) VALUES(%s,%s,%s,%s,%s)",
                    (idea_id,evidence.page_number,evidence.quote,evidence.evidence_kind,Jsonb(evidence.locator)))
        for item in parsed.coverage:
            conn.execute("INSERT INTO section_coverage(version_id,section_id,revision,status,notes) VALUES(%s,%s,%s,%s,%s)",
                (version_id,item.section_id,job["revision"],item.status,item.notes))
        conn.execute("DELETE FROM paper_references WHERE version_id=%s", (version_id,))
        refs = {}
        for item in parsed.references:
            ref_id = uuid5(UUID(str(version_id)),"reference:"+item.citation_key)
            refs[item.citation_key] = ref_id
            conn.execute("INSERT INTO paper_references(id,version_id,citation_key,title,arxiv_id,doi,url) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (ref_id,version_id,item.citation_key,item.title,item.arxiv_id,item.doi,item.url))
        for item in parsed.predecessors:
            conn.execute("""INSERT INTO predecessor_nominations(idea_id,reference_id,evidence_page,evidence_quote)
                VALUES(%s,%s,%s,%s)""", (idea_ids[item.idea_local_id],refs[item.citation_key],item.evidence_page,item.evidence_quote))
        conn.execute("""UPDATE paper_versions SET state='extracted',extraction_revision=%s,reviewed_revision=NULL,
            extractor_id=%s,reviewer_id=NULL,relevance_score=NULL,relevance_reason=NULL WHERE id=%s""", (job["revision"],agent_id,version_id))
        conn.execute("INSERT INTO jobs(version_id,kind,revision) VALUES(%s,'review',%s) ON CONFLICT DO NOTHING", (version_id,job["revision"]))
        result = _record_submission(conn,job,digest,value)
        result["ideas"] = len(parsed.ideas)
        return result


def _revision(conn, version, reason):
    version_id = version["id"]
    conn.execute("UPDATE paper_versions SET state='revision_required',reviewed_revision=NULL,reviewer_id=NULL,relevance_score=NULL,relevance_reason=NULL WHERE id=%s", (version_id,))
    conn.execute("UPDATE ideas SET relevance_score=NULL,relevance_reason=NULL WHERE version_id=%s", (version_id,))
    conn.execute("DELETE FROM embeddings WHERE idea_id IN (SELECT id FROM ideas WHERE version_id=%s)", (version_id,))
    conn.execute("""UPDATE predecessor_nominations SET state='pending',reviewer_id=NULL,reason=NULL,target_paper_id=NULL
        WHERE idea_id IN (SELECT id FROM ideas WHERE version_id=%s)""", (version_id,))
    return conn.execute("""INSERT INTO jobs(version_id,kind,revision,error) VALUES(%s,'extract',%s,%s)
        ON CONFLICT(version_id,kind,revision) DO UPDATE SET error=EXCLUDED.error RETURNING *""",
        (version_id,version["extraction_revision"]+1,reason)).fetchone()


def review(conn, job_id, agent_id, token, payload):
    parsed,value,digest = _payload(ReviewPayload,payload)
    with conn.transaction():
        job,corpus = _job(conn,job_id,agent_id,token,kind="review",allow_done=True)
        replay = _replay(conn,job,digest)
        if replay:
            return replay
        version_id = job["version_id"]
        version = conn.execute("""SELECT v.*,p.source_depth FROM paper_versions v JOIN papers p ON p.id=v.paper_id
            WHERE v.id=%s FOR UPDATE OF v""", (version_id,)).fetchone()
        if _extraction_participant(conn,version_id,agent_id):
            raise WorkflowError("Review must be performed by a different agent from every extraction participant")
        if version["extraction_revision"] != job["revision"]:
            raise WorkflowError("Review targets a stale extraction revision")
        ideas = {row["local_id"]:row for row in conn.execute("SELECT * FROM ideas WHERE version_id=%s AND active", (version_id,))}
        nominations = conn.execute("""SELECT n.*,i.local_id,r.citation_key FROM predecessor_nominations n
            JOIN ideas i ON i.id=n.idea_id JOIN paper_references r ON r.id=n.reference_id
            WHERE i.version_id=%s AND i.active""", (version_id,)).fetchall()
        scores = {item.local_id:item for item in parsed.idea_relevance}
        decisions = {(item.idea_local_id,item.citation_key):item for item in parsed.nomination_decisions}
        if parsed.accepted:
            if not ideas or set(scores) != set(ideas):
                raise WorkflowError("Accepted review requires at least one idea and exactly one score for every active idea")
            uncovered = conn.execute("""SELECT s.id FROM sections s LEFT JOIN section_coverage sc
                ON sc.section_id=s.id AND sc.version_id=s.version_id AND sc.revision=%s
                WHERE s.version_id=%s AND (sc.status IS NULL OR sc.status<>'covered')""", (job["revision"],version_id)).fetchall()
            if uncovered:
                raise WorkflowError("Unreadable or unaccounted sections block acceptance")
            pages = {row["page_number"]:row for row in conn.execute("SELECT * FROM pages WHERE version_id=%s", (version_id,))}
            if any("visual_review_required" in row["flags"] for row in pages.values()) and not parsed.visual_evidence_confirmed:
                raise WorkflowError("Accepted review must explicitly confirm visual inspection of every flagged page")
            evidence = conn.execute("""SELECT e.* FROM idea_evidence e JOIN ideas i ON i.id=e.idea_id
                WHERE i.version_id=%s AND i.active""", (version_id,)).fetchall()
            for item in evidence:
                if item["evidence_kind"] == "text":
                    _quote(pages,item["page_number"],item["quote"])
                elif not parsed.visual_evidence_confirmed:
                    raise WorkflowError("Accepted review must explicitly confirm all visual evidence")
            if set(decisions) != {(item["local_id"],item["citation_key"]) for item in nominations}:
                raise WorkflowError("Accepted review must decide every predecessor nomination exactly once")
            for item in nominations:
                _quote(pages,item["evidence_page"],item["evidence_quote"],"Predecessor")
                decision = decisions[(item["local_id"],item["citation_key"])]
                approved = decision.approved and scores[item["local_id"]].score >= 50 and version["source_depth"] == 0
                reason = decision.reason
                if decision.approved and version["source_depth"] != 0:
                    reason = "One-hop boundary: added paper cannot expand the corpus. " + reason
                elif decision.approved and scores[item["local_id"]].score < 50:
                    reason = "Relevance below foundational threshold (50). " + reason
                conn.execute("UPDATE predecessor_nominations SET state=%s,reviewer_id=%s,reason=%s WHERE id=%s",
                    ("approved" if approved else "rejected",agent_id,reason,item["id"]))
            for local_id,item in scores.items():
                conn.execute("UPDATE ideas SET relevance_score=%s,relevance_reason=%s WHERE id=%s", (item.score,item.reason,ideas[local_id]["id"]))
            conn.execute("""UPDATE paper_versions SET state='reviewed',reviewed_revision=%s,reviewer_id=%s,
                relevance_score=%s,relevance_reason=%s WHERE id=%s""", (job["revision"],agent_id,parsed.paper_relevance.score,parsed.paper_relevance.reason,version_id))
        else:
            _revision(conn,version,"\n".join(parsed.issues))
        _invalidate_graph(conn,corpus["id"])
        conn.execute("INSERT INTO reviews(version_id,revision,reviewer_id,accepted,payload) VALUES(%s,%s,%s,%s,%s)",
            (version_id,job["revision"],agent_id,parsed.accepted,Jsonb(value)))
        result = _record_submission(conn,job,digest,value)
        result["accepted"] = parsed.accepted
        return result


def fail(conn, job_id, agent_id, token, error, retry=True):
    if not isinstance(error,str) or not error.strip():
        raise WorkflowError("A failure reason is required")
    with conn.transaction():
        _job(conn,job_id,agent_id,token)
        conn.execute("UPDATE job_attempts SET state=%s,error=%s,finished_at=clock_timestamp() WHERE token=%s", ("retry" if retry else "blocked",error,token))
        return conn.execute("""UPDATE jobs SET state=%s,error=%s,lease_until=NULL,updated_at=clock_timestamp()
            WHERE id=%s RETURNING *""", ("queued" if retry else "blocked",error,job_id)).fetchone()


def reopen(conn, corpus_id, reason):
    if not isinstance(reason,str) or not reason.strip():
        raise WorkflowError("Reopening requires an audit reason")
    with conn.transaction():
        row = conn.execute("SELECT * FROM corpus WHERE id=%s FOR UPDATE", (corpus_id,)).fetchone()
        if not row:
            raise WorkflowError("Unknown corpus")
        _invalidate_graph(conn,corpus_id)
        conn.execute("UPDATE corpus SET state='open',expansion_closed=false WHERE id=%s", (corpus_id,))
        conn.execute("INSERT INTO corpus_events(corpus_id,action,reason) VALUES(%s,'reopen',%s)", (corpus_id,reason))
        return {"corpus_id": corpus_id, "state": "open", "reason": reason}


def retry_job(conn, job_id, reason):
    """Explicitly resume a blocked job after its source/problem has been resolved."""
    if not isinstance(reason,str) or not reason.strip():
        raise WorkflowError('Retrying a blocked job requires an audit reason')
    with conn.transaction():
        initial = conn.execute('SELECT version_id FROM jobs WHERE id=%s',(job_id,)).fetchone()
        if not initial:
            raise WorkflowError('Unknown job')
        corpus = _lock_corpus(conn,initial['version_id'])
        job = conn.execute('SELECT * FROM jobs WHERE id=%s FOR UPDATE',(job_id,)).fetchone()
        version = conn.execute('SELECT extraction_revision FROM paper_versions WHERE id=%s FOR UPDATE',(job['version_id'],)).fetchone()
        expected = version['extraction_revision'] + (1 if job['kind']=='extract' else 0)
        if job['state'] != 'blocked' or job['revision'] != expected:
            raise WorkflowError('Only a blocked job for the current workflow revision may be retried')
        conn.execute('INSERT INTO corpus_events(corpus_id,action,reason) VALUES(%s,%s,%s)',
                     (corpus['id'],'retry_job',f'{job_id}: {reason}'))
        return conn.execute("""UPDATE jobs SET state='queued',token=NULL,agent_id=NULL,
            lease_until=NULL,error=%s,updated_at=clock_timestamp() WHERE id=%s RETURNING *""",
            (reason,job_id)).fetchone()


def request_revision(conn, version_id, reason):
    if not isinstance(reason,str) or not reason.strip():
        raise WorkflowError("Revision requires an audit reason")
    with conn.transaction():
        corpus = _lock_corpus(conn,version_id)
        version = conn.execute("SELECT * FROM paper_versions WHERE id=%s FOR UPDATE", (version_id,)).fetchone()
        if version["state"] not in ("reviewed","complete","revision_required"):
            raise WorkflowError("Only a reviewed or complete paper can be explicitly reopened for correction")
        _invalidate_graph(conn,corpus["id"])
        conn.execute("UPDATE corpus SET expansion_closed=false WHERE id=%s", (corpus["id"],))
        conn.execute("INSERT INTO corpus_events(corpus_id,action,reason) VALUES(%s,'request_revision',%s)", (corpus["id"],reason))
        return _revision(conn,version,reason)


def status(conn):
    return {
        "corpora": conn.execute("SELECT * FROM corpus ORDER BY created_at").fetchall(),
        "papers": conn.execute("SELECT source_depth,count(*) AS count FROM papers GROUP BY source_depth ORDER BY source_depth").fetchall(),
        "paper_completion": conn.execute("""SELECT source_depth,count(*) AS total,
            count(*) FILTER (WHERE complete) AS complete FROM (
                SELECT p.id,p.source_depth,count(v.id)>0 AND bool_and(v.state='complete') AS complete
                FROM papers p LEFT JOIN paper_versions v ON v.paper_id=p.id
                GROUP BY p.id,p.source_depth
            ) progress GROUP BY source_depth ORDER BY source_depth""").fetchall(),
        "versions": conn.execute("SELECT state,count(*) AS count FROM paper_versions GROUP BY state ORDER BY state").fetchall(),
        "jobs": conn.execute("SELECT kind,state,count(*) AS count FROM jobs GROUP BY kind,state ORDER BY kind,state").fetchall(),
        "active_workers": conn.execute("SELECT id,version_id,kind,agent_id,lease_until FROM jobs WHERE state='leased' AND lease_until>clock_timestamp() ORDER BY lease_until").fetchall(),
        "expired_leases": conn.execute("SELECT count(*) AS count FROM jobs WHERE state='leased' AND lease_until<=clock_timestamp()").fetchone()["count"],
        "ideas": conn.execute("SELECT count(*) AS count FROM ideas WHERE active").fetchone()["count"],
        "nominations": conn.execute("SELECT state,count(*) AS count FROM predecessor_nominations GROUP BY state ORDER BY state").fetchall(),
        "source_uncertainties": conn.execute("""SELECT
            count(*) FILTER (WHERE version_label IS NULL) AS unknown_pdf_versions,
            count(*) FILTER (WHERE NOT date_verified) AS unverified_version_dates
            FROM paper_versions""").fetchone(),
        "withdrawn_sources": conn.execute("SELECT count(*) AS n FROM papers WHERE withdrawal_note IS NOT NULL").fetchone()["n"],
        "unreadable_coverage": conn.execute("""SELECT sc.version_id,sc.section_id,sc.notes
            FROM section_coverage sc JOIN paper_versions v ON v.id=sc.version_id
            WHERE sc.revision=v.extraction_revision AND sc.status='unreadable'""").fetchall(),
        "unavailable_predecessors": conn.execute("""SELECT n.id,r.title,n.reason FROM predecessor_nominations n
            JOIN paper_references r ON r.id=n.reference_id WHERE n.state='unavailable'""").fetchall(),
        "blocked": conn.execute("SELECT id,version_id,kind,error FROM jobs WHERE state='blocked' ORDER BY created_at").fetchall(),
    }
