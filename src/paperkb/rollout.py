"""Pilot-gated work dispatch. Codex, not a daemon, launches fresh worker contexts."""
import hashlib

from . import documents, workflow
from .config import settings

PILOT_IDS = ['2304.08467','2307.06945','2310.05736','2310.06839',
             '2403.12968','2407.15504','2405.16635','2404.17433']

def pilot_status(conn):
    rows = conn.execute('''SELECT p.arxiv_id,p.title,v.id AS version_id,v.state,
        v.extraction_revision,v.reviewed_revision,v.extractor_id,v.reviewer_id,v.relevance_score
        FROM papers p JOIN paper_versions v ON v.paper_id=p.id
        WHERE p.source_depth=0 AND p.arxiv_id=ANY(%s) ORDER BY p.arxiv_id''', (PILOT_IDS,)).fetchall()
    present = {r['arxiv_id'] for r in rows}
    config = settings()
    for row in rows:
        blockers = []
        revision, version_id = row['extraction_revision'], row['version_id']
        if row['state'] != 'complete':
            blockers.append('Paper is not complete')
        if (not revision or row['reviewed_revision'] != revision or not row['reviewer_id']
                or row['reviewer_id'] == row['extractor_id']):
            blockers.append('Current independent review is missing')
        if not conn.execute('''SELECT 1 FROM reviews WHERE version_id=%s AND revision=%s
                AND accepted AND reviewer_id=%s''', (version_id,revision,row['reviewer_id'])).fetchone():
            blockers.append('Accepted current review record is missing')
        if row['relevance_score'] is None:
            blockers.append('Paper relevance is missing')
        coverage = conn.execute('''SELECT count(*) AS total,
            count(*) FILTER(WHERE sc.status='covered') AS covered FROM sections s
            LEFT JOIN section_coverage sc ON sc.section_id=s.id AND sc.version_id=s.version_id
            AND sc.revision=%s WHERE s.version_id=%s''', (revision,version_id)).fetchone()
        if not coverage['total'] or coverage['total'] != coverage['covered']:
            blockers.append('Section coverage is incomplete or unreadable')
        ideas = conn.execute('''SELECT i.revision,i.relevance_score,i.embedding_text,
            e.revision AS embedded_revision,e.model,e.model_revision,e.text_sha256
            FROM ideas i LEFT JOIN embeddings e ON e.idea_id=i.id WHERE i.version_id=%s AND i.active''',
            (version_id,)).fetchall()
        if not ideas:
            blockers.append('No active ideas')
        for idea in ideas:
            if idea['revision'] != revision or idea['relevance_score'] is None:
                blockers.append('An idea is stale or unranked')
                break
        if any(idea['embedded_revision'] != revision or idea['model'] != config['embedding_model']
               or idea['model_revision'] != config['embedding_revision']
               or idea['text_sha256'] != hashlib.sha256(idea['embedding_text'].encode()).hexdigest() for idea in ideas):
            blockers.append('An embedding does not match the current model, revision, or text')
        row['ready'], row['blockers'] = not blockers, blockers
    return {'ready': len(present)==8 and all(r['ready'] for r in rows),
            'missing': sorted(set(PILOT_IDS)-present), 'papers': rows}

def initialize(conn):
    report = pilot_status(conn)
    if report['missing']:
        raise ValueError('Import corpus before initializing pilot: missing ' + ', '.join(report['missing']))
    outputs = []
    for row in report['papers']:
        if row['state']=='registered':
            outputs.append(documents.prepare(conn,row['version_id']))
    return {'prepared': outputs, 'pilot': pilot_status(conn)}

def next_work(conn, agent_id, lease_seconds=1800):
    workflow._identity(agent_id, lease_seconds)
    pilot = pilot_status(conn)
    if pilot['missing']:
        raise ValueError('The required pilot corpus is not registered')
    versions = None if pilot['ready'] else [r['version_id'] for r in pilot['papers']]
    def no_work(action='Embed reviewed papers; resolve review issues or wait for active workers.'):
        return {'phase':'corpus' if pilot['ready'] else 'pilot','job':None,
                'pilot':pilot,'next_action':action}

    # Prepare at most one new paper per request. Without this bound, a full
    # worker pool used to recursively prepare the entire corpus while no claim
    # could succeed, retaining database locks throughout that recursion.
    for prepared in (False, True):
        active = conn.execute("""SELECT agent_id FROM jobs WHERE state='leased'
            AND lease_until>clock_timestamp()""").fetchall()
        if len(active) >= 3:
            return no_work('Three workers are active; wait for a lease to become available.')
        if any(row['agent_id'] == agent_id for row in active):
            raise ValueError('A worker may hold only one paper job at a time')
        # Finish reviews first; claim serializes and rechecks the global cap.
        for kind in ('review','extract'):
            rows = conn.execute('''SELECT j.version_id FROM jobs j JOIN paper_versions v ON v.id=j.version_id
                JOIN papers p ON p.id=v.paper_id JOIN corpus c ON c.id=p.corpus_id
                WHERE j.kind=%s AND c.state='open'
                AND j.revision=v.extraction_revision+CASE WHEN j.kind='extract' THEN 1 ELSE 0 END
                AND (j.state='queued' OR (j.state='leased' AND j.lease_until<=clock_timestamp()))
                AND (%s::uuid[] IS NULL OR j.version_id=ANY(%s::uuid[]))
                ORDER BY (v.state='revision_required') DESC,(j.state='leased') DESC,j.created_at,j.id''',
                (kind,versions,versions)).fetchall()
            for row in rows:
                job = workflow.claim(conn,agent_id,kind,row['version_id'],lease_seconds)
                if job:
                    return {'phase':'corpus' if pilot['ready'] else 'pilot','job':job,
                            'packet':documents.packet(conn,row['version_id']),
                            'instruction':'Launch a fresh agent for this single paper. Read WORKER_PROTOCOL.md. Persist section checkpoints.'}
        if not pilot['ready'] or prepared:
            return no_work()
        # Capacity can change during the preceding claim attempts.
        if conn.execute("""SELECT count(*) AS n FROM jobs WHERE state='leased'
            AND lease_until>clock_timestamp()""").fetchone()['n'] >= 3:
            return no_work('Three workers are active; wait for a lease to become available.')
        row = conn.execute("""SELECT v.id FROM paper_versions v JOIN papers p ON p.id=v.paper_id
            JOIN corpus c ON c.id=p.corpus_id WHERE v.state='registered' AND c.state='open'
            ORDER BY v.created_at,v.id LIMIT 1""").fetchone()
        if not row:
            return no_work()
        documents.prepare(conn,row['id'])
    return no_work()
