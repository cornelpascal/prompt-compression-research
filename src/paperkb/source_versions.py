"""Acquire a chronologically eligible PDF version of a reviewed predecessor.

The paper is already admitted; this adds immutable source bytes, not a new hop,
replacement extraction, or ancestry edge. Every additional version must pass the
ordinary preparation, extraction, review, ranking and embedding workflow.
"""
from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from psycopg.types.json import Jsonb
from pydantic import Field

from . import documents, expansion
from .config import ROOT
from .models import Payload, normalized_text


class SourceVersion(Payload):
    resolver_id: str = Field(min_length=1)
    matched_title: str = Field(min_length=1)
    version_label: str = Field(pattern=r'^v[1-9][0-9]*$')
    metadata_url: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def _inputs(conn, nomination_id, request):
    rows = expansion._approved_nominations(
        conn, states=('admitted', 'existing'), nomination_id=nomination_id)
    if not rows or rows[0]['source_depth'] != 0:
        raise ValueError('An additional version requires a resolved original-paper nomination')
    nomination = rows[0]
    expansion._validate_nomination(conn, nomination)
    target = conn.execute('SELECT * FROM papers WHERE id=%s',
                          (nomination['target_paper_id'],)).fetchone()
    child = conn.execute('SELECT * FROM paper_versions WHERE id=%s',
                         (nomination['version_id'],)).fetchone()
    if not target or target['corpus_id'] != nomination['corpus_id'] or not target['arxiv_id']:
        raise ValueError('A resolved arXiv predecessor in the same corpus is required')
    if not child['date_verified'] or child['available_at'] is None:
        raise ValueError('Verify the citing PDF version date before acquiring an earlier version')
    title_key = lambda value: re.sub(r'[^\w]', '', normalized_text(value).casefold())
    if title_key(request.matched_title) != title_key(nomination['reference_title']):
        raise ValueError('Verified metadata title does not match the reviewed reference')
    cited_id, cited_version = expansion._reference_identity({
        'arxiv_id': nomination['reference_arxiv_id'], 'url': nomination['reference_url']})
    if ((cited_id and cited_id != target['arxiv_id']) or
            (cited_version and cited_version != request.version_label)):
        raise ValueError('Requested version conflicts with the reviewed citation identity')
    metadata = urlparse(request.metadata_url)
    if (metadata.hostname not in {'arxiv.org', 'export.arxiv.org'} or
            not metadata.path.startswith(('/abs/', '/html/')) or
            documents.normalize_arxiv(request.metadata_url) !=
            (target['arxiv_id'], request.version_label)):
        raise ValueError('Provide the matching exact-version primary arXiv metadata URL')
    corpus = conn.execute('SELECT state,expansion_closed FROM corpus WHERE id=%s',
                          (nomination['corpus_id'],)).fetchone()
    if corpus['state'] != 'open' or corpus['expansion_closed']:
        raise ValueError('Corpus is frozen or citation expansion is closed')
    # Only identities and chronology participate in the acquisition fence.
    target_identity = {key: target[key] for key in ('id', 'arxiv_id', 'corpus_id', 'source_depth')}
    child_identity = {key: child[key] for key in ('id', 'sha256', 'available_at', 'date_verified')}
    return nomination, target_identity, child_identity


def _existing(conn, target, child, request):
    rows = conn.execute('SELECT * FROM paper_versions WHERE paper_id=%s AND version_label=%s',
                        (target['id'], request.version_label)).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError('Multiple PDF sources have this version label; inspect their provenance')
    row = rows[0]
    if not row['date_verified'] or row['available_at'] is None:
        raise ValueError('Verify the existing source version date before reusing it')
    if row['available_at'] >= child['available_at']:
        raise ValueError('Predecessor version must be strictly earlier than the citing PDF')
    if documents.sha256_file(Path(row['file_path'])) != row['sha256']:
        raise ValueError('Existing source PDF checksum mismatch')
    return {'paper_id': str(target['id']), 'version_id': str(row['id']),
            'version_label': row['version_label'], 'available_at': row['available_at'],
            'date_verified': True, 'existing': True,
            'requires_preparation': not row['preparation'].get('complete', False)}


def acquire(conn, nomination_id, payload):
    request = SourceVersion.model_validate(payload)
    expansion._safe_url(request.metadata_url)
    # Share acquisition serialization and pacing with normal one-hop expansion.
    # No corpus/workflow row lock is held during metadata or PDF downloads.
    conn.execute('SELECT pg_advisory_xact_lock(%s)', (78128472101,))
    nomination, target, child = _inputs(conn, nomination_id, request)
    existing = _existing(conn, target, child, request)
    if existing:
        return existing
    exact_id = target['arxiv_id'] + request.version_label
    feed_url, feed = documents._fetch_exact_version_feed([exact_id])
    date = documents.parse_exact_version_feed(feed, target['arxiv_id'], request.version_label)
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    title_key = lambda value: re.sub(r'[^\w]', '', normalized_text(value).casefold())
    entries = ET.fromstring(feed).findall('a:entry', ns)
    titles = [entry.findtext('a:title', '', ns) for entry in entries
              if documents.normalize_arxiv(entry.findtext('a:id', '', ns)) ==
              (target['arxiv_id'], request.version_label)]
    if len(titles) != 1 or title_key(titles[0]) != title_key(request.matched_title):
        raise ValueError('Exact-version metadata title does not match the reviewed reference')
    if date >= child['available_at']:
        raise ValueError('Predecessor version must be strictly earlier than the citing PDF')
    url = f'https://arxiv.org/pdf/{exact_id}'
    path, sha, stamp, final_url = expansion._download_pdf(url, Path(ROOT) / 'papers/predecessors')
    if documents.normalize_arxiv(final_url) != (target['arxiv_id'], request.version_label):
        raise ValueError('Downloaded PDF final URL does not preserve the exact requested version')
    if ((stamp.get('arxiv_id') and stamp['arxiv_id'] != target['arxiv_id']) or
            (stamp.get('version_label') and stamp['version_label'] != request.version_label)):
        raise ValueError('Downloaded PDF stamp conflicts with the requested source version')
    if documents.sha256_file(path) != sha:
        raise ValueError('Downloaded PDF checksum changed before admission')

    with conn.transaction():
        documents._open_corpus(conn, nomination['corpus_id'], expansion=True)
        current, current_target, current_child = _inputs(conn, nomination_id, request)
        comparable = lambda row: {k: v for k, v in row.items() if k != 'version_state'}
        if (comparable(current) != comparable(nomination) or current_target != target or
                current_child != child):
            raise ValueError('Reviewed dependency or source identity changed during acquisition')
        existing = _existing(conn, target, child, request)
        if existing:
            return existing
        if conn.execute('SELECT 1 FROM paper_versions WHERE sha256=%s', (sha,)).fetchone():
            raise ValueError('Identical PDF is registered with a different source identity')
        version_id = uuid.uuid5(documents.NAMESPACE, f'pdf:{sha}')
        admitted_at = datetime.now(timezone.utc).isoformat()
        evidence = {
            'source_version_id': str(nomination['version_id']),
            'source_extraction_revision': nomination['revision'],
            'source_idea_id': str(nomination['idea_id']),
            'reference_id': str(nomination['reference_id']),
            'citation_key': nomination['reference_citation_key'],
            'page_number': nomination['evidence_page'], 'quote': nomination['evidence_quote'],
            'reviewer_id': nomination['reviewer_id'],
            'relevance_score': nomination['relevance_score'], 'review_reason': nomination['reason'],
        }
        download = {'url': final_url, 'requested_url': url, 'sha256': sha,
                    'file': str(path), 'downloaded_at': admitted_at}
        verification = {'nomination_id': str(nomination_id), 'request': request.model_dump(),
                        'citing_available_at': child['available_at'].isoformat(),
                        'recorded_at': admitted_at}
        preparation = {'provenance': stamp, 'download': download,
                       'admission_evidence': evidence, 'additional_version': verification}
        conn.execute('''INSERT INTO paper_versions(id,paper_id,sha256,file_path,version_label,preparation)
            VALUES(%s,%s,%s,%s,%s,%s)''',
            (version_id, target['id'], sha, str(path), request.version_label, Jsonb(preparation)))
        documents._record_version_date(conn, documents._version(conn, version_id),
                                       request.version_label, None, feed_url, feed)
        audit = {'paper_id': str(target['id']), 'version_id': str(version_id),
                 'download': download, 'admission_evidence': evidence,
                 'verification': verification, 'available_at': date.isoformat()}
        event = conn.execute('''INSERT INTO corpus_events(corpus_id,action,reason)
            VALUES(%s,'acquire_source_version',%s) RETURNING id''',
            (nomination['corpus_id'], json.dumps(audit, ensure_ascii=False))).fetchone()
        return {'paper_id': str(target['id']), 'version_id': str(version_id),
                'version_label': request.version_label, 'available_at': date,
                'date_verified': True, 'existing': False, 'requires_preparation': True,
                'audit_event_id': str(event['id'])}
