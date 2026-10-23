import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from paperkb import cli, db, documents, expansion, graph, source_versions, workflow
from test_expansion import nomination


def feed(version='v2', date='2023-08-01T00:00:00Z', title='Predecessor'):
    return (f'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            f'<id>http://arxiv.org/abs/2307.06945{version}</id><title>{title}</title>'
            f'<updated>{date}</updated></entry></feed>').encode()


def request(**updates):
    return dict(resolver_id='version-resolver', matched_title='Predecessor',
                version_label='v2', metadata_url='https://arxiv.org/abs/2307.06945v2',
                reason='Primary history verifies an earlier version of the cited work.') | updates


def seed_resolved(conn, tmp_path, monkeypatch, **nomination_options):
    corpus, original, child, nid = nomination(conn, **nomination_options)
    conn.execute("UPDATE paper_versions SET version_label='v1',available_at='2024-01-01',date_verified=true WHERE id=%s", (child,))
    target = conn.execute("INSERT INTO papers(corpus_id,arxiv_id,title,source_depth,metadata) VALUES(%s,'2307.06945','Predecessor',1,%s) RETURNING id",
                         (corpus, Jsonb({'keep': 'original admission metadata'}))).fetchone()['id']
    old_pdf = tmp_path / 'already-admitted-v3.pdf'
    old_pdf.write_bytes(b'immutable existing PDF fixture')
    old = conn.execute("INSERT INTO paper_versions(paper_id,sha256,file_path,version_label,available_at,date_verified) VALUES(%s,%s,%s,'v3','2024-02-01',true) RETURNING id",
                       (target, hashlib.sha256(old_pdf.read_bytes()).hexdigest(), str(old_pdf))).fetchone()['id']
    conn.execute("UPDATE predecessor_nominations SET state='admitted',target_paper_id=%s WHERE id=%s", (target, nid))
    pdf = tmp_path / 'earlier-v2.pdf'
    pdf.write_bytes(b'immutable earlier PDF fixture')
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(expansion, '_safe_url', lambda url: url)
    monkeypatch.setattr(documents, '_fetch_exact_version_feed', lambda ids: ('https://export.arxiv.org/api/query', feed()))
    monkeypatch.setattr(documents, 'settings', lambda: {'artifact_dir': str(tmp_path / 'artifacts')})
    monkeypatch.setattr(expansion, '_download_pdf', lambda *args: (
        pdf, sha, {'arxiv_id': '2307.06945', 'version_label': 'v2'}, 'https://arxiv.org/pdf/2307.06945v2'))
    return {'corpus': corpus, 'original': original, 'child': child, 'nomination': nid,
            'target': target, 'old': old, 'old_pdf': old_pdf, 'pdf': pdf, 'sha': sha}


def test_earlier_version_preserves_existing_sources_and_is_idempotent(conn, tmp_path, monkeypatch):
    s = seed_resolved(conn, tmp_path, monkeypatch)
    before = conn.execute('SELECT * FROM paper_versions WHERE id=%s', (s['old'],)).fetchone()
    result = source_versions.acquire(conn, s['nomination'], request())
    assert not result['existing'] and result['requires_preparation']
    assert result['paper_id'] == str(s['target']) and result['version_id'] != str(s['old'])
    assert conn.execute('SELECT * FROM paper_versions WHERE id=%s', (s['old'],)).fetchone() == before
    assert s['old_pdf'].read_bytes() == b'immutable existing PDF fixture'
    assert conn.execute('SELECT count(*) AS n FROM papers').fetchone()['n'] == 2
    assert conn.execute('SELECT metadata,source_depth FROM papers WHERE id=%s', (s['target'],)).fetchone() == {
        'metadata': {'keep': 'original admission metadata'}, 'source_depth': 1}
    row = conn.execute('SELECT * FROM paper_versions WHERE id=%s', (result['version_id'],)).fetchone()
    assert row['state'] == 'registered' and row['extraction_revision'] == 0 and row['date_verified']
    assert row['reviewed_revision'] is None and row['relevance_score'] is None
    assert row['preparation']['download']['sha256'] == s['sha']
    assert row['preparation']['admission_evidence']['source_version_id'] == str(s['child'])
    assert Path(row['preparation']['date_verification']['artifact']).read_bytes() == feed()
    assert not graph.barrier(conn, s['corpus'])['ready']
    assert any('Predecessor: state is registered' in reason for reason in graph.barrier(conn, s['corpus'])['reasons'])
    monkeypatch.setattr(expansion, '_download_pdf', lambda *a: pytest.fail('Retry must reuse the exact existing PDF'))
    assert source_versions.acquire(conn, s['nomination'], request())['existing']
    assert conn.execute("SELECT count(*) AS n FROM corpus_events WHERE action='acquire_source_version'").fetchone()['n'] == 1
    assert conn.execute('SELECT count(*) AS n FROM graph_parents').fetchone()['n'] == 0


@pytest.mark.parametrize('change', ['unreviewed', 'rejected', 'low_score', 'depth_one', 'child_date', 'closed', 'frozen'])
def test_version_acquisition_enforces_current_review_hop_date_and_corpus_gates(conn, tmp_path, monkeypatch, change):
    s = seed_resolved(conn, tmp_path, monkeypatch)
    if change == 'unreviewed':
        conn.execute('DELETE FROM reviews WHERE version_id=%s', (s['child'],))
    elif change == 'rejected':
        conn.execute("UPDATE predecessor_nominations SET state='rejected' WHERE id=%s", (s['nomination'],))
    elif change == 'low_score':
        conn.execute('UPDATE ideas SET relevance_score=49')
    elif change == 'depth_one':
        conn.execute('UPDATE papers SET source_depth=1 WHERE id=%s', (s['original'],))
    elif change == 'child_date':
        conn.execute('UPDATE paper_versions SET date_verified=false WHERE id=%s', (s['child'],))
    elif change == 'closed':
        conn.execute('UPDATE corpus SET expansion_closed=true')
    else:
        conn.execute("UPDATE corpus SET state='frozen'")
    monkeypatch.setattr(expansion, '_download_pdf', lambda *a: pytest.fail('Invalid request must not download'))
    with pytest.raises(ValueError):
        source_versions.acquire(conn, s['nomination'], request())
    assert conn.execute('SELECT count(*) AS n FROM paper_versions').fetchone()['n'] == 2


@pytest.mark.parametrize('response', [feed(date='2024-01-01T00:00:00Z'), feed(date='2024-06-01T00:00:00Z'), feed('v3'), feed(title='A different work')])
def test_version_metadata_must_match_and_strictly_predate_citing_source(conn, tmp_path, monkeypatch, response):
    s = seed_resolved(conn, tmp_path, monkeypatch)
    monkeypatch.setattr(documents, '_fetch_exact_version_feed', lambda ids: ('https://export.arxiv.org/api/query', response))
    monkeypatch.setattr(expansion, '_download_pdf', lambda *a: pytest.fail('Invalid metadata must not download'))
    with pytest.raises(ValueError):
        source_versions.acquire(conn, s['nomination'], request())
    assert conn.execute('SELECT count(*) AS n FROM paper_versions').fetchone()['n'] == 2


@pytest.mark.parametrize('stamp,url', [
    ({'arxiv_id': '2307.06945', 'version_label': 'v3'}, 'https://arxiv.org/pdf/2307.06945v2'),
    ({}, 'https://arxiv.org/pdf/2307.06945v3'),
    ({}, 'https://arxiv.org/pdf/2310.05736v2'),
])
def test_wrong_pdf_identity_never_overwrites_or_registers_source(conn, tmp_path, monkeypatch, stamp, url):
    s = seed_resolved(conn, tmp_path, monkeypatch)
    monkeypatch.setattr(expansion, '_download_pdf', lambda *a: (s['pdf'], s['sha'], stamp, url))
    with pytest.raises(ValueError):
        source_versions.acquire(conn, s['nomination'], request())
    assert conn.execute('SELECT count(*) AS n FROM paper_versions').fetchone()['n'] == 2
    assert s['old_pdf'].read_bytes() == b'immutable existing PDF fixture'


def test_explicit_citation_version_cannot_be_replaced(conn, tmp_path, monkeypatch):
    s = seed_resolved(conn, tmp_path, monkeypatch)
    conn.execute("UPDATE paper_references SET url='https://arxiv.org/abs/2307.06945v1'")
    with pytest.raises(ValueError, match='conflicts'):
        source_versions.acquire(conn, s['nomination'], request())


@pytest.mark.parametrize('change', [None, 'revoke', 'reference', 'date'])
def test_version_download_does_not_lock_workflow_and_fences_concurrent_changes(conn, tmp_path, monkeypatch, change):
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    from paperkb.config import settings
    s = seed_resolved(conn, tmp_path, monkeypatch)
    schema = conn.execute('SELECT current_schema() AS name').fetchone()['name']
    conn.commit()
    def download(*args):
        options = conninfo_to_dict(settings()['database_url'])
        options['dbname'] = 'paperkb_test'
        with db.connect(make_conninfo(**options)) as other:
            other.execute(sql.SQL('SET search_path TO {},public').format(sql.Identifier(schema)))
            other.execute("SET lock_timeout='500ms'")
            other.execute('SELECT id FROM corpus WHERE id=%s FOR UPDATE', (s['corpus'],))
            if change == 'revoke':
                other.execute("UPDATE predecessor_nominations SET state='rejected' WHERE id=%s", (s['nomination'],))
            elif change == 'reference':
                other.execute("UPDATE paper_references SET title='Changed reference'")
            elif change == 'date':
                other.execute("UPDATE paper_versions SET available_at='2024-01-02' WHERE id=%s", (s['child'],))
        return s['pdf'], s['sha'], {}, 'https://arxiv.org/pdf/2307.06945v2'
    monkeypatch.setattr(expansion, '_download_pdf', download)
    if change:
        with pytest.raises(ValueError):
            source_versions.acquire(conn, s['nomination'], request())
        assert conn.execute('SELECT count(*) AS n FROM paper_versions').fetchone()['n'] == 2
    else:
        assert source_versions.acquire(conn, s['nomination'], request())['date_verified']


def test_acquisition_failure_rolls_back_new_version_and_event(conn, tmp_path, monkeypatch):
    s = seed_resolved(conn, tmp_path, monkeypatch)
    def fail(*args):
        raise ValueError('Synthetic date persistence failure')
    monkeypatch.setattr(documents, '_record_version_date', fail)
    with pytest.raises(ValueError, match='persistence'):
        source_versions.acquire(conn, s['nomination'], request())
    assert conn.execute('SELECT count(*) AS n FROM paper_versions').fetchone()['n'] == 2
    assert conn.execute('SELECT count(*) AS n FROM corpus_events').fetchone()['n'] == 0


def test_acquire_version_cli_accepts_validated_json_and_emits_receipt(conn, tmp_path, monkeypatch, capsys):
    s = seed_resolved(conn, tmp_path, monkeypatch)
    payload = tmp_path / 'request.json'
    payload.write_text(json.dumps(request()))
    monkeypatch.setattr(db, 'connect', lambda: nullcontext(conn))
    assert cli.main(['acquire-version', str(s['nomination']), '--file', str(payload)]) == 0
    assert json.loads(capsys.readouterr().out)['requires_preparation']


def test_preparation_preserves_per_version_admission_and_counts_all_versions(conn, tmp_path, monkeypatch):
    from pypdf import PdfWriter
    s = seed_resolved(conn, tmp_path, monkeypatch)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(s['pdf'])
    sha = hashlib.sha256(s['pdf'].read_bytes()).hexdigest()
    monkeypatch.setattr(expansion, '_download_pdf', lambda *a: (
        s['pdf'], sha, {}, 'https://arxiv.org/pdf/2307.06945v2'))
    result = source_versions.acquire(conn, s['nomination'], request())
    before = conn.execute('SELECT preparation FROM paper_versions WHERE id=%s',
                          (result['version_id'],)).fetchone()['preparation']
    prepared = documents.prepare(conn, result['version_id'])
    for key in ('download', 'admission_evidence', 'additional_version', 'date_verification'):
        assert prepared['version']['preparation'][key] == before[key]
    assert len(prepared['pages']) == len(prepared['sections']) == 1
    assert prepared['jobs'][0]['state'] == 'queued' and not prepared['ideas']
    conn.execute("UPDATE paper_versions SET state='complete' WHERE id=%s", (s['old'],))
    progress = workflow.status(conn)['paper_completion']
    assert next(row for row in progress if row['source_depth'] == 1) == {
        'source_depth': 1, 'total': 1, 'complete': 0}
