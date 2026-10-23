import hashlib
import http.client
import io
import json
import urllib.error
import uuid

import pytest

from paperkb import expansion


def nomination(conn, *, depth=0, score=70, target="2307.06945", state="approved"):
    corpus = conn.execute("INSERT INTO corpus(name) VALUES(%s) RETURNING id", (str(uuid.uuid4()),)).fetchone()["id"]
    paper = conn.execute("INSERT INTO papers(corpus_id,arxiv_id,title,source_depth) VALUES(%s,'2405.16635','Original',%s) RETURNING id", (corpus, depth)).fetchone()["id"]
    version = conn.execute("INSERT INTO paper_versions(paper_id,sha256,file_path,state,extraction_revision,reviewed_revision,extractor_id,reviewer_id) VALUES(%s,%s,'unused.pdf','reviewed',1,1,'extractor','reviewer') RETURNING id", (paper, "1" * 64)).fetchone()["id"]
    conn.execute("INSERT INTO reviews(version_id,revision,reviewer_id,accepted,payload) VALUES(%s,1,'reviewer',true,'{}')", (version,))
    idea = uuid.uuid4()
    conn.execute("INSERT INTO ideas(id,version_id,local_id,revision,kind,title,statement,attribution,confidence,relevance_score,embedding_text) VALUES(%s,%s,'i1',1,'method','Method','Explicit extension','original',0.9,%s,'Method')", (idea, version, score))
    reference = conn.execute("INSERT INTO paper_references(version_id,citation_key,title,arxiv_id) VALUES(%s,'R1','Predecessor',%s) RETURNING id", (version, target)).fetchone()["id"]
    nid = conn.execute("INSERT INTO predecessor_nominations(idea_id,reference_id,evidence_page,evidence_quote,state,reviewer_id) VALUES(%s,%s,1,'Explicit extension',%s,'reviewer') RETURNING id", (idea, reference, state)).fetchone()["id"]
    return corpus, paper, version, nid


def test_doi_and_source_canonicalization():
    assert expansion.normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert expansion.normalize_doi("invalid") is None
    assert expansion.source_urls({"arxiv_id": "2307.06945v2", "url": "https://arxiv.org/abs/2307.06945"}) == ["https://arxiv.org/pdf/2307.06945v2"]
    assert expansion.source_urls({"doi": "10.1234/a"}) == []
    assert expansion.source_urls({"arxiv_id":"2307.06945","url":"https://arxiv.org/pdf/2307.06945v2"}) == ['https://arxiv.org/pdf/2307.06945v2']
    with pytest.raises(ValueError,match='Conflicting'):
        expansion.source_urls({"arxiv_id":"2307.06945v1","url":"https://arxiv.org/pdf/2307.06945v2"})


def test_verified_arxiv_pdf_fallback_preserves_exact_version():
    assert expansion.source_urls({"arxiv_id": "2404.09193v1",
        "url": "https://export.arxiv.org/pdf/2404.09193v1"}) == [
            "https://arxiv.org/pdf/2404.09193v1", "https://export.arxiv.org/pdf/2404.09193v1"]
    assert expansion.source_urls({"arxiv_id": "2404.09193v1",
        "url": "https://export.arxiv.org/pdf/2404.09193"}) == [
            "https://arxiv.org/pdf/2404.09193v1"]
    with pytest.raises(ValueError, match="Conflicting"):
        expansion.source_urls({"arxiv_id": "2404.09193v1",
            "url": "https://export.arxiv.org/pdf/2404.09193v2"})


def resolution_payload(**updates):
    return dict(resolver_id='metadata-resolver',matched_title='Predecessor',
                metadata_url='https://arxiv.org/abs/2307.06945',
                reason='Primary metadata title and author identity checked.',
                arxiv_id='2307.06945')|updates


def test_resolved_missing_reference_retries_acquisition_without_new_extraction(conn,monkeypatch):
    _,_,version,nid=nomination(conn,target=None,state='unavailable')
    reference=conn.execute('SELECT id FROM paper_references').fetchone()['id']
    monkeypatch.setattr(expansion,'_safe_url',lambda url:url)
    result=expansion.resolve_reference(conn,reference,resolution_payload())
    assert not result['admitted'] and result['eligible_nominations']==1
    assert conn.execute('SELECT state FROM predecessor_nominations WHERE id=%s',(nid,)).fetchone()['state']=='approved'
    row=conn.execute('SELECT state,extraction_revision,reviewed_revision FROM paper_versions WHERE id=%s',(version,)).fetchone()
    assert row=={'state':'reviewed','extraction_revision':1,'reviewed_revision':1}
    audit=json.loads(conn.execute("SELECT reason FROM corpus_events WHERE action='resolve_reference'").fetchone()['reason'])
    assert audit['original_identifiers']['arxiv_id'] is None
    assert audit['resolved_identifiers']['arxiv_id']=='2307.06945'
    assert audit['verification']['metadata_url']=='https://arxiv.org/abs/2307.06945'


@pytest.mark.parametrize('updates,message',[
    ({'matched_title':'A different paper'},'title'),
    ({'arxiv_id':'2304.08467'},'conflicting'),
])
def test_source_resolution_rejects_identity_changes(conn,monkeypatch,updates,message):
    nomination(conn)
    reference=conn.execute('SELECT id FROM paper_references').fetchone()['id']
    monkeypatch.setattr(expansion,'_safe_url',lambda url:url)
    with pytest.raises(ValueError,match=message):
        expansion.resolve_reference(conn,reference,resolution_payload(**updates))
    assert conn.execute('SELECT count(*) AS n FROM corpus_events').fetchone()['n']==0


@pytest.mark.parametrize('depth,score,state',[(1,70,'approved'),(0,20,'approved'),(0,70,'rejected')])
def test_source_resolution_preserves_review_relevance_and_hop_gates(conn,monkeypatch,depth,score,state):
    nomination(conn,depth=depth,score=score,state=state)
    reference=conn.execute('SELECT id FROM paper_references').fetchone()['id']
    monkeypatch.setattr(expansion,'_safe_url',lambda url:url)
    with pytest.raises(ValueError):
        expansion.resolve_reference(conn,reference,resolution_payload())
    assert conn.execute('SELECT count(*) AS n FROM corpus_events').fetchone()['n']==0


def test_missing_identifier_requires_lookup_before_declaring_gap(conn,monkeypatch):
    nomination(conn,target=None)
    assert expansion.expand(conn)['results'][0]['needs_source_resolution']
    with pytest.raises(ValueError,match='Unresolved'):
        expansion.close_expansion(conn)
    reference=conn.execute('SELECT id FROM paper_references').fetchone()['id']
    monkeypatch.setattr(expansion,'_safe_url',lambda url:url)
    payload=resolution_payload(arxiv_id=None,availability='unavailable')
    with pytest.raises(ValueError,match='URLs checked'):
        expansion.resolve_reference(conn,reference,payload)
    payload['checked_urls']=['https://publisher.example/article']
    payload['reason']='Primary publication record and author repository checked; no accessible PDF found.'
    assert expansion.resolve_reference(conn,reference,payload)['availability']=='unavailable'
    assert expansion.close_expansion(conn)['closed']


def test_private_and_non_https_urls_rejected(monkeypatch):
    for url in ["file:///etc/passwd", "http://example.com/a.pdf", "https://localhost/a.pdf", "https://me:secret@example.com/a.pdf"]:
        with pytest.raises(ValueError):
            expansion._safe_url(url)
    monkeypatch.setattr(expansion.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="non-public"):
        expansion._safe_url("https://example.com/a.pdf")


def test_existing_target_deduplicated_without_download(conn, monkeypatch):
    corpus, _, _, nid = nomination(conn)
    target = conn.execute("INSERT INTO papers(corpus_id,arxiv_id,title,source_depth) VALUES(%s,'2307.06945','Existing',0) RETURNING id", (corpus,)).fetchone()["id"]
    monkeypatch.setattr(expansion, "_download_pdf", lambda *args: pytest.fail("Existing papers must not download"))
    result = expansion.expand(conn)
    assert result["results"][0]["state"] == "existing"
    row = conn.execute("SELECT * FROM predecessor_nominations WHERE id=%s", (nid,)).fetchone()
    assert row["target_paper_id"] == target


def test_depth_one_nomination_cannot_expand(conn, monkeypatch):
    _, _, _, nid = nomination(conn, depth=1)
    monkeypatch.setattr(expansion, "_download_pdf", lambda *args: pytest.fail("One-hop limit must prevent download"))
    assert expansion.expand(conn)["results"][0]["state"] == "rejected"
    assert conn.execute("SELECT state FROM predecessor_nominations WHERE id=%s", (nid,)).fetchone()["state"] == "rejected"


def test_low_relevance_and_stale_review_block_expansion(conn):
    nomination(conn, score=49)
    with pytest.raises(ValueError, match="qualifying review"):
        expansion.expand(conn)


def test_pending_nomination_not_downloaded_and_blocks_closure(conn):
    nomination(conn, state="pending")
    assert expansion.expand(conn)["processed"] == 0
    with pytest.raises(ValueError, match="Unresolved"):
        expansion.close_expansion(conn)


def test_admit_acquired_pdf_depth_one_with_unknown_date(conn, tmp_path, monkeypatch):
    corpus, _, _, nid = nomination(conn)
    pdf = tmp_path / "download.pdf"
    pdf.write_bytes(b"verified-by-download-helper")
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(expansion, "ROOT", tmp_path)
    monkeypatch.setattr(expansion, "_download_pdf", lambda *args: (pdf, sha, {"arxiv_id": "2307.06945", "version_label": "v2"}, "https://arxiv.org/pdf/2307.06945v2"))
    result = expansion.expand(conn)["results"][0]
    assert result["state"] == "admitted" and result["requires_preparation"]
    row = conn.execute("SELECT p.source_depth,v.available_at,v.date_verified,v.version_label FROM paper_versions v JOIN papers p ON p.id=v.paper_id WHERE v.id=%s", (result["version_id"],)).fetchone()
    assert row == {"source_depth": 1, "available_at": None, "date_verified": False, "version_label": "v2"}
    assert expansion.expand(conn)["processed"] == 0
    assert expansion.close_expansion(conn)["closed"] == [str(corpus)]


def test_admission_keeps_dependency_provenance_after_current_nomination_is_replaced(conn,tmp_path,monkeypatch):
    _,_,source_version,nid=nomination(conn)
    pdf=tmp_path/'source.pdf';pdf.write_bytes(b'already validated fixture')
    sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(expansion,'_download_pdf',lambda *args:(pdf,sha,{},'https://arxiv.org/pdf/2307.06945v1'))
    admitted=expansion.expand(conn)['results'][0]
    conn.execute('DELETE FROM predecessor_nominations WHERE id=%s',(nid,))
    metadata=conn.execute('SELECT metadata FROM papers WHERE id=%s',(admitted['paper_id'],)).fetchone()['metadata']
    evidence=metadata['admission_evidence']
    assert evidence['source_version_id']==str(source_version)
    assert evidence['source_extraction_revision']==1 and evidence['citation_key']=='R1'
    assert evidence['quote']=='Explicit extension' and evidence['page_number']==1
    assert evidence['reviewer_id']=='reviewer' and evidence['relevance_score']==70


def test_missing_pdf_records_gap_without_admission(conn, monkeypatch):
    corpus, _, _, nid = nomination(conn)
    def missing(*args):
        raise urllib.error.HTTPError("https://arxiv.org/pdf/2307.06945", 404, "missing", {}, None)
    monkeypatch.setattr(expansion, "_download_pdf", missing)
    result = expansion.expand(conn)["results"][0]
    assert result["state"] == "unavailable"
    assert conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"] == 1
    assert expansion.close_expansion(conn)["closed"] == [str(corpus)]


def test_transient_download_is_retryable_and_blocks_closure(conn, monkeypatch):
    nomination(conn)
    def temporary(*args):
        raise urllib.error.URLError("timed out")
    monkeypatch.setattr(expansion, "_download_pdf", temporary)
    assert expansion.expand(conn)["results"][0]["state"] == "approved"
    with pytest.raises(ValueError, match="Unresolved"):
        expansion.close_expansion(conn)


def test_incomplete_http_read_remains_retryable(conn, monkeypatch):
    nomination(conn)
    def interrupted(*args):
        raise http.client.IncompleteRead(b"%PDF-partial", 400)
    monkeypatch.setattr(expansion, "_download_pdf", interrupted)
    result = expansion.expand(conn)["results"][0]
    assert result["state"] == "approved"
    assert "IncompleteRead" in result["reason"]
    assert conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"] == 1
    with pytest.raises(ValueError, match="Unresolved"):
        expansion.close_expansion(conn)


@pytest.mark.parametrize("body,declared,status,message", [
    (b"%PDF-partial\n%%EOF", 1000, 200, "received"),
    (b"%PDF-partial\n%%EOF", 0, 206, "partial PDF response"),
    (b"%PDF-partial", 0, 200, "missing end marker"),
])
def test_incomplete_download_is_cleaned_and_never_validated_as_complete(
        tmp_path, monkeypatch, body, declared, status, message):
    class Response(io.BytesIO):
        headers = {"Content-Length": str(declared)}
        def geturl(self):
            return "https://arxiv.org/pdf/2307.06945v1"
    response = Response(body)
    response.status = status
    class Opener:
        def open(self, request, timeout):
            return response
    monkeypatch.setattr(expansion, "_safe_url", lambda url: url)
    monkeypatch.setattr(expansion.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(expansion.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(ConnectionError, match=message):
        expansion._download_pdf(response.geturl(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unreviewed_original_blocks_closure(conn):
    _, _, vid, _ = nomination(conn, state="rejected")
    conn.execute("UPDATE paper_versions SET reviewed_revision=NULL,state='extracted' WHERE id=%s", (vid,))
    with pytest.raises(ValueError, match="not independently reviewed"):
        expansion.close_expansion(conn)


def test_generic_oa_fallback_never_inherits_requested_arxiv_version(conn,tmp_path,monkeypatch):
    _,_,vid,_=nomination(conn,target='2307.06945v2')
    conn.execute("UPDATE paper_references SET url='https://example.org/paper.pdf' WHERE version_id=%s",(vid,))
    pdf=tmp_path/'source.pdf';pdf.write_bytes(b'synthetic already-validated PDF')
    sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
    def download(url,directory):
        if 'arxiv.org' in url:
            raise urllib.error.HTTPError(url,404,'missing',{},None)
        return pdf,sha,{},'https://example.org/paper.pdf'
    monkeypatch.setattr(expansion,'_download_pdf',download)
    result=expansion.expand(conn)['results'][0]
    row=conn.execute('SELECT version_label,date_verified FROM paper_versions WHERE id=%s',(result['version_id'],)).fetchone()
    assert row=={'version_label':None,'date_verified':False}


def test_actual_versioned_download_url_proves_version_without_stamp(conn,tmp_path,monkeypatch):
    nomination(conn,target='2307.06945v2')
    pdf=tmp_path/'source.pdf';pdf.write_bytes(b'synthetic already-validated PDF')
    sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(expansion,'_download_pdf',lambda *args:(pdf,sha,{},'https://arxiv.org/pdf/2307.06945v2'))
    result=expansion.expand(conn)['results'][0]
    assert conn.execute('SELECT version_label FROM paper_versions WHERE id=%s',(result['version_id'],)).fetchone()['version_label']=='v2'


@pytest.mark.parametrize('final_url',['https://arxiv.org/pdf/2307.06945v3','https://arxiv.org/pdf/2310.05736v2'])
def test_redirected_wrong_source_is_not_admitted(conn,tmp_path,monkeypatch,final_url):
    nomination(conn,target='2307.06945v2')
    pdf=tmp_path/'source.pdf';pdf.write_bytes(b'synthetic already-validated PDF')
    sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(expansion,'_download_pdf',lambda *args:(pdf,sha,{},final_url))
    assert expansion.expand(conn)['results'][0]['state']=='unavailable'
    assert conn.execute('SELECT count(*) AS n FROM papers').fetchone()['n']==1


def test_conflicting_citation_identifiers_do_not_choose_arbitrary_existing_paper(conn,monkeypatch):
    corpus,_,vid,_=nomination(conn)
    conn.execute("INSERT INTO papers(corpus_id,arxiv_id,doi,title,source_depth) VALUES(%s,'2307.06945','10.1234/a','A',0)",(corpus,))
    conn.execute("INSERT INTO papers(corpus_id,arxiv_id,doi,title,source_depth) VALUES(%s,'2310.05736','10.1234/b','B',0)",(corpus,))
    conn.execute("UPDATE paper_references SET doi='10.1234/b' WHERE version_id=%s",(vid,))
    monkeypatch.setattr(expansion,'_download_pdf',lambda *args:pytest.fail('Conflicting identities must not download'))
    with pytest.raises(ValueError,match='Conflicting citation identifiers'):
        expansion.expand(conn)


def test_approval_requires_actual_current_accepted_review(conn):
    _,_,vid,_=nomination(conn)
    conn.execute('DELETE FROM reviews WHERE version_id=%s',(vid,))
    with pytest.raises(ValueError,match='qualifying review'):
        expansion.expand(conn)
    with pytest.raises(ValueError,match='not independently reviewed'):
        expansion.close_expansion(conn)


def test_original_without_admitted_pdf_blocks_expansion_closure(conn):
    corpus,_,_,_=nomination(conn,state='rejected')
    conn.execute("INSERT INTO papers(corpus_id,title,source_depth) VALUES(%s,'Missing original',0)",(corpus,))
    with pytest.raises(ValueError,match='not independently reviewed'):
        expansion.close_expansion(conn)


@pytest.mark.parametrize('concurrent_change',[None,'revoke','reference'])
def test_download_releases_workflow_locks_and_fences_changed_nomination(conn,tmp_path,monkeypatch,concurrent_change):
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict,make_conninfo
    from paperkb.config import settings
    from paperkb.db import connect

    corpus,_,_,nid=nomination(conn)
    schema=conn.execute('SELECT current_schema() AS name').fetchone()['name']
    conn.commit()
    pdf=tmp_path/'acquired.pdf';pdf.write_bytes(b'already validated source fixture')
    sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
    def download(*args):
        options=conninfo_to_dict(settings()['database_url'])
        options['dbname']=conn.info.dbname
        with connect(make_conninfo(**options)) as other:
            other.execute(sql.SQL('SET search_path TO {},public').format(sql.Identifier(schema)))
            # A worker must still acquire the corpus lock while HTTP is waiting.
            other.execute('SELECT id FROM corpus WHERE id=%s FOR UPDATE NOWAIT',(corpus,))
            if concurrent_change=='revoke':
                other.execute("UPDATE predecessor_nominations SET state='rejected' WHERE id=%s",(nid,))
            elif concurrent_change=='reference':
                other.execute("UPDATE paper_references SET arxiv_id='2310.05736' WHERE id=(SELECT reference_id FROM predecessor_nominations WHERE id=%s)",(nid,))
        return pdf,sha,{},'https://arxiv.org/pdf/2307.06945v1'
    monkeypatch.setattr(expansion,'_download_pdf',download)
    result=expansion.expand(conn)['results'][0]
    if concurrent_change:
        assert result['state']=='superseded'
        assert conn.execute('SELECT count(*) AS n FROM papers').fetchone()['n']==1
        if concurrent_change=='revoke':
            assert conn.execute('SELECT state FROM predecessor_nominations WHERE id=%s',(nid,)).fetchone()['state']=='rejected'
    else:
        assert result['state']=='admitted'
