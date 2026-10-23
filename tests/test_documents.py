"""Document tests use small synthetic PDFs, never require the downloaded corpus."""

import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from paperkb import documents


def test_logical_section_boundaries_preserve_exact_offsets_and_appendix():
    pages=[{'page_number':1,'text':'Title\nAbstract\nSummary\n1 Introduction\nBody\n'},
           {'page_number':2,'text':'2 Method\nDetails\nAppendix A\nProof\n'}]
    sections=documents.section_boundaries(pages)
    assert [s['title'] for s in sections][1:]==['Abstract','1 Introduction','2 Method','Appendix A']
    assert sections[0]['boundary_status']=='fallback'
    assert sections[2]['page_end']==2
    for section in sections[1:]:
        text=pages[section['page_start']-1]['text']
        assert text[section['text_start']:].startswith(section['title'])
    assert sections[-1]['text_end_exclusive']==len(pages[-1]['text'])


def make_pdf(path, texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        contents = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        contents.set_data(f"BT /F1 10 Tf 40 740 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(contents)
    writer.write(path)
    return path


def local_manifest(tmp_path, texts=None, note=None):
    pdf = make_pdf(tmp_path / "2405.16635v1.pdf", texts or ["arXiv:2405.16635v1 [cs.CL] 26 May 2024"])
    sha = documents.sha256_file(pdf)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([{"arxiv_id": "2405.16635", "versioned_id": "2405.16635v3", "title": "A withdrawn prompt-compression paper", "updated": "2026-01-01T00:00:00Z", "published": "2024-05-26T00:00:00Z"}]))
    downloads = tmp_path / "downloads.json"
    downloads.write_text(json.dumps({"2405.16635": {"status": "downloaded", "file": pdf.name, "sha256": sha, "bytes": pdf.stat().st_size, "url": "https://arxiv.org/pdf/2405.16635v1", "note": note}}))
    return metadata, downloads, pdf


def test_arxiv_normalization_and_stamp():
    assert documents.normalize_arxiv("https://arxiv.org/pdf/2405.16635v1.pdf") == ("2405.16635", "v1")
    assert documents.normalize_arxiv("hep-th/9901001v2") == ("hep-th/9901001", "v2")
    assert documents.normalize_arxiv("https://example.org/pdf/2405.16635v1") == (None, None)
    text = "arXiv:2405.16635v1 [cs.CL] 26 May 2024"
    assert documents.printed_provenance(text)["version_label"] == "v1"
    assert documents.printed_provenance(text, "2310.05736") == {}


def test_exact_version_feed_rejects_latest_substitution():
    xml = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2405.16635v3</id><published>2024-05-26T12:00:00Z</published><updated>2026-01-01T12:00:00Z</updated></entry></feed>'''
    with pytest.raises(ValueError, match="exact version"):
        documents.parse_exact_version_feed(xml, "2405.16635", "v1")
    exact = xml.replace(b"v3", b"v1").replace(b"2026-01-01", b"2024-05-26")
    assert documents.parse_exact_version_feed(exact, "2405.16635", "v1") == datetime(2024, 5, 26, 12, tzinfo=timezone.utc)


def test_non_pdf_rejected(tmp_path):
    target = tmp_path / "wrong.pdf"
    target.write_text("<html>temporarily unavailable</html>")
    with pytest.raises(ValueError, match="Not a PDF"):
        documents.validate_pdf(target)


def test_import_uses_actual_withdrawn_version_and_is_idempotent(conn, tmp_path):
    meta, manifest, _ = local_manifest(tmp_path, note="Earlier version; current version withdrawn")
    first = documents.import_corpus(conn, meta, manifest)
    assert first["registered"] == 1 and first["withdrawn"] == 1
    assert documents.import_corpus(conn, meta, manifest)["existing"] == 1
    version = conn.execute("SELECT * FROM paper_versions").fetchone()
    assert version["version_label"] == "v1"
    assert version["available_at"] is None and version["date_verified"] is False
    paper = conn.execute("SELECT * FROM papers").fetchone()
    assert "withdrawn" in paper["withdrawal_note"]
    assert paper["metadata"]["versioned_id"] == "2405.16635v3"


def test_import_checksum_mismatch_rejected(conn, tmp_path):
    meta, manifest, pdf = local_manifest(tmp_path)
    pdf.write_bytes(pdf.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        documents.import_corpus(conn, meta, manifest)


def test_import_rejects_pdf_or_url_identifier_conflicts(conn, tmp_path):
    meta, manifest, _ = local_manifest(tmp_path, texts=["arXiv:2307.06945v1 [cs.CL] 26 May 2024"])
    with pytest.raises(ValueError, match="Printed PDF identifier"):
        documents.import_corpus(conn, meta, manifest)
    meta, manifest, _ = local_manifest(tmp_path)
    downloads=json.loads(manifest.read_text())
    downloads['2405.16635']['url']='https://arxiv.org/pdf/2307.06945v1'
    manifest.write_text(json.dumps(downloads))
    with pytest.raises(ValueError, match="Download URL identifier"):
        documents.import_corpus(conn, meta, manifest)


def test_preparation_covers_blank_appendix_and_resumes(conn, tmp_path, monkeypatch):
    meta, manifest, _ = local_manifest(tmp_path, texts=["arXiv:2405.16635v1 [cs.CL] 26 May 2024 Figure 1 contains results and equation", ""])
    monkeypatch.setattr(documents, "settings", lambda: {"artifact_dir": str(tmp_path / "artifacts")})
    documents.import_corpus(conn, meta, manifest)
    vid = conn.execute("SELECT id FROM paper_versions").fetchone()["id"]
    first = documents.prepare(conn, vid)
    second = documents.prepare(conn, vid)
    assert len(first["pages"]) == 2 and len(first["sections"]) == 2
    assert first["sections"] == second["sections"]
    assert len(second["jobs"]) == 1 and second["jobs"][0]["revision"] == 1
    assert "little_or_no_extractable_text" in first["pages"][1]["flags"]
    assert all(section["page_start"] == section["page_end"] for section in first["sections"])
    assert first["version"]["preparation"]["complete"] is True
    assert first["pages"][0]["text_sha256"] == hashlib.sha256(first["pages"][0]["text"].encode()).hexdigest()
    assert conn.execute("SELECT count(*) AS n FROM graph_parents").fetchone()["n"] == 0
    rendered = documents.render_page(conn, vid, 1)
    assert rendered["path"].endswith("page-0001.png")
    with pytest.raises(ValueError, match="exceeds"):
        documents.render_page(conn, vid, 3)


def test_frozen_corpus_blocks_preparation(conn, tmp_path):
    meta, manifest, _ = local_manifest(tmp_path)
    documents.import_corpus(conn, meta, manifest)
    vid = conn.execute("SELECT id FROM paper_versions").fetchone()["id"]
    conn.execute("UPDATE corpus SET state='frozen'")
    with pytest.raises(ValueError, match="frozen"):
        documents.prepare(conn, vid)


def test_parser_failure_recovers_text_with_audited_visual_fallback(conn, tmp_path, monkeypatch):
    from pypdf import PageObject
    meta, manifest, _ = local_manifest(tmp_path)
    def fail(*args,**kwargs):
        raise ValueError('Broken primary font mapping')
    monkeypatch.setattr(PageObject,'extract_text',fail)
    monkeypatch.setattr(documents,'settings',lambda:{'artifact_dir':str(tmp_path/'artifacts')})
    documents.import_corpus(conn,meta,manifest)
    version=conn.execute('SELECT id,version_label FROM paper_versions').fetchone()
    assert version['version_label']=='v1'
    packet=documents.prepare(conn,version['id'])
    page=packet['pages'][0]
    assert 'arXiv:2405.16635v1' in page['text']
    assert 'text_extracted_with_pdfium' in page['flags']
    assert 'visual_review_required' in page['flags']
    provenance=packet['version']['preparation']['fallback_text_pages']['1']
    assert provenance['engine']=='pdfium' and 'font mapping' in provenance['primary_error']


@pytest.mark.parametrize('invalid', ['Math \ud835 value', 'Emoji \ud83d\ude00 value'])
def test_invalid_primary_unicode_recovers_text_and_retains_diagnostics(conn,tmp_path,monkeypatch,invalid):
    from pypdf import PageObject
    meta,manifest,_=local_manifest(tmp_path)
    monkeypatch.setattr(PageObject,'extract_text',lambda *args,**kwargs:invalid)
    monkeypatch.setattr(documents,'settings',lambda:{'artifact_dir':str(tmp_path/'artifacts')})
    documents.import_corpus(conn,meta,manifest)
    version=conn.execute('SELECT id,version_label FROM paper_versions').fetchone()
    assert version['version_label']=='v1'
    packet=documents.prepare(conn,version['id'])
    page=packet['pages'][0]
    assert 'arXiv:2405.16635v1' in page['text']
    assert page['text_sha256']==hashlib.sha256(page['text'].encode()).hexdigest()
    assert 'text_extracted_with_pdfium' in page['flags']
    assert 'visual_review_required' in page['flags']
    provenance=packet['version']['preparation']['fallback_text_pages']['1']
    assert 'UnicodeEncodeError' in provenance['primary_error']
    assert json.loads(Path(packet['version']['preparation']['packet_path']).read_text())['pages'][0]['text']==page['text']


def test_invalid_unicode_from_both_parsers_remains_explicitly_unavailable(tmp_path,monkeypatch):
    import pypdfium2 as pdfium
    from pypdf import PageObject
    pdf=make_pdf(tmp_path/'source.pdf',['Readable rendered text'])
    monkeypatch.setattr(PageObject,'extract_text',lambda *args,**kwargs:'Bad \ud835')
    monkeypatch.setattr(pdfium.PdfTextPage,'get_text_bounded',lambda *args,**kwargs:'Also bad \ud835')
    result=documents.extract_page_text(documents.validate_pdf(pdf).pages[0],pdf,1)
    assert result['engine']=='unavailable' and result['text']==''
    assert 'UnicodeEncodeError' in result['primary_error']
    assert 'UnicodeEncodeError' in result['fallback_error']


def test_exact_date_enrichment_preserves_evidence(conn, tmp_path, monkeypatch):
    meta, manifest, _ = local_manifest(tmp_path)
    documents.import_corpus(conn, meta, manifest)
    vid = conn.execute("SELECT id FROM paper_versions").fetchone()["id"]
    monkeypatch.setattr(documents, "settings", lambda: {"artifact_dir": str(tmp_path / "artifacts")})
    xml = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2405.16635v1</id><updated>2024-05-26T12:01:02Z</updated></entry></feed>'
    monkeypatch.setattr(documents.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(xml))
    result = documents.verify_version_date(conn, vid)
    assert result["date_verified"]
    row = conn.execute("SELECT * FROM paper_versions WHERE id=%s", (vid,)).fetchone()
    assert row["available_at"].year == 2024
    assert row["preparation"]["date_verification"]["response_sha256"] == hashlib.sha256(xml).hexdigest()


def test_date_verification_can_recover_unknown_version_from_actual_pdf(conn,tmp_path,monkeypatch):
    meta,manifest,_=local_manifest(tmp_path)
    documents.import_corpus(conn,meta,manifest)
    vid=conn.execute('SELECT id FROM paper_versions').fetchone()['id']
    conn.execute('UPDATE paper_versions SET version_label=NULL WHERE id=%s',(vid,))
    xml=b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2405.16635v1</id><updated>2024-05-26T12:01:02Z</updated></entry></feed>'
    monkeypatch.setattr(documents.urllib.request,'urlopen',lambda *args,**kwargs:io.BytesIO(xml))
    monkeypatch.setattr(documents.time,'sleep',lambda _:None)
    assert documents.verify_version_date(conn,vid)['date_verified']
    row=conn.execute('SELECT version_label,preparation FROM paper_versions WHERE id=%s',(vid,)).fetchone()
    assert row['version_label']=='v1'
    assert row['preparation']['provenance_recovered_from_checksum_verified_pdf']


@pytest.mark.parametrize('batch', [False, True])
def test_date_network_wait_does_not_lock_workers_or_overwrite_their_preparation(conn,tmp_path,monkeypatch,batch):
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict,make_conninfo
    from paperkb.db import connect
    from paperkb.config import settings
    meta,manifest,_=local_manifest(tmp_path)
    documents.import_corpus(conn,meta,manifest)
    vid=conn.execute('SELECT id FROM paper_versions').fetchone()['id']
    schema=conn.execute('SELECT current_schema() AS name').fetchone()['name']
    conn.commit()
    options=conninfo_to_dict(settings()['database_url'])
    options['dbname']=conn.info.dbname
    xml=b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2405.16635v1</id><updated>2024-05-26T12:01:02Z</updated></entry></feed>'
    def fetch(*args,**kwargs):
        with connect(make_conninfo(**options)) as worker:
            worker.execute(sql.SQL('SET search_path TO {},public').format(sql.Identifier(schema)))
            worker.execute('SELECT id FROM corpus FOR UPDATE NOWAIT').fetchall()
            worker.execute("UPDATE paper_versions SET preparation=preparation||'{\"concurrent_worker\":\"preserved\"}'::jsonb WHERE id=%s",(vid,))
        return io.BytesIO(xml)
    monkeypatch.setattr(documents.urllib.request,'urlopen',fetch)
    monkeypatch.setattr(documents.time,'sleep',lambda _:None)
    monkeypatch.setattr(documents,'settings',lambda:{'artifact_dir':str(tmp_path/'artifacts')})
    if batch:
        assert documents.verify_version_dates(conn,20)['verified']==1
    else:
        assert documents.verify_version_date(conn,vid)['date_verified']
    preparation=conn.execute('SELECT preparation FROM paper_versions WHERE id=%s',(vid,)).fetchone()['preparation']
    assert preparation['concurrent_worker']=='preserved'
    assert preparation['date_verification']['response_sha256']==hashlib.sha256(xml).hexdigest()


@pytest.mark.parametrize('second_version,verified', [('v2', 2), ('v3', 1)])
def test_batch_dates_match_each_exact_version_in_one_request(conn,tmp_path,monkeypatch,second_version,verified):
    from urllib.parse import parse_qs,urlparse
    meta,manifest,_=local_manifest(tmp_path)
    documents.import_corpus(conn,meta,manifest)
    corpus=conn.execute('SELECT id FROM corpus').fetchone()['id']
    paper=conn.execute("INSERT INTO papers(corpus_id,arxiv_id,title,source_depth) VALUES(%s,'2404.12345','Second',0) RETURNING id",(corpus,)).fetchone()['id']
    vid=conn.execute("INSERT INTO paper_versions(paper_id,sha256,file_path,version_label) VALUES(%s,%s,'unused.pdf','v2') RETURNING id",(paper,'2'*64)).fetchone()['id']
    xml=(f'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2404.12345{second_version}</id><updated>2024-06-02T00:00:00Z</updated></entry>'
         '<entry><id>http://arxiv.org/abs/2405.16635v1</id><updated>2024-05-26T12:01:02Z</updated></entry></feed>').encode()
    calls=[]
    def fetch(request,**kwargs):
        calls.append(parse_qs(urlparse(request.full_url).query))
        return io.BytesIO(xml)
    monkeypatch.setattr(documents.urllib.request,'urlopen',fetch)
    monkeypatch.setattr(documents.time,'sleep',lambda _:None)
    monkeypatch.setattr(documents,'settings',lambda:{'artifact_dir':str(tmp_path/'artifacts')})
    result=documents.verify_version_dates(conn,20)
    assert result['requested']==2 and result['verified']==verified and result['failed']==2-verified
    assert len(calls)==1 and calls[0]['max_results']==['2']
    assert set(calls[0]['id_list'][0].split(','))=={'2404.12345v2','2405.16635v1'}
    rows=conn.execute('SELECT id,date_verified,preparation FROM paper_versions').fetchall()
    for row in rows:
        if str(row['id'])==str(vid) and second_version=='v3':
            assert not row['date_verified'] and 'date_verification' not in row['preparation']
        else:
            proof=row['preparation']['date_verification']
            assert proof['response_sha256']==hashlib.sha256(xml).hexdigest()
            assert Path(proof['artifact']).read_bytes()==xml


def test_batch_dates_skip_unknown_versions_and_stop_on_rate_limit(conn,tmp_path,monkeypatch):
    import urllib.error
    meta,manifest,_=local_manifest(tmp_path)
    documents.import_corpus(conn,meta,manifest)
    def limited(*args,**kwargs):
        raise urllib.error.HTTPError('https://export.arxiv.org/api/query',429,'rate limit',{},None)
    monkeypatch.setattr(documents.urllib.request,'urlopen',limited)
    monkeypatch.setattr(documents.time,'sleep',lambda _:None)
    with pytest.raises(urllib.error.HTTPError):
        documents.verify_version_dates(conn)
    assert not conn.execute('SELECT date_verified FROM paper_versions').fetchone()['date_verified']
    conn.execute('UPDATE paper_versions SET version_label=NULL')
    assert documents.verify_version_dates(conn)['requested']==0


def test_batch_dates_recheck_corpus_freeze_after_fetch(conn,tmp_path,monkeypatch):
    meta,manifest,_=local_manifest(tmp_path)
    documents.import_corpus(conn,meta,manifest)
    xml=b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2405.16635v1</id><updated>2024-05-26T12:01:02Z</updated></entry></feed>'
    def fetch(*args,**kwargs):
        conn.execute("UPDATE corpus SET state='frozen'")
        return io.BytesIO(xml)
    monkeypatch.setattr(documents.urllib.request,'urlopen',fetch)
    monkeypatch.setattr(documents.time,'sleep',lambda _:None)
    result=documents.verify_version_dates(conn)
    assert result['verified']==0 and result['failed']==1
    assert 'frozen' in result['results'][0]['error']
    assert not conn.execute('SELECT date_verified FROM paper_versions').fetchone()['date_verified']


@pytest.mark.parametrize('case', ['match','parser_fallback','different_bytes','redirected_version','conflicting_stamp','frozen'])
def test_unknown_version_recovery_requires_actual_checksum_and_exact_identity(conn,tmp_path,monkeypatch,case):
    from paperkb import expansion
    meta,manifest,pdf=local_manifest(tmp_path)
    documents.import_corpus(conn,meta,manifest)
    vid=conn.execute('SELECT id FROM paper_versions').fetchone()['id']
    conn.execute('UPDATE paper_versions SET version_label=NULL')
    digest=documents.sha256_file(pdf)
    calls=[]
    def download(url,directory):
        calls.append(url)
        conn.execute("UPDATE paper_versions SET preparation=preparation||'{\"concurrent_worker\":true}'::jsonb")
        if case=='frozen':
            conn.execute("UPDATE corpus SET state='frozen'")
        if case=='different_bytes':
            other=make_pdf(tmp_path/'different.pdf',['Different revision'])
            return other,documents.sha256_file(other),{},url
        if case=='redirected_version':
            return pdf,digest,{},url[:-1]+'2'
        if case=='conflicting_stamp':
            return pdf,digest,{'arxiv_id':'2405.16635','version_label':'v2'},url
        if case=='parser_fallback':
            return pdf,digest,{'text_extraction':{'engine':'pdfium','primary_error':'Font mapping failed'}},url
        return pdf,digest,{},url
    monkeypatch.setattr(expansion,'_download_pdf',download)
    monkeypatch.setattr(documents,'settings',lambda:{'artifact_dir':str(tmp_path/'artifacts')})
    if case in ('match','parser_fallback'):
        result=documents.recover_version(conn,vid,'v1')
        assert result['recovered'] and result['proof']['sha256']==digest
        row=conn.execute('SELECT * FROM paper_versions').fetchone()
        assert row['version_label']=='v1' and not row['date_verified']
        assert row['preparation']['concurrent_worker']
        if case=='parser_fallback':
            assert result['proof']['download_provenance']['text_extraction']['engine']=='pdfium'
        audit=json.loads(conn.execute("SELECT reason FROM corpus_events WHERE action='recover_version'").fetchone()['reason'])
        assert audit['proof']['final_url']=='https://export.arxiv.org/pdf/2405.16635v1'
        assert documents.recover_version(conn,vid,'v1')['already_known']
        assert len(calls)==1
    else:
        with pytest.raises(ValueError):
            documents.recover_version(conn,vid,'v1')
        assert conn.execute('SELECT version_label FROM paper_versions').fetchone()['version_label'] is None
        assert conn.execute("SELECT count(*) AS n FROM corpus_events WHERE action='recover_version'").fetchone()['n']==0


def test_packet_contains_review_and_correction_context(conn, tmp_path, monkeypatch):
    meta, manifest, _ = local_manifest(tmp_path)
    monkeypatch.setattr(documents, "settings", lambda: {"artifact_dir": str(tmp_path / "artifacts")})
    documents.import_corpus(conn, meta, manifest)
    vid = conn.execute("SELECT id FROM paper_versions").fetchone()["id"]
    prepared = documents.prepare(conn, vid)
    idea_id = uuid.uuid4()
    conn.execute("INSERT INTO ideas(id,version_id,local_id,revision,kind,title,statement,attribution,confidence,embedding_text) VALUES(%s,%s,'method',1,'method','Method','Statement','original',0.9,'Statement')", (idea_id, vid))
    conn.execute("INSERT INTO idea_evidence(idea_id,page_number,quote,evidence_kind) VALUES(%s,1,'source text','text')", (idea_id,))
    ref = conn.execute("INSERT INTO paper_references(version_id,citation_key,title,arxiv_id) VALUES(%s,'Smith2024','Earlier method','2307.06945') RETURNING id", (vid,)).fetchone()["id"]
    conn.execute("INSERT INTO predecessor_nominations(idea_id,reference_id,evidence_page,evidence_quote) VALUES(%s,%s,1,'source text')", (idea_id, ref))
    conn.execute("INSERT INTO reviews(version_id,revision,reviewer_id,accepted,payload) VALUES(%s,1,'other-agent',false,'{\"issues\":[\"check equation\"]}')", (vid,))
    conn.execute("INSERT INTO submissions(job_id,payload_hash,payload) VALUES(%s,'hash','{\"ideas\":[]}')", (prepared["jobs"][0]["id"],))
    result = documents.packet(conn, vid)
    assert result["ideas"][0]["evidence"][0]["quote"] == "source text"
    assert result["references"][0]["citation_key"] == "Smith2024"
    assert result["nominations"][0]["idea_local_id"] == "method"
    assert result["nominations"][0]["citation_key"] == "Smith2024"
    assert result["reviews"][0]["payload"]["issues"] == ["check equation"]
    assert result["latest_submission"]["revision"] == 1
