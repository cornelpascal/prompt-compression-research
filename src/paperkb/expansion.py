"""Reviewer-approved, one-hop open-access predecessor acquisition.

No extraction is performed here. Downloaded PDFs receive immutable provenance,
then the orchestrator prepares them and assigns a fresh extraction worker.
"""

from __future__ import annotations

import ipaddress
import http.client
import json
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import Field, model_validator

from .config import ROOT
from .documents import NAMESPACE, USER_AGENT, _open_corpus, extract_page_text, normalize_arxiv, normalize_doi, printed_provenance, sha256_file, validate_pdf
from .models import Payload, normalized_text

MAX_PDF_BYTES = 256 * 1024 * 1024
REQUEST_INTERVAL = 3.1
_last_request = 0.0


class ReferenceSource(Payload):
    resolver_id: str = Field(min_length=1)
    matched_title: str = Field(min_length=1)
    metadata_url: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    availability: Literal['available','unavailable'] = 'available'
    checked_urls: list[str] = Field(default_factory=list)
    arxiv_id: str | None = None
    doi: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def source_required(self):
        if self.availability=='available' and not self.arxiv_id and not self.url:
            raise ValueError('Provide a verified arXiv identifier or direct full-text URL')
        if self.availability=='unavailable' and not self.checked_urls:
            raise ValueError('A source gap requires an audit of the URLs checked for full text')
        if self.arxiv_id and not normalize_arxiv(self.arxiv_id)[0]:
            raise ValueError('Invalid arXiv identifier')
        if self.doi and not normalize_doi(self.doi):
            raise ValueError('Invalid DOI')
        return self


def resolve_reference(conn, reference_id, payload):
    """Attach checked retrieval metadata without changing reviewed source claims.

    The trusted orchestrator verifies the linked primary metadata. Existing
    scientific review and relevance requirements remain mandatory for admission.
    Original bibliography/submission data and the resolution audit are retained.
    """
    parsed = ReferenceSource.model_validate(payload)
    _safe_url(parsed.metadata_url)
    if parsed.url:
        _safe_url(parsed.url)
    for url in parsed.checked_urls:
        _safe_url(url)
    with conn.transaction():
        reference = conn.execute('''SELECT r.*,p.corpus_id,p.source_depth,v.extraction_revision,
            v.reviewed_revision,v.reviewer_id,v.extractor_id FROM paper_references r
            JOIN paper_versions v ON v.id=r.version_id JOIN papers p ON p.id=v.paper_id
            WHERE r.id=%s''',(reference_id,)).fetchone()
        if not reference:
            raise ValueError('Unknown reference')
        _open_corpus(conn,reference['corpus_id'],expansion=True)
        # Re-read after the corpus lock: corrections may have replaced references.
        current=conn.execute('SELECT * FROM paper_references WHERE id=%s FOR UPDATE',(reference_id,)).fetchone()
        if not current or any(current[key]!=reference[key] for key in ('version_id','title','arxiv_id','doi','url')):
            raise ValueError('Reference changed while resolving its source; refresh the packet')
        version=conn.execute('SELECT * FROM paper_versions WHERE id=%s',(reference['version_id'],)).fetchone()
        if (reference['source_depth']!=0 or version['reviewed_revision']!=version['extraction_revision']
                or not version['reviewer_id'] or version['reviewer_id']==version['extractor_id']
                or not conn.execute('''SELECT 1 FROM reviews WHERE version_id=%s AND revision=%s
                    AND reviewer_id=%s AND accepted''',(version['id'],version['extraction_revision'],version['reviewer_id'])).fetchone()):
            raise ValueError('Reference source resolution requires a current independent original-paper review')
        title_key=lambda title:re.sub(r'[^\w]','',normalized_text(title).casefold())
        if title_key(parsed.matched_title)!=title_key(reference['title']):
            raise ValueError('Verified metadata title does not match the reviewed reference')
        old_arxiv,old_version=_reference_identity(reference)
        supplied_arxiv,supplied_version=_reference_identity(parsed.model_dump())
        old_doi,new_doi=normalize_doi(reference['doi']),normalize_doi(parsed.doi)
        if ((old_arxiv and supplied_arxiv and old_arxiv!=supplied_arxiv)
                or (old_version and supplied_version and old_version!=supplied_version)
                or (old_doi and new_doi and old_doi!=new_doi)):
            raise ValueError('Source resolution cannot replace a conflicting existing identifier')
        arxiv_id=supplied_arxiv or old_arxiv
        version_label=supplied_version or old_version
        arxiv_value=arxiv_id+(version_label or '') if arxiv_id else None
        updated={'arxiv_id':arxiv_value,'doi':new_doi or old_doi,'url':parsed.url or reference['url']}
        _reference_identity(updated)
        nominations=conn.execute('''SELECT n.*,i.relevance_score,i.active FROM predecessor_nominations n
            JOIN ideas i ON i.id=n.idea_id WHERE n.reference_id=%s''',(reference_id,)).fetchall()
        eligible=[n for n in nominations if n['state'] in ('approved','unavailable') and n['active']
                  and n['relevance_score'] is not None and n['relevance_score']>=50
                  and n['reviewer_id']==version['reviewer_id']]
        if not eligible:
            raise ValueError('No unresolved independently approved qualifying nomination uses this reference')
        if any(n['target_paper_id'] for n in nominations):
            raise ValueError('Already resolved predecessors cannot be retargeted')
        conn.execute('UPDATE paper_references SET arxiv_id=%s,doi=%s,url=%s WHERE id=%s',
                     (updated['arxiv_id'],updated['doi'],updated['url'],reference_id))
        audit={'reference_id':str(reference_id),'version_id':str(version['id']),
               'extraction_revision':version['extraction_revision'],'citation_key':reference['citation_key'],
               'original_identifiers':{k:reference[k] for k in ('arxiv_id','doi','url')},
               'resolved_identifiers':updated,'verification':parsed.model_dump(),
               'prior_nomination_states':[{'id':str(n['id']),'state':n['state'],'reason':n['reason']} for n in eligible]}
        event=conn.execute('''INSERT INTO corpus_events(corpus_id,action,reason)
            VALUES(%s,'resolve_reference',%s) RETURNING id''',
            (reference['corpus_id'],json.dumps(audit,ensure_ascii=False))).fetchone()
        for nomination in eligible:
            if parsed.availability=='unavailable':
                _mark(conn,nomination['id'],'unavailable',parsed.reason)
            elif nomination['state']=='unavailable':
                _mark(conn,nomination['id'],'approved','Verified full-text source attached; retry acquisition')
        return {'reference_id':str(reference_id),'audit_event_id':str(event['id']),
                'identifiers':updated,'eligible_nominations':len(eligible),
                'availability':parsed.availability,'admitted':False}


def _safe_url(url):
    """Reject local/private endpoints and non-web URLs, including redirects."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTPS open-access source URLs are allowed")
    if parsed.port not in (None, 443):
        raise ValueError("Nonstandard network ports are not supported")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Private/local source URLs are not allowed")
    for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(item[4][0])
        if not ip.is_global:
            raise ValueError("Source URL resolves to a non-public address")
    return url


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _reference_identity(reference):
    arxiv_id, version = normalize_arxiv(reference.get("arxiv_id"))
    url_id, url_version = normalize_arxiv(reference.get("url"))
    if ((arxiv_id and url_id and arxiv_id != url_id)
            or (version and url_version and version != url_version)):
        raise ValueError("Conflicting citation identifiers or versions in the arXiv field and URL")
    return arxiv_id or url_id, version or url_version


def _conflicting_identifiers(arxiv_id, doi, paper):
    return ((arxiv_id and paper["arxiv_id"] and arxiv_id != paper["arxiv_id"])
            or (doi and paper["doi"] and doi != normalize_doi(paper["doi"])))


def source_urls(reference):
    arxiv_id, version = _reference_identity(reference)
    urls = []
    if arxiv_id:
        urls.append(f"https://arxiv.org/pdf/{arxiv_id}{version or ''}")
    url = reference.get("url")
    if url:
        url_id, url_version = normalize_arxiv(url)
        if not url_id:
            # The worker supplies a direct legitimate OA URL. No login, cookie,
            # paywall bypass, scraping, or DOI-to-publisher access is attempted.
            urls.append(url)
        elif (url not in urls and url_version == version
              and urllib.parse.urlparse(url).path.startswith('/pdf/')):
            # Retain a verified arXiv PDF mirror/URL variant. An unversioned
            # fallback must never replace a specifically requested version.
            urls.append(url)
    return urls


def _download_pdf(url, directory):
    global _last_request
    _safe_url(url)
    # A new CLI process must also pause after the preceding process released
    # the database acquisition lock; its process-local clock has no history.
    time.sleep(REQUEST_INTERVAL if not _last_request else max(0, REQUEST_INTERVAL - (time.monotonic() - _last_request)))
    _last_request = time.monotonic()
    opener = urllib.request.build_opener(_SafeRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf, */*;q=0.8"})
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as output:
        temporary = Path(output.name)
        try:
            with opener.open(request, timeout=120) as response:
                _safe_url(response.geturl())
                if response.status == 206:
                    raise ConnectionError("Server returned a partial PDF response to a full-file request")
                declared_size = int(response.headers.get("Content-Length", "0"))
                if declared_size > MAX_PDF_BYTES:
                    raise ValueError("PDF exceeds the 256 MiB acquisition limit")
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise ValueError("PDF exceeds the 256 MiB acquisition limit")
                    output.write(chunk)
                final_url = response.geturl()
                if declared_size and total != declared_size:
                    raise ConnectionError(f"Incomplete PDF transfer: received {total} of {declared_size} bytes")
            output.flush()
            try:
                reader = validate_pdf(temporary)
            except ValueError as exc:
                if str(exc).startswith("Incomplete PDF (no end marker)"):
                    raise ConnectionError("Incomplete PDF transfer: missing end marker") from exc
                raise
            extracted=extract_page_text(reader.pages[0],temporary,1)
            stamp=printed_provenance(extracted['text'])
            if extracted['engine']!='pypdf':
                stamp['text_extraction']={key:value for key,value in extracted.items() if key!='text'}
            sha = sha256_file(temporary)
            target = directory / f"{sha}.pdf"
            if target.exists():
                if sha256_file(target) != sha:
                    raise ValueError("Existing predecessor PDF checksum mismatch")
                temporary.unlink()
            else:
                temporary.replace(target)
            return target, sha, stamp, final_url
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _mark(conn, nomination_id, state, reason, paper_id=None):
    conn.execute(
        "UPDATE predecessor_nominations SET state=%s,reason=%s,target_paper_id=%s WHERE id=%s",
        (state, reason, paper_id, nomination_id),
    )


def _approved_nominations(conn, limit=None, *, states=('approved',), nomination_id=None):
    params = [list(states)]
    restriction = ''
    if nomination_id is not None:
        restriction = ' AND n.id=%s'
        params.append(nomination_id)
    if limit is not None:
        params.append(int(limit))
    return conn.execute(
        "SELECT n.*,r.citation_key AS reference_citation_key,r.title AS reference_title,r.arxiv_id AS reference_arxiv_id,r.doi AS reference_doi,r.url AS reference_url,"
        "r.version_id AS reference_version_id,i.version_id,i.active,i.relevance_score,i.revision,"
        "v.reviewed_revision,v.extraction_revision,v.extractor_id,v.reviewer_id AS current_reviewer_id,"
        "v.state AS version_state,p.source_depth,p.corpus_id "
        "FROM predecessor_nominations n JOIN ideas i ON i.id=n.idea_id "
        "JOIN paper_references r ON r.id=n.reference_id JOIN paper_versions v ON v.id=i.version_id "
        "JOIN papers p ON p.id=v.paper_id WHERE n.state=ANY(%s)" + restriction + " ORDER BY n.id"
        + (" LIMIT %s" if limit is not None else ""),
        params,
    ).fetchall()


def _validate_nomination(conn, nomination):
    if (not nomination["active"] or nomination["relevance_score"] is None or nomination["relevance_score"] < 50
            or nomination["revision"] != nomination["extraction_revision"]
            or nomination["reviewed_revision"] != nomination["extraction_revision"]
            or nomination["version_state"] not in {"reviewed", "complete"}
            or nomination["reference_version_id"] != nomination["version_id"]
            or not nomination["reviewer_id"] or nomination["reviewer_id"] == nomination["extractor_id"]
            or nomination["reviewer_id"] != nomination["current_reviewer_id"]
            or not conn.execute("SELECT 1 FROM reviews WHERE version_id=%s AND revision=%s AND accepted AND reviewer_id=%s",
                (nomination["version_id"], nomination["extraction_revision"], nomination["reviewer_id"])).fetchone()):
        raise ValueError(f"Nomination {nomination['id']} is not supported by a current independent qualifying review")


def _acquire_source(urls, directory):
    errors, transient, downloaded = [], False, None
    for url in urls:
        try:
            downloaded = _download_pdf(url, directory)
            break
        except urllib.error.HTTPError as exc:
            errors.append(f"HTTP {exc.code}: {url}")
            transient |= exc.code in {408, 429} or exc.code >= 500
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, ConnectionError, OSError) as exc:
            errors.append(f"Temporary download failure: {type(exc).__name__}: {exc}")
            transient = True
        except ValueError as exc:
            errors.append(f"Source rejected: {exc}")
    return errors, transient, downloaded


def expand(conn, limit=None):
    """Acquire approved sources without holding worker workflow row locks.

    A transaction advisory lock serializes acquisition and request pacing. All
    network reads precede the short corpus-locked admission phase, which rechecks
    exact nomination snapshots so a concurrent correction cannot admit stale work.
    """
    if limit is not None and int(limit) < 1:
        raise ValueError("Expansion limit must be positive")
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (78128472101,))
    snapshots = _approved_nominations(conn, limit)
    directory = Path(ROOT) / "papers" / "predecessors"
    acquired, cache = {}, {}
    for nomination in snapshots:
        corpus = conn.execute("SELECT state,expansion_closed FROM corpus WHERE id=%s",
                              (nomination["corpus_id"],)).fetchone()
        if corpus["state"] != "open" or corpus["expansion_closed"]:
            raise ValueError("Corpus is frozen or citation expansion is closed")
        if nomination["source_depth"] != 0:
            continue
        _validate_nomination(conn, nomination)
        reference = dict(title=nomination["reference_title"], arxiv_id=nomination["reference_arxiv_id"],
                         doi=nomination["reference_doi"], url=nomination["reference_url"])
        arxiv_id, _ = _reference_identity(reference)
        doi = normalize_doi(reference["doi"])
        # Existing identities are resolved and conflict-checked under locks below.
        if conn.execute("SELECT 1 FROM papers WHERE arxiv_id=%s OR lower(doi)=%s",
                        (arxiv_id, doi)).fetchone():
            continue
        urls = source_urls(reference)
        if urls:
            key = tuple(urls)
            if key not in cache:
                cache[key] = _acquire_source(urls, directory)
            acquired[nomination["id"]] = cache[key]

    conn.execute("SELECT id FROM corpus ORDER BY id FOR UPDATE").fetchall()
    current = {row["id"]: row for row in _approved_nominations(conn)}
    nominations, results = [], []
    for snapshot in snapshots:
        nomination = current.get(snapshot["id"])
        # Embedding completion may change reviewed -> complete without changing
        # the reviewed dependency. Every other input must remain identical.
        comparable = lambda row: {key:value for key,value in row.items() if key != "version_state"}
        if nomination is None or comparable(nomination) != comparable(snapshot):
            results.append({"nomination_id":str(snapshot["id"]), "state":"superseded",
                            "reason":"Nomination changed during acquisition; refresh before retrying"})
            continue
        nominations.append(nomination)
    for nomination in nominations:
        corpus_id = nomination["corpus_id"]
        _open_corpus(conn, corpus_id, expansion=True)
        # A stored approval is not enough if extraction/review has since changed.
        if nomination["source_depth"] != 0:
            _mark(conn, nomination["id"], "rejected", "One-hop limit: added predecessors cannot nominate further papers")
            results.append({"nomination_id": str(nomination["id"]), "state": "rejected"})
            continue
        _validate_nomination(conn, nomination)
        reference = dict(title=nomination["reference_title"], arxiv_id=nomination["reference_arxiv_id"], doi=nomination["reference_doi"], url=nomination["reference_url"])
        arxiv_id, requested_version = _reference_identity(reference)
        doi = normalize_doi(reference["doi"])
        matches = conn.execute(
            "SELECT id,corpus_id,arxiv_id,doi FROM papers WHERE arxiv_id=%s OR lower(doi)=%s ORDER BY id",
            (arxiv_id, doi),
        ).fetchall()
        if len(matches) > 1:
            raise ValueError("Conflicting citation identifiers resolve to different predecessor papers")
        existing = matches[0] if matches else None
        if existing:
            if _conflicting_identifiers(arxiv_id, doi, existing):
                raise ValueError("Conflicting citation identifiers disagree with the existing predecessor")
            if existing["corpus_id"] != corpus_id:
                raise ValueError("Predecessor exists in a different corpus; cross-corpus admission requires an explicit import")
            _mark(conn, nomination["id"], "existing", "Already admitted to this corpus", existing["id"])
            results.append({"nomination_id": str(nomination["id"]), "state": "existing", "paper_id": str(existing["id"])})
            continue
        urls = source_urls(reference)
        if not urls:
            reason='Primary metadata and open-access full-text lookup required; no source URL has been resolved'
            _mark(conn,nomination['id'],'approved',reason)
            results.append({'nomination_id':str(nomination['id']),'reference_id':str(nomination['reference_id']),
                            'state':'approved','needs_source_resolution':True,'reason':reason})
            continue
        acquisition = acquired.get(nomination["id"])
        if acquisition is None:
            results.append({"nomination_id":str(nomination["id"]), "state":"approved",
                            "reason":"Source availability changed during acquisition; retry"})
            continue
        errors, transient, downloaded = acquisition
        if downloaded is None:
            reason = "; ".join(errors) or "No arXiv identifier or direct open-access PDF URL was provided"
            state = "approved" if transient else "unavailable"
            _mark(conn, nomination["id"], state, reason)
            results.append({"nomination_id": str(nomination["id"]), "state": state, "reason": reason})
            continue
        path, sha, stamp, final_url = downloaded
        final_arxiv, final_version = normalize_arxiv(final_url)
        observed_arxiv = stamp.get("arxiv_id") or final_arxiv
        if ((arxiv_id and observed_arxiv and observed_arxiv != arxiv_id)
                or (stamp.get("arxiv_id") and final_arxiv and stamp["arxiv_id"] != final_arxiv)):
            _mark(conn, nomination["id"], "unavailable", "Downloaded PDF stamp or final URL does not match nominated identifier")
            results.append({"nomination_id": str(nomination["id"]), "state": "unavailable"})
            continue
        observed_version = stamp.get("version_label") or final_version
        if ((requested_version and observed_version and requested_version != observed_version)
                or (stamp.get("version_label") and final_version and stamp["version_label"] != final_version)):
            _mark(conn, nomination["id"], "unavailable", "Downloaded PDF version does not match nominated source version")
            results.append({"nomination_id": str(nomination["id"]), "state": "unavailable"})
            continue
        arxiv_id = arxiv_id or observed_arxiv
        # A requested citation version is not proof of the downloaded bytes.
        # Generic OA fallbacks without an arXiv stamp retain an unknown version.
        version_label = observed_version
        if arxiv_id:
            known_paper = conn.execute("SELECT id,corpus_id,arxiv_id,doi FROM papers WHERE arxiv_id=%s", (arxiv_id,)).fetchone()
            if known_paper:
                if _conflicting_identifiers(arxiv_id, doi, known_paper):
                    raise ValueError("Conflicting citation identifiers disagree with the downloaded predecessor")
                if known_paper["corpus_id"] != corpus_id:
                    raise ValueError("Identified arXiv predecessor is already assigned to another corpus")
                _mark(conn, nomination["id"], "existing", "Downloaded source identifies a paper already admitted", known_paper["id"])
                results.append({"nomination_id": str(nomination["id"]), "state": "existing", "paper_id": str(known_paper["id"])})
                continue
        # The same content may have been nominated through a different URL/DOI.
        known = conn.execute("SELECT v.paper_id,p.corpus_id,p.arxiv_id,p.doi FROM paper_versions v JOIN papers p ON p.id=v.paper_id WHERE v.sha256=%s", (sha,)).fetchone()
        if known:
            if _conflicting_identifiers(arxiv_id, doi, known):
                raise ValueError("Conflicting citation identifiers disagree with the identical admitted PDF")
            if known["corpus_id"] != corpus_id:
                raise ValueError("Downloaded predecessor is already assigned to another corpus")
            _mark(conn, nomination["id"], "existing", "Identical source PDF already admitted", known["paper_id"])
            results.append({"nomination_id": str(nomination["id"]), "state": "existing", "paper_id": str(known["paper_id"])})
            continue
        identity = f"arxiv:{arxiv_id}" if arxiv_id else f"doi:{doi}" if doi else f"pdf:{sha}"
        paper_id = uuid.uuid5(NAMESPACE, identity)
        version_id = uuid.uuid5(NAMESPACE, f"pdf:{sha}")
        metadata = {"reference": reference, "download": {"url": final_url, "sha256": sha, "file": str(path), "downloaded_at": datetime.now(timezone.utc).isoformat()}, "admitted_from_nomination": str(nomination["id"]), "one_hop": True}
        # Corrections replace current nominations. Preserve the precise admitted
        # dependency independently of those mutable workflow rows.
        metadata["admission_evidence"] = {
            "source_version_id": str(nomination["version_id"]),
            "source_extraction_revision": nomination["revision"],
            "source_idea_id": str(nomination["idea_id"]),
            "reference_id": str(nomination["reference_id"]),
            "citation_key": nomination["reference_citation_key"],
            "page_number": nomination["evidence_page"],
            "quote": nomination["evidence_quote"],
            "reviewer_id": nomination["reviewer_id"],
            "relevance_score": nomination["relevance_score"],
            "review_reason": nomination["reason"],
        }
        conn.execute(
            "INSERT INTO papers(id,corpus_id,arxiv_id,doi,title,metadata,source_depth) VALUES(%s,%s,%s,%s,%s,%s,1)",
            (paper_id, corpus_id, arxiv_id, doi, reference["title"], Jsonb(metadata)),
        )
        conn.execute(
            "INSERT INTO paper_versions(id,paper_id,sha256,file_path,version_label,preparation) VALUES(%s,%s,%s,%s,%s,%s)",
            (version_id, paper_id, sha, str(path), version_label, Jsonb({"provenance": stamp})),
        )
        _mark(conn, nomination["id"], "admitted", "Independent review approved this explicit predecessor; full text acquired", paper_id)
        results.append({"nomination_id": str(nomination["id"]), "state": "admitted", "paper_id": str(paper_id), "version_id": str(version_id), "requires_preparation": True})
    return {"processed": len(results), "results": results}


def close_expansion(conn):
    """Close every open corpus only after its original reviews and nominations resolve."""
    corpora = conn.execute("SELECT * FROM corpus ORDER BY created_at FOR UPDATE").fetchall()
    if not corpora:
        raise ValueError("No corpus has been imported")
    changed = []
    # Validate the entire operation before changing any row.
    for corpus in corpora:
        if corpus["expansion_closed"]:
            continue
        _open_corpus(conn, corpus["id"])
        incomplete = conn.execute(
            "SELECT v.id,p.id AS paper_id FROM papers p LEFT JOIN paper_versions v ON v.paper_id=p.id "
            "WHERE p.corpus_id=%s AND p.source_depth=0 AND "
            "(v.id IS NULL OR v.extraction_revision=0 OR v.state NOT IN ('reviewed','complete') "
            "OR v.reviewed_revision IS DISTINCT FROM v.extraction_revision OR v.reviewed_revision IS NULL "
            "OR v.reviewer_id IS NULL OR v.reviewer_id=v.extractor_id OR NOT EXISTS "
            "(SELECT 1 FROM reviews r WHERE r.version_id=v.id AND r.revision=v.extraction_revision "
            "AND r.accepted AND r.reviewer_id=v.reviewer_id)) LIMIT 1", (corpus["id"],),
        ).fetchone()
        if incomplete:
            raise ValueError(f"Original paper {incomplete['id'] or incomplete['paper_id']} is not independently reviewed")
        pending = conn.execute(
            "SELECT n.id FROM predecessor_nominations n JOIN ideas i ON i.id=n.idea_id "
            "JOIN paper_versions v ON v.id=i.version_id JOIN papers p ON p.id=v.paper_id "
            "WHERE p.corpus_id=%s AND n.state IN ('pending','approved') LIMIT 1", (corpus["id"],),
        ).fetchone()
        if pending:
            raise ValueError(f"Unresolved predecessor nomination {pending['id']} blocks expansion closure")
        changed.append(corpus["id"])
    for corpus_id in changed:
        conn.execute("UPDATE corpus SET expansion_closed=true WHERE id=%s", (corpus_id,))
    return {"closed": [str(value) for value in changed], "already_closed": len(corpora) - len(changed)}
