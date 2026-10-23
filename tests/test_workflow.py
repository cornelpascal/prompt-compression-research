import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from pydantic import ValidationError

from paperkb.models import ExtractionPayload, ReviewPayload, normalized_text
from paperkb import workflow
from paperkb.config import settings


def seed(conn, *, source_depth=0, text="Compression removes redundant tokens. Our method extends Prior."):
    corpus = conn.execute("INSERT INTO corpus(name) VALUES(%s) RETURNING id", (str(uuid4()),)).fetchone()["id"]
    paper = conn.execute("INSERT INTO papers(corpus_id,title,source_depth) VALUES(%s,'Example paper',%s) RETURNING id", (corpus,source_depth)).fetchone()["id"]
    version = conn.execute("INSERT INTO paper_versions(paper_id,sha256,file_path,state) VALUES(%s,%s,'fixture.pdf','prepared') RETURNING id", (paper,hashlib.sha256(str(uuid4()).encode()).hexdigest())).fetchone()["id"]
    conn.execute("INSERT INTO pages(version_id,page_number,text,text_sha256) VALUES(%s,1,%s,%s)", (version,text,hashlib.sha256(text.encode()).hexdigest()))
    section = conn.execute("INSERT INTO sections(version_id,ordinal,title,page_start,page_end) VALUES(%s,1,'Page 1',1,1) RETURNING id", (version,)).fetchone()["id"]
    job = conn.execute("INSERT INTO jobs(version_id,kind,revision) VALUES(%s,'extract',1) RETURNING id", (version,)).fetchone()["id"]
    conn.commit()
    return {"corpus":corpus,"paper":paper,"version":version,"section":section,"job":job}


def extraction(seed, *, visual=False, nomination=False, unreadable=False):
    payload = {
        "ideas": [{"local_id":"compression","kind":"method","title":"Token removal",
                   "statement":"Compression removes redundant tokens.","conditions":"", "attribution":"original",
                   "confidence":0.9,"embedding_text":"Token removal. Compression removes redundant tokens.",
                   "evidence":[{"page_number":1,"quote":"Compression removes redundant tokens.",
                                "evidence_kind":"visual" if visual else "text",
                                "locator":{"figure":"1"} if visual else {}}]}],
        "coverage":[{"section_id":str(seed["section"]),"status":"unreadable" if unreadable else "covered","notes":"Reviewed the complete page including visual elements."}],
        "references":[],"predecessors":[],
    }
    if nomination:
        payload["references"]=[{"citation_key":"prior","title":"Prior","arxiv_id":"2301.00001"}]
        payload["predecessors"]=[{"idea_local_id":"compression","citation_key":"prior","evidence_page":1,"evidence_quote":"Our method extends Prior."}]
    return payload


def accepted(*, visual=False, nomination=False, score=95):
    return {"accepted":True,"notes":"Evidence, coverage, attribution and equations checked independently.",
            "paper_relevance":{"score":95,"reason":"Central topic is prompt compression."},
            "idea_relevance":[{"local_id":"compression","score":score,"reason":"Token removal is a compression mechanism."}],
            "issues":[],"nomination_decisions":[{"idea_local_id":"compression","citation_key":"prior","approved":True,"reason":"Explicit method dependency."}] if nomination else [],
            "visual_evidence_confirmed":visual}


def submit_seed(conn, seeded, **kwargs):
    job = workflow.claim(conn,"extractor",version_id=seeded["version"])
    result = workflow.submit(conn,job["id"],"extractor",job["token"],extraction(seeded,**kwargs))
    conn.commit()
    return job,result


def test_models_reject_untrusted_parent_and_duplicate_ids():
    item = extraction({"section":uuid4()})
    item["ideas"][0]["parent_id"] = str(uuid4())
    with pytest.raises(ValidationError):
        ExtractionPayload.model_validate(item)
    del item["ideas"][0]["parent_id"]
    item["ideas"].append(item["ideas"][0].copy())
    with pytest.raises(ValidationError,match="Duplicate"):
        ExtractionPayload.model_validate(item)


def test_blocked_job_can_resume_without_losing_checkpoint_or_attempt(conn):
    item=seed(conn)
    old=workflow.claim(conn,'first',version_id=item['version'])
    workflow.checkpoint(conn,old['id'],'first',old['token'],item['section'],{'remaining':'Inspect recovered page'})
    workflow.fail(conn,old['id'],'first',old['token'],'Source rendering unavailable',retry=False)
    workflow.retry_job(conn,old['id'],'Renderer repaired; source checksum unchanged')
    new=workflow.claim(conn,'replacement',version_id=item['version'])
    assert new['attempts']==2 and new['token']!=old['token']
    assert conn.execute('SELECT count(*) AS n FROM checkpoints WHERE job_id=%s',(old['id'],)).fetchone()['n']==1
    with pytest.raises(workflow.WorkflowError):
        workflow.submit(conn,old['id'],'first',old['token'],extraction(item))


def test_models_reject_nonfinite_confidence_and_coerced_scores():
    item = extraction({"section":uuid4()})
    item["ideas"][0]["confidence"] = float("nan")
    with pytest.raises(ValidationError):
        ExtractionPayload.model_validate(item)
    review = accepted()
    review["idea_relevance"][0]["score"] = True
    with pytest.raises(ValidationError):
        ReviewPayload.model_validate(review)
    assert normalized_text("A\n  ﬁgure") == "A figure"


def test_submission_is_transactional_and_exact_replay_is_idempotent(conn):
    item = seed(conn)
    job = workflow.claim(conn,"extractor",version_id=item["version"])
    bad = extraction(item)
    bad["ideas"][0]["evidence"][0]["quote"] = "This claim is fabricated."
    with pytest.raises(workflow.WorkflowError,match="quote"):
        workflow.submit(conn,job["id"],"extractor",job["token"],bad)
    assert conn.execute("SELECT count(*) AS n FROM ideas").fetchone()["n"] == 0
    good = extraction(item)
    first = workflow.submit(conn,job["id"],"extractor",job["token"],good)
    second = workflow.submit(conn,job["id"],"extractor",job["token"],good)
    assert first["submission_id"] == second["submission_id"]
    assert second["idempotent"] is True
    assert conn.execute("SELECT count(*) AS n FROM ideas").fetchone()["n"] == 1
    good["ideas"][0]["statement"] = "Changed statement"
    with pytest.raises(workflow.WorkflowError,match="different submission"):
        workflow.submit(conn,job["id"],"extractor",job["token"],good)


def test_expired_lease_fences_stale_worker_and_preserves_checkpoint(conn):
    item = seed(conn)
    old = workflow.claim(conn,"crashed-worker",version_id=item["version"])
    workflow.checkpoint(conn,old["id"],"crashed-worker",old["token"],item["section"],{"next_step":"check equation"})
    conn.execute("UPDATE jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s", (old["id"],))
    conn.commit()
    new = workflow.claim(conn,"replacement-worker",version_id=item["version"])
    assert new["id"] == old["id"] and new["token"] != old["token"] and new["attempts"] == 2
    with pytest.raises(workflow.WorkflowError,match="fencing"):
        workflow.heartbeat(conn,old["id"],"crashed-worker",old["token"])
    with pytest.raises(workflow.WorkflowError,match="fencing"):
        workflow.submit(conn,old["id"],"crashed-worker",old["token"],extraction(item))
    assert conn.execute("SELECT payload FROM checkpoints WHERE job_id=%s", (new["id"],)).fetchone()["payload"]["next_step"] == "check equation"
    workflow.submit(conn,new["id"],"replacement-worker",new["token"],extraction(item))
    attempts = conn.execute("SELECT state FROM job_attempts WHERE job_id=%s ORDER BY attempt_number", (old["id"],)).fetchall()
    assert [row["state"] for row in attempts] == ["expired","done"]


def test_global_concurrency_cap_with_concurrent_claims(conn):
    for _ in range(5):
        seed(conn)
    schema = conn.execute("SELECT current_schema() AS name").fetchone()["name"]
    options = conninfo_to_dict(os.environ.get("PAPERKB_TEST_DATABASE_URL") or settings()["database_url"])
    options["dbname"] = conn.info.dbname
    dsn = make_conninfo(**options)
    conn.commit()

    def get_job(index):
        with psycopg.connect(dsn,row_factory=dict_row) as worker_conn:
            worker_conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema)))
            return workflow.claim(worker_conn,f"parallel-{index}",kind="extract")

    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = list(pool.map(get_job,range(5)))
    claimed = [item for item in jobs if item]
    assert len(claimed) == 3
    assert len({item["id"] for item in claimed}) == 3


def test_single_worker_one_paper_and_checkpoint_scope(conn):
    first,second = seed(conn),seed(conn)
    job = workflow.claim(conn,"worker",version_id=first["version"])
    with pytest.raises(workflow.WorkflowError,match="one paper"):
        workflow.claim(conn,"worker",version_id=second["version"])
    with pytest.raises(workflow.WorkflowError,match="does not belong"):
        workflow.checkpoint(conn,job["id"],"worker",job["token"],second["section"],{})


def test_every_section_and_independent_review_required(conn):
    item = seed(conn)
    job,_ = submit_seed(conn,item)
    assert workflow.claim(conn,"extractor",kind="review",version_id=item["version"]) is None
    reviewer = workflow.claim(conn,"reviewer",kind="review",version_id=item["version"])
    bad = accepted()
    bad["idea_relevance"] = []
    with pytest.raises(workflow.WorkflowError,match="every active idea"):
        workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],bad)
    result = workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted())
    assert result["accepted"] is True
    assert conn.execute("SELECT state FROM paper_versions WHERE id=%s", (item["version"],)).fetchone()["state"] == "reviewed"
    assert conn.execute("SELECT parent_id FROM ideas_current").fetchone()["parent_id"] is None
    assert workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted())["idempotent"] is True


def test_missing_coverage_rejected_and_unreadable_blocks_review(conn):
    item = seed(conn)
    job = workflow.claim(conn,"extractor",version_id=item["version"])
    wrong = extraction(item)
    wrong["coverage"][0]["section_id"] = str(uuid4())
    with pytest.raises(workflow.WorkflowError,match="every prepared section"):
        workflow.submit(conn,job["id"],"extractor",job["token"],wrong)
    workflow.submit(conn,job["id"],"extractor",job["token"],extraction(item,unreadable=True))
    reviewer = workflow.claim(conn,"reviewer",kind="review",version_id=item["version"])
    with pytest.raises(workflow.WorkflowError,match="Unreadable"):
        workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted())


def test_visual_evidence_requires_explicit_review_confirmation(conn):
    item = seed(conn)
    submit_seed(conn,item,visual=True)
    reviewer = workflow.claim(conn,"reviewer",kind="review",version_id=item["version"])
    with pytest.raises(workflow.WorkflowError,match="visual evidence"):
        workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted())
    workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted(visual=True))


def test_flagged_visual_page_cannot_be_silently_skipped(conn):
    item = seed(conn)
    conn.execute("UPDATE pages SET flags='[\"visual_review_required\",\"figures_or_tables\"]' WHERE version_id=%s", (item["version"],))
    conn.commit()
    submit_seed(conn,item)
    reviewer = workflow.claim(conn,"reviewer",kind="review",version_id=item["version"])
    with pytest.raises(workflow.WorkflowError,match="flagged page"):
        workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted())
    workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted(visual=True))


def test_rejected_review_queues_correction_and_stable_idea_uuid(conn):
    item = seed(conn)
    submit_seed(conn,item)
    idea = conn.execute("SELECT id FROM ideas").fetchone()["id"]
    reviewer = workflow.claim(conn,"reviewer",kind="review",version_id=item["version"])
    result = workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],{"accepted":False,"notes":"Needs clarification.","issues":["Specify assumptions for removal."]})
    assert result["accepted"] is False
    corrected_job = workflow.claim(conn,"correction-worker",kind="extract",version_id=item["version"])
    assert corrected_job["revision"] == 2
    corrected = extraction(item)
    corrected["ideas"][0]["conditions"] = "Only redundant tokens may be removed."
    workflow.submit(conn,corrected_job["id"],"correction-worker",corrected_job["token"],corrected)
    current = conn.execute("SELECT id,revision,relevance_score FROM ideas WHERE active").fetchone()
    assert current == {"id":idea,"revision":2,"relevance_score":None}
    assert conn.execute("SELECT count(*) AS n FROM section_coverage").fetchone()["n"] == 2


def test_original_extractor_cannot_review_replacement_correction(conn):
    item=seed(conn)
    submit_seed(conn,item)
    first_review=workflow.claim(conn,'independent-reviewer',kind='review',version_id=item['version'])
    workflow.review(conn,first_review['id'],'independent-reviewer',first_review['token'],
                    {'accepted':False,'notes':'Clarify conditions.','issues':['Add the precise condition.']})
    correction=workflow.claim(conn,'corrector',kind='extract',version_id=item['version'])
    workflow.submit(conn,correction['id'],'corrector',correction['token'],extraction(item))
    assert workflow.claim(conn,'extractor',kind='review',version_id=item['version']) is None
    second_review=workflow.claim(conn,'independent-reviewer',kind='review',version_id=item['version'])
    # A lease granted by the previous implementation must also fail at ingestion.
    conn.execute('UPDATE jobs SET agent_id=%s WHERE id=%s',('extractor',second_review['id']))
    with pytest.raises(workflow.WorkflowError,match='every extraction participant'):
        workflow.review(conn,second_review['id'],'extractor',second_review['token'],accepted())
    conn.execute('UPDATE jobs SET agent_id=%s WHERE id=%s',('independent-reviewer',second_review['id']))
    assert workflow.review(conn,second_review['id'],'independent-reviewer',second_review['token'],accepted())['accepted']


def test_expired_extraction_participant_cannot_review_resumed_draft(conn):
    item=seed(conn)
    original=workflow.claim(conn,'original',kind='extract',version_id=item['version'])
    workflow.checkpoint(conn,original['id'],'original',original['token'],item['section'],{'draft_snapshot':extraction(item)})
    conn.execute("UPDATE jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",(original['id'],))
    replacement=workflow.claim(conn,'replacement',kind='extract',version_id=item['version'])
    workflow.submit(conn,replacement['id'],'replacement',replacement['token'],extraction(item))
    assert workflow.claim(conn,'original',kind='review',version_id=item['version']) is None
    independent=workflow.claim(conn,'independent',kind='review',version_id=item['version'])
    assert workflow.review(conn,independent['id'],'independent',independent['token'],accepted())['accepted']


@pytest.mark.parametrize("depth,score,expected",[(0,95,"approved"),(0,49,"rejected"),(1,95,"rejected")])
def test_nomination_requires_original_source_and_foundational_relevance(conn,depth,score,expected):
    item = seed(conn,source_depth=depth)
    submit_seed(conn,item,nomination=True)
    reviewer = workflow.claim(conn,"reviewer",kind="review",version_id=item["version"])
    with pytest.raises(workflow.WorkflowError,match="every predecessor"):
        workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted(score=score))
    workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted(nomination=True,score=score))
    assert conn.execute("SELECT state FROM predecessor_nominations").fetchone()["state"] == expected


def test_revision_invalidates_rankings_embeddings_and_published_graph(conn):
    item = seed(conn)
    submit_seed(conn,item)
    reviewer = workflow.claim(conn,"reviewer",kind="review",version_id=item["version"])
    workflow.review(conn,reviewer["id"],"reviewer",reviewer["token"],accepted())
    idea = conn.execute("SELECT id FROM ideas").fetchone()["id"]
    vector = "[1," + ",".join(["0"] * 1023) + "]"
    conn.execute("INSERT INTO embeddings(idea_id,revision,model,model_revision,text_sha256,embedding) VALUES(%s,1,'test','test',%s,%s::vector)",(idea,"a"*64,vector))
    graph = conn.execute("INSERT INTO graph_builds(corpus_id,state,snapshot) VALUES(%s,'published','{}') RETURNING id", (item["corpus"],)).fetchone()["id"]
    conn.execute("UPDATE corpus SET state='frozen',current_graph_id=%s,expansion_closed=true WHERE id=%s", (graph,item["corpus"]))
    conn.commit()
    with pytest.raises(workflow.WorkflowError,match="frozen"):
        workflow.request_revision(conn,item["version"],"Correct assumptions")
    workflow.reopen(conn,item["corpus"],"Reopen for evidence correction")
    correction = workflow.request_revision(conn,item["version"],"Correct assumptions")
    assert correction["revision"] == 2
    assert conn.execute("SELECT count(*) AS n FROM embeddings").fetchone()["n"] == 0
    assert conn.execute("SELECT relevance_score FROM ideas").fetchone()["relevance_score"] is None
    assert conn.execute("SELECT current_graph_id FROM corpus WHERE id=%s", (item["corpus"],)).fetchone()["current_graph_id"] is None
    assert conn.execute("SELECT state FROM graph_builds WHERE id=%s", (graph,)).fetchone()["state"] == "superseded"


def test_fail_retry_and_blocked_are_durable(conn):
    item = seed(conn)
    original = workflow.claim(conn,"first",version_id=item["version"])
    workflow.fail(conn,original["id"],"first",original["token"],"Temporary parser failure")
    replacement = workflow.claim(conn,"second",version_id=item["version"])
    assert replacement["attempts"] == 2
    workflow.fail(conn,replacement["id"],"second",replacement["token"],"Unreadable source requires action",retry=False)
    assert workflow.claim(conn,"third",version_id=item["version"]) is None
    assert workflow.status(conn)["blocked"][0]["id"] == original["id"]


def test_other_pdf_version_extractor_cannot_review_same_paper(conn):
    first=seed(conn)
    submit_seed(conn,first)
    second=seed(conn)
    conn.execute('UPDATE paper_versions SET paper_id=%s WHERE id=%s',(first['paper'],second['version']))
    job=workflow.claim(conn,'new-version-extractor',kind='extract',version_id=second['version'])
    workflow.submit(conn,job['id'],'new-version-extractor',job['token'],extraction(second))
    assert workflow.claim(conn,'extractor',kind='review',version_id=second['version']) is None
    independent=workflow.claim(conn,'fresh-reviewer',kind='review',version_id=second['version'])
    assert workflow.review(conn,independent['id'],'fresh-reviewer',independent['token'],accepted())['accepted']
