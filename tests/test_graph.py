import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from paperkb import graph


def seed_ready(conn, monkeypatch, count=3, with_coverage=True):
    """Synthetic reviewed sources, isolated from the downloaded research corpus."""
    monkeypatch.setenv("PAPERKB_EMBEDDING_REVISION", "test-model-revision")
    corpus = uuid4()
    conn.execute("INSERT INTO corpus(id,name,expansion_closed) VALUES(%s,%s,true)", (corpus, str(corpus)))
    documents = []
    for n in range(count):
        paper, version, idea, review = (uuid4() for _ in range(4))
        statement = f"Compression method {n} selects relevant context tokens."
        page = statement + " This method extends the compression method of earlier work."
        text_hash = hashlib.sha256(statement.encode()).hexdigest()
        conn.execute("""INSERT INTO papers(id,corpus_id,arxiv_id,title,source_depth)
            VALUES(%s,%s,%s,%s,0)""", (paper, corpus, f"230{n}.00001", f"Paper {n}"))
        conn.execute("""INSERT INTO paper_versions(id,paper_id,sha256,file_path,version_label,available_at,
            date_verified,state,extraction_revision,reviewed_revision,extractor_id,reviewer_id,relevance_score,relevance_reason)
            VALUES(%s,%s,%s,'/synthetic.pdf','v1',%s,true,'complete',1,1,'extractor','reviewer',90,'Directly relevant')""",
            (version,paper,hashlib.sha256(str(version).encode()).hexdigest(),datetime(2023, n + 1, 1, tzinfo=timezone.utc)))
        conn.execute("INSERT INTO pages(version_id,page_number,text,text_sha256) VALUES(%s,1,%s,%s)",
                     (version,page,hashlib.sha256(page.encode()).hexdigest()))
        if with_coverage:
            section = conn.execute("""INSERT INTO sections(version_id,ordinal,title,page_start,page_end)
                VALUES(%s,1,'Page 1',1,1) RETURNING id""", (version,)).fetchone()["id"]
            conn.execute("""INSERT INTO section_coverage(version_id,section_id,revision,status,notes)
                VALUES(%s,%s,1,'covered','Complete synthetic coverage')""", (version,section))
        conn.execute("""INSERT INTO ideas(id,version_id,local_id,revision,kind,title,statement,attribution,
            confidence,relevance_score,relevance_reason,embedding_text)
            VALUES(%s,%s,'method',1,'method',%s,%s,'original',0.9,90,'Direct method',%s)""",
            (idea,version,f"Compression method {n}",statement,statement))
        conn.execute("""INSERT INTO idea_evidence(idea_id,page_number,quote,evidence_kind)
            VALUES(%s,1,%s,'text')""", (idea,statement))
        conn.execute("INSERT INTO reviews(id,version_id,revision,reviewer_id,accepted,payload) VALUES(%s,%s,1,'reviewer',true,%s)",
                     (review,version,Jsonb({"notes": "Synthetic accepted review"})))
        vector = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
        conn.execute("""INSERT INTO embeddings(idea_id,revision,model,model_revision,text_sha256,embedding)
            VALUES(%s,1,'BAAI/bge-m3','test-model-revision',%s,%s::vector)""", (idea,text_hash,vector))
        documents.append({"paper":paper,"version":version,"idea":idea,"statement":statement,"page":page,"arxiv":f"230{n}.00001"})
    return corpus, documents


def payload_for(conn, child, parent, **overrides):
    reference = conn.execute("""INSERT INTO paper_references(version_id,citation_key,title,arxiv_id)
        VALUES(%s,%s,'Predecessor paper',%s) ON CONFLICT(version_id,citation_key)
        DO UPDATE SET title=EXCLUDED.title RETURNING id""",
        (child["version"],str(parent["paper"]),parent["arxiv"])).fetchone()["id"]
    payload = {"child_id":str(child["idea"]),"parent_id":str(parent["idea"]),
               "relation_type":"intellectual_predecessor","confidence":0.95,"confirmed":True,
               "is_primary":True,"rationale":"Explicitly builds on the cited earlier method.",
               "proposer_id":"lineage-proposer","reviewer_id":"lineage-reviewer",
               "evidence":{"child":{"page_number":1,"quote":"This method extends the compression method of earlier work."},
                           "parent":{"page_number":1,"quote":parent["statement"]},
                           "reference_id":str(reference),"directional_claim":"Child explicitly extends the parent method."}}
    return payload | overrides


def test_barrier_checks_embeddings_jobs_and_expansion(conn, monkeypatch):
    corpus, documents = seed_ready(conn, monkeypatch)
    assert graph.barrier(conn,corpus)["ready"]
    conn.execute("UPDATE embeddings SET text_sha256='outdated' WHERE idea_id=%s", (documents[0]["idea"],))
    assert any("embedding" in reason for reason in graph.barrier(conn,corpus)["reasons"])
    conn.execute("UPDATE corpus SET expansion_closed=false WHERE id=%s", (corpus,))
    conn.execute("INSERT INTO jobs(version_id,kind,revision) VALUES(%s,'review',1)", (documents[0]["version"],))
    reasons = graph.barrier(conn,corpus)["reasons"]
    assert any("expansion" in reason for reason in reasons)
    assert any("unfinished" in reason for reason in reasons)
    with pytest.raises(graph.GraphError, match="barrier"):
        graph.begin_build(conn,corpus)


def test_begin_is_idempotent_and_freezes(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    first = graph.begin_build(conn,corpus)
    assert graph.begin_build(conn,corpus)["id"] == first["id"]
    assert conn.execute("SELECT state FROM corpus WHERE id=%s", (corpus,)).fetchone()["state"] == "frozen"
    frozen = {row['id']:row for row in first['snapshot']['versions']}
    assert set(frozen) == {str(doc['version']) for doc in docs}
    assert all(row['extraction_revision']==1 and row['reviewed_revision']==1 and len(row['sha256'])==64 for row in frozen.values())


def test_publish_chain_and_keep_secondary_evidence(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    first = payload_for(conn,docs[1],docs[0])
    second = payload_for(conn,docs[2],docs[1])
    secondary = payload_for(conn,docs[2],docs[0],relation_type="uses",is_primary=False)
    build = graph.begin_build(conn,corpus)
    for payload in (first,second,secondary):
        graph.add_relation(conn,build["id"],payload)
    # Re-ingestion is idempotent.
    graph.add_relation(conn,build["id"],first)
    published = graph.publish(conn,build["id"])
    assert published["parent_count"] == 2
    assert graph.publish(conn,build["id"])["already_published"]
    assert conn.execute("SELECT count(*) AS n FROM idea_relations").fetchone()["n"] == 3
    rows = conn.execute("SELECT idea_id,parent_id FROM graph_parents WHERE graph_id=%s", (build["id"],)).fetchall()
    assert {row["idea_id"]:row["parent_id"] for row in rows} == {
        docs[0]["idea"]:None, docs[1]["idea"]:docs[0]["idea"], docs[2]["idea"]:docs[1]["idea"]}
    assert conn.execute("SELECT state FROM corpus WHERE id=%s", (corpus,)).fetchone()["state"] == "published"
    with pytest.raises(graph.GraphError,match="building"):
        graph.add_relation(conn,build["id"],first)


def test_ambiguous_parents_remain_null(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    payloads = [payload_for(conn,docs[2],parent) for parent in docs[:2]]
    build = graph.begin_build(conn,corpus)
    for payload in payloads:
        graph.add_relation(conn,build["id"],payload)
    result = graph.publish(conn,build["id"])
    assert result["ambiguous_count"] == 1
    row = conn.execute("SELECT * FROM graph_parents WHERE graph_id=%s AND idea_id=%s", (build["id"],docs[2]["idea"])).fetchone()
    assert row["parent_id"] is None and "Ambiguous" in row["rationale"]
    assert conn.execute("SELECT count(*) AS n FROM idea_relations WHERE is_primary").fetchone()["n"] == 0
    assert conn.execute("SELECT count(*) AS n FROM idea_relations WHERE confirmed").fetchone()["n"] == 2


def test_quotes_dates_citations_and_independence_are_validated(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    payload = payload_for(conn,docs[2],docs[0])
    build = graph.begin_build(conn,corpus)
    bad = dict(payload) | {"evidence":dict(payload["evidence"]) | {"child":{"page_number":1,"quote":"Invented attribution."}}}
    with pytest.raises(graph.GraphError,match="quotation"):
        graph.add_relation(conn,build["id"],bad)
    with pytest.raises(graph.GraphError,match="extractor"):
        graph.add_relation(conn,build["id"],payload | {"reviewer_id":"extractor"})
    with pytest.raises(ValidationError,match="distinct"):
        graph.add_relation(conn,build["id"],payload | {"reviewer_id":payload["proposer_id"]})
    conn.execute("UPDATE paper_versions SET date_verified=false WHERE id=%s", (docs[0]["version"],))
    with pytest.raises(graph.GraphError,match="verified strictly earlier"):
        graph.add_relation(conn,build["id"],payload)
    conn.execute("UPDATE paper_versions SET date_verified=true,available_at='2030-01-01' WHERE id=%s", (docs[0]["version"],))
    with pytest.raises(graph.GraphError,match="verified strictly earlier"):
        graph.add_relation(conn,build["id"],payload)


def test_wrong_citation_target_rejected(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    payload = payload_for(conn,docs[2],docs[0])
    payload["parent_id"] = str(docs[1]["idea"])
    payload["evidence"]["parent"]["quote"] = docs[1]["statement"]
    build = graph.begin_build(conn,corpus)
    with pytest.raises(graph.GraphError,match="resolved|conflicting"):
        graph.add_relation(conn,build["id"],payload)


def test_snapshot_change_blocks_publication(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    build = graph.begin_build(conn,corpus)
    conn.execute("UPDATE ideas SET statement=statement||' Changed claim.' WHERE id=%s", (docs[0]["idea"],))
    with pytest.raises(graph.GraphError,match="inputs changed"):
        graph.publish(conn,build["id"])
    assert conn.execute("SELECT current_graph_id FROM corpus WHERE id=%s", (corpus,)).fetchone()["current_graph_id"] is None
    assert conn.execute("SELECT count(*) AS n FROM graph_parents").fetchone()["n"] == 0


def test_publish_rolls_back_entire_publication_on_pointer_failure(conn, monkeypatch):
    corpus, _ = seed_ready(conn,monkeypatch)
    old = graph.begin_build(conn,corpus)
    graph.publish(conn,old["id"])
    new = graph.begin_build(conn,corpus)
    conn.execute("""CREATE FUNCTION reject_pointer() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN IF NEW.current_graph_id IS DISTINCT FROM OLD.current_graph_id THEN
        RAISE EXCEPTION 'synthetic pointer failure'; END IF; RETURN NEW; END $$""")
    conn.execute("CREATE TRIGGER reject_pointer BEFORE UPDATE ON corpus FOR EACH ROW EXECUTE FUNCTION reject_pointer()")
    with pytest.raises(Exception,match="synthetic pointer failure"):
        graph.publish(conn,new["id"])
    assert conn.execute("SELECT current_graph_id FROM corpus WHERE id=%s", (corpus,)).fetchone()["current_graph_id"] == old["id"]
    assert conn.execute("SELECT state FROM graph_builds WHERE id=%s", (old["id"],)).fetchone()["state"] == "published"
    assert conn.execute("SELECT state FROM graph_builds WHERE id=%s", (new["id"],)).fetchone()["state"] == "building"
    assert conn.execute("SELECT count(*) AS n FROM graph_parents WHERE graph_id=%s", (new["id"],)).fetchone()["n"] == 0


def test_candidates_and_groups_do_not_create_parent_edges(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    payload_for(conn,docs[2],docs[0])
    candidates = graph.candidates(conn,docs[2]["idea"])
    assert candidates["purpose"] == "retrieval_only"
    assert candidates["results"] and all(not row["confirmed"] for row in candidates["results"])
    assert any("citation" in row["retrieval_sources"] for row in candidates["results"])
    ids = [doc["idea"] for doc in docs]
    grouping = graph.canonical_group(conn,"Token selection","Explicitly equivalent terminology",ids,"Reviewed conceptual equivalence")
    assert graph.canonical_group(conn,"Token selection","Explicitly equivalent terminology",ids,"Reviewed conceptual equivalence")["concept_id"] == grouping["concept_id"]
    assert conn.execute("SELECT count(*) AS n FROM ideas").fetchone()["n"] == 3
    assert conn.execute("SELECT count(*) AS n FROM idea_relations").fetchone()["n"] == 0
    assert conn.execute("SELECT count(*) AS n FROM graph_parents").fetchone()["n"] == 0


def test_cycle_detection_and_unconfirmed_primary_rejected():
    with pytest.raises(graph.GraphError,match="cycle"):
        graph._assert_acyclic({"a":"b","b":"c","c":"a"})
    graph._assert_acyclic({"a":"b","b":None})
    with pytest.raises(ValidationError):
        graph.RelationPayload.model_validate({"child_id":str(uuid4()),"parent_id":str(uuid4()),
            "relation_type":"similarity","confidence":1,"confirmed":False,"is_primary":True,
            "rationale":"Similar embeddings","proposer_id":"a","evidence":{"child":{"page_number":1,"quote":"q"},
            "parent":{"page_number":1,"quote":"q"},"directional_claim":"similarity"}})


def test_barrier_rechecks_coverage_ideas_and_admitted_predecessors(conn,monkeypatch):
    corpus,docs=seed_ready(conn,monkeypatch)
    conn.execute("UPDATE section_coverage SET status='unreadable' WHERE version_id=%s",(docs[0]['version'],))
    conn.execute("UPDATE ideas SET active=false WHERE id=%s",(docs[1]['idea'],))
    conn.execute("UPDATE papers SET source_depth=1 WHERE id=%s",(docs[2]['paper'],))
    conn.execute("UPDATE paper_versions SET state='registered' WHERE id=%s",(docs[2]['version'],))
    reasons=graph.barrier(conn,corpus)['reasons']
    assert any('coverage' in reason for reason in reasons)
    assert any('no active' in reason for reason in reasons)
    assert any('registered' in reason for reason in reasons)


def test_closed_flag_cannot_hide_unresolved_nomination(conn,monkeypatch):
    corpus,docs=seed_ready(conn,monkeypatch)
    payload=payload_for(conn,docs[2],docs[0])
    conn.execute("""INSERT INTO predecessor_nominations(idea_id,reference_id,evidence_page,evidence_quote,state,reviewer_id)
        VALUES(%s,%s,1,'Explicit source','approved','reviewer')""",(docs[2]['idea'],payload['evidence']['reference_id']))
    assert any('Unresolved predecessor' in reason for reason in graph.barrier(conn,corpus)['reasons'])
    with pytest.raises(graph.GraphError,match='barrier'):
        graph.begin_build(conn,corpus)


def test_relation_resolves_canonical_arxiv_url_without_id_field(conn,monkeypatch):
    corpus,docs=seed_ready(conn,monkeypatch)
    payload=payload_for(conn,docs[2],docs[0])
    conn.execute('UPDATE paper_references SET arxiv_id=NULL,url=%s WHERE id=%s',
                 (f"https://arxiv.org/pdf/{docs[0]['arxiv']}v1.pdf",payload['evidence']['reference_id']))
    build=graph.begin_build(conn,corpus)
    assert graph.add_relation(conn,build['id'],payload)['confirmed']


def test_relation_rejects_conflicting_arxiv_and_doi(conn,monkeypatch):
    corpus,docs=seed_ready(conn,monkeypatch)
    payload=payload_for(conn,docs[2],docs[0])
    conn.execute("UPDATE papers SET doi='10.1234/actual' WHERE id=%s",(docs[0]['paper'],))
    conn.execute("UPDATE paper_references SET doi='10.1234/wrong' WHERE id=%s",(payload['evidence']['reference_id'],))
    build=graph.begin_build(conn,corpus)
    with pytest.raises(graph.GraphError,match='conflicting'):
        graph.add_relation(conn,build['id'],payload)


def test_relation_rejects_conflicting_citation_versions(conn,monkeypatch):
    corpus,docs=seed_ready(conn,monkeypatch)
    payload=payload_for(conn,docs[2],docs[0])
    conn.execute('UPDATE paper_references SET arxiv_id=%s,url=%s WHERE id=%s',
                 (docs[0]['arxiv']+'v1',f"https://arxiv.org/pdf/{docs[0]['arxiv']}v2.pdf",
                  payload['evidence']['reference_id']))
    build=graph.begin_build(conn,corpus)
    with pytest.raises(graph.GraphError,match='conflicting'):
        graph.add_relation(conn,build['id'],payload)


@pytest.mark.parametrize('citation_location',['field','url'])
@pytest.mark.parametrize('parent_version',['v2',None])
def test_explicit_citation_requires_matching_parent_version(conn,monkeypatch,citation_location,parent_version):
    corpus,docs=seed_ready(conn,monkeypatch)
    payload=payload_for(conn,docs[2],docs[0])
    conn.execute('UPDATE paper_references SET arxiv_id=%s,url=%s WHERE id=%s',
                 (docs[0]['arxiv']+'v1' if citation_location=='field' else None,
                  f"https://arxiv.org/pdf/{docs[0]['arxiv']}v1.pdf" if citation_location=='url' else None,
                  payload['evidence']['reference_id']))
    conn.execute('UPDATE paper_versions SET version_label=%s WHERE id=%s',(parent_version,docs[0]['version']))
    build=graph.begin_build(conn,corpus)
    with pytest.raises(graph.GraphError,match='parent source version'):
        graph.add_relation(conn,build['id'],payload)
    # An unresolved version leaves ancestry null without preventing graph publication.
    assert graph.publish(conn,build['id'])['parent_count']==0


@pytest.mark.parametrize('citation_location',['field','url'])
def test_matching_explicit_citation_version_can_publish(conn,monkeypatch,citation_location):
    corpus,docs=seed_ready(conn,monkeypatch)
    payload=payload_for(conn,docs[2],docs[0])
    conn.execute('UPDATE paper_references SET arxiv_id=%s,url=%s WHERE id=%s',
                 (docs[0]['arxiv']+'v1' if citation_location=='field' else None,
                  f"https://arxiv.org/pdf/{docs[0]['arxiv']}v1.pdf" if citation_location=='url' else None,
                  payload['evidence']['reference_id']))
    build=graph.begin_build(conn,corpus)
    graph.add_relation(conn,build['id'],payload)
    assert graph.publish(conn,build['id'])['parent_count']==1


def test_publish_rechecks_explicit_citation_version_of_stored_relations(conn,monkeypatch):
    corpus,docs=seed_ready(conn,monkeypatch)
    payload=payload_for(conn,docs[2],docs[0])
    conn.execute('UPDATE paper_references SET arxiv_id=%s WHERE id=%s',
                 (docs[0]['arxiv']+'v2',payload['evidence']['reference_id']))
    build=graph.begin_build(conn,corpus)
    # Simulate a relation saved by a previous implementation before the version check.
    evidence=payload['evidence']|{'proposer_id':payload['proposer_id'],'primary_requested':True}
    conn.execute('''INSERT INTO idea_relations
        (graph_id,child_id,parent_id,relation_type,evidence,confidence,reviewer_id,confirmed,is_primary,rationale)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
        (build['id'],payload['child_id'],payload['parent_id'],payload['relation_type'],Jsonb(evidence),
         payload['confidence'],payload['reviewer_id'],True,True,payload['rationale']))
    with pytest.raises(graph.GraphError,match='parent source version'):
        graph.publish(conn,build['id'])
    assert conn.execute('SELECT current_graph_id FROM corpus WHERE id=%s',(corpus,)).fetchone()['current_graph_id'] is None
    assert conn.execute('SELECT count(*) AS n FROM graph_parents').fetchone()['n']==0


def test_snapshot_includes_accepted_review_content(conn,monkeypatch):
    corpus,docs=seed_ready(conn,monkeypatch)
    build=graph.begin_build(conn,corpus)
    conn.execute("UPDATE reviews SET payload=payload||'{\"notes\":\"Changed audit decision\"}'::jsonb WHERE version_id=%s",(docs[0]['version'],))
    with pytest.raises(graph.GraphError,match='inputs changed'):
        graph.publish(conn,build['id'])
