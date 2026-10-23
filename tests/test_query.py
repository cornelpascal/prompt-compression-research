import json
import io
import os

import pytest

from paperkb import graph, query
from test_graph import payload_for, seed_ready


def test_keyword_hybrid_and_minimum_relevance(conn, monkeypatch):
    _, docs = seed_ready(conn,monkeypatch)
    keyword = query.search(conn,"compression")
    assert keyword["mode"] == "keyword" and len(keyword["results"]) == 3
    hybrid = query.search(conn,"compression",embedding=[1] + [0] * 1023,limit=2)
    assert hybrid["mode"] == "hybrid" and len(hybrid["results"]) == 2
    assert all(set(row["retrieval_sources"]) == {"keyword","vector"} for row in hybrid["results"])
    assert query.search(conn,"compression",min_relevance=99)["results"] == []
    conn.execute("UPDATE ideas SET active=false WHERE id=%s", (docs[0]["idea"],))
    assert len(query.search(conn,"compression")["results"]) == 2
    conn.execute("UPDATE paper_versions SET reviewed_revision=NULL WHERE id=%s", (docs[1]["version"],))
    assert len(query.search(conn,"compression")["results"]) == 1


def test_sql_metacharacters_are_query_data(conn, monkeypatch):
    seed_ready(conn,monkeypatch)
    query.search(conn,"compression'; DROP TABLE ideas; --")
    assert conn.execute("SELECT count(*) AS n FROM ideas").fetchone()["n"] == 3


def test_inspect_lineage_export_preserve_provenance(conn, monkeypatch):
    corpus, docs = seed_ready(conn,monkeypatch)
    before = query.lineage(conn,docs[2]["idea"])
    assert before["state"] == "unpublished" and not before["complete"]
    payload = payload_for(conn,docs[2],docs[0])
    graph.canonical_group(conn,"Compression","Related source occurrences",[docs[0]["idea"],docs[2]["idea"]],"Explicit review")
    build = graph.begin_build(conn,corpus)
    graph.add_relation(conn,build["id"],payload)
    graph.publish(conn,build["id"])
    detail = query.inspect_idea(conn,docs[2]["idea"])
    assert detail["parent_id"] == docs[0]["idea"]
    assert detail["evidence"][0]["quote"] == docs[2]["statement"]
    assert detail["sha256"] and detail["concepts"] and detail["relations"]
    lineage = query.lineage(conn,docs[2]["idea"])
    assert lineage["complete"] and [row["id"] for row in lineage["chain"]] == [docs[2]["idea"],docs[0]["idea"]]
    exported = query.export_data(conn)
    assert len(exported["data"]["ideas"]) == 3
    assert len(exported["data"]["graph_parents"]) == 3
    assert not exported["vector_values_included"]
    json.dumps(exported)  # Export can be written directly as JSON.


@pytest.mark.parametrize("embedding", [[1,2], [0]*1024, [float('nan')]*1024, [True]*1024, "[1]"])
def test_vector_validation(embedding):
    with pytest.raises(ValueError,match="embedding"):
        query._vector_literal(embedding)


def test_query_validation_without_database():
    for kwargs in ({"query":""},{"query":"a","limit":0},{"query":"a","min_relevance":101}):
        with pytest.raises(ValueError):
            query.search(None,**kwargs)


def test_stream_export_matches_schema_with_bounded_server_cursors(conn,monkeypatch):
    seed_ready(conn,monkeypatch)
    expected=query.export_data(conn)
    conn.commit()
    fetch_sizes=[]
    class Cursor:
        def __init__(self,real):self.real=real
        def __enter__(self):self.real.__enter__();return self
        def __exit__(self,*args):return self.real.__exit__(*args)
        def execute(self,sql):return self.real.execute(sql)
        def fetchmany(self,size):
            fetch_sizes.append(size)
            return self.real.fetchmany(size)
    class Connection:
        info=conn.info
        def transaction(self):return conn.transaction()
        def execute(self,*args):return conn.execute(*args)
        def cursor(self,*,name):
            assert name.startswith('paperkb_export_')
            return Cursor(conn.cursor(name=name))
    output=io.StringIO()
    counts=query.stream_export(Connection(),output,batch_size=2)
    assert json.loads(output.getvalue())==expected
    assert counts['row_counts']['papers']==3
    assert fetch_sizes and set(fetch_sizes)=={2}
    assert conn.info.transaction_status.name=='IDLE'


def test_stream_export_sees_one_snapshot_across_concurrent_commits(conn,monkeypatch):
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    from paperkb.config import settings
    from paperkb.db import connect

    _,docs=seed_ready(conn,monkeypatch)
    schema=conn.execute('SELECT current_schema() AS name').fetchone()['name']
    options=conninfo_to_dict(os.environ.get('PAPERKB_TEST_DATABASE_URL') or settings()['database_url'])
    options['dbname']=conn.info.dbname
    conn.commit()
    with connect(make_conninfo(**options)) as writer:
        writer.execute(sql.SQL('SET search_path TO {},public').format(sql.Identifier(schema)))
        writer.commit()
        class DuringExport(io.StringIO):
            changed=False
            def write(self,text):
                if text=='"ideas":[' and not self.changed:
                    with writer.transaction():
                        writer.execute("UPDATE papers SET title='Concurrent new title' WHERE id=%s",(docs[0]['paper'],))
                        writer.execute("UPDATE ideas SET statement='Concurrent new claim' WHERE id=%s",(docs[0]['idea'],))
                    self.changed=True
                return super().write(text)
        output=DuringExport()
        query.stream_export(conn,output,batch_size=1)
        exported=json.loads(output.getvalue())['data']
        assert output.changed
        assert next(row for row in exported['papers'] if row['id']==str(docs[0]['paper']))['title']=='Paper 0'
        assert next(row for row in exported['ideas'] if row['id']==str(docs[0]['idea']))['statement']==docs[0]['statement']
    assert conn.execute('SELECT statement FROM ideas WHERE id=%s',(docs[0]['idea'],)).fetchone()['statement']=='Concurrent new claim'


def test_stream_export_preserves_existing_caller_transaction(conn,monkeypatch):
    seed_ready(conn,monkeypatch)
    output=io.StringIO()
    with pytest.raises(ValueError,match='idle connection'):
        query.stream_export(conn,output)
    assert output.getvalue()==''
    assert conn.execute('SELECT count(*) AS n FROM papers').fetchone()['n']==3


def test_export_file_publishes_only_complete_json(conn,monkeypatch,tmp_path):
    seed_ready(conn,monkeypatch)
    conn.commit()
    target=tmp_path/'corpus.json'
    result=query.export_to_file(conn,target,batch_size=1)
    assert result=={'path':str(target)}
    assert len(json.loads(target.read_text())['data']['papers'])==3
    with pytest.raises(FileExistsError):
        query.export_to_file(conn,target)
    assert list(tmp_path.iterdir())==[target]


def test_export_failure_cleans_partial_file(conn,monkeypatch,tmp_path):
    def fail(conn,output,batch_size):
        output.write('{"partial":')
        raise RuntimeError('synthetic export failure')
    monkeypatch.setattr(query,'stream_export',fail)
    with pytest.raises(RuntimeError,match='synthetic export failure'):
        query.export_to_file(conn,tmp_path/'corpus.json')
    assert list(tmp_path.iterdir())==[]
