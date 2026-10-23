import hashlib
import numpy as np
import pytest
from paperkb import embeddings, workflow
from test_workflow import seed, submit_seed, accepted

LOCK={'model':'BAAI/bge-m3','revision':'a'*40,'dimensions':1024,'normalized':True,'similarity':'cosine'}

def reviewed(conn):
    item=seed(conn)
    submit_seed(conn,item)
    job=workflow.claim(conn,'independent-reviewer',kind='review',version_id=item['version'])
    workflow.review(conn,job['id'],'independent-reviewer',job['token'],accepted())
    conn.commit()
    return item

def fake(monkeypatch):
    monkeypatch.setattr(embeddings,'load_model',lambda:(None,LOCK))
    monkeypatch.setattr(embeddings,'encode',lambda model,texts,batch_size=8:[[1.0]+[0.0]*1023 for _ in texts])

def test_current_embeddings_complete_and_are_idempotent(conn,monkeypatch):
    item=reviewed(conn);fake(monkeypatch)
    result=embeddings.embed(conn,item['version'])
    assert result['embedded_ideas']==1
    assert str(item['version']) in result['completed_versions']
    row=conn.execute('SELECT * FROM embeddings').fetchone()
    idea=conn.execute('SELECT embedding_text FROM ideas').fetchone()
    assert row['text_sha256']==hashlib.sha256(idea['embedding_text'].encode()).hexdigest()
    assert row['model_revision']==LOCK['revision']
    assert embeddings.embed(conn,item['version'])['embedded_ideas']==0

def test_changed_revision_during_inference_never_persists_stale_vector(conn,monkeypatch):
    item=reviewed(conn);fake(monkeypatch)
    def race(model,texts,batch_size=8):
        workflow.request_revision(conn,item['version'],'Correction during inference')
        conn.commit()
        return [[1.0]+[0.0]*1023]
    monkeypatch.setattr(embeddings,'encode',race)
    with pytest.raises(ValueError,match='changed'):
        embeddings.embed(conn,item['version'])
    assert conn.execute('SELECT count(*) AS n FROM embeddings').fetchone()['n']==0
    assert conn.execute('SELECT reviewed_revision FROM paper_versions WHERE id=%s',(item['version'],)).fetchone()['reviewed_revision'] is None

def test_frozen_corpus_is_not_mutated(conn,monkeypatch):
    item=reviewed(conn);fake(monkeypatch)
    conn.execute("UPDATE corpus SET state='frozen' WHERE id=%s",(item['corpus'],));conn.commit()
    assert embeddings.embed(conn,item['version'])['completed_versions']==[]
    assert conn.execute('SELECT count(*) AS n FROM embeddings').fetchone()['n']==0

def test_text_truncation_and_invalid_vectors_are_rejected():
    class Tokenizer:
        def encode(self,text,truncation=False):return list(range(len(text)))
    class Model:
        tokenizer=Tokenizer();max_seq_length=3
        def encode(self,texts,**kwargs):return np.zeros((len(texts),1024))
    with pytest.raises(ValueError,match='context'):embeddings.encode(Model(),['long'])
    with pytest.raises(ValueError,match='normalization'):embeddings.encode(Model(),['ok'])


def test_symbolic_revision_cannot_drift_or_disagree_with_database(monkeypatch,tmp_path):
    monkeypatch.setattr(embeddings,'ROOT',tmp_path)
    monkeypatch.setattr(embeddings,'settings',lambda:{'embedding_model':'BAAI/bge-m3','embedding_revision':'main'})
    with pytest.raises(ValueError,match='immutable'):
        embeddings.pin_model()
    assert not (tmp_path/'.paperkb/model.lock.json').exists()


def test_pinned_revision_requires_no_registry_lookup(monkeypatch,tmp_path):
    monkeypatch.setattr(embeddings,'ROOT',tmp_path)
    monkeypatch.setattr(embeddings,'settings',lambda:{'embedding_model':'BAAI/bge-m3','embedding_revision':LOCK['revision']})
    result=embeddings.pin_model()
    assert result['revision']==LOCK['revision']
