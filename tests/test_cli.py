import json
import hashlib
from contextlib import nullcontext

from paperkb import cli, db, query, documents, workflow
from test_graph import seed_ready
from test_workflow import seed, extraction


def test_export_stdout_is_one_json_document(conn,monkeypatch,capsys):
    seed_ready(conn,monkeypatch)
    conn.commit()
    monkeypatch.setattr(db,'connect',lambda:nullcontext(conn))
    assert cli.main(['export'])==0
    captured=capsys.readouterr()
    result=json.loads(captured.out)  # A trailing null/second response fails here.
    assert result['format']=='paperkb-evidence-and-graph'
    assert len(result['data']['papers'])==3
    assert captured.err==''


def test_export_output_file_returns_only_receipt(conn,monkeypatch,capsys,tmp_path):
    seed_ready(conn,monkeypatch)
    conn.commit()
    monkeypatch.setattr(db,'connect',lambda:nullcontext(conn))
    target=tmp_path/'export.json'
    assert cli.main(['export','--output',str(target)])==0
    captured=capsys.readouterr()
    assert json.loads(captured.out)=={'path':str(target)}
    assert len(json.loads(target.read_text())['data']['papers'])==3


def test_export_file_failure_has_no_success_receipt(conn,monkeypatch,capsys,tmp_path):
    monkeypatch.setattr(db,'connect',lambda:nullcontext(conn))
    def fail(conn,output,batch_size):
        output.write('{"partial":')
        raise RuntimeError('synthetic export failure')
    monkeypatch.setattr(query,'stream_export',fail)
    target=tmp_path/'export.json'
    assert cli.main(['export','--output',str(target)])==1
    captured=capsys.readouterr()
    assert captured.out==''
    assert json.loads(captured.err)['error']=='RuntimeError'
    assert not target.exists()


def test_checkpoint_recovers_full_draft_without_local_file(conn,monkeypatch,capsys,tmp_path):
    source=seed(conn)
    job=workflow.claim(conn,'draft-worker',version_id=source['version'])
    draft=extraction(source)
    draft_file=tmp_path/'extraction.json'
    draft_file.write_text(json.dumps(draft))
    checkpoint_file=tmp_path/'checkpoint.json'
    checkpoint_file.write_text(json.dumps({'completed_pages':[1],'remaining':'Independent review'}))
    monkeypatch.setattr(db,'connect',lambda:nullcontext(conn))
    assert cli.main(['checkpoint',str(job['id']),'--agent','draft-worker','--token',str(job['token']),
                     '--section',str(source['section']),'--file',str(checkpoint_file),
                     '--draft-file',str(draft_file)])==0
    assert json.loads(capsys.readouterr().out)['job_id']==str(job['id'])
    draft_file.unlink()
    recovered=documents.packet(conn,source['version'])['checkpoints'][0]['payload']
    assert recovered['draft_snapshot']==draft
    assert recovered['draft_sha256']==hashlib.sha256(json.dumps(draft,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert recovered['completed_pages']==[1]
