import pytest

from paperkb import rollout,workflow
from test_workflow import seed,submit_seed
from test_graph import seed_ready

def test_initial_extraction_revision_zero_is_dispatchable(conn,monkeypatch):
    item=seed(conn)
    monkeypatch.setattr(rollout,'pilot_status',lambda c:{'missing':[],'ready':False,'papers':[{'version_id':item['version']}]})
    result=rollout.next_work(conn,'pilot-worker')
    assert result['job']['revision']==1
    assert result['job']['version_id']==item['version']

def test_pilot_gate_excludes_other_queued_papers(conn,monkeypatch):
    pilot=seed(conn);other=seed(conn)
    monkeypatch.setattr(rollout,'pilot_status',lambda c:{'missing':[],'ready':False,'papers':[{'version_id':pilot['version']}]})
    workflow.claim(conn,'busy',version_id=pilot['version'])
    result=rollout.next_work(conn,'available')
    assert result['job'] is None
    assert conn.execute('SELECT state FROM jobs WHERE id=%s',(other['job'],)).fetchone()['state']=='queued'


def test_full_pool_does_not_prepare_registered_papers(conn,monkeypatch):
    for number in range(3):
        item=seed(conn)
        workflow.claim(conn,f'worker-{number}',version_id=item['version'])
    waiting=seed(conn)
    conn.execute("UPDATE paper_versions SET state='registered' WHERE id=%s",(waiting['version'],))
    monkeypatch.setattr(rollout,'pilot_status',lambda c:{'missing':[],'ready':True,'papers':[]})
    monkeypatch.setattr(rollout.documents,'prepare',lambda *args:(_ for _ in ()).throw(AssertionError('must not prepare')))
    assert rollout.next_work(conn,'fourth-worker')['job'] is None
    assert conn.execute('SELECT state FROM paper_versions WHERE id=%s',(waiting['version'],)).fetchone()['state']=='registered'


def test_frozen_corpus_is_not_prepared(conn,monkeypatch):
    item=seed(conn)
    conn.execute("UPDATE paper_versions SET state='registered' WHERE id=%s",(item['version'],))
    conn.execute("UPDATE corpus SET state='frozen' WHERE id=%s",(item['corpus'],))
    monkeypatch.setattr(rollout,'pilot_status',lambda c:{'missing':[],'ready':True,'papers':[]})
    monkeypatch.setattr(rollout.documents,'prepare',lambda *args:(_ for _ in ()).throw(AssertionError('must not prepare')))
    assert rollout.next_work(conn,'new-worker')['job'] is None


def test_pilot_requires_current_model_review_and_coverage(conn,monkeypatch):
    _,docs=seed_ready(conn,monkeypatch,count=8,with_coverage=False)
    for arxiv,doc in zip(rollout.PILOT_IDS,docs):
        conn.execute('UPDATE papers SET arxiv_id=%s WHERE id=%s',(arxiv,doc['paper']))
        section=conn.execute("INSERT INTO sections(version_id,ordinal,title,page_start,page_end) VALUES(%s,1,'Page 1',1,1) RETURNING id",(doc['version'],)).fetchone()['id']
        conn.execute("INSERT INTO section_coverage(version_id,section_id,revision,status,notes) VALUES(%s,%s,1,'covered','Complete')",(doc['version'],section))
    assert rollout.pilot_status(conn)['ready']
    monkeypatch.setenv('PAPERKB_EMBEDDING_REVISION','changed-model-revision')
    report=rollout.pilot_status(conn)
    assert not report['ready']
    assert all(any('embedding' in reason for reason in paper['blockers']) for paper in report['papers'])
    monkeypatch.setenv('PAPERKB_EMBEDDING_REVISION','test-model-revision')
    conn.execute("UPDATE section_coverage SET status='unreadable' WHERE version_id=%s",(docs[0]['version'],))
    assert not rollout.pilot_status(conn)['ready']


@pytest.mark.parametrize('dispatch', ['rollout', 'claim'])
@pytest.mark.parametrize('unfinished', ['correction', 'expired'])
def test_unfinished_paper_precedes_prepared_backlog(conn,monkeypatch,dispatch,unfinished):
    waiting=seed(conn)
    item=seed(conn)
    if unfinished=='correction':
        submit_seed(conn,item)
        review=workflow.claim(conn,'reviewer',kind='review',version_id=item['version'])
        workflow.review(conn,review['id'],'reviewer',review['token'],{
            'accepted':False,'notes':'A source claim needs correction.',
            'issues':['Correct the statement.'],
            'visual_evidence_confirmed':False,
        })
    else:
        old=workflow.claim(conn,'interrupted',version_id=item['version'])
        workflow.checkpoint(conn,old['id'],'interrupted',old['token'],item['section'],{'remaining':'Continue this page'})
        conn.execute("UPDATE jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",(old['id'],))
    conn.execute("UPDATE jobs SET created_at=clock_timestamp()-interval '1 day' WHERE id=%s",(waiting['job'],))
    conn.commit()
    monkeypatch.setattr(rollout,'pilot_status',lambda c:{'missing':[],'ready':True,'papers':[]})
    job=(rollout.next_work(conn,'next-worker')['job'] if dispatch=='rollout'
         else workflow.claim(conn,'next-worker'))
    assert job['version_id']==item['version']
    assert conn.execute('SELECT state FROM jobs WHERE id=%s',(waiting['job'],)).fetchone()['state']=='queued'
    if unfinished=='expired':
        assert job['attempts']==2 and job['token']!=old['token']
        assert conn.execute('SELECT payload FROM checkpoints WHERE job_id=%s',(job['id'],)).fetchone()['payload']['remaining']=='Continue this page'
