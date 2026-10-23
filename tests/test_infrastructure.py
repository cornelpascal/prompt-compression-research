import subprocess

import pytest

from paperkb import infrastructure


def test_failed_backup_does_not_publish_partial_archive(monkeypatch,tmp_path):
    target=tmp_path/'backup.dump'
    def fail(*args,**kwargs):
        kwargs['stdout'].write(b'partial')
        raise subprocess.CalledProcessError(1,['pg_dump'])
    monkeypatch.setattr(infrastructure,'docker',fail)
    with pytest.raises(subprocess.CalledProcessError):
        infrastructure.backup(target)
    assert not target.exists()
    assert list(tmp_path.iterdir())==[]


def test_backup_is_complete_and_never_overwrites(monkeypatch,tmp_path):
    target=tmp_path/'backup.dump'
    monkeypatch.setattr(infrastructure,'docker',lambda *args,**kwargs:kwargs['stdout'].write(b'complete dump'))
    result=infrastructure.backup(target)
    assert result['bytes']==len(b'complete dump') and target.read_bytes()==b'complete dump'
    with pytest.raises(FileExistsError):
        infrastructure.backup(target)
    assert target.read_bytes()==b'complete dump'


def test_missing_restore_input_has_no_database_side_effect(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(infrastructure,'docker',lambda *args,**kwargs:calls.append(args))
    with pytest.raises(FileNotFoundError):
        infrastructure.restore(tmp_path/'missing.dump','new_database')
    assert not calls


def test_restore_is_transactional_and_protects_active_database(monkeypatch,tmp_path):
    path=tmp_path/'backup.dump';path.write_bytes(b'fixture')
    calls=[]
    monkeypatch.setattr(infrastructure,'docker',lambda *args,**kwargs:calls.append(args))
    with pytest.raises(ValueError,match='protected'):
        infrastructure.restore(path,'paperkb')
    infrastructure.restore(path,'new_database')
    assert len(calls)==2 and '--single-transaction' in calls[1] and '--exit-on-error' in calls[1]
