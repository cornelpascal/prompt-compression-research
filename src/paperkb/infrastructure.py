import os
import json
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from .config import ROOT, settings

def init_env():
    path = ROOT / '.env'
    if not path.exists():
        password = secrets.token_urlsafe(32)
        data = f'PAPERKB_DB_PASSWORD={password}\nPAPERKB_DATABASE_URL=postgresql://paperkb:{password}@127.0.0.1:55432/paperkb\n'
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(data)
    return {'environment_file': str(path), 'credentials_displayed': False}

def docker(*args, **kwargs):
    # Anonymous public-image access without inheriting a broken desktop credential helper.
    config_dir = ROOT / '.paperkb' / 'docker'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / 'config.json'
    config_file.write_text(json.dumps({'cliPluginsExtraDirs': [
        '/opt/homebrew/lib/docker/cli-plugins', '/usr/local/lib/docker/cli-plugins']}))
    socket = Path.home() / '.colima' / 'paperkb' / 'docker.sock'
    return subprocess.run(['docker', '--config', str(config_dir), '--host', f'unix://{socket}', *args],
                          cwd=ROOT, check=True, **kwargs)

def up():
    init_env()
    subprocess.run(['colima', 'start', 'paperkb', '--cpu', '4', '--memory', '6', '--disk', '40',
                    '--arch', 'aarch64', '--activate=false'], check=True, stdout=sys.stderr)
    docker('compose', 'up', '-d', '--wait', stdout=sys.stderr)
    return {'status': 'ready', 'host': '127.0.0.1', 'port': 55432}

def backup(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Backup destination already exists: {path}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix=f'.{path.name}.', suffix='.part', delete=False) as f:
            temporary = Path(f.name)
            docker('compose', 'exec', '-T', 'postgres', 'pg_dump', '-U', 'paperkb', '-Fc', 'paperkb', stdout=f)
            f.flush()
            os.fsync(f.fileno())
        # Exclusive publication: never overwrite a destination created while
        # pg_dump was running. The completed archive appears atomically.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'path': str(path), 'bytes': path.stat().st_size}

def restore(path, database):
    if database == 'paperkb':
        raise ValueError('Restore requires a new database name; active paperkb is protected.')
    if not database.replace('_', '').isalnum():
        raise ValueError('Database name must contain only letters, digits, and underscores.')
    # Validate/open input before creating any database. A failed restore leaves
    # an empty new database rather than a partially restored schema.
    with Path(path).open('rb') as f:
        docker('compose', 'exec', '-T', 'postgres', 'createdb', '-U', 'paperkb', database)
        docker('compose', 'exec', '-T', 'postgres', 'pg_restore', '-U', 'paperkb', '-d', database,
               '--single-transaction', '--exit-on-error', stdin=f)
    return {'database': database, 'source': str(Path(path).resolve())}
