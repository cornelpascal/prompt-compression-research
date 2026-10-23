import json
import os
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]

def settings():
    env = {**dotenv_values(ROOT / '.env'), **os.environ}
    model_lock = ROOT / '.paperkb' / 'model.lock.json'
    if not model_lock.exists():
        model_lock = ROOT / 'embedding-model.lock.json'
    locked = json.loads(model_lock.read_text()) if model_lock.exists() else {}
    return {
        'database_url': env.get('PAPERKB_DATABASE_URL', 'postgresql://paperkb@127.0.0.1:55432/paperkb'),
        'embedding_model': 'BAAI/bge-m3',
        'embedding_revision': env.get('PAPERKB_EMBEDDING_REVISION') or locked.get('revision'),
        'embedding_device': env.get('PAPERKB_EMBEDDING_DEVICE', 'cpu'),
        'artifact_dir': str(ROOT / '.paperkb' / 'artifacts'),
        'docker_context': 'colima-paperkb',
    }
