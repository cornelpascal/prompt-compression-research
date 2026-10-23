"""One local BGE-M3 process, pinned provenance, and revision-safe persistence."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .config import ROOT, settings

LOCK_KEY = (731991, 3)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def pin_model() -> dict:
    """Resolve once, then use the immutable hub commit for every operation."""
    config = settings()
    revision = config['embedding_revision']
    if revision and not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('PAPERKB_EMBEDDING_REVISION must be an immutable 40-character commit SHA; omit it to pin the current model once')
    if not revision:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(config['embedding_model'], revision='main').sha
        if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Model registry did not return an immutable commit SHA')
    lock = {'model': config['embedding_model'], 'revision': revision,
            'dimensions': 1024, 'normalized': True, 'similarity': 'cosine'}
    path = ROOT / '.paperkb' / 'model.lock.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(lock, indent=2) + '\n')
    tmp.replace(path)
    return lock


def load_model():
    from sentence_transformers import SentenceTransformer
    config = settings()
    lock = pin_model()
    model = SentenceTransformer(config['embedding_model'], revision=lock['revision'],
                                cache_folder=str(ROOT / '.paperkb' / 'models'),
                                device=config['embedding_device'], trust_remote_code=False)
    if model.get_sentence_embedding_dimension() != 1024:
        raise ValueError('BGE-M3 must return 1,024 dimensions')
    return model, lock


def encode(model, texts: list[str], batch_size: int = 8) -> list[list[float]]:
    import numpy as np
    # Reject truncation instead of silently dropping part of a concept statement.
    for text in texts:
        if len(model.tokenizer.encode(text, truncation=False)) > model.max_seq_length:
            raise ValueError('Embedding text exceeds model context; revise the atomic idea text')
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding shape or nonfinite values')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('Embedding normalization failed')
    return vectors.tolist()


def _lock(conn):
    # Session lock covers downloads, model initialization, inference and persistence.
    if not conn.execute('SELECT pg_try_advisory_lock(%s,%s) AS acquired', LOCK_KEY).fetchone()['acquired']:
        raise ValueError('Another embedding process is active; retry when it finishes')
    conn.commit()


def _unlock(conn):
    conn.rollback()
    conn.execute('SELECT pg_advisory_unlock(%s,%s)', LOCK_KEY)
    conn.commit()


def initialize(conn) -> dict:
    _lock(conn)
    try:
        model, lock = load_model()
        vector = encode(model, ['Prompt compression reduces context tokens while preserving task information.'])[0]
        return {**lock, 'smoke_test_dimensions': len(vector), 'local': True}
    finally:
        _unlock(conn)


def query_vector(conn, text: str) -> tuple[list[float], dict]:
    _lock(conn)
    try:
        model, lock = load_model()
        return encode(model, [text])[0], lock
    finally:
        _unlock(conn)


def embed(conn, version_id=None, batch_size: int = 8) -> dict:
    if batch_size < 1 or batch_size > 64:
        raise ValueError('batch_size must be between 1 and 64')
    _lock(conn)
    try:
        model, lock = load_model()
        versions = conn.execute('''SELECT v.id,p.corpus_id,v.extraction_revision FROM paper_versions v
            JOIN papers p ON p.id=v.paper_id JOIN corpus c ON c.id=p.corpus_id
            WHERE v.state IN ('reviewed','complete') AND v.reviewed_revision=v.extraction_revision
              AND c.state='open' AND (%s::uuid IS NULL OR v.id=%s::uuid)
            ORDER BY v.id''', (version_id,version_id)).fetchall()
        conn.commit()
        embedded, completed = 0, []
        for version in versions:
            rows = conn.execute('''SELECT i.*,e.model AS embedded_model,e.model_revision,
                e.text_sha256,e.revision AS embedded_revision FROM ideas i
                LEFT JOIN embeddings e ON e.idea_id=i.id WHERE i.version_id=%s AND i.active ORDER BY i.id''',
                (version['id'],)).fetchall()
            conn.commit()
            missing = [r for r in rows if r['embedded_model'] != lock['model'] or
                       r['model_revision'] != lock['revision'] or r['text_sha256'] != text_hash(r['embedding_text']) or
                       r['embedded_revision'] != r['revision']]
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start+batch_size]
                vectors = encode(model, [r['embedding_text'] for r in batch], batch_size)
                with conn.transaction():
                    corpus = conn.execute('SELECT state FROM corpus WHERE id=%s FOR UPDATE', (version['corpus_id'],)).fetchone()
                    current = conn.execute('SELECT * FROM paper_versions WHERE id=%s FOR UPDATE', (version['id'],)).fetchone()
                    if corpus['state'] != 'open' or current['reviewed_revision'] != version['extraction_revision'] or current['extraction_revision'] != version['extraction_revision']:
                        raise ValueError('Corpus or extraction changed during embedding; stale vectors were discarded')
                    for row, vector in zip(batch, vectors):
                        actual = conn.execute('SELECT embedding_text,revision,active FROM ideas WHERE id=%s FOR UPDATE', (row['id'],)).fetchone()
                        if not actual['active'] or actual['revision'] != row['revision'] or actual['embedding_text'] != row['embedding_text']:
                            raise ValueError('Idea changed during embedding')
                        conn.execute('''INSERT INTO embeddings(idea_id,revision,model,model_revision,text_sha256,embedding)
                            VALUES(%s,%s,%s,%s,%s,%s::vector) ON CONFLICT(idea_id) DO UPDATE SET
                            revision=excluded.revision,model=excluded.model,model_revision=excluded.model_revision,
                            text_sha256=excluded.text_sha256,embedding=excluded.embedding,created_at=now()''',
                            (row['id'],row['revision'],lock['model'],lock['revision'],text_hash(row['embedding_text']),json.dumps(vector)))
                    embedded += len(batch)
            with conn.transaction():
                corpus = conn.execute('SELECT state FROM corpus WHERE id=%s FOR UPDATE', (version['corpus_id'],)).fetchone()
                current = conn.execute('SELECT * FROM paper_versions WHERE id=%s FOR UPDATE', (version['id'],)).fetchone()
                if corpus['state'] != 'open' or current['reviewed_revision'] != version['extraction_revision'] or current['extraction_revision'] != version['extraction_revision']:
                    raise ValueError('Completion refused after a concurrent corpus/revision change')
                all_rows = conn.execute('''SELECT i.*,e.revision AS er,e.model,e.model_revision,e.text_sha256
                    FROM ideas i LEFT JOIN embeddings e ON e.idea_id=i.id WHERE i.version_id=%s AND i.active''',
                    (version['id'],)).fetchall()
                accepted = conn.execute('SELECT 1 FROM reviews WHERE version_id=%s AND revision=%s AND accepted',
                    (version['id'],version['extraction_revision'])).fetchone()
                if accepted and current['relevance_score'] is not None and all(
                    r['relevance_score'] is not None and r['er']==r['revision'] and r['model']==lock['model'] and
                    r['model_revision']==lock['revision'] and r['text_sha256']==text_hash(r['embedding_text']) for r in all_rows):
                    conn.execute("UPDATE paper_versions SET state='complete' WHERE id=%s", (version['id'],))
                    completed.append(str(version['id']))
        return {'embedded_ideas': embedded, 'completed_versions': completed, **lock}
    finally:
        _unlock(conn)
