"""PostgreSQL connections and checksum-verified, transactional migrations."""
from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from .config import ROOT, settings


def connect(database_url: str | None = None):
    return psycopg.connect(database_url or settings()["database_url"], row_factory=dict_row)


def migrate(conn=None) -> list[str]:
    if conn is None:
        with connect() as owned:
            return migrate(owned)
    applied: list[str] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(731991, 1)")
        # Keep the shared extension outside disposable test schemas.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            name text PRIMARY KEY, sha256 text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now())""")
        for path in sorted((Path(ROOT) / "migrations").glob("*.sql")):
            sql = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(sql.encode()).hexdigest()
            previous = conn.execute("SELECT sha256 FROM schema_migrations WHERE name=%s", (path.name,)).fetchone()
            if previous:
                if previous["sha256"] != digest:
                    raise RuntimeError(f"Applied migration was changed: {path.name}")
                continue
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations(name,sha256) VALUES(%s,%s)", (path.name,digest))
            applied.append(path.name)
    return applied
