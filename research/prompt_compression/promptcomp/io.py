from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import urllib.request


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(path):
    root = Path(path)
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(canonical(value) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def source_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    versions = {}
    for package in ("numpy", "pydantic", "torch", "transformers", "peft", "accelerate", "safetensors", "llmlingua", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


@contextmanager
def run_directory(path, identity):
    """Process-safe lock; abandoned processes release it automatically."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Run directory is already in use") from None
        manifest = root / "manifest.json"
        signature = digest(identity)
        if manifest.exists() and read_json(manifest)["identity"] != signature:
            raise ValueError("Run inputs changed; use a new output directory")
        if not manifest.exists():
            atomic_json(manifest, {"identity": signature, "inputs": identity, "environment": environment(), "complete": False})
        try:
            yield root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def complete_run(root, **extra):
    path = Path(root) / "manifest.json"
    atomic_json(path, {**read_json(path), **extra, "complete": True})


def fetch_json(url, attempts=3):
    """Only public metadata/datasets. Do not put credentials in URLs."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "promptcomp/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt + 1 == attempts:
                raise RuntimeError("Public JSON download failed; retry or supply local files") from None
            time.sleep(2 ** attempt)

