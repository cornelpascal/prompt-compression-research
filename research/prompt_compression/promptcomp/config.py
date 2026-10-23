from pathlib import Path
import re

from .io import atomic_json, fetch_json, read_json, tree_hash
from .schema import StudyConfig


def load_config(path, locked=True):
    cfg = StudyConfig.model_validate(read_json(path))
    if locked:
        for spec in cfg.models.values():
            if Path(spec.model).is_dir():
                if spec.revision != "local:" + tree_hash(spec.model):
                    raise ValueError("Local model artifact changed or was not locked")
            elif not re.fullmatch(r"[0-9a-f]{40}", spec.revision):
                raise ValueError("All remote models must be pinned; run lock-models first")
    return cfg


def lock_models(config_path, output):
    cfg = load_config(config_path, locked=False).model_dump(mode="json")
    resolved = {}
    for spec in cfg["models"].values():
        name = spec["model"]
        if Path(name).is_dir():
            revision = "local:" + tree_hash(name)
        elif name in resolved:
            revision = resolved[name]
        else:
            revision = fetch_json("https://huggingface.co/api/models/" + name)["sha"]
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Model metadata did not return a commit hash")
        resolved[name] = revision
        spec["revision"] = revision
    atomic_json(output, cfg)
    return {"models": resolved, "output": str(output)}
