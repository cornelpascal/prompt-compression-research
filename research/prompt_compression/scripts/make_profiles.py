"""Derive bounded profiles without changing architecture or evaluation defaults."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for prefix in ("", "attention."):
    for suffix in ("", ".locked"):
        source = root / "configs" / f"{prefix}full{suffix}.json"
        if not source.exists():
            continue
        for name, size in (("smoke", 32), ("pilot", 512)):
            config = json.loads(source.read_text())
            config.update(profile=name, train_limit=size, eval_limit=size, seeds=[11])
            (root / "configs" / f"{prefix}{name}{suffix}.json").write_text(json.dumps(config, indent=2) + "\n")

