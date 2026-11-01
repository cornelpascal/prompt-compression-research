"""Offline attention workflow with random tiny weights and fixture solvers."""
import argparse
import os
from pathlib import Path

from .io import atomic_json, canonical, digest


def run(output):
    import torch
    from .attention import AttentionBackend, attention_records
    from .backends import TokenCounter
    from .calibration import calibrate
    from .data import load_examples
    from .evaluation import evaluate
    from .reporting import report
    from .runtime import checkpoint_registry, factory
    from .smoke import EngineeringReward, FixtureTarget, FixtureTeacher, fixture_data, smoke_config, tiny_models
    from .supervision import generate_supervision, load_supervision
    from .training import train
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Attention smoke output must be a new directory")
    specs = tiny_models(root / "models")
    base = smoke_config(specs)
    cfg = base.model_copy(update={"models": {**base.models, "attention": specs["attention"]},
        "ratios": (.25, .5, .75, .95, .98), "methods": ("original", "attention_frozen", "attention_sft"),
        "attention_key_block": 16, "attention_query_block": 2})
    atomic_json(root / "config.json", cfg.model_dump(mode="json"))
    data = fixture_data(root / "data")
    rows = load_examples(data / "train.jsonl")
    counter = TokenCounter.from_spec(specs["causal"])
    generate_supervision(rows, cfg, FixtureTeacher(), EngineeringReward(), counter, root / "supervision")
    raw, identity = load_supervision(root / "supervision", "delete")
    records = attention_records(raw, rows)
    backend = AttentionBackend(specs["attention"], device=os.environ.get("PROMPTCOMP_DEVICE", "cpu"), key_block=16, query_block=2)
    before = {n: p.detach().clone() for n, p in backend.model.named_parameters()}
    train(backend, records, cfg, root / "attention" / "sft", "attention", "sft", 11,
          digest({"supervision": identity, "records": records}), checkpoint_every=1)
    changed = [n for n, p in backend.model.named_parameters() if not torch.equal(p, before[n])]
    if not changed or any(not n.startswith(("q_proj.", "k_proj.")) for n in changed):
        raise RuntimeError("Expected only query/key weight updates")
    registry_file = root / "registry.json"
    atomic_json(registry_file, {"attention_sft:11": str(root / "attention" / "sft")})
    registry, artifacts = checkpoint_registry(registry_file)
    targets = [FixtureTarget(t, cfg.models[t], counter) for t in (*cfg.training_targets, cfg.transfer_target)]
    build = factory(cfg, counter, registry)
    evaluate(load_examples(data / "validation.jsonl"), cfg, counter, targets[:2], build, root / "validation", artifacts)
    calibration = calibrate(root / "validation", root / "calibration.json", repetitions=100)
    evaluate(load_examples(data / "test.jsonl"), cfg, counter, targets, build, root / "test", artifacts, calibration=calibration)
    report(root / "validation", root / "validation-report", 100)
    report(root / "test", root / "test-report", 100)
    result = {"engineering_only": True, "changed_parameters": changed,
        "calibration_status": {k: e["status"] for k, e in calibration["entries"].items()}, "output": str(root)}
    atomic_json(root / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    print(canonical(run(parser.parse_args().output)))
