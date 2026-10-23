"""Dataset-level attention calibration; no per-prompt accuracy guarantees."""
from __future__ import annotations

import math
from pathlib import Path

from .io import atomic_json, digest, read_json, source_hash
from .metrics import clustered_interval

METHODS = ("attention_frozen", "attention_sft")


def scorer_identity(cfg, method, seed, artifacts):
    if method not in METHODS:
        raise ValueError("Calibration requires an attention method")
    identity = {"model": cfg.models["attention"].model_dump(mode="json"), "method": method,
        "seed": seed, "key_block": cfg.attention_key_block, "query_block": cfg.attention_query_block,
        "precision": "float32", "code": source_hash(),
        "tokenizer": cfg.models[cfg.reference_tokenizer].model_dump(mode="json")}
    if method == "attention_sft":
        checkpoint = artifacts[f"attention_sft:{seed}"]
        identity["checkpoint"] = {k: checkpoint[k] for k in ("manifest", "state")}
    return identity


def quality_identity(cfg):
    return {"models": {t: cfg.models[t].model_dump(mode="json") for t in (*cfg.training_targets, cfg.transfer_target)},
            "answer_max_tokens": cfg.answer_max_tokens, "training_targets": list(cfg.training_targets),
            "transfer_target": cfg.transfer_target}


def candidate_metrics(rows, method, seed, ratio, targets, threshold=.98, repetitions=2000):
    metrics = {}
    fractions = []
    for target in targets:
        base = [r for r in rows if r["method"] == "original" and r["seed"] == seed and r["target"] == target]
        candidates = [r for r in rows if (r["method"], r["seed"], r["target"], r["ratio"]) == (method, seed, target, ratio)]
        baseline, selected = {r["id"]: r for r in base}, {r["id"]: r for r in candidates}
        if not baseline or len(baseline) != len(base) or len(selected) != len(candidates) or baseline.keys() != selected.keys():
            raise ValueError("Calibration requires complete paired baseline and candidate results")
        pairs = [(selected[key], baseline[key]) for key in sorted(baseline)]
        if any(r["group_id"] != b["group_id"] for r, b in pairs):
            raise ValueError("Candidate and baseline article groups differ")
        original = sum(b["score"]["f1"] for _, b in pairs) / len(pairs)
        compressed = sum(r["budgeted_score"]["f1"] for r, _ in pairs) / len(pairs)
        if original <= 0:
            raise ValueError("A positive baseline F1 is required for relative calibration")
        margin = [r["budgeted_score"]["f1"] - threshold * b["score"]["f1"] for r, b in pairs]
        metrics[target] = {"original_f1": original, "compressed_f1": compressed,
            "relative_f1": compressed / original, "passes": compressed + 1e-12 >= threshold * original,
            "threshold_margin": clustered_interval(margin, [r["group_id"] for r, _ in pairs], repetitions),
            "examples": len(pairs)}
        fractions.append(sum(r["compression"]["output_tokens"] for r, _ in pairs)
                         / sum(r["compression"]["input_tokens"] for r, _ in pairs))
    return {"ratio": ratio, "retained_fraction": sum(fractions) / len(fractions), "targets": metrics,
            "passes": all(v["passes"] for v in metrics.values())}


def calibrate(validation, output, threshold=.98, repetitions=2000):
    from .evaluation import load_results
    from .schema import StudyConfig
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Relative quality threshold must be in (0, 1]")
    rows = load_results(validation)
    manifest = read_json(Path(validation) / "manifest.json")
    inputs = manifest["inputs"]
    if not rows or {r["split"] for r in rows} != {"validation"} or inputs["variant"] != "canonical":
        raise ValueError("Calibration accepts canonical validation results only")
    if inputs["code"] != source_hash():
        raise ValueError("Calibration requires results from the current scorer code")
    cfg = StudyConfig.model_validate(inputs["config"])
    if set(inputs["targets"]) != set(cfg.training_targets):
        raise ValueError("Calibration requires both training targets and excludes transfer targets")
    methods = [m for m in inputs["methods"] if m in METHODS]
    if not methods or "original" not in inputs["methods"]:
        raise ValueError("Calibration requires attention and original-prompt results")
    entries = {}
    for method in methods:
        for seed in inputs["seeds"]:
            candidates = [candidate_metrics(rows, method, seed, ratio, cfg.training_targets, threshold, repetitions)
                          for ratio in cfg.ratios if ratio < 1]
            eligible = [c for c in candidates if c["passes"] and c["retained_fraction"] < 1]
            winner = min(eligible, key=lambda c: (c["retained_fraction"], c["ratio"])) if eligible else None
            entries[f"{method}:{seed}"] = {"status": "qualified" if winner else "no_qualifying_compression",
                "ratio": winner["ratio"] if winner else None, "candidates": candidates,
                "scorer": scorer_identity(cfg, method, seed, inputs["artifacts"])}
    payload = {"kind": "attention-calibration-v1", "threshold": threshold, "entries": entries,
        "validation_identity": manifest["identity"], "quality_identity": quality_identity(cfg),
        "profile": cfg.profile, "scientific_result": cfg.profile != "offline-engineering",
        "meaning": "Empirical dataset-level relative F1; not per-prompt confidence or a future-data guarantee."}
    artifact = {**payload, "identity": digest(payload)}
    atomic_json(output, artifact)
    return artifact


def validate_calibration(artifact, cfg, method, seed, artifacts):
    payload = {k: v for k, v in artifact.items() if k != "identity"}
    if artifact.get("kind") != "attention-calibration-v1" or artifact.get("identity") != digest(payload):
        raise ValueError("Calibration artifact checksum or kind mismatch")
    if artifact["quality_identity"] != quality_identity(cfg) or artifact["profile"] != cfg.profile:
        raise ValueError("Calibration receiver configuration/profile mismatch")
    entry = artifact["entries"][f"{method}:{seed}"]
    if entry["scorer"] != scorer_identity(cfg, method, seed, artifacts):
        raise ValueError("Calibration scorer/checkpoint identity mismatch")
    ratio = entry["ratio"]
    if entry["status"] == "no_qualifying_compression" and ratio is None:
        return None
    if entry["status"] != "qualified" or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ValueError("Invalid calibrated ratio")
    return ratio


def threshold_report(directory, repetitions=2000):
    """Assess the locked operating points without making another selection."""
    from .evaluation import load_results
    inputs = read_json(Path(directory) / "manifest.json")["inputs"]
    artifact = inputs.get("calibration")
    if artifact is None:
        return None
    rows = load_results(directory)
    output = {}
    for method in inputs["methods"]:
        if method not in METHODS:
            continue
        for seed in inputs["seeds"]:
            key = f"{method}:{seed}"
            entry = artifact["entries"][key]
            if entry["ratio"] is None:
                output[key] = {"status": "no_qualifying_compression", "targets": {}}
                continue
            metric = candidate_metrics(rows, method, seed, entry["ratio"], inputs["targets"], artifact["threshold"], repetitions)
            output[key] = {"status": "passed" if metric["passes"] else "failed", **metric}
    return {"threshold": artifact["threshold"], "calibration_identity": artifact["identity"],
            "entries": output, "interpretation": "Measured held-out dataset performance; no per-prompt guarantee."}
