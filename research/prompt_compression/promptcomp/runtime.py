from __future__ import annotations

import os
from pathlib import Path

from .backends import CausalBackend, SolverPool, Target, TokenCounter, generator
from .compressors import BaselineCompressor, DeleteCompressor, RewriteCompressor, SelectorBackend
from .io import digest, read_json
from .training import checkpoint_identity, load_checkpoint


def solver_pool(cfg, cache):
    return SolverPool([Target(alias, cfg.models[alias], Path(cache) / alias, cfg.answer_max_tokens)
        for alias in cfg.training_targets])


def checkpoint_registry(path):
    records = read_json(path) if path else {}
    identity = {name: {"path": str(Path(folder).resolve()), **checkpoint_identity(folder)} for name, folder in records.items()}
    return records, identity


def factory(cfg, counter, registry, selection=None):
    def build(method, seed):
        if method in ("question_only",):
            return None
        if method in ("original", "truncate"):
            return BaselineCompressor(method, counter)
        if method == "llmlingua2":
            from huggingface_hub import snapshot_download
            from llmlingua import PromptCompressor
            spec = cfg.models["llmlingua2"]
            local = snapshot_download(spec.model, revision=spec.revision)
            lingua = PromptCompressor(model_name=local, use_llmlingua2=True,
                device_map="cuda" if __import__("torch").cuda.is_available() else "cpu")
            return BaselineCompressor(method, counter, lingua)
        if method == "untrained_rewrite":
            return RewriteCompressor(generator("rewriter", cfg.models["rewriter"]), counter)
        if method in ("attention_frozen", "attention_sft"):
            from .attention import AttentionBackend, AttentionCompressor
            if method == "attention_sft":
                checkpoint = registry[f"attention_sft:{seed}"]
                manifest = read_json(Path(checkpoint) / "manifest.json")
                inputs = manifest["inputs"]
                if not manifest["complete"] or (inputs["method"], inputs["stage"], inputs["seed"]) != ("attention", "sft", seed):
                    raise ValueError("Attention checkpoint method, seed, or stage mismatch")
                expected, actual = cfg.models["attention"], inputs["config"]["models"]["attention"]
                if (actual["model"], actual["revision"]) != (expected.model, expected.revision):
                    raise ValueError("Attention checkpoint base model/revision mismatch")
            backend = AttentionBackend(cfg.models["attention"], key_block=cfg.attention_key_block,
                                       query_block=cfg.attention_query_block)
            if method == "attention_sft":
                load_checkpoint(backend, checkpoint)
            return AttentionCompressor(backend, counter)
        if method not in ("rewrite", "rewrite_sft", "delete", "delete_sft", "rewrite_selected", "delete_selected", "direct_answer"):
            raise ValueError("Unknown evaluation method")
        family = "delete" if method.startswith("delete") else "rewrite"
        if method.endswith("_selected") or (method == "direct_answer" and selection):
            if not selection:
                raise ValueError("Selected methods require a validation selection file")
            entry = selection["selected"][f"{family}:{seed}"]
            checkpoint = entry["path"]
            if checkpoint_identity(checkpoint) != entry["checkpoint_identity"]:
                raise ValueError("Selected checkpoint changed after validation selection")
            checkpoint_method = entry["method"]
        else:
            checkpoint_method = family if method == "direct_answer" else method
            checkpoint = registry[f"{checkpoint_method}:{seed}"]
        manifest = read_json(Path(checkpoint) / "manifest.json")
        if not manifest["complete"] or manifest["inputs"]["method"] != family or manifest["inputs"]["seed"] != seed:
            raise ValueError("Checkpoint method, seed, or completion mismatch")
        expected_stage = "sft" if checkpoint_method.endswith("_sft") else "rl"
        if manifest["inputs"]["stage"] != expected_stage:
            raise ValueError("Checkpoint stage does not match its registry label")
        alias = "rewriter" if family == "rewrite" else "selector"
        expected = cfg.models[alias].model_dump(mode="json")
        actual = manifest["inputs"]["config"]["models"][alias]
        if (actual["model"], actual["revision"]) != (expected["model"], expected["revision"]):
            raise ValueError("Checkpoint base model/revision mismatch")
        if family == "rewrite":
            backend = CausalBackend(cfg.models["rewriter"], trainable=True)
            load_checkpoint(backend, checkpoint)
            if method == "direct_answer":
                return backend
            return RewriteCompressor(backend, counter)
        backend = SelectorBackend(cfg.models["selector"])
        load_checkpoint(backend, checkpoint)
        return DeleteCompressor(backend, counter)
    return build


def preflight(cfg):
    import torch
    from .devices import resolve_device
    from .io import environment
    selected = resolve_device() if os.environ.get("PROMPTCOMP_DEVICE") == "xla" else None
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    return {"environment": environment(), "cuda_available": torch.cuda.is_available(),
        "xla_available": selected is not None and selected.type == "xla",
        "selected_device": str(selected) if selected is not None else None,
        "cuda_version": torch.version.cuda, "devices": devices,
        "configured_endpoints": {alias: bool(spec.endpoint_env and os.environ.get(spec.endpoint_env)) for alias, spec in cfg.models.items()},
        "training_targets": cfg.training_targets, "transfer_target": cfg.transfer_target,
        "note": "Hardware discovery is not a model memory or distributed execution validation."}


def estimate(cfg, data_directory, pilot_directory):
    manifest = read_json(Path(data_directory) / "manifest.json")
    count = manifest["files"]["train.jsonl"]["examples"]
    count = min(count, cfg.train_limit) if cfg.train_limit else count
    budgets = len(cfg.ratios)
    teacher_calls = count * budgets * 2 * cfg.candidates
    estimates = []
    for path in sorted(Path(pilot_directory).rglob("state.json")):
        if not (path.parent / "manifest.json").exists():
            continue
        state = read_json(path)
        run = read_json(path.parent / "manifest.json")["inputs"]
        if "stage" not in run or run["stage"] not in ("sft", "rl") or state["step"] < 1:
            continue
        epochs = cfg.sft_epochs if run["stage"] == "sft" else cfg.rl_epochs
        world = run["world_size"]
        # SFT upper bound assumes an accepted example at every budget.
        steps = __import__("math").ceil(count * budgets / world) * epochs
        seconds = state["elapsed_seconds"] / state["step"] * steps
        estimates.append({"method": run["method"], "stage": run["stage"], "seed": run["seed"],
            "estimated_steps": steps, "estimated_seconds": seconds, "source_device_type": run["device_type"],
            "estimated_training_gpu_hours": seconds * world / 3600 if run["device_type"] == "cuda" else None,
            "estimated_training_tpu_hours": seconds * world / 3600 if run["device_type"] == "xla" else None,
            "world_size": world, "source": str(path), "measured_peak_memory_bytes": state["peak_memory_bytes"]})
    return {"training_examples": count, "teacher_generation_calls": teacher_calls,
        "supervision_solver_calls_upper_bound": teacher_calls * len(cfg.training_targets),
        "rl_solver_calls": count * budgets * cfg.candidates * len(cfg.training_targets) * 2 * len(cfg.seeds) * cfg.rl_epochs,
        "training_estimates": estimates,
        "limitations": ["Estimates assume the same hardware and topology as the pilot.",
            "GPU hours exclude teacher/solver serving unless measured separately.",
            "Caches may reduce actual solver work; SFT acceptance affects step counts."]}
