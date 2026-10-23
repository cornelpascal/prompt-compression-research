from pathlib import Path
import pytest

from promptcomp.backends import Generation, Target, TokenCounter
from promptcomp.compressors import BaselineCompressor, result
from promptcomp.data import load_examples
from promptcomp.evaluation import evaluate, load_results, select_checkpoints
from promptcomp.io import atomic_json, read_json
from promptcomp.schema import CompressionRequest
from promptcomp.smoke import FixtureTarget, smoke_config


def test_evaluation_same_compression_for_targets_and_resume(pc_models, pc_data, tmp_path):
    cfg = smoke_config(pc_models)
    counter = TokenCounter.from_spec(pc_models["causal"])
    rows = load_examples(pc_data / "test.jsonl")
    targets = [FixtureTarget(t, cfg.models[t], counter) for t in (*cfg.training_targets, cfg.transfer_target)]
    seen = []
    class Whole:
        def compress(self, request):
            seen.append(request)
            return result(request, request.prompt, counter)
    build = lambda method,seed: Whole()
    output = tmp_path / "eval"
    evaluate(rows, cfg, counter, targets, build, output, {}, methods=["rewrite"])
    assert len(seen) == len(rows)
    records = load_results(output)
    assert all(r["compression"]["status"] == "overflow" for r in records)
    assert all(r["budgeted_score"]["f1"] == 0 for r in records)
    assert all(r["score"]["f1"] == 1 for r in records)
    times = {p:p.stat().st_mtime_ns for p in (output / "cases").glob("*.json")}
    evaluate(rows, cfg, counter, targets, build, output, {}, methods=["rewrite"])
    assert len(seen) == len(rows)
    assert all(p.stat().st_mtime_ns == timestamp for p,timestamp in times.items())


def test_validation_cannot_use_transfer_target(pc_models, pc_data, tmp_path):
    cfg = smoke_config(pc_models)
    counter = TokenCounter.from_spec(pc_models["causal"])
    target = FixtureTarget(cfg.transfer_target, cfg.models[cfg.transfer_target], counter)
    with pytest.raises(ValueError, match="Transfer target"):
        evaluate(load_examples(pc_data / "validation.jsonl"), cfg, counter, [target], None, tmp_path / "bad", {})


def test_failure_then_resume_and_tamper_detection(pc_models, pc_data, tmp_path):
    cfg = smoke_config(pc_models)
    counter = TokenCounter.from_spec(pc_models["causal"])
    target = FixtureTarget("qwen", cfg.models["qwen"], counter)
    original = target.answer
    calls = []
    def fail_second(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("service interrupted")
        return original(prompt)
    target.answer = fail_second
    factory = lambda method,seed: BaselineCompressor(method, counter)
    path = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate(load_examples(pc_data / "test.jsonl"), cfg, counter, [target], factory, path, {}, methods=["original"])
    assert not read_json(path / "manifest.json")["complete"]
    assert len(list((path / "cases").glob("*.json"))) == 1
    target.answer = original
    evaluate(load_examples(pc_data / "test.jsonl"), cfg, counter, [target], factory, path, {}, methods=["original"])
    record = next((path / "cases").glob("*.json"))
    record.write_text(record.read_text()+"\n")
    with pytest.raises(ValueError, match="checksum"):
        load_results(path)


def test_target_cache_identity_and_no_labels(pc_models, tmp_path):
    counter = TokenCounter.from_spec(pc_models["causal"])
    class Backend:
        def __init__(self):
            self.counter, self.calls = counter, []
        def generate(self, prompt, max_new_tokens):
            self.calls.append(prompt)
            return Generation("blue", (1,), 3, 1, .1)
    backend = Backend()
    target = Target("qwen", pc_models["causal"], tmp_path / "cache", backend=backend)
    first, second = target.answer("public prompt"), target.answer("public prompt")
    assert not first["cache_hit"] and second["cache_hit"]
    assert backend.calls == ["public prompt"]
    target.answer("different prompt")
    assert len(backend.calls) == 2
    target.answer("public prompt", use_cache=False)
    assert backend.calls == ["public prompt", "different prompt", "public prompt"]
    assert target.answer("public prompt")["cache_hit"]


def test_selection_requires_full_validation_matrix(pc_models, pc_data, tmp_path):
    from promptcomp.training import checkpoint_identity
    cfg = smoke_config(pc_models)
    counter = TokenCounter.from_spec(pc_models["causal"])
    targets = [FixtureTarget(t, cfg.models[t], counter) for t in cfg.training_targets]
    build = lambda method,seed: BaselineCompressor("truncate", counter)
    output = tmp_path / "validation"
    mapping = {f"{method}:11": str(tmp_path / "checkpoints" / method) for method in ("rewrite_sft", "rewrite", "delete_sft", "delete")}
    for key, path in mapping.items():
        atomic_json(Path(path) / "manifest.json", {"complete": True, "identity": key})
        atomic_json(Path(path) / "state.json", {"step": 1})
    artifacts = {key: {"path": path, **checkpoint_identity(path)} for key,path in mapping.items()}
    evaluate(load_examples(pc_data / "validation.jsonl"), cfg, counter, targets, build, output, artifacts,
        methods=["rewrite_sft", "rewrite", "delete_sft", "delete"])
    selected = select_checkpoints(output, mapping, tmp_path / "selected.json")
    assert selected["rewrite:11"]["method"] == "rewrite_sft"
    assert selected["delete:11"]["method"] == "delete_sft"
    atomic_json(Path(mapping["rewrite_sft:11"]) / "state.json", {"step": 999})
    with pytest.raises(ValueError, match="evaluated artifact"):
        select_checkpoints(output, mapping, tmp_path / "bad-selection.json")


def test_report_outputs_real_plot_files(pc_models, pc_data, tmp_path):
    from promptcomp.reporting import report
    cfg = smoke_config(pc_models)
    counter = TokenCounter.from_spec(pc_models["causal"])
    targets = [FixtureTarget("qwen", cfg.models["qwen"], counter)]
    evaluate(load_examples(pc_data / "test.jsonl"), cfg, counter, targets,
        lambda method,seed: BaselineCompressor(method, counter), tmp_path / "eval", {}, methods=["original", "truncate"])
    report(tmp_path / "eval", tmp_path / "report", 100)
    assert (tmp_path / "report" / "tradeoff-qwen.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "report" / "summary.csv").exists()
