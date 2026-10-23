import copy

import pytest

from promptcomp.backends import TokenCounter
from promptcomp.calibration import calibrate, candidate_metrics, threshold_report, validate_calibration
from promptcomp.compressors import result
from promptcomp.data import load_examples
from promptcomp.evaluation import evaluate, load_results
from promptcomp.smoke import FixtureTarget, smoke_config


def validation_fixture(pc_models, pc_data, tmp_path, passing=True):
    base = smoke_config(pc_models)
    cfg = base.model_copy(update={"models": {**base.models, "attention": pc_models["causal"]},
        "ratios": (.25, .5, .75), "methods": ("original", "attention_frozen", "attention_sft")})
    counter = TokenCounter.from_spec(pc_models["causal"])
    targets = [FixtureTarget(t, cfg.models[t], counter) for t in cfg.training_targets]
    artifacts = {"attention_sft:11": {"manifest": "m", "state": "s"}}
    seen = []
    class Compressor:
        def __init__(self, method):
            self.method = method
        def compress(self, request):
            if self.method == "original":
                return result(request, request.prompt, counter)
            seen.append(request)
            fraction = request.budget_tokens / counter.count(request.prompt)
            # Deliberately non-monotonic quality: the middle retention setting fails.
            text = "blue" if passing and (fraction < .4 or fraction > .65) else "red"
            return result(request, text, counter)
    factory = lambda method, seed: Compressor(method)
    directory = tmp_path / "validation"
    evaluate(load_examples(pc_data / "validation.jsonl"), cfg, counter, targets, factory,
             directory, artifacts)
    assert seen and all(r.focus_span is not None for r in seen)
    artifact = calibrate(directory, tmp_path / "calibration.json", repetitions=50)
    return cfg, counter, artifacts, factory, artifact, directory


def test_nonmonotonic_selection_locked_test_and_identities(pc_models, pc_data, tmp_path):
    cfg, counter, artifacts, factory, artifact, directory = validation_fixture(pc_models, pc_data, tmp_path)
    entry = artifact["entries"]["attention_frozen:11"]
    assert [c["passes"] for c in entry["candidates"]] == [True, False, True]
    assert entry["ratio"] == .25
    assert not artifact["scientific_result"]
    assert validate_calibration(artifact, cfg, "attention_sft", 11, artifacts) == .25
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_calibration(artifact, cfg, "attention_sft", 11, {"attention_sft:11": {"manifest": "m", "state": "changed"}})
    edited = copy.deepcopy(artifact)
    edited["entries"]["attention_frozen:11"]["ratio"] = .9
    with pytest.raises(ValueError, match="checksum"):
        validate_calibration(edited, cfg, "attention_frozen", 11, artifacts)
    targets = [FixtureTarget(t, cfg.models[t], counter) for t in (*cfg.training_targets, cfg.transfer_target)]
    final = tmp_path / "test"
    evaluate(load_examples(pc_data / "test.jsonl"), cfg, counter, targets, factory,
             final, artifacts, calibration=artifact)
    assert {r["ratio"] for r in load_results(final) if r["method"] != "original"} == {.25}
    report = threshold_report(final, 50)
    assert report["entries"]["attention_sft:11"]["status"] == "passed"
    assert set(report["entries"]["attention_sft:11"]["targets"]) == {"qwen", "mistral", "phi"}
    with pytest.raises(ValueError, match="validation"):
        calibrate(final, tmp_path / "bad.json", repetitions=50)
    with pytest.raises(ValueError, match="final test"):
        evaluate(load_examples(pc_data / "validation.jsonl"), cfg, counter, targets[:2], factory,
                 tmp_path / "bad-eval", artifacts, calibration=artifact)


def test_no_qualifying_compression_is_explicit(pc_models, pc_data, tmp_path):
    cfg, counter, artifacts, factory, artifact, _ = validation_fixture(pc_models, pc_data, tmp_path, passing=False)
    assert all(e["status"] == "no_qualifying_compression" and e["ratio"] is None for e in artifact["entries"].values())
    assert validate_calibration(artifact, cfg, "attention_frozen", 11, artifacts) is None
    targets = [FixtureTarget(t, cfg.models[t], counter) for t in (*cfg.training_targets, cfg.transfer_target)]
    final = tmp_path / "test"
    evaluate(load_examples(pc_data / "test.jsonl"), cfg, counter, targets, factory, final, artifacts, calibration=artifact)
    assert {r["method"] for r in load_results(final)} == {"original"}
    assert threshold_report(final, 50)["entries"]["attention_sft:11"]["status"] == "no_qualifying_compression"


def test_calibration_rejects_missing_pairs_and_zero_baseline(pc_models, pc_data, tmp_path):
    cfg, _, _, _, _, directory = validation_fixture(pc_models, pc_data, tmp_path)
    rows = load_results(directory)
    incomplete = [r for r in rows if not (r["method"] == "attention_frozen" and r["ratio"] == .25 and r["target"] == "mistral")]
    with pytest.raises(ValueError, match="complete paired"):
        candidate_metrics(incomplete, "attention_frozen", 11, .25, cfg.training_targets, repetitions=50)
    for row in rows:
        if row["method"] == "original":
            row["score"]["f1"] = 0
    with pytest.raises(ValueError, match="positive baseline"):
        candidate_metrics(rows, "attention_frozen", 11, .25, cfg.training_targets, repetitions=50)


def test_benchmark_bypasses_answer_cache(pc_models, pc_data, tmp_path):
    from promptcomp.attention_benchmark import benchmark
    cfg, counter, artifacts, factory, artifact, _ = validation_fixture(pc_models, pc_data, tmp_path)
    calls = []
    class UncachedTarget(FixtureTarget):
        def answer(self, prompt, *, use_cache=True):
            assert use_cache is False
            calls.append((self.alias, prompt))
            return super().answer(prompt)
    targets = [UncachedTarget(t, cfg.models[t], counter) for t in cfg.training_targets]
    rows = load_examples(pc_data / "validation.jsonl")
    measured = benchmark(rows, cfg, counter, targets, factory, artifacts, artifact,
                         "attention_frozen", 11, tmp_path / "timing", repeats=2, warmup=1)
    assert len(calls) == 2 * (1 + 2 * len(rows) * 2)
    assert len(measured["summary"]) == 4
    assert all(c["solver_seconds"] >= 0 and c["total_seconds"] >= c["solver_seconds"] for c in measured["cases"])
    assert not measured["scientific_result"]


def test_final_failure_does_not_reselect(pc_models, pc_data, tmp_path):
    cfg, counter, artifacts, factory, artifact, _ = validation_fixture(pc_models, pc_data, tmp_path)
    class TransferFailure(FixtureTarget):
        def answer(self, prompt):
            answer = super().answer(prompt)
            if len(prompt.split()) < 10:
                answer["text"] = "wrong"
            return answer
    targets = [FixtureTarget(t, cfg.models[t], counter) for t in cfg.training_targets]
    targets.append(TransferFailure(cfg.transfer_target, cfg.models[cfg.transfer_target], counter))
    final = tmp_path / "final-failure"
    evaluate(load_examples(pc_data / "test.jsonl"), cfg, counter, targets, factory, final, artifacts, calibration=artifact)
    metrics = threshold_report(final, 50)["entries"]["attention_frozen:11"]
    assert metrics["status"] == "failed" and metrics["ratio"] == .25
    assert metrics["targets"]["qwen"]["passes"] and not metrics["targets"]["phi"]["passes"]
