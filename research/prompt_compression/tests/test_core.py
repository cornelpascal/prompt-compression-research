import json
from pathlib import Path
import pytest

from promptcomp.compressors import align_extraction, render_mask, word_spans
from promptcomp.data import load_examples, perturb, prepare_squad
from promptcomp.io import atomic_json, read_json, run_directory
from promptcomp.metrics import compression_reward, normalize, squad_score, clustered_interval
from promptcomp.schema import CompressionRequest


def test_official_squad_normalization_and_multi_reference():
    assert normalize(" The, CAT! ") == "cat"
    assert squad_score("the blue sky", ["blue", "blue sky"]) == {"em": 1., "f1": 1.}
    assert squad_score("red blue", ["blue sky"])["f1"] == .5
    assert squad_score("", ["blue"]) == {"em": 0., "f1": 0.}
    assert squad_score("a", ["the"]) == {"em": 1., "f1": 0.}


def test_grouped_data_and_integrity(pc_data):
    groups = {}
    for split in ("train", "validation", "test"):
        for row in load_examples(pc_data / f"{split}.jsonl", expected_split=split):
            assert groups.get(row.group_id, split) == split
            groups[row.group_id] = split
            assert row.context[row.answers[0].start:row.answers[0].start+4] == "blue"
    path = pc_data / "train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_examples(path)


def test_source_offsets_and_article_overlap_rejected(pc_data, tmp_path):
    train = read_json(pc_data.parent / "train.json")
    train["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 0
    path = tmp_path / "bad.json"
    atomic_json(path, train)
    with pytest.raises(ValueError, match="offset"):
        prepare_squad(path, pc_data.parent / "dev.json", tmp_path / "bad-data")


def test_inference_schema_rejects_labels():
    with pytest.raises(ValueError):
        CompressionRequest(prompt="question", budget_tokens=2, answer="secret")
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            CompressionRequest(prompt="question", budget_tokens=invalid)


def test_extract_alignment_order_and_unicode():
    text = "Never change café or API-v2."
    mask = align_extraction(text, "Never café API-v2.")
    assert render_mask(text, word_spans(text), mask) == "Never café API-v2."
    with pytest.raises(ValueError):
        align_extraction(text, "API-v2. Never")
    with pytest.raises(ValueError):
        align_extraction(text, "Invented")


def test_reward_and_cluster_intervals():
    assert compression_reward([{"f1": .9}, {"f1": .6}], 15, 10) == pytest.approx(-.4)
    assert compression_reward([{"f1": 1}], 0, 10, empty=True) == -1
    result = clustered_interval([1, 0, 0], ["a", "a", "b"], repetitions=100)
    assert result["mean"] == pytest.approx(1/3)
    assert result["articles"] == 2


def test_run_lock_and_resume_identity(tmp_path):
    with run_directory(tmp_path / "run", {"input": 1}):
        with pytest.raises(RuntimeError, match="already in use"):
            with run_directory(tmp_path / "run", {"input": 1}):
                pass
    with pytest.raises(ValueError, match="inputs changed"):
        with run_directory(tmp_path / "run", {"input": 2}):
            pass


def test_counterfactual_changes_labels_without_changing_canonical(pc_data):
    row = load_examples(pc_data / "test.jsonl")[0]
    modified = perturb(row, "entity")
    assert modified.answers[0].text.startswith("Entity")
    assert modified.context[modified.answers[0].start:].startswith(modified.answers[0].text)
    assert row.answers[0].text == "blue"
    assert perturb(row, "paraphrase").context == row.context


def test_workload_estimate_does_not_treat_cpu_timing_as_gpu_time(pc_models, pc_data, tmp_path):
    from promptcomp.runtime import estimate
    from promptcomp.smoke import smoke_config
    cfg = smoke_config(pc_models)
    pilot = tmp_path / "pilot"
    for device in ("cpu", "cuda"):
        folder = pilot / device
        atomic_json(folder / "manifest.json", {"inputs": {"stage": "sft", "method": "rewrite", "seed": 11, "world_size": 2, "device_type": device}})
        atomic_json(folder / "state.json", {"step": 2, "elapsed_seconds": 10., "peak_memory_bytes": {"allocated": 0, "reserved": 0}})
    estimates = {e["source_device_type"]:e for e in estimate(cfg, pc_data, pilot)["training_estimates"]}
    assert estimates["cpu"]["estimated_training_gpu_hours"] is None
    assert estimates["cuda"]["estimated_training_gpu_hours"] > 0
