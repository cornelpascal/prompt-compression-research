import json
from pathlib import Path
import pytest

from leaklab.cli import main
from leaklab.data import generate_dataset,generate_split
from leaklab.io import atomic_json,read_jsonl
from leaklab.metrics import paired_difference
from leaklab.runner import run,export_attack_data,read_case
from leaklab.schema import ExperimentConfig


def test_end_to_end_resume_and_train_only_export(tmp_path):
    data = tmp_path/"data"
    generate_dataset(data,size=6)
    config = tmp_path/"config.json"
    atomic_json(config,ExperimentConfig(seeds=(11,),ratios=(.5,),bootstrap_samples=100).model_dump())
    directory = tmp_path/"run"
    result = run(config,data,directory)
    assert result["scientific_status"] == "engineering_pilot"
    assert result["evaluation_rows"] == 12*4*2*2
    cases = sorted((directory/"cases").glob("*.json"))
    times = {p.name:p.stat().st_mtime_ns for p in cases}
    assert run(config,data,directory) == result
    assert times == {p.name:p.stat().st_mtime_ns for p in cases}
    rows = json.loads((directory/"results.json").read_text())
    diff = paired_difference(rows,"redact_compress","original",samples=100)
    assert diff["mean_token_difference"] < 0
    export = tmp_path/"attack.jsonl"
    export_attack_data(directory,export)
    attacks = read_jsonl(export)
    assert {r["split"] for r in attacks} == {"train"}
    assert not {r["id"] for r in attacks}&{r["id"] for r in rows}
    for row in attacks:
        if row["observation"]["access"] == "public":
            assert row["labels"]["secrets"]["private_code"] not in json.dumps(row["observation"])
    cfg = json.loads(config.read_text())
    cfg["no_query"] = True
    atomic_json(config,cfg)
    with pytest.raises(ValueError,match="changed"):
        run(config,data,directory)


def test_paired_comparison_rejects_missing_or_duplicate_cases():
    a = {"id":"1","seed":11,"model":"rules","ratio":.5,"access":"context","attacker":"copy",
         "method":"a","pair_id":"p","context_tokens":10,"metrics":{"secret_recovery":1.}}
    b = {**a,"method":"b","metrics":{"secret_recovery":0.}}
    assert paired_difference([a,b],"a","b",samples=100)["mean"] == 1.
    with pytest.raises(ValueError,match="matching"):
        paired_difference([a],"a","b")
    with pytest.raises(ValueError,match="Duplicate"):
        paired_difference([a,a,b],"a","b")


def test_cli_invalid_config_returns_error_without_traceback(tmp_path,capsys):
    path = tmp_path/"bad.json"
    path.write_text('{"backend":"rules","methods":["fake"]}')
    assert main(["run","--config",str(path),"--data",str(tmp_path),"--output",str(tmp_path/"out")]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_lora_requires_explicit_training_seed_adapters():
    with pytest.raises(ValueError,match="per training seed"):
        ExperimentConfig(backend="hf",task_models=("real",),methods=("lora",))


def test_checkpoint_corruption_is_not_treated_as_completed(tmp_path):
    from leaklab.io import digest
    p = tmp_path/"case.json"
    value = {"answer":"eligible"}
    atomic_json(p,{**value,"checkpoint_sha256":digest(value)})
    assert read_case(p) == value
    atomic_json(p,{"answer":"changed","checkpoint_sha256":digest(value)})
    with pytest.raises(ValueError,match="checksum"):
        read_case(p)


def test_comparison_keeps_budget_cells_separate():
    from leaklab.metrics import grouped_comparison
    base = {"id":"one","seed":11,"model":"rules","access":"context","attacker":"copy","pair_id":"p",
        "domain":"support","regime":"independent","context_tokens":10,"budget":10,"metrics":{"secret_recovery":0.,"utility":1.}}
    rows = [{**base,"method":m,"ratio":r} for m in ("a","b") for r in (.25,.5)]
    results = grouped_comparison(rows,"a","b",samples=100)
    assert len(results)==2
    assert all(r["length_match_within_5_percent"] and r["utility_match_within_2_percentage_points"] for r in results)


def test_external_import_requires_annotations_and_preserves_unknown_regime(tmp_path):
    from leaklab.data import import_external,load_examples
    from leaklab.io import write_jsonl
    source = tmp_path/"source.jsonl"
    mapping = {k:k for k in ("id","entity_id","template_id","context","query","answer","secrets","required_facts")}
    atomic_json(tmp_path/"map.json",mapping)
    write_jsonl(source,[{"id":"external-1","entity_id":"external-entity","template_id":"external-template",
        "context":"A fictional person has code 81 and paid 10 credits.","query":"How much was paid?",
        "answer":"10 credits","secrets":{"code":"81"},"required_facts":["10 credits"]}])
    output = tmp_path/"imported.jsonl"
    import_external(source,output,tmp_path/"map.json","https://example.org/dataset")
    assert load_examples(output)[0].regime == "unspecified"
    assert load_examples(output)[0].split == "external"
