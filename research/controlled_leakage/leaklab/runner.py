from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import random
import time
import numpy as np

from .attacks import AttributeAttack, DirectCopyAttack
from .backends import HFBackend, RuleTask, task_prompt
from .compressors import ByteTokenizer, Compressor
from .data import load_examples, validate_splits
from .io import atomic_json, digest, environment, file_hash, read_jsonl, source_hash, write_jsonl
from .metrics import scores, summarize
from .observations import alr_features, alr_rank_diagnostic, observe, probability_view
from .schema import ExperimentConfig, Observation


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another process owns this run directory") from e
        try:
            yield directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def scope_id(seed, model, method, ratio, access):
    return digest({"seed": seed, "model": model, "method": method, "ratio": ratio, "access": access})[:24]


def read_case(path):
    value = json.loads(Path(path).read_text())
    expected = value.pop("checkpoint_sha256",None)
    if expected is None or digest(value) != expected:
        raise ValueError("Case checkpoint checksum mismatch")
    return value


def save_probabilities(directory, case_key, full, accesses, hidden_size):
    paths = {}
    for access in accesses:
        if access == "text":
            continue
        view = probability_view(full, access)
        relative = Path("probabilities") / f"{case_key}-{access}.npz"
        path = directory / relative
        path.parent.mkdir(exist_ok=True)
        if access == "full":
            rng = np.random.default_rng(2026)
            coordinates = rng.choice(view.vocabulary_size, min(hidden_size+1, view.vocabulary_size), replace=False)
            np.savez_compressed(path, features=alr_features(view, coordinates), selected_tokens=coordinates,
                vocabulary_size=view.vocabulary_size, access=access)
        else:
            np.savez_compressed(path, token_ids=view.token_ids, logprobs=view.logprobs,
                vocabulary_size=view.vocabulary_size, access=access)
        paths[access] = {"path": str(relative), "sha256": file_hash(path),
                         "representation": "alr_selected_coordinates" if access == "full" else "unrenormalized_topk"}
    return paths


def collect_case(directory, example, compressor, task, config, seed, model, method, ratio, protocol_id):
    key = digest([example.id, seed, model, method, ratio])
    path = directory / "cases" / f"{key}.json"
    if path.exists():
        cached = read_case(path)
        for trace in cached["probabilities"].values():
            if file_hash(directory / trace["path"]) != trace["sha256"]:
                raise ValueError("Corrupted probability checkpoint")
        return cached
    started = time.monotonic()
    request = example.request
    original = len(compressor.tokenizer.encode(request.context))
    budget = math.floor(original * ratio)
    compressed = compressor.compress(request.context, "" if config.no_query else request.query, request.privacy_policy, budget)
    if config.backend == "rules":
        answer = task.answer(compressed.text, request.query)
        inputs = len(compressor.tokenizer.encode(task_prompt(compressed.text, request.query)))
        outputs = len(compressor.tokenizer.encode(answer))
        probabilities = {}
    else:
        generated = task.answer(compressed.text, request.query, trace=any(a != "text" for a in config.probability_access), max_new_tokens=config.max_new_tokens)
        answer, inputs, outputs = generated.text, generated.input_tokens, generated.output_tokens
        probabilities = save_probabilities(directory, key, generated.logprobs, config.probability_access,
            task.model.config.hidden_size) if generated.logprobs is not None else {}
    observations = {access: observe(request, access, compressed.text, answer).model_dump()
                    for access in ("public", "context", *config.probability_access)}
    value = {"id": example.id, "split": example.split, "pair_id": example.pair_id, "domain": example.domain,
        "protocol_id":protocol_id,
        "regime": example.regime, "seed": seed, "model": model, "method": method, "ratio": ratio,
        "compression": compressed.model_dump(), "answer": answer, "observations": observations,
        "probabilities": probabilities, "original_tokens": original, "request_input_tokens": inputs,
        "request_output_tokens": outputs, "elapsed_seconds": time.monotonic()-started,
        # Explicit evaluator-only section; never passed to inference adapters.
        "evaluation": {"answer": example.answer, "secrets": example.secrets,
            "required_facts": example.required_facts, "compressed_target": compressed.text}}
    atomic_json(path, {**value,"checkpoint_sha256":digest(value)})
    return value


def result_row(case, example, access, attacker, guesses, public_guesses):
    c = case["compression"]
    return {**{k: case[k] for k in ("id", "pair_id", "domain", "regime", "seed", "model", "method", "ratio")},
        "access": access, "attacker": attacker, "prediction": guesses,
        "metrics": scores(example, c["text"], case["answer"], guesses, public_guesses),
        "context_tokens": c["token_count"], "original_tokens": case["original_tokens"], "budget": c["requested_budget"],
        "request_input_tokens": case["request_input_tokens"], "request_output_tokens": case["request_output_tokens"],
        "compressor_input_tokens": c["compressor_input_tokens"], "compressor_output_tokens": c["compressor_output_tokens"],
        "tokenizer": c["tokenizer"]}


def run(config_path, data_directory, output):
    config = ExperimentConfig.model_validate_json(Path(config_path).read_text())
    data_directory = Path(data_directory)
    train_path = data_directory / "train.jsonl"
    eval_path = Path(config.evaluation_file) if config.evaluation_file else data_directory / f"{config.split}.jsonl"
    train, evaluation = load_examples(train_path), load_examples(eval_path)
    if any(r.split != "train" for r in train) or any(r.split != config.split for r in evaluation):
        raise ValueError("Dataset split labels disagree with configuration")
    validate_splits(train + evaluation)
    policies = None
    if config.adaptive_policy:
        policies = json.loads(Path(config.adaptive_policy).read_text())
        if {e.id for e in evaluation} & set(policies["training_ids"]+policies["validation_ids"]):
            raise ValueError("Adaptive policy evaluation overlaps optimization data")
    with run_lock(output) as directory:
        adapter_hashes = {}
        for seed, adapter in config.compressor_adapters.items():
            adapter_hashes[seed] = {str(p.relative_to(adapter)): file_hash(p)
                for p in sorted(Path(adapter).rglob("*")) if p.is_file()}
            if not adapter_hashes[seed]:
                raise ValueError("Compressor adapter does not exist")
        identity = {"config": config.model_dump(), "code_sha256": source_hash(), "train_sha256": file_hash(train_path),
            "evaluation_sha256": file_hash(eval_path), "adapter_hashes": adapter_hashes,
            "policy_sha256":file_hash(config.adaptive_policy) if config.adaptive_policy else None}
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["identity"] != json.loads(json.dumps(identity)):
                raise ValueError("Run inputs/code changed; use a new output directory")
        else:
            atomic_json(manifest_path, {"schema_version": 1, "identity": identity, "environment": environment(),
                "status": "running", "scientific_status": "engineering_pilot" if config.backend == "rules" else "experiment",
                "limitations": ["Structured redaction is a synthetic baseline", "Adaptive uses independent reflective prompt search, not published GEPA reproduction", "Probability attacks require train-inverter and evaluate-attacker", "Deterministic baselines repeat across seeds; only separate adapter checkpoints reflect training variation"]})
        (directory / "cases").mkdir(exist_ok=True)
        rows = []
        all_cases = []
        for model in config.task_models:
            task = RuleTask() if config.backend == "rules" else HFBackend(model, config.revisions.get(model))
            tokenizer = ByteTokenizer() if config.backend == "rules" else task.counter
            if config.backend == "hf" and "full" in config.probability_access:
                head = task.model.get_output_embeddings().weight
                coordinates = np.random.default_rng(2026).choice(head.shape[0],min(task.model.config.hidden_size+1,head.shape[0]),replace=False)
                diagnostic = alr_rank_diagnostic(head,coordinates)
                atomic_json(directory/f"alr-rank-{digest(model)[:12]}.json",{"model":model,"revision":task.revision,
                    "selected_tokens":coordinates.tolist(),**diagnostic})
            rewriter = None
            if set(config.methods) & {"summarize", "adaptive"}:
                rewriter = HFBackend(config.compressor_model, config.revisions.get(config.compressor_model))
            lingua = None
            if "llmlingua2" in config.methods:
                from llmlingua import PromptCompressor
                lingua_model = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
                revision = config.revisions.get(lingua_model)
                if not revision or len(revision) != 40:
                    raise ValueError("Pin the LLMLingua model revision")
                lingua = PromptCompressor(model_name=lingua_model, use_llmlingua2=True, model_config={"revision":revision,"trust_remote_code":False})
            for seed in config.seeds:
                learned_rewriter = HFBackend(config.compressor_model, config.revisions.get(config.compressor_model),
                    adapter=config.compressor_adapters[str(seed)]) if "lora" in config.methods else None
                for method in config.methods:
                    compressor = Compressor(method, tokenizer, learned_rewriter if method == "lora" else rewriter, lingua)
                    for ratio in config.ratios:
                        if method == "adaptive":
                            compressor.instruction = policies["policies"][f"{model}|{ratio}"]
                        seed_everything(seed)
                        protocol_id = digest({"train_sha256":identity["train_sha256"],"code_sha256":identity["code_sha256"],
                            "backend":config.backend,"model":model,"revision":config.revisions.get(model),
                            "method":method,"seed":seed,"ratio":ratio,"no_query":config.no_query,
                            "compressor_model":config.compressor_model,"compressor_revision":config.revisions.get(config.compressor_model),
                            "adapter":adapter_hashes.get(str(seed)) if method=="lora" else None,
                            "policy_sha256":identity["policy_sha256"] if method=="adaptive" else None,
                            "lingua_revision":config.revisions.get("microsoft/llmlingua-2-xlm-roberta-large-meetingbank") if method=="llmlingua2" else None,
                            "max_new_tokens":config.max_new_tokens})
                        training_cases = [collect_case(directory, e, compressor, task, config, seed, model, method, ratio,protocol_id) for e in train]
                        evaluation_cases = [collect_case(directory, e, compressor, task, config, seed, model, method, ratio,protocol_id) for e in evaluation]
                        all_cases.extend(training_cases + evaluation_cases)
                        prior = AttributeAttack().fit((Observation.model_validate(c["observations"]["public"]), e.secrets) for c,e in zip(training_cases, train))
                        for access in ("context", *config.probability_access):
                            # Text baselines intentionally use only the textual part of a probability observation.
                            attack = AttributeAttack().fit((Observation.model_validate(c["observations"][access]), e.secrets) for c,e in zip(training_cases, train))
                            for case, e in zip(evaluation_cases, evaluation):
                                obs = Observation.model_validate(case["observations"][access])
                                public_guess = prior.predict(Observation.model_validate(case["observations"]["public"]))
                                for a in (DirectCopyAttack(), attack):
                                    name = a.name + ("-text_component_only" if access in ("top5","top20","full") else "")
                                    prior_guess = a.predict(Observation.model_validate(case["observations"]["public"])) if isinstance(a,DirectCopyAttack) else public_guess
                                    rows.append(result_row(case, e, access, name, a.predict(obs), prior_guess))
                        atomic_json(directory / "progress.json", {"model": model, "seed": seed, "method": method, "ratio": ratio, "completed_cases": len(all_cases)})
                if learned_rewriter is not None:
                    del compressor
                    del learned_rewriter
            if rewriter is not None:
                del rewriter
            del task
        # Rewrite aggregate artifacts atomically; per-case checkpoints are durable.
        atomic_json(directory / "results.json", rows)
        summaries = summarize(rows, directory, config.bootstrap_samples)
        manifest = json.loads(manifest_path.read_text())
        manifest["status"], manifest["evaluation_rows"] = "complete", len(rows)
        manifest["completed_attackers"] = ["direct_copy", "attribute_nb"]
        manifest["results_sha256"] = file_hash(directory/"results.json")
        atomic_json(manifest_path, manifest)
        return {"output": str(directory), "evaluation_rows": len(rows), "summary_groups": len(summaries), "scientific_status": manifest["scientific_status"]}


def export_attack_data(run_directory, output):
    """Trusted training export; labels stay outside the serialized Observation."""
    run_directory = Path(run_directory)
    rows = []
    for path in sorted((run_directory / "cases").glob("*.json")):
        case = read_case(path)
        if case["split"] != "train":
            continue
        for access, obs in case["observations"].items():
            rows.append({"split": "train", "id": case["id"], "scope": scope_id(case["seed"], case["model"], case["method"], case["ratio"], access),
                "protocol_id":case["protocol_id"],
                "scope_metadata": {k: case[k] for k in ("seed", "model", "method", "ratio")},
                "observation": Observation.model_validate(obs).model_dump(),
                "labels": {"secrets": case["evaluation"]["secrets"], "compressed_text": case["compression"]["text"]},
                "probabilities": case["probabilities"].get(access)})
    if not rows:
        raise ValueError("No completed training cases")
    write_jsonl(output, rows)
    atomic_json(str(output) + ".manifest.json", {"run_manifest_sha256": file_hash(run_directory / "manifest.json"),
        "run_directory": str(run_directory.resolve()), "sha256": file_hash(output), "rows": len(rows), "allowed_split": "train"})
    return {"rows": len(rows), "output": str(output)}
