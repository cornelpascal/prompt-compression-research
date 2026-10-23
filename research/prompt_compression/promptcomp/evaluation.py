from __future__ import annotations

import math
from pathlib import Path

from .compressors import result
from .data import ADAPTERS, perturb
from .io import atomic_json, complete_run, digest, file_hash, read_json, run_directory, source_hash
from .schema import CompressionRequest, CompressionResult
from .supervision import seed_all


def diagnostics(example, compressed):
    # These are lexical audit flags, not semantic-fidelity guarantees.
    import re
    original, text = example.prompt, compressed.text
    negations = {x.lower() for x in re.findall(r"\b(?:not|never|no|except|unless)\b", original, re.I)}
    present = {x.lower() for x in re.findall(r"\b(?:not|never|no|except|unless)\b", text, re.I)}
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    source_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", original))
    from .metrics import normalize
    return {"missing_negation_words": sorted(negations - present),
        "new_numeric_strings": sorted(numbers - source_numbers),
        "has_question_mark": "?" in text,
        "answer_only_match": any(normalize(text) == normalize(a.text) for a in example.answers),
        "answer_string_retained": any(a.text.lower() in text.lower() for a in example.answers)}


def evaluate(rows, cfg, counter, targets, method_factory, output, artifact_identity,
             methods=None, seeds=None, variant="canonical", calibration=None):
    if not rows or len({r.split for r in rows}) != 1 or rows[0].split not in ("validation", "test"):
        raise ValueError("Evaluation requires one non-training split")
    methods, seeds = tuple(methods or cfg.methods), tuple(seeds or cfg.seeds)
    if rows[0].split == "validation" and any(t.alias == cfg.transfer_target for t in targets):
        raise ValueError("Transfer target must remain excluded from validation and selection")
    policies = {}
    if calibration is not None:
        from .calibration import METHODS, validate_calibration
        if rows[0].split != "test" or variant != "canonical" or "original" not in methods or any(m not in (*METHODS, "original") for m in methods):
            raise ValueError("Calibrated evaluation requires canonical final test, attention methods, and original baseline")
        if {t.alias for t in targets} != {*cfg.training_targets, cfg.transfer_target}:
            raise ValueError("Calibrated final test requires all three receivers")
        for method in methods:
            if method in METHODS:
                for seed in seeds:
                    policies[method, seed] = validate_calibration(calibration, cfg, method, seed, artifact_identity)
    originals = rows
    if variant != "canonical":
        rows = [modified for row in rows if (modified := perturb(row, variant)) is not None]
        if not rows:
            raise ValueError("No examples eligible for the diagnostic variant")
    identity = {"stage": "evaluation", "config": cfg.model_dump(mode="json"), "methods": methods,
        "seeds": seeds, "data": digest([r.model_dump() for r in originals]), "variant": variant,
        "counter": counter.name, "artifacts": artifact_identity, "targets": [t.alias for t in targets], "code": source_hash()}
    if calibration is not None:
        identity["calibration"] = calibration
    with run_directory(output, identity) as root:
        if read_json(root / "manifest.json")["complete"]:
            records = load_results(root)
            return {"cases": len(records), "eligible_examples": read_json(root / "index.json")["eligible_examples"]}
        keys = []
        for method in methods:
            for seed in seeds:
                if (method, seed) in policies and policies[method, seed] is None:
                    continue
                seed_all(seed)
                compressor = method_factory(method, seed)
                for row in rows:
                    ratios = (1.,) if method in ("original", "question_only", "direct_answer") else cfg.ratios
                    if (method, seed) in policies:
                        ratios = (policies[method, seed],)
                    for ratio in ratios:
                        request = CompressionRequest(prompt=row.prompt,
                            budget_tokens=max(1, math.floor(counter.count(row.prompt) * ratio)))
                        if method.startswith("attention_"):
                            from .attention import question_span
                            request = request.model_copy(update={"focus_span": question_span(row)})
                        compression_key = digest([method, seed, row.id, ratio])
                        compression_path = root / "compressions" / f"{compression_key}.json"
                        if compression_path.exists():
                            compressed = CompressionResult.model_validate(read_json(compression_path))
                        elif method == "question_only":
                            text = "Answer the question. Return only the shortest answer, with no explanation.\nQuestion:\n" + row.question
                            compressed = result(request, text, counter)
                            atomic_json(compression_path, compressed.model_dump())
                        elif method == "direct_answer":
                            generated = compressor.generate(row.prompt, cfg.answer_max_tokens)
                            compressed = result(request, generated.text, counter, generated.seconds)
                            atomic_json(compression_path, compressed.model_dump())
                        else:
                            compressed = compressor.compress(request)
                            atomic_json(compression_path, compressed.model_dump())
                        selected_targets = (None,) if method == "direct_answer" else targets
                        for target in selected_targets:
                            alias = target.alias if target else "compressor_direct"
                            key = digest([method, seed, row.id, ratio, alias])
                            keys.append(key)
                            path = root / "cases" / f"{key}.json"
                            if path.exists():
                                continue
                            if target:
                                # Infrastructure failures stop the stage and remain retryable;
                                # they never become low-quality model results or cached answers.
                                answer = target.answer(compressed.text)
                                original_input_tokens = len(target.counter.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": row.prompt}], tokenize=True,
                                    add_generation_prompt=True, return_dict=False, **target.spec.chat_options))
                            else:
                                answer = {"text": compressed.text, "input_tokens": counter.count(row.prompt),
                                    "output_tokens": compressed.output_tokens, "seconds": compressed.seconds, "cache_hit": False}
                                original_input_tokens = answer["input_tokens"]
                            score = ADAPTERS[row.dataset].score(row, answer["text"])
                            valid = compressed.status == "ok" or method in ("original", "question_only", "direct_answer")
                            record = {"id": row.id, "group_id": row.group_id, "dataset": row.dataset,
                                "split": row.split, "variant": variant, "method": method, "seed": seed,
                                "target": alias, "ratio": ratio, "compression_key": compression_key,
                                "compression": compressed.model_dump(), "answer": answer,
                                "score": score, "budget_valid": valid,
                                "budgeted_score": score if valid else {"em": 0., "f1": 0.},
                                "original_request_tokens": original_input_tokens,
                                "total_seconds": compressed.seconds + answer["seconds"] if target else answer["seconds"],
                                "diagnostics": diagnostics(row, compressed)}
                            atomic_json(path, record)
                            if target and not target.endpoint:
                                target.unload()
                del compressor
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        atomic_json(root / "index.json", {"keys": keys,
            "checksums": {key: file_hash(root / "cases" / f"{key}.json") for key in keys},
            "eligible_examples": len(rows), "source_examples": len(originals)})
        complete_run(root, cases=len(keys))
        return {"cases": len(keys), "eligible_examples": len(rows)}


def load_results(directory):
    root = Path(directory)
    if not read_json(root / "manifest.json")["complete"]:
        raise ValueError("Evaluation is incomplete; resume it before reporting or selection")
    index = read_json(root / "index.json")
    if any(file_hash(root / "cases" / f"{key}.json") != index["checksums"][key] for key in index["keys"]):
        raise ValueError("Evaluation result checksum mismatch")
    return [read_json(root / "cases" / f"{key}.json") for key in index["keys"]]


def select_checkpoints(validation_directory, checkpoint_map, output):
    """Select SFT vs RL per method/seed, averaging all budgets and both solvers."""
    rows = load_results(validation_directory)
    manifest = read_json(Path(validation_directory) / "manifest.json")
    cfg = manifest["inputs"]["config"]
    if any(r["split"] != "validation" or r["variant"] != "canonical" for r in rows):
        raise ValueError("Selection requires canonical validation only")
    if set(manifest["inputs"]["targets"]) != set(cfg["training_targets"]):
        raise ValueError("Selection requires both training targets and excludes transfer")
    selected = {}
    for family in ("rewrite", "delete"):
        for seed in cfg["seeds"]:
            choices = []
            for method in (family + "_sft", family):
                cells = [r for r in rows if r["method"] == method and r["seed"] == seed]
                expected = {(t, ratio) for t in cfg["training_targets"] for ratio in cfg["ratios"]}
                if {(r["target"], r["ratio"]) for r in cells} != expected:
                    raise ValueError("Selection requires every budget and training target")
                score = sum(r["budgeted_score"]["f1"] for r in cells) / len(cells)
                choices.append((score, method))
            # Prefer cheaper supervised checkpoint on an exact tie.
            score, winner = max(choices, key=lambda x: (x[0], x[1].endswith("_sft")))
            key = f"{winner}:{seed}"
            recorded = manifest["inputs"]["artifacts"].get(key)
            from .training import checkpoint_identity
            current = {"path": str(Path(checkpoint_map[key]).resolve()), **checkpoint_identity(checkpoint_map[key])}
            if recorded != current:
                raise ValueError("Selection checkpoint does not match the evaluated artifact")
            selected[f"{family}:{seed}"] = {"method": winner, "path": checkpoint_map[f"{winner}:{seed}"],
                "checkpoint_identity": {k:v for k,v in current.items() if k != "path"}, "validation_f1": score}
    atomic_json(output, {"validation_identity": manifest["identity"], "selected": selected})
    return selected

