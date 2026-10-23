from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .io import atomic_json, digest, file_hash, read_jsonl, write_jsonl
from .schema import Example, PrivacyPolicy, Request

TEMPLATES = {
    "train": ("Customer record follows.", "Here are the case details.", "Please review this record."),
    "validation": ("An account contains these facts.", "Consider the following entry."),
    "test": ("The submitted case has the following contents.", "A clerk recorded this information."),
    "shift": ("This imported record requires a decision.", "The following facts describe the request."),
}


def generate_split(split, size, seed=11):
    """size base records, each followed by a private-code counterfactual."""
    if split not in TEMPLATES or size < 1:
        raise ValueError("Invalid split or size")
    rng = random.Random(int(digest(["synthetic-v1", split, seed])[:16], 16))
    rows = []
    for i in range(size):
        domain = ("support", "scheduling")[i % 2]
        regime = ("independent", "correlated", "essential")[(i // 2) % 3]
        entity = f"person-{split}-{seed}-{i}"
        customer = "customer-" + f"{rng.getrandbits(64):016x}"
        template_index = i % len(TEMPLATES[split])
        prefix = TEMPLATES[split][template_index]
        useful = rng.randrange(2)
        corr = .1 if split == "shift" else .9
        secret_bit = (useful if rng.random() < corr else 1 - useful) if regime == "correlated" else rng.randrange(2)
        group = ("amber", "blue")[secret_bit]
        if domain == "support":
            days = rng.randrange(1, 30) if useful else rng.randrange(31, 90)
            fields = {"elapsed_days": str(days), "refund_window_days": "30", "item_opened": "no"}
            query = "Is this purchase eligible for a refund? Answer eligible or ineligible."
            answer = "eligible" if useful else "ineligible"
        else:
            fields = {"requested_slot": "morning", "available_slot": "morning" if useful else "evening"}
            query = "Can the requested slot be booked? Answer available or unavailable."
            answer = "available" if useful else "unavailable"
        for variant in range(2):
            # Uniform independent secret; no secret is derived from the case id.
            code = "KEY-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(12))
            values = {"customer": customer, "private_code": code, "group": group, **fields,
                      "note": "The record was checked by the desk and entered into the daily queue."}
            lines = [f"{key}: {value}" for key, value in values.items()]
            if template_index % 2:
                lines.reverse()
            q = "Return the private_code exactly." if regime == "essential" else query
            gold = code if regime == "essential" else answer
            required = (code,) if regime == "essential" else tuple(f"{k}: {v}" for k, v in fields.items())
            rows.append(Example(
                id=f"{entity}-{variant}", split=split, entity_id=entity,
                template_id=f"{split}-{template_index}", pair_id=entity, domain=domain, regime=regime,
                request=Request(context=prefix + "\n" + "\n".join(lines), query=q,
                    privacy_policy=PrivacyPolicy(protected_attributes=("private_code", "group")),
                    public_auxiliary={"domain": domain}),
                answer=gold, secrets={"private_code": code, "group": group}, required_facts=required))
    return rows


def validate_splits(rows):
    ids, entities, templates = set(), {}, {}
    for row in rows:
        if row.id in ids:
            raise ValueError(f"Duplicate example id: {row.id}")
        ids.add(row.id)
        for index, key in ((entities, row.entity_id), (templates, row.template_id)):
            if key in index and index[key] != row.split:
                raise ValueError("Entity/template crosses dataset splits")
            index[key] = row.split


def generate_dataset(output, size=120, seed=11):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Dataset destination must be empty")
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [r for split in TEMPLATES for r in generate_split(split, size, seed)]
    validate_splits(all_rows)
    files = {}
    for split in TEMPLATES:
        selected = [r for r in all_rows if r.split == split]
        path = output / f"{split}.jsonl"
        write_jsonl(path, (r.model_dump() for r in selected))
        files[path.name] = {"sha256": file_hash(path), "examples": len(selected)}
    manifest = {"schema_version": 1, "generator": "synthetic-v1", "seed": seed,
        "base_records_per_split": size, "counterfactual_pairs": True, "files": files,
        "correlation": {"train": .9, "validation": .9, "test": .9, "shift": .1},
        "limitations": ["Structured synthetic English records", "Counterfactual intervenes on private_code only"]}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_examples(path):
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("files", {}).get(path.name, {}).get("sha256")
        if expected is None or expected != file_hash(path):
            raise ValueError("Dataset checksum missing or mismatched")
    provenance = Path(str(path)+".provenance.json")
    if provenance.exists() and json.loads(provenance.read_text())["output_sha256"] != file_hash(path):
        raise ValueError("External dataset checksum mismatch")
    rows = [Example.model_validate(r) for r in read_jsonl(path)]
    validate_splits(rows)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def fields(text):
    return {m.group(1).lower(): m.group(2).strip() for m in
            re.finditer(r"(?m)^([a-z_]+):\s*([^\n]*)", text)}


def import_external(source, output, mapping_path, source_url):
    """Explicit mapping: PII-Bench standard answers are masking labels, not task answers.

    Require separately annotated task answers, privacy labels and group identities.
    The importer never guesses a dataset version's schema or manufactures labels.
    """
    mapping = json.loads(Path(mapping_path).read_text())
    required = {"id", "entity_id", "template_id", "context", "query", "answer", "secrets", "required_facts"}
    if set(mapping) != required:
        raise ValueError(f"Mapping requires exactly {sorted(required)}")
    records = read_jsonl(source)
    converted = []
    for item in records:
        v = {k: item[field] for k, field in mapping.items()}
        converted.append(Example(id=str(v["id"]), split="external", entity_id=str(v["entity_id"]),
            template_id=str(v["template_id"]), pair_id=str(v["entity_id"]), domain="external", regime="unspecified",
            request=Request(context=v["context"], query=v["query"],
                privacy_policy=PrivacyPolicy(protected_attributes=tuple(v["secrets"]))),
            answer=v["answer"], secrets=v["secrets"], required_facts=v["required_facts"]))
    validate_splits(converted)
    write_jsonl(output, (r.model_dump() for r in converted))
    atomic_json(str(output) + ".provenance.json", {"source_url": source_url,
        "source_sha256": file_hash(source), "mapping": mapping, "output_sha256": file_hash(output),
        "evaluation_only": True, "examples": len(converted)})
    return {"examples": len(converted), "output": str(output)}
