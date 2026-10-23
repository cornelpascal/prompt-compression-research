from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import re

from .io import atomic_json, digest, fetch_json, file_hash, read_json, read_jsonl, write_jsonl
from .schema import Example

SOURCE = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
INSTRUCTION = "Answer the question using the passage. Return only the shortest answer span, with no explanation."


def make_prompt(context, question, instruction=INSTRUCTION):
    return f"{instruction}\n\nPassage:\n{context}\n\nQuestion:\n{question}"


def prepare_squad(train_path, dev_path, output, seed=11):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Dataset output must be empty; prepared datasets are immutable")
    train, dev = read_json(train_path), read_json(dev_path)
    if any(str(x.get("version")) not in ("1.1", "v1.1") for x in (train, dev)):
        raise ValueError("Expected SQuAD 1.1 input")
    train_titles = [x["title"] for x in train["data"]]
    dev_titles = [x["title"] for x in dev["data"]]
    if len(set(train_titles)) != len(train_titles) or len(set(dev_titles)) != len(dev_titles) or set(train_titles) & set(dev_titles):
        raise ValueError("Duplicate or overlapping article titles")
    if len(train_titles) < 2:
        raise ValueError("At least two training articles are needed for a grouped split")
    titles = sorted(train_titles, key=lambda t: digest([seed, t]))
    validation_titles = set(titles[:max(1, math.ceil(len(titles) * .1))])
    buckets = defaultdict(list)
    seen_ids, context_splits = set(), {}
    for source, external in ((train, False), (dev, True)):
        for article in source["data"]:
            title = article["title"]
            split = "test" if external else ("validation" if title in validation_titles else "train")
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                context_hash = digest(context)
                if context_hash in context_splits and context_splits[context_hash] != split:
                    raise ValueError("An identical passage crosses splits")
                context_splits[context_hash] = split
                for qa in paragraph["qas"]:
                    if qa["id"] in seen_ids:
                        raise ValueError("Duplicate question id")
                    seen_ids.add(qa["id"])
                    row = Example(id=qa["id"], group_id=title, split=split,
                        prompt=make_prompt(context, qa["question"]), context=context, question=qa["question"],
                        answers=[{"text": a["text"], "start": a["answer_start"]} for a in qa["answers"]])
                    buckets[split].append(row.model_dump())
    root.mkdir(parents=True)
    files = {}
    for split in ("train", "validation", "test"):
        rows = sorted(buckets[split], key=lambda r: r["id"])
        path = root / f"{split}.jsonl"
        write_jsonl(path, rows)
        files[path.name] = {"sha256": file_hash(path), "examples": len(rows), "groups": len({r["group_id"] for r in rows})}
    manifest = {"dataset": "squad_v1", "version": "1.1", "seed": seed, "files": files,
        "source": {"train": {"url": SOURCE + "train-v1.1.json", "sha256": file_hash(train_path)},
                   "dev": {"url": SOURCE + "dev-v1.1.json", "sha256": file_hash(dev_path)}},
        "license": "CC-BY-SA-4.0", "test_is_official_dev": True,
        "split_policy": "90/10 article-grouped official train; official dev held out"}
    atomic_json(root / "manifest.json", manifest)
    return manifest


def download_squad(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("train-v1.1.json", "dev-v1.1.json"):
        path = root / name
        if not path.exists():
            atomic_json(path, fetch_json(SOURCE + name))
    return root / "train-v1.1.json", root / "dev-v1.1.json"


def load_examples(path, limit=None, seed=11, expected_split=None):
    path = Path(path)
    manifest = read_json(path.parent / "manifest.json")
    expected = manifest["files"].get(path.name, {}).get("sha256")
    if not expected or file_hash(path) != expected:
        raise ValueError("Dataset checksum mismatch")
    adapter = ADAPTERS[manifest["dataset"]]
    rows = [adapter.example_type.model_validate(r) for r in read_jsonl(path)]
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError("Empty data or duplicate example ids")
    if expected_split and any(r.split != expected_split for r in rows):
        raise ValueError("Wrong dataset split for this stage")
    rows.sort(key=lambda r: digest([seed, r.id]))
    return rows if limit is None else rows[:limit]


def perturb(row, variant):
    """Separate diagnostic sets; never mutate canonical SQuAD examples."""
    if variant == "paraphrase":
        instruction = "Use the passage to answer the question below. Output just the minimal answer text; omit commentary."
        return row.model_copy(update={"id": row.id + ":paraphrase", "prompt": make_prompt(row.context, row.question, instruction)})
    if variant != "entity":
        raise ValueError("Unknown diagnostic variant")
    # Replace one unambiguous, nonoverlapping answer string everywhere in the
    # passage and question. Cases with alternate surface forms are ineligible.
    values = {a.text for a in row.answers}
    if len(values) != 1:
        return None
    old = next(iter(values))
    if len(old) < 3 or not re.search(r"[A-Za-z]", old):
        return None
    replacement = "Entity" + digest(row.id)[:10]
    context = row.context.replace(old, replacement)
    question = row.question.replace(old, replacement)
    starts = [m.start() for m in re.finditer(re.escape(replacement), context)]
    if not starts:
        return None
    return Example(id=row.id + ":entity", group_id=row.group_id, split=row.split,
        prompt=make_prompt(context, question), context=context, question=question,
        answers=[{"text": replacement, "start": s} for s in starts])


class SquadAdapter:
    example_type = Example

    @staticmethod
    def score(example, prediction):
        from .metrics import squad_score
        return squad_score(prediction, [a.text for a in example.answers])

    build_prompt = staticmethod(make_prompt)
    perturb = staticmethod(perturb)


ADAPTERS = {"squad_v1": SquadAdapter()}


def register_adapter(name, adapter):
    if name in ADAPTERS:
        raise ValueError("Dataset adapter already registered")
    ADAPTERS[name] = adapter

