"""Exploratory, small real-receiver pilot; not a locked study result.

Uses SQuAD 1.1 *training* examples so the project's official-dev holdout stays
untouched. Selectors never receive gold answers. Run on a Colab T4 with the
promptcomp training extra installed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path

DATA_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
METHODS = ("full", "prefix", "random_sentences", "bm25_sentences")
STOP = frozenset("a an and are as at be by did do does for from how in is it of on or the to was were what when where which who why with".split())


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def sentences(context: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", context) if part.strip()]


def normalized(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [word for word in text.split() if word not in {"a", "an", "the"}]


def answer_scores(prediction: str, answers: list[str]) -> tuple[float, float]:
    guess = normalized(prediction)
    em = max(float(guess == normalized(answer)) for answer in answers)
    f1 = 0.0
    for answer in answers:
        gold = normalized(answer)
        common = sum((collections.Counter(guess) & collections.Counter(gold)).values())
        score = 1.0 if not guess and not gold else 2 * common / (len(guess) + len(gold)) if common else 0.0
        f1 = max(f1, score)
    return em, f1


def bm25_order(parts: list[str], question: str) -> list[int]:
    query = set(words(question))
    documents = [collections.Counter(words(part)) for part in parts]
    n = len(documents)
    avg_len = sum(map(lambda d: sum(d.values()), documents)) / n or 1.0
    df = collections.Counter(term for document in documents for term in document)
    scores = []
    for i, document in enumerate(documents):
        length = sum(document.values())
        score = 0.0
        for term in query:
            tf = document[term]
            if tf:
                idf = math.log1p((n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / avg_len))
        scores.append(score)
    return sorted(range(n), key=lambda i: (-scores[i], i))


def select_sentences(parts: list[str], order: list[int], budget: int, tokenizer) -> tuple[str, bool]:
    lengths = [len(tokenizer.encode(part, add_special_tokens=False)) for part in parts]
    kept, used = [], 0
    for i in order:
        if used + lengths[i] <= budget:
            kept.append(i)
            used += lengths[i]
    if not kept:
        # Rare fallback keeps the budget hard even when every sentence is long.
        first = tokenizer.encode(parts[order[0]], add_special_tokens=False)[:budget]
        return tokenizer.decode(first, skip_special_tokens=True), True
    return " ".join(parts[i] for i in sorted(kept)), False


def sample_examples(dataset: dict, tokenizer, count: int, seed: int) -> list[dict]:
    candidates = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if len(sentences(context)) < 4 or len(tokenizer.encode(context, add_special_tokens=False)) < 80:
                continue
            for qa in paragraph["qas"]:
                answers = [a["text"] for a in qa["answers"]]
                if answers:
                    candidates.append({"id": qa["id"], "article": article["title"], "context": context,
                                       "question": qa["question"], "answers": answers})
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected, articles = [], set()
    for row in candidates:
        if row["article"] not in articles:
            selected.append(row)
            articles.add(row["article"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible distinct articles for {count} examples")
    return selected


def run(output: Path, count: int, seed: int) -> dict:
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot needs a CUDA GPU")
    revision = model_info(MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    dataset_path = output / "train-v1.1.json"
    if not dataset_path.exists():
        urllib.request.urlretrieve(DATA_URL, dataset_path)
    raw = dataset_path.read_bytes()
    examples = sample_examples(json.loads(raw), tokenizer, count, seed)
    rows = []
    for case_index, example in enumerate(examples):
        context = example["context"]
        parts = sentences(context)
        budget = max(1, int(len(tokenizer.encode(context, add_special_tokens=False)) * 0.5))
        variants = {"full": (context, False, 0.0)}
        started = time.perf_counter()
        ids = tokenizer.encode(context, add_special_tokens=False)[:budget]
        variants["prefix"] = (tokenizer.decode(ids, skip_special_tokens=True), False,
                              (time.perf_counter() - started) * 1000)
        rng = random.Random(seed * 100000 + case_index)
        order = list(range(len(parts)))
        started = time.perf_counter()
        rng.shuffle(order)
        selected, fallback = select_sentences(parts, order, budget, tokenizer)
        variants["random_sentences"] = (selected, fallback, (time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        selected, fallback = select_sentences(parts, bm25_order(parts, example["question"]), budget, tokenizer)
        variants["bm25_sentences"] = (selected, fallback, (time.perf_counter() - started) * 1000)
        for method in METHODS:
            selected, fallback, selection_ms = variants[method]
            messages = [
                {"role": "system", "content": "Answer the question using only the passage. Return only the shortest answer span."},
                {"role": "user", "content": f"Passage:\n{selected}\n\nQuestion: {example['question']}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            new_tokens = generated[0, inputs.input_ids.shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            em, f1 = answer_scores(answer, example["answers"])
            rows.append({"id": example["id"], "article": example["article"], "method": method,
                         "context_tokens": len(tokenizer.encode(selected, add_special_tokens=False)),
                         "full_context_tokens": len(tokenizer.encode(context, add_special_tokens=False)),
                         "input_tokens": int(inputs.input_ids.shape[1]), "output_tokens": int(new_tokens.shape[0]),
                         "seconds": seconds, "selection_ms": selection_ms, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                         "answer_in_selected_context": any(a.lower() in selected.lower() for a in example["answers"]),
                         "sentence_fallback": fallback, "em": em, "f1": f1, "prediction": answer,
                         "gold_answers": example["answers"]})
        print(f"{case_index + 1}/{count}: {example['id']}", flush=True)
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        mean = lambda key: sum(row[key] for row in subset) / len(subset)
        summary[method] = {"n": len(subset), "em": mean("em"), "f1": mean("f1"),
                           "input_tokens": mean("input_tokens"), "output_tokens": mean("output_tokens"),
                           "seconds": mean("seconds"), "selection_ms": mean("selection_ms"),
                           "answer_recall": mean("answer_in_selected_context"),
                           "peak_gpu_bytes": max(row["peak_gpu_bytes"] for row in subset),
                           "sentence_fallbacks": sum(row["sentence_fallback"] for row in subset)}
    result = {"status": "exploratory_real_receiver_pilot", "dataset": "SQuAD 1.1 official training split",
              "dataset_url": DATA_URL, "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "sample_ids": [item["id"] for item in examples], "sample_seed": seed,
              "model": MODEL, "model_revision": revision, "dtype": "float16", "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "count": count, "context_budget_ratio": 0.5,
              "decode": {"max_new_tokens": 24, "do_sample": False},
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
              "summary": summary, "cases": rows,
              "limitations": ["Small exploratory sample from training articles, one seed and one receiver.",
                              "No tuning or conclusion about held-out SQuAD dev, larger receivers, or other tasks.",
                              "Token reduction is a proxy for prefill work; measured wall time includes runtime overhead.",
                              "Sentence selector uses query words only; it does not see gold answers."]}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run(args.output, args.count, args.seed)
