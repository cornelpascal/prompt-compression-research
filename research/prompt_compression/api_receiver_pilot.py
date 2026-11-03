"""Exploratory SQuAD receiver comparison using GPT-6 Sol medium via Responses API."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
import urllib.request
from pathlib import Path

from colab_receiver_pilot import (DATA_URL, METHODS, answer_scores, bm25_order,
                                  sample_examples, select_sentences, sentences)
from secure_openai_config import DEFAULT_PATH, load_config


class BudgetTokenizer:
    """Local selection proxy; API usage supplies authoritative token counts."""

    def __init__(self):
        import tiktoken
        self.codec = tiktoken.get_encoding("o200k_base")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.codec.encode(text)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.codec.decode(ids)


def _summary(rows: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        complete = [row for row in subset if row["status"] == "completed" and row["em"] is not None]
        mean = lambda key: sum(row[key] for row in complete) / len(complete) if complete else None
        summary[method] = {"n": len(subset), "completed": len(complete),
                           "em": mean("em"), "f1": mean("f1"),
                           "input_tokens": mean("input_tokens"),
                           "output_tokens": mean("output_tokens"),
                           "reasoning_tokens": mean("reasoning_tokens"),
                           "seconds": mean("seconds"),
                           "selection_ms": mean("selection_ms"),
                           "answer_recall": mean("answer_in_selected_context")}
    return summary


def run(output: Path, config_path: Path, count: int, seed: int, max_output_tokens: int) -> dict:
    from openai import OpenAI

    if count < 1 or count > 100:
        raise ValueError("Count must be 1..100")
    if max_output_tokens < 128 or max_output_tokens > 8192:
        raise ValueError("max_output_tokens must be 128..8192")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    config = load_config(config_path)
    client = OpenAI(api_key=config["api_key"], max_retries=2, timeout=120)
    tokenizer = BudgetTokenizer()
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "train-v1.1.json"
    urllib.request.urlretrieve(DATA_URL, dataset_path)
    raw = dataset_path.read_bytes()
    examples = sample_examples(json.loads(raw), tokenizer, count, seed)
    rows = []
    journal_path = output / "cases.jsonl"
    for case_index, example in enumerate(examples):
        context = example["context"]
        parts = sentences(context)
        budget = max(1, len(tokenizer.encode(context)) // 2)
        variants = {"full": (context, False, 0.0)}
        started = time.perf_counter()
        variants["prefix"] = (tokenizer.decode(tokenizer.encode(context)[:budget]), False,
                              (time.perf_counter() - started) * 1000)
        order = list(range(len(parts)))
        started = time.perf_counter()
        random.Random(seed * 100000 + case_index).shuffle(order)
        selected, fallback = select_sentences(parts, order, budget, tokenizer)
        variants["random_sentences"] = (selected, fallback, (time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        selected, fallback = select_sentences(parts, bm25_order(parts, example["question"]), budget, tokenizer)
        variants["bm25_sentences"] = (selected, fallback, (time.perf_counter() - started) * 1000)
        for method in METHODS:
            selected, fallback, selection_ms = variants[method]
            started = time.perf_counter()
            response = client.responses.create(
                model="gpt-6-sol", reasoning={"effort": "medium"},
                instructions="Answer the question using only the passage. Return only the shortest answer span.",
                input=f"Passage:\n{selected}\n\nQuestion: {example['question']}",
                max_output_tokens=max_output_tokens, store=False)
            seconds = time.perf_counter() - started
            usage = response.usage
            output_details = getattr(usage, "output_tokens_details", None)
            status = str(response.status)
            prediction = (response.output_text or "").strip()
            em, f1 = answer_scores(prediction, example["answers"]) if status == "completed" and prediction else (None, None)
            row = {"id": example["id"], "article": example["article"], "method": method,
                         "status": status, "response_model": response.model,
                         "incomplete_reason": getattr(response.incomplete_details, "reason", None),
                         "context_tokens_proxy": len(tokenizer.encode(selected)),
                         "input_tokens": usage.input_tokens if usage else None,
                         "output_tokens": usage.output_tokens if usage else None,
                         "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
                         "seconds": seconds, "selection_ms": selection_ms,
                         "answer_in_selected_context": any(a.lower() in selected.lower() for a in example["answers"]),
                         "sentence_fallback": fallback, "em": em, "f1": f1,
                         "prediction": prediction, "gold_answers": example["answers"]}
            rows.append(row)
            with journal_path.open("a") as journal:
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                journal.flush()
        print(f"{case_index + 1}/{count}: {example['id']} complete", flush=True)
    result = {"status": "exploratory_api_receiver_pilot", "dataset": "SQuAD 1.1 official training split",
              "dataset_url": DATA_URL, "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "sample_ids": [item["id"] for item in examples], "sample_seed": seed,
              "model": "gpt-6-sol", "reasoning_effort": "medium",
              "token_budget_proxy": "tiktoken o200k_base; API usage is authoritative",
              "count": count, "context_budget_ratio": 0.5, "max_output_tokens": max_output_tokens,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
              "summary": _summary(rows), "cases": rows,
              "limitations": ["Exploratory sample from training articles, one seed and one receiver.",
                              "Selection uses a local token proxy; achieved API input lengths are reported separately.",
                              "Reasoning tokens are billed output tokens and do not appear in prediction text.",
                              "API latency includes network time and is not local GPU latency."]}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    args = parser.parse_args()
    run(args.output, args.config, args.count, args.seed, args.max_output_tokens)
