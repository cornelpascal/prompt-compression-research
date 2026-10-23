from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import re
import numpy as np

from .io import atomic_json


def normalize(text):
    return " ".join(re.findall(r"\w+", str(text).casefold()))


def exact(prediction, target):
    """Literal string recovery: case and punctuation are part of the secret."""
    return float(isinstance(prediction,str) and prediction == target)


def answer_accuracy(example, answer):
    if example.regime == "essential":
        return exact(answer,example.answer)
    return float(answer is not None and normalize(answer) == normalize(example.answer))


def scores(example, compressed, answer, guesses, public_guesses):
    recovery = {key: exact(guesses.get(key), value) for key, value in example.secrets.items()}
    prior = {key: exact(public_guesses.get(key), value) for key, value in example.secrets.items()}
    return {"utility": answer_accuracy(example,answer),
        "required_fact_recall": sum(normalize(x) in normalize(compressed) for x in example.required_facts) / max(1, len(example.required_facts)),
        "secret_recovery": float(np.mean(list(recovery.values()))),
        "public_recovery": float(np.mean(list(prior.values()))),
        "excess_recovery": float(np.mean([recovery[k] - prior[k] for k in recovery])),
        "attribute_recovery": recovery, "attribute_public_recovery": prior}


def cluster_interval(values, clusters, seed=0, samples=2000):
    if len(values) != len(clusters) or not values:
        raise ValueError("Aligned nonempty observations required")
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        if not math.isfinite(value):
            raise ValueError("Non-finite metric")
        grouped[cluster].append(value)
    # Equal entity weighting; repeated seeds and counterfactuals stay together.
    means = np.asarray([np.mean(v) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    chunk = max(1,min(samples,1000000//len(means)))
    boot = np.concatenate([means[rng.integers(0,len(means),(min(chunk,samples-start),len(means)))].mean(1)
        for start in range(0,samples,chunk)])
    return {"mean": float(means.mean()), "low": float(np.quantile(boot, .025)),
            "high": float(np.quantile(boot, .975)), "clusters": len(means), "rows": len(values),
            "interval": "paired-entity cluster bootstrap, 95%"}


GROUP = ("model", "method", "ratio", "access", "attacker", "domain", "regime")


def summarize(rows, output, bootstrap_samples=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP)].append(row)
    summaries = []
    for key, items in sorted(groups.items()):
        result = dict(zip(GROUP, key))
        attrs = sorted({a for x in items for a in x["metrics"]["attribute_recovery"]})
        for metric in ("utility", "required_fact_recall", "secret_recovery", "public_recovery", "excess_recovery"):
            result[metric] = cluster_interval([r["metrics"][metric] for r in items], [r["pair_id"] for r in items], samples=bootstrap_samples)
        result["attribute_recovery"] = {a: cluster_interval(
            [r["metrics"]["attribute_recovery"][a] for r in items if a in r["metrics"]["attribute_recovery"]],
            [r["pair_id"] for r in items if a in r["metrics"]["attribute_recovery"]], samples=bootstrap_samples) for a in attrs}
        for metric in ("context_tokens", "original_tokens", "request_input_tokens", "request_output_tokens", "compressor_input_tokens", "compressor_output_tokens"):
            result[metric] = float(np.mean([r[metric] for r in items]))
        result["budget_compliance"] = float(np.mean([r["context_tokens"] <= r["budget"] for r in items]))
        result["achieved_retention"] = float(np.mean([r["context_tokens"] / max(1, r["original_tokens"]) for r in items]))
        result["seed_means"] = {str(s): {m: float(np.mean([r["metrics"][m] for r in items if r["seed"] == s]))
            for m in ("utility", "secret_recovery")} for s in sorted({r["seed"] for r in items})}
        summaries.append(result)
    output = Path(output)
    atomic_json(output / "summary.json", summaries)
    if summaries:
        with (output / "summary.csv").open("w", newline="") as f:
            names = list(GROUP) + ["utility", "secret_recovery", "excess_recovery", "achieved_retention", "budget_compliance", "context_tokens"]
            writer = csv.DictWriter(f, fieldnames=names)
            writer.writeheader()
            for item in summaries:
                writer.writerow({k: item[k]["mean"] if isinstance(item[k], dict) and "mean" in item[k] else item[k] for k in names})
    return summaries


def paired_difference(rows, method, baseline, metric="secret_recovery", samples=2000):
    """Preserves pairing. It does NOT imply the methods have matched lengths."""
    keys = ("id", "seed", "model", "ratio", "access", "attacker")
    def indexed(name):
        result = {}
        for row in rows:
            if row["method"] != name:
                continue
            key = tuple(row[k] for k in keys)
            if key in result:
                raise ValueError("Duplicate paired observation")
            result[key] = row
        return result
    left, right = indexed(method), indexed(baseline)
    if not left or set(left) != set(right):
        raise ValueError("Methods must have exactly matching evaluation cases")
    ordered = sorted(left)
    result = cluster_interval([left[k]["metrics"][metric] - right[k]["metrics"][metric] for k in ordered],
        [left[k]["pair_id"] for k in ordered], samples=samples)
    result.update(method=method, baseline=baseline, metric=metric,
        mean_token_difference=float(np.mean([left[k]["context_tokens"] - right[k]["context_tokens"] for k in ordered])),
        interpretation="Paired difference; inspect lengths and utility before claiming a frontier improvement")
    return result


def grouped_comparison(rows,method,baseline,metric="secret_recovery",samples=2000):
    groups = defaultdict(list)
    keys = tuple(k for k in GROUP if k != "method")
    for row in rows:
        if row["method"] in (method,baseline):
            groups[tuple(row[k] for k in keys)].append(row)
    results = []
    for group,items in sorted(groups.items()):
        result = paired_difference(items,method,baseline,metric,samples)
        result.update(dict(zip(keys,group)))
        utility = paired_difference(items,method,baseline,"utility",samples)
        target = [r for r in items if r["method"]==method]
        reference = [r for r in items if r["method"]==baseline]
        tokens = float(np.mean([r["context_tokens"] for r in reference]))
        result["utility_difference"] = utility
        result["length_match_within_5_percent"] = abs(result["mean_token_difference"]) <= .05*max(1,tokens)
        result["utility_match_within_2_percentage_points"] = abs(utility["mean"]) <= .02
        result["all_target_cases_within_budget"] = all(r["context_tokens"] <= r["budget"] for r in target)
        result["scientific_status"] = "exploratory; no multiplicity correction or automatic superiority claim"
        results.append(result)
    if not results:
        raise ValueError("No requested methods in results")
    return results
