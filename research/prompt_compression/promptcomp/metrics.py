from collections import Counter
import re
import string
import numpy as np


def normalize(text):
    # SQuAD 1.1 evaluation normalization, including ASCII punctuation only.
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def squad_score(prediction, references):
    def score(answer):
        p, a = normalize(prediction), normalize(answer)
        p_tokens, a_tokens = p.split(), a.split()
        overlap = sum((Counter(p_tokens) & Counter(a_tokens)).values())
        # Official v1.1 F1 is zero when there are no overlapping tokens.
        f1 = 2 * overlap / (len(p_tokens) + len(a_tokens)) if overlap else 0.
        return float(p == a), f1
    scores = [score(a) for a in references]
    if not scores:
        raise ValueError("At least one gold answer required")
    return {"em": max(x[0] for x in scores), "f1": max(x[1] for x in scores)}


def compression_reward(scores, output_tokens, budget, empty=False):
    if budget < 1 or not scores:
        raise ValueError("Positive budget and target scores required")
    if empty:
        return -1.
    return min(s["f1"] for s in scores) - 2 * max(0., output_tokens / budget - 1.)


def clustered_interval(values, groups, repetitions=2000, seed=11):
    """Resample articles, weighting their constituent cases (not article means)."""
    if not values or len(values) != len(groups):
        raise ValueError("Nonempty paired values/groups required")
    unique = sorted(set(groups))
    totals, counts = [], []
    for group in unique:
        selected = [v for v, g in zip(values, groups) if g == group]
        totals.append(sum(selected))
        counts.append(len(selected))
    totals, counts = np.asarray(totals), np.asarray(counts)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(unique), len(unique))
        samples.append(float(totals[chosen].sum() / counts[chosen].sum()))
    low, high = np.quantile(samples, [.025, .975])
    return {"mean": float(np.mean(values)), "low": float(low), "high": float(high), "articles": len(unique)}

