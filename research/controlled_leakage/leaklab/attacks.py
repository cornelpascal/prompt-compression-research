from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re

from .observations import observation_prompt, probability_text
from .schema import Observation


def parse_prediction(text, attributes):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        match = re.search(r"\{[^{}]*\}", text)
        try:
            value = json.loads(match.group()) if match else {}
        except ValueError:
            value = {}
    if not isinstance(value, dict):
        value = {}
    return {a: value.get(a) if isinstance(value.get(a), str) else None for a in attributes}


class DirectCopyAttack:
    name = "direct_copy"

    def predict(self, observation: Observation):
        text = "\n".join([observation.query,*observation.public_auxiliary.values(),observation.visible_text])
        result = {}
        for attr in observation.protected_attributes:
            match = re.search(r"(?im)^" + re.escape(attr) + r"\s*:\s*([^\n]+)", text)
            value = match.group(1).strip() if match else None
            if attr == "private_code":
                code = re.search(r"\bKEY-[A-Z2-9]{12}\b", text)
                value = code.group() if code else None
            elif attr == "group" and value is None:
                group = re.search(r"\b(amber|blue)\b", text, re.I)
                value = group.group().lower() if group else None
            result[attr] = None if value == "[REDACTED]" else value
        return result


def features(observation):
    return Counter(re.findall(r"[a-z_]+", json.dumps(observation.model_dump(), sort_keys=True).lower()))


class AttributeAttack:
    """Train-only multinomial naive Bayes, for low-cardinality attributes.

    This is a diagnostic baseline; high-cardinality reconstruction uses the
    separate sequence model. It never consults evaluation labels at prediction.
    """
    name = "attribute_nb"

    def fit(self, pairs):
        pairs = list(pairs)
        if not pairs:
            raise ValueError("Attacker training is empty")
        self.stats, self.vocabulary = {}, set()
        attrs = pairs[0][0].protected_attributes
        for attr in attrs:
            values = {labels[attr] for _, labels in pairs}
            if len(values) > 32:
                continue
            counts, documents = defaultdict(Counter), Counter()
            for obs, labels in pairs:
                f = features(obs)
                self.vocabulary.update(f)
                counts[labels[attr]].update(f)
                documents[labels[attr]] += 1
            self.stats[attr] = (counts, documents)
        return self

    def predict(self, observation):
        f = features(observation)
        result = {}
        for attr in observation.protected_attributes:
            if attr not in self.stats:
                result[attr] = None
                continue
            counts, docs = self.stats[attr]
            scores = {}
            for label, n in docs.items():
                total = sum(counts[label].values()) + max(1, len(self.vocabulary))
                scores[label] = math.log(n / sum(docs.values())) + sum(
                    frequency * math.log((counts[label][token] + 1) / total)
                    for token, frequency in f.items() if token in self.vocabulary)
            result[attr] = max(sorted(scores), key=scores.get)
        return result


class NeuralAttack:
    name = "learned_reconstruction"

    def __init__(self, backend):
        self.backend = backend

    def predict(self, observation, probability_view=None):
        prompt = observation_prompt(observation)
        if probability_view is not None:
            if probability_view.access != observation.access:
                raise ValueError("Probability access and observation disagree")
            prompt += "\nProbability observations: " + probability_text(probability_view)
        text = self.backend.generate(prompt, max_new_tokens=128).text
        return parse_prediction(text, observation.protected_attributes)
