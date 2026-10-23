from __future__ import annotations

import json
import re
from typing import Protocol

from .schema import Compression, PrivacyPolicy


class Tokenizer(Protocol):
    name: str
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Offline test instrument. Byte counts are never advertised as LLM tokens."""
    name = "utf8-bytes-pilot-only"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="ignore")


class HFTokenizer:
    def __init__(self, tokenizer, name):
        self.tokenizer, self.name = tokenizer, name

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def bounded_text(text, budget, tokenizer):
    if type(budget) is not int or budget < 0:
        raise ValueError("token_budget must be a nonnegative integer")
    ids = tokenizer.encode(text)
    if len(ids) <= budget:
        return text
    # Decode/re-encode can change BPE boundaries. Validate the actual final text.
    candidate = tokenizer.decode(ids[:budget])
    while len(tokenizer.encode(candidate)) > budget:
        candidate = candidate[:-1]
    return candidate


def redact(context, policy, query):
    """Transparent structured-data baseline; reads no evaluator labels.

    Query-relevant attributes remain visible, exposing the privacy conflict.
    For arbitrary natural text use the neural adaptive baseline.
    """
    for attribute in policy.protected_attributes:
        if re.search(r"\b" + re.escape(attribute) + r"\b", query, re.I):
            continue
        context = re.sub(r"(?im)^(" + re.escape(attribute) + r"\s*:)\s*[^\n]*", r"\1 [REDACTED]", context)
    return context


def select_relevant(context, query):
    keys = ("private_code",) if "private_code" in query else (
        ("elapsed_days", "refund_window_days", "item_opened") if "refund" in query.lower()
        else ("requested_slot", "available_slot") if "slot" in query.lower() else ())
    lines = context.splitlines()
    preferred = [line for line in lines if any(line.lower().startswith(k + ":") for k in keys)]
    return "\n".join(preferred + [line for line in lines if line not in preferred])


def rewrite_prompt(context, query, policy, budget, mode):
    instruction = {
        "summarize": "Summarize the context for answering the query. Preserve necessary facts.",
        "adaptive": "Anonymize and compress the context. Preserve facts needed for the query while minimizing recoverability of protected attributes, including indirect clues.",
        "lora": "Compress the context for the query while minimizing disclosure of protected attributes.",
    }[mode]
    # JSON delimits evidence; the model still needs empirical injection testing.
    return instruction + " Return only the rewritten context.\n" + json.dumps({
        "context": context, "query": query, "protected_attributes": list(policy.protected_attributes),
        "maximum_target_tokens": budget}, ensure_ascii=False)


class Compressor:
    def __init__(self, method, tokenizer, backend=None, llmlingua=None, instruction=None):
        self.method, self.tokenizer, self.backend, self.llmlingua = method, tokenizer, backend, llmlingua
        self.instruction = instruction

    def compress(self, context: str, query: str, privacy_policy: PrivacyPolicy, token_budget: int) -> Compression:
        if type(token_budget) is not int or token_budget < 0:
            raise ValueError("token_budget must be a nonnegative integer")
        input_tokens = output_tokens = 0
        if self.method == "original":
            text = context  # Control deliberately preserves original length.
        elif self.method == "empty":
            text = ""
        elif self.method in ("redact", "redact_compress"):
            text = redact(context, privacy_policy, query)
            if self.method == "redact_compress":
                text = select_relevant(text, query)
        elif self.method in ("summarize", "adaptive", "lora"):
            if self.backend is None:
                raise ValueError("A neural rewriting backend is required")
            prompt = rewrite_prompt(context, query, privacy_policy, token_budget, self.method)
            if self.method == "adaptive":
                if not self.instruction:
                    raise ValueError("Adaptive baseline requires a validation-selected instruction")
                prompt = self.instruction + "\n" + prompt
            generated = self.backend.generate(prompt, max_new_tokens=max(1, min(2048, token_budget * 2)), trace=False)
            text = generated.text
            input_tokens, output_tokens = generated.input_tokens, generated.output_tokens
        elif self.method == "llmlingua2":
            if self.llmlingua is None:
                raise ValueError("An actual LLMLingua-2 model is required")
            rate = min(1.0, max(.001, token_budget / max(1, len(self.tokenizer.encode(context)))))
            text = self.llmlingua.compress_prompt(context, rate=rate)["compressed_prompt"]
            input_tokens = len(self.llmlingua.tokenizer.encode(context, add_special_tokens=False))
        else:
            raise ValueError(f"Unknown compressor: {self.method}")
        raw = len(self.tokenizer.encode(text))
        # Preserve original/redaction as honest unconstrained controls.
        if self.method not in ("original", "redact"):
            text = bounded_text(text, token_budget, self.tokenizer)
        return Compression(text=text, token_count=len(self.tokenizer.encode(text)),
            requested_budget=token_budget, raw_token_count=raw, tokenizer=self.tokenizer.name,
            compressor_input_tokens=input_tokens, compressor_output_tokens=output_tokens)
