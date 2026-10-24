from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.request

from .io import atomic_json, digest, read_json
from .devices import resolve_device, synchronize


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    input_tokens: int
    output_tokens: int
    seconds: float
    exhausted: bool = False


class TokenCounter:
    def __init__(self, tokenizer, name="local"):
        self.tokenizer, self.name = tokenizer, name

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def count(self, text):
        return len(self.encode(text))

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @classmethod
    def from_spec(cls, spec):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(spec.model,
            revision=None if spec.revision.startswith("local:") else spec.revision,
            trust_remote_code=False)
        return cls(tokenizer, spec.model + "@" + spec.revision)


def chat_ids(tokenizer, prompt, options=None):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_dict=False, **(options or {}))


class CausalBackend:
    def __init__(self, spec, trainable=False, device=None, lora=True):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
        self.spec = spec
        self.counter = TokenCounter.from_spec(spec)
        self.tokenizer = self.counter.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = resolve_device(device)
        dtype = torch.bfloat16 if self.device.type == "xla" or (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        configuration = AutoConfig.from_pretrained(spec.model,
            revision=None if spec.revision.startswith("local:") else spec.revision, trust_remote_code=False)
        loader = AutoModelForImageTextToText if configuration.model_type == "mistral3" else AutoModelForCausalLM
        self.model = loader.from_pretrained(spec.model,
            revision=None if spec.revision.startswith("local:") else spec.revision,
            torch_dtype=dtype, trust_remote_code=False, attn_implementation="eager").to(self.device)
        if trainable and lora:
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(r=16, lora_alpha=32,
                target_modules="all-linear", lora_dropout=0., task_type="CAUSAL_LM"))
        if not trainable:
            self.model.requires_grad_(False)
        self.model.eval()

    def ids(self, prompt):
        return chat_ids(self.tokenizer, prompt, self.spec.chat_options)

    def context_limit(self):
        config = getattr(self.model.config, "text_config", self.model.config)
        return getattr(config, "max_position_embeddings", 8192)

    def generate(self, prompt, max_new_tokens=64, sample=False):
        import torch
        tokens = self.ids(prompt)
        if max_new_tokens < 1 or len(tokens) + max_new_tokens > self.context_limit():
            raise ValueError("Request exceeds model context; no silent truncation")
        ids = torch.tensor([tokens], device=self.device)
        synchronize(self.device)
        start = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens, do_sample=sample,
                temperature=1. if sample else None, top_p=1. if sample else None,
                top_k=0 if sample else None, pad_token_id=self.tokenizer.pad_token_id)
        synchronize(self.device)
        output = tuple(generated[0, len(tokens):].tolist())
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        exhausted = len(output) == max_new_tokens and (not output or output[-1] not in eos)
        return Generation(self.tokenizer.decode(output, skip_special_tokens=True), output,
            len(tokens), len(output), time.perf_counter() - start, exhausted)

    def token_logps(self, prompt, response_ids, model=None):
        import torch
        if not response_ids:
            raise ValueError("Response ids must be nonempty")
        prefix = self.ids(prompt)
        sequence = prefix + list(response_ids)
        if len(sequence) > self.context_limit():
            raise ValueError("Training sequence exceeds context")
        ids = torch.tensor([sequence], device=self.device)
        logits = (model or self.model)(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, len(prefix)-1:-1].float()
        targets = torch.tensor(response_ids, device=self.device)
        return logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)

    def response_ids(self, text):
        return tuple(self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id])


class Target:
    """Frozen solver, locally loaded or served at a revision-named endpoint."""
    def __init__(self, alias, spec, cache_dir, max_tokens=64, backend=None):
        self.alias, self.spec, self.max_tokens = alias, spec, max_tokens
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.counter = backend.counter if backend else TokenCounter.from_spec(spec)
        self.backend = backend
        self.endpoint = os.environ.get(spec.endpoint_env, "") if spec.endpoint_env else ""

    def answer(self, prompt, *, use_cache=True):
        if not use_cache:
            return {**self._answer_fresh(prompt), "cache_hit": False}
        identity = {"model": self.spec.model, "revision": self.spec.revision,
            "prompt": prompt, "max_tokens": self.max_tokens, "chat_options": self.spec.chat_options,
            "decoding": "greedy", "schema": 1}
        key = digest(identity)
        path = self.cache / f"{key}.json"
        with (self.cache / f"{key}.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                saved = read_json(path)
                if saved["identity"] != identity:
                    raise ValueError("Inference cache identity mismatch")
                return {**saved["result"], "cache_hit": True}
            result = self._answer_fresh(prompt)
            atomic_json(path, {"identity": identity, "result": result})
            return {**result, "cache_hit": False}

    def _answer_fresh(self, prompt):
        if self.endpoint:
            return self._remote(prompt)
        if self.backend is None:
            self.backend = CausalBackend(self.spec)
        generated = self.backend.generate(prompt, self.max_tokens)
        if generated.exhausted:
            raise ValueError("Target output limit exhausted; raise answer_max_tokens in a new run")
        return {"text": generated.text, "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens, "seconds": generated.seconds}

    def _remote(self, prompt):
        payload = {"model": self.alias + "@" + self.spec.revision,
            "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": self.max_tokens, "chat_template_kwargs": self.spec.chat_options}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("PROMPTCOMP_SERVER_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["PROMPTCOMP_SERVER_TOKEN"]
        for attempt in range(3):
            start = time.perf_counter()
            try:
                request = urllib.request.Request(self.endpoint.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.load(response)
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Target output limit exhausted")
                return {"text": choice["message"]["content"], "input_tokens": body["usage"]["prompt_tokens"],
                    "output_tokens": body["usage"]["completion_tokens"], "seconds": time.perf_counter()-start}
            except (OSError, KeyError, json.JSONDecodeError):
                if attempt == 2:
                    raise RuntimeError("Target service failed after three attempts; no failed response was cached") from None
                time.sleep(2 ** attempt)

    def unload(self):
        self.backend = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SolverPool:
    def __init__(self, targets, unload_local=True):
        self.targets, self.unload_local = targets, unload_local

    def score(self, example, prompt):
        from .data import ADAPTERS
        scores = []
        for target in self.targets:
            answer = target.answer(prompt)
            scores.append(ADAPTERS[example.dataset].score(example, answer["text"]))
            if self.unload_local and not target.endpoint:
                target.unload()
        return scores


class RemoteGenerator:
    """For teacher/baseline generation only; never used for policy likelihoods."""
    def __init__(self, alias, spec):
        self.alias, self.spec = alias, spec
        self.counter = TokenCounter.from_spec(spec)
        self.tokenizer = self.counter.tokenizer
        self.endpoint = os.environ[spec.endpoint_env]

    def generate(self, prompt, max_new_tokens=64, sample=False):
        import torch
        payload = {"model": self.alias + "@" + self.spec.revision,
            "messages": [{"role": "user", "content": prompt}], "temperature": 1. if sample else 0.,
            "top_p": 1., "top_k": -1, "max_tokens": max_new_tokens,
            "seed": int(torch.initial_seed() % (2**31)), "chat_template_kwargs": self.spec.chat_options}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("PROMPTCOMP_SERVER_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["PROMPTCOMP_SERVER_TOKEN"]
        for attempt in range(3):
            start = time.perf_counter()
            try:
                request = urllib.request.Request(self.endpoint.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.load(response)
                choice = body["choices"][0]
                text = choice["message"]["content"]
                return Generation(text, tuple(self.counter.encode(text)), body["usage"]["prompt_tokens"],
                    body["usage"]["completion_tokens"], time.perf_counter()-start, choice.get("finish_reason") == "length")
            except (OSError, KeyError, json.JSONDecodeError):
                if attempt == 2:
                    raise RuntimeError("Generation service failed; retry the stage") from None
                time.sleep(2 ** attempt)


def generator(alias, spec, device=None):
    if spec.endpoint_env and os.environ.get(spec.endpoint_env):
        return RemoteGenerator(alias, spec)
    return CausalBackend(spec, device=device)

