"""First-layer Q/K scoring. No value projections, MLPs, or language-model head."""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left, bisect_right
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .compressors import render_mask, result, word_spans
from .io import read_json
from .schema import CompressionRequest
from .devices import resolve_device, synchronize


def question_span(example):
    """Use the adapter's known suffix, never search for an ambiguous delimiter."""
    if not example.question.strip() or not example.prompt.endswith("Question:\n" + example.question):
        raise ValueError("Question is not an identifiable prompt suffix")
    return len(example.prompt) - len(example.question), len(example.prompt)


class AttentionScorer(nn.Module):
    def __init__(self, config, key_block=256, query_block=32):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
        if config.model_type != "qwen3" or config.layer_types[0] != "full_attention":
            raise ValueError("Attention scorer requires Qwen3 with a full-attention first layer")
        if key_block < 1 or query_block < 1:
            raise ValueError("Attention block sizes must be positive")
        self.config, self.key_block, self.query_block = config, key_block, query_block
        self.head_dim = config.head_dim
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        if self.heads % self.kv_heads:
            raise ValueError("Query heads must be divisible by key heads")
        # Large tensors are allocated only when their selected checkpoint tensors load.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device="meta")
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim,
                                bias=config.attention_bias, device="meta")
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim,
                                bias=config.attention_bias, device="meta")
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def project(self, ids, source_length):
        from transformers.models.qwen3.modeling_qwen3 import rotate_half, repeat_kv
        if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < source_length < ids.shape[1]:
            raise ValueError("Expected one source sequence followed by nonempty focus tokens")
        if ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Source plus focus exceeds context; no silent truncation")
        hidden = self.input_layernorm(self.embed_tokens(ids))
        q = self.q_norm(self.q_proj(hidden[:, source_length:]).view(1, -1, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden).view(1, -1, self.kv_heads, self.head_dim)).transpose(1, 2)
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden, positions)
        q = q * cos[:, source_length:].unsqueeze(1) + rotate_half(q) * sin[:, source_length:].unsqueeze(1)
        k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
        return q[0], repeat_kv(k, self.heads // self.kv_heads)[0]

    def token_scores(self, q, k, source_length):
        """Globally normalized causal attention, reduced without an N x N allocation.

        Recompute block intermediates in backward so training doesn't retain every
        attention block. Keys include the appended focus for exact normalization.
        """
        total = q.new_zeros(source_length, dtype=torch.float32)
        for qs in range(0, q.shape[1], self.query_block):
            query = q[:, qs:qs + self.query_block]
            qpos = torch.arange(source_length + qs, source_length + qs + query.shape[1], device=q.device)

            def logits(query, key, kpos, qpos):
                scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.head_dim ** -.5
                return scores.masked_fill(kpos[None, :] > qpos[:, None], torch.finfo(scores.dtype).min)

            def normalizer(query, key, kpos, qpos):
                return torch.logsumexp(logits(query, key, kpos, qpos), dim=-1)

            def probabilities(query, key, kpos, qpos, log_z):
                return (logits(query, key, kpos, qpos) - log_z[..., None]).exp().sum(dim=(0, 1))

            def run(fn, *args):
                if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
                    return checkpoint(fn, *args, use_reentrant=False)
                return fn(*args)

            log_z = None
            for ks in range(0, k.shape[1], self.key_block):
                key = k[:, ks:ks + self.key_block]
                kpos = torch.arange(ks, ks + key.shape[1], device=k.device)
                value = run(normalizer, query, key, kpos, qpos)
                log_z = value if log_z is None else torch.logaddexp(log_z, value)
            pieces = []
            for ks in range(0, source_length, self.key_block):
                end = min(source_length, ks + self.key_block)
                kpos = torch.arange(ks, end, device=k.device)
                pieces.append(run(probabilities, query, k[:, ks:end], kpos, qpos, log_z))
            total = total + torch.cat(pieces)
        return total / (q.shape[0] * q.shape[1])

    def forward(self, ids, source_length):
        return self.token_scores(*self.project(ids, source_length), source_length)


def _selected_weights(spec, model):
    """Read only the named tensors, even if their download shards contain more."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors import safe_open

    def fetch(name, optional=False):
        if Path(spec.model).is_dir():
            path = Path(spec.model) / name
            if path.exists():
                return path
            if optional:
                return None
            raise FileNotFoundError(f"Missing attention checkpoint file: {name}")
        try:
            return Path(hf_hub_download(spec.model, name, revision=spec.revision))
        except EntryNotFoundError:
            if optional:
                return None
            raise

    mapping = {}
    for name in model.state_dict():
        if name.startswith("embed_tokens."):
            original = "model." + name
        elif name.startswith("input_layernorm."):
            original = "model.layers.0." + name
        else:
            original = "model.layers.0.self_attn." + name
        mapping[name] = original
    index = fetch("model.safetensors.index.json", optional=True)
    files = defaultdict(list)
    weight_map = read_json(index)["weight_map"] if index else None
    for name, original in mapping.items():
        filename = weight_map[original] if weight_map else "model.safetensors"
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Invalid safetensors shard name")
        files[filename].append((name, original))
    state = {}
    for filename, names in files.items():
        with safe_open(fetch(filename), framework="pt", device="cpu") as handle:
            for name, original in names:
                state[name] = handle.get_tensor(original).float()
    model.load_state_dict(state, strict=True, assign=True)


class AttentionBackend:
    def __init__(self, spec, device=None, key_block=256, query_block=32):
        from transformers import AutoConfig, AutoTokenizer
        self.spec = spec
        self.device = resolve_device(device)
        kwargs = {"revision": None if spec.revision.startswith("local:") else spec.revision, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model, **kwargs)
        config = AutoConfig.from_pretrained(spec.model, **kwargs)
        self.model = AttentionScorer(config, key_block, query_block)
        _selected_weights(spec, self.model)
        self.model.to(self.device).eval().requires_grad_(False)
        # Same parameter set for training and checkpoint loading; inference uses no_grad.
        self.model.q_proj.requires_grad_(True)
        self.model.k_proj.requires_grad_(True)

    def word_scores(self, request):
        if request.focus_span is None:
            raise ValueError("Attention compression requires a focus_span")
        encoded = self.tokenizer(request.prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded["offset_mapping"]
        a, b = request.focus_span
        focus = [i for i, (start, end) in enumerate(offsets) if end > start and end > a and start < b]
        if not focus:
            raise ValueError("Focus span has no tokens")
        original = encoded["input_ids"]
        ids = torch.tensor([original + [original[i] for i in focus]], device=self.device)
        token_scores = self.model(ids, len(original))
        spans = word_spans(request.prompt)
        starts, ends = [a for a, _ in offsets], [b for _, b in offsets]
        words = []
        for start, end in spans:
            selected = [i for i in range(bisect_right(ends, start), bisect_left(starts, end))
                        if ends[i] > starts[i]]
            if not selected:
                raise ValueError("Word has no tokenizer offset mapping")
            words.append(token_scores[selected].mean())
        return torch.stack(words), spans


class AttentionCompressor:
    def __init__(self, backend, counter):
        self.backend, self.counter = backend, counter

    def compress(self, request):
        if request.focus_span is None:
            raise ValueError("Attention compression requires a focus_span")
        if self.backend.device.type == "cuda":
            torch.cuda.synchronize(self.backend.device)
            torch.cuda.reset_peak_memory_stats(self.backend.device)
        else:
            synchronize(self.backend.device)
        started = time.perf_counter()
        count = self.counter.count(request.prompt)
        spans = word_spans(request.prompt)
        if count <= request.budget_tokens:
            text, chosen = request.prompt, tuple(spans)
        else:
            with torch.no_grad():
                scores, spans = self.backend.word_scores(request)
            if not torch.isfinite(scores).all():
                raise RuntimeError("Non-finite attention scores")
            values = scores.detach().cpu().tolist()
            mask = [True] * len(spans)
            text = render_mask(request.prompt, spans, mask)
            for index in sorted(range(len(spans)), key=lambda i: (values[i], i)):
                if self.counter.count(text) <= request.budget_tokens:
                    break
                mask[index] = False
                text = render_mask(request.prompt, spans, mask)
            chosen = tuple(span for span, keep in zip(spans, mask) if keep)
        memory = 0
        synchronize(self.backend.device)
        if self.backend.device.type == "cuda":
            memory = torch.cuda.max_memory_allocated(self.backend.device)
        return result(request, text, self.counter, time.perf_counter() - started, count, chosen).model_copy(
            update={"peak_memory_bytes": memory})


def attention_records(records, rows):
    """Aggregate accepted deletion masks across budgets, one target per prompt."""
    by_id, grouped = {row.id: row for row in rows}, defaultdict(list)
    for record in records:
        if record["example_id"] not in by_id:
            continue
        row = by_id[record["example_id"]]
        if record["split"] != "train" or row.split != "train" or record["method"] != "delete":
            raise ValueError("Attention supervision requires training deletion masks")
        if record["request"]["prompt"] != row.prompt:
            raise ValueError("Supervision prompt differs from dataset")
        mask = record["selected"]["mask"]
        if len(mask) != len(word_spans(row.prompt)) or not any(mask) or any(type(x) is not bool for x in mask):
            raise ValueError("Invalid supervised word mask")
        grouped[row.id].append([int(x) / sum(mask) for x in mask])
    output = []
    for identity in sorted(grouped):
        row, masks = by_id[identity], grouped[identity]
        target = [sum(values) / len(masks) for values in zip(*masks)]
        request = CompressionRequest(prompt=row.prompt, budget_tokens=1, focus_span=question_span(row))
        output.append({"example_id": identity, "split": "train", "request": request.model_dump(), "attention_target": target})
    if not output:
        raise ValueError("No accepted training masks for attention")
    return output


def attention_loss(backend, record):
    scores, _ = backend.word_scores(CompressionRequest.model_validate(record["request"]))
    distribution = scores / scores.sum()
    target = torch.tensor(record["attention_target"], device=scores.device, dtype=scores.dtype)
    if target.shape != scores.shape or not torch.isfinite(target).all() or (target < 0).any() or not torch.isclose(target.sum(), target.new_tensor(1.)):
        raise ValueError("Invalid attention target distribution")
    return -(target * distribution.clamp_min(torch.finfo(scores.dtype).tiny).log()).sum()
