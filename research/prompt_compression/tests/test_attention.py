import copy
from pathlib import Path

import pytest
import torch

from promptcomp.attention import AttentionBackend, AttentionCompressor, attention_records, question_span
from promptcomp.backends import TokenCounter
from promptcomp.data import load_examples
from promptcomp.io import digest, tree_hash
from promptcomp.schema import CompressionRequest, ModelSpec
from promptcomp.smoke import smoke_config
from promptcomp.training import load_checkpoint, train


@pytest.fixture
def attention_model(pc_models, tmp_path):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(pc_models["causal"].model)
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=512, attention_bias=True, attention_dropout=0.,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    path = tmp_path / "qwen"
    model.save_pretrained(path, max_shard_size="5KB")
    tokenizer.save_pretrained(path)
    return ModelSpec(model=str(path), revision="local:" + tree_hash(path)), model


def test_subset_loader_and_exact_first_layer(attention_model, monkeypatch):
    spec, full = attention_model
    from transformers import AutoModelForCausalLM, AutoModel
    def forbidden(*a, **kw):
        raise AssertionError("Full model loader must not run")
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden)
    backend = AttentionBackend(spec, device="cpu", key_block=3, query_block=2)
    assert not any(x in n for n, _ in backend.model.named_parameters() for x in ("v_proj", "o_proj", "mlp", "lm_head"))
    source = torch.tensor([[5, 8, 9, 12, 13, 15, 16]])
    ids = torch.cat([source, source[:, -3:]], dim=1)
    with torch.no_grad():
        expected = full(ids, output_attentions=True, use_cache=False).attentions[0][0, :, -3:, :7].mean(dim=(0, 1))
        actual = backend.model(ids, 7)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_blockwise_gradients_match_dense(attention_model):
    spec, _ = attention_model
    backend = AttentionBackend(spec, device="cpu", key_block=2, query_block=1)
    dense = copy.deepcopy(backend.model)
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12]])
    actual = backend.model(ids, 8)
    q, k = dense.project(ids, 8)
    logits = q @ k.transpose(-1, -2) * dense.head_dim ** -.5
    logits = logits.masked_fill(torch.arange(12)[None, :] > torch.arange(8, 12)[:, None], torch.finfo(logits.dtype).min)
    expected = logits.softmax(-1)[..., :8].mean(dim=(0, 1))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    weights = torch.arange(1., 9.)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for (name, p), (_, ref) in zip(backend.model.named_parameters(), dense.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, ref.grad, rtol=3e-4, atol=2e-6)


def test_attention_training_freezes_non_qk_and_recovers(attention_model, pc_models, pc_data, tmp_path):
    spec, _ = attention_model
    cfg = smoke_config(pc_models).model_copy(update={"models": {**smoke_config(pc_models).models, "attention": spec},
        "attention_key_block": 16, "attention_query_block": 2})
    rows = load_examples(pc_data / "train.jsonl")
    supervised = []
    for row in rows:
        n = len(row.prompt.split())
        for offset in (0, 1):
            mask = [i % 2 == offset for i in range(n)]
            supervised.append({"example_id": row.id, "split": "train", "method": "delete",
                "request": {"prompt": row.prompt}, "selected": {"mask": mask}})
    records = attention_records(supervised, rows)
    assert len(records) == len(rows)
    assert sum(records[0]["attention_target"]) == pytest.approx(1.)
    def backend():
        return AttentionBackend(spec, device="cpu", key_block=16, query_block=2)
    a, b = backend(), backend()
    before = {n: p.detach().clone() for n, p in a.model.named_parameters()}
    train(a, records, cfg, tmp_path / "full", "attention", "sft", 11, digest(records), checkpoint_every=1)
    train(b, records, cfg, tmp_path / "resume", "attention", "sft", 11, digest(records), checkpoint_every=1, stop_after=1)
    b = backend()
    train(b, records, cfg, tmp_path / "resume", "attention", "sft", 11, digest(records), checkpoint_every=1)
    reloaded = backend()
    load_checkpoint(reloaded, tmp_path / "full")
    changed = []
    for (name, p), (_, q), (_, saved) in zip(a.model.named_parameters(), b.model.named_parameters(), reloaded.model.named_parameters()):
        assert torch.equal(p, q) and torch.equal(p, saved)
        if not torch.equal(before[name], p):
            changed.append(name)
        if not name.startswith(("q_proj.", "k_proj.")):
            assert torch.equal(before[name], p)
    assert any(n.startswith("q_proj.") for n in changed)
    assert any(n.startswith("k_proj.") for n in changed)
    with pytest.raises(ValueError, match="supervised"):
        train(b, records, cfg, tmp_path / "bad", "attention", "rl", 11, "x")


def test_extraction_unicode_identity_and_tight_budgets(attention_model):
    spec, _ = attention_model
    backend = AttentionBackend(spec, device="cpu", key_block=4)
    counter = TokenCounter.from_spec(spec)
    compressor = AttentionCompressor(backend, counter)
    prompt = "Café  blue\n sky? 日本語"
    for budget in (1, 2, 3, 20):
        request = CompressionRequest(prompt=prompt, budget_tokens=budget, focus_span=(12, len(prompt)))
        compressed = compressor.compress(request)
        assert compressed.output_tokens <= budget
        assert compressed.spans == tuple(sorted(compressed.spans))
        if budget == 20:
            assert compressed.text == prompt
        else:
            assert compressed.text == " ".join(prompt[a:b] for a, b in compressed.spans)
    class Expensive:
        def count(self, text):
            return 5 * len(text.split())
    empty = AttentionCompressor(backend, Expensive()).compress(CompressionRequest(prompt="blue sky?", budget_tokens=1, focus_span=(5, 9)))
    assert empty.status == "empty" and not empty.text
    with pytest.raises(ValueError, match="focus_span"):
        compressor.compress(CompressionRequest(prompt=prompt, budget_tokens=1))
    for span in ((-1, 2), (0, 100), (2, 2), (4, 6), (True, 2)):
        with pytest.raises(ValueError):
            CompressionRequest(prompt=prompt, budget_tokens=1, focus_span=span)


def test_context_and_supervision_boundaries(attention_model, pc_data):
    spec, _ = attention_model
    backend = AttentionBackend(spec, device="cpu")
    with pytest.raises(ValueError, match="exceeds context"):
        backend.model(torch.ones((1, 513), dtype=torch.long), 510)
    row = load_examples(pc_data / "test.jsonl")[0]
    record = {"example_id": row.id, "split": "test", "method": "delete", "request": {"prompt": row.prompt}, "selected": {"mask": [True] * len(row.prompt.split())}}
    with pytest.raises(ValueError, match="training"):
        attention_records([record], [row])
    assert row.prompt[slice(*question_span(row))] == row.question
