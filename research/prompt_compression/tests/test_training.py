import copy
from pathlib import Path
import pytest
import torch

from promptcomp.backends import CausalBackend, TokenCounter
from promptcomp.compressors import DeleteCompressor, RewriteCompressor, SelectorBackend
from promptcomp.data import load_examples
from promptcomp.io import digest, read_json
from promptcomp.schema import CompressionRequest
from promptcomp.smoke import EngineeringReward, FixtureTeacher, smoke_config
from promptcomp.supervision import generate_supervision, load_supervision, seed_all
from promptcomp.training import grpo_loss, group_advantages, load_checkpoint, train


def test_grpo_clipping_gradient_and_zero_variance():
    old = torch.tensor([-.5, -.5])
    current = torch.tensor([-.5, -.4], requires_grad=True)
    loss = grpo_loss(current, old, old, torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(current.grad).all() and current.grad.abs().sum() > 0
    assert torch.equal(group_advantages([1., 1.]), torch.zeros(2))


@pytest.mark.parametrize("method", ["rewrite", "delete"])
def test_real_updates_exact_resume_and_budget(method, pc_models, pc_data, tmp_path):
    cfg = smoke_config(pc_models)
    counter = TokenCounter.from_spec(pc_models["causal"])
    rows = load_examples(pc_data / "train.jsonl")
    reward = EngineeringReward()
    generate_supervision(rows, cfg, FixtureTeacher(), reward, counter, tmp_path / "supervision")
    records, identity = load_supervision(tmp_path / "supervision", method)
    def backend():
        seed_all(11)
        return CausalBackend(pc_models["causal"], trainable=True, device="cpu", lora=False) if method == "rewrite" else SelectorBackend(pc_models["encoder"], device="cpu")
    a, b = backend(), backend()
    before = {n:p.detach().clone() for n,p in a.model.named_parameters()}
    train(a, records, cfg, tmp_path / "uninterrupted", method, "sft", 11, identity, checkpoint_every=1)
    train(b, records, cfg, tmp_path / "resumed", method, "sft", 11, identity, checkpoint_every=1, stop_after=1)
    b = backend()
    train(b, records, cfg, tmp_path / "resumed", method, "sft", 11, identity, checkpoint_every=1)
    assert any(not torch.equal(before[n], p) for n,p in a.model.named_parameters())
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(p, q)
    with pytest.raises(ValueError, match="inputs changed"):
        train(b, records, cfg, tmp_path / "resumed", method, "sft", 11, "different-data")
    before_rl = {n:p.detach().clone() for n,p in b.model.named_parameters()}
    train(b, rows, cfg, tmp_path / "rl", method, "rl", 11, digest([r.model_dump() for r in rows]),
        counter, reward, tmp_path / "resumed", checkpoint_every=1)
    assert any(not torch.equal(before_rl[n], p) for n,p in b.model.named_parameters())
    c = backend()
    load_checkpoint(c, tmp_path / "rl")
    for p, q in zip(b.model.parameters(), c.model.parameters()):
        assert torch.equal(p, q)
    compressor = RewriteCompressor(c, counter) if method == "rewrite" else DeleteCompressor(c, counter)
    compressed = compressor.compress(CompressionRequest(prompt=rows[0].prompt, budget_tokens=4))
    assert compressed.output_tokens == counter.count(compressed.text)
    if method == "delete":
        assert compressed.output_tokens <= 4
        assert " ".join(rows[0].prompt[a:b] for a,b in compressed.spans) == compressed.text


def test_lora_update_and_checkpoint(pc_models, tmp_path):
    pytest.importorskip("peft")
    cfg = smoke_config(pc_models)
    backend = CausalBackend(pc_models["causal"], trainable=True, device="cpu", lora=True)
    records = [{"split": "train", "request": {"prompt": "sky blue", "budget_tokens": 2}, "selected": {"text": "blue"}}]
    train(backend, records, cfg, tmp_path / "lora", "rewrite", "sft", 11, "fixture")
    assert any("lora_B" in name and value.abs().sum() > 0 for name, value in backend.model.named_parameters())
    reloaded = CausalBackend(pc_models["causal"], trainable=True, device="cpu", lora=True)
    load_checkpoint(reloaded, tmp_path / "lora")
    assert all(torch.equal(p,q) for p,q in zip(backend.model.parameters(), reloaded.model.parameters()))


def test_supervision_resume_and_training_split_boundary(pc_models, pc_data, tmp_path):
    cfg = smoke_config(pc_models)
    counter = TokenCounter.from_spec(pc_models["causal"])
    rows = load_examples(pc_data / "train.jsonl")
    output = tmp_path / "supervision"
    generate_supervision(rows, cfg, FixtureTeacher(), EngineeringReward(), counter, output)
    timestamps = {p: p.stat().st_mtime_ns for p in (output / "cases").glob("*.json")}
    generate_supervision(rows, cfg, FixtureTeacher(), EngineeringReward(), counter, output)
    assert all(p.stat().st_mtime_ns == timestamp for p,timestamp in timestamps.items())
    with pytest.raises(ValueError, match="training examples only"):
        generate_supervision(load_examples(pc_data / "test.jsonl"), cfg, FixtureTeacher(), EngineeringReward(), counter, tmp_path / "bad")
