"""Offline engineering exercise with randomly initialized models, NOT a benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from .io import atomic_json, digest, read_json, tree_hash
from .schema import ModelSpec, StudyConfig


def tiny_models(root):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(7)
    root = Path(root)
    words = ["<pad>", "<s>", "</s>", "<unk>", "<assistant>", "Answer", "the", "question", "using", "passage.", "Return",
        "only", "shortest", "answer", "span,", "with", "no", "explanation.", "Passage:", "Question:", "The", "sky", "is", "blue.",
        "What", "color", "sky?", "blue", "red", "green", "Retained", "token", "budget:", "Prompt:", "4", "8", "12", "16", "20", "32"]
    raw = Tokenizer(WordLevel({w: i for i,w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tokenizer.chat_template = "{{ messages[0]['content'] }} <assistant>"
    paths = {}
    for name in ("causal", "encoder", "attention"):
        path = root / name
        path.mkdir(parents=True)
        tokenizer.save_pretrained(path)
        if name == "causal":
            model = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
                bos_token_id=1, eos_token_id=2, pad_token_id=0))
        elif name == "encoder":
            model = ModernBertModel(ModernBertConfig(vocab_size=len(words), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=512,
                local_attention=32, global_attn_every_n_layers=1, pad_token_id=0,
                attn_implementation="eager", reference_compile=False))
        else:
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(words), hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=0))
        model.save_pretrained(path)
        paths[name] = ModelSpec(model=str(path), revision="local:" + tree_hash(path))
    return paths


def fixture_data(root):
    from .data import prepare_squad
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    def article(index):
        context = f"The sky is blue. {index}"
        return {"title": f"article-{index}", "paragraphs": [{"context": context,
            "qas": [{"id": f"q-{index}", "question": "What color is the sky?", "answers": [{"text": "blue", "answer_start": 11}]}]}]}
    train, dev = root / "train.json", root / "dev.json"
    atomic_json(train, {"version": "1.1", "data": [article(i) for i in range(3)]})
    atomic_json(dev, {"version": "1.1", "data": [article(i) for i in range(3, 5)]})
    prepare_squad(train, dev, root / "prepared")
    return root / "prepared"


class FixtureTeacher:
    def generate(self, prompt, max_new_tokens, sample):
        import json
        from .backends import Generation
        payload = json.loads(prompt.split("\n")[-1])
        words = payload["prompt"].split()
        text = " ".join(words[::2][:payload["maximum_tokens"]])
        return Generation(text, (1,), len(prompt.split()), len(text.split()), .001)


class EngineeringReward:
    """Deterministic nonsemantic reward to exercise stochastic policy gradients."""
    def score(self, example, prompt):
        value = (sum(map(ord, prompt)) % 19) / 19
        return [{"f1": value}, {"f1": value * .9}]


class FixtureTarget:
    def __init__(self, alias, spec, counter):
        self.alias, self.spec, self.counter = alias, spec, counter
        self.endpoint = "offline-fixture"

    def answer(self, prompt):
        return {"text": "blue" if "blue" in prompt else "red", "input_tokens": len(self.counter.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)),
            "output_tokens": 1, "seconds": .001, "cache_hit": False}


def smoke_config(specs):
    models = {alias: ModelSpec(model="fixture/" + alias) for alias in ("teacher", "qwen", "mistral", "phi", "llmlingua2")}
    models.update(rewriter=specs["causal"], selector=specs["encoder"])
    return StudyConfig(profile="offline-engineering", models=models, seeds=(11,), ratios=(.5,),
        candidates=4, learning_rate=.01, selector_learning_rate=.01, bootstrap_samples=100)


def run(output):
    from .backends import CausalBackend, TokenCounter
    from .compressors import BaselineCompressor, DeleteCompressor, RewriteCompressor, SelectorBackend
    from .data import load_examples
    from .evaluation import evaluate
    from .reporting import report
    from .supervision import generate_supervision, load_supervision, seed_all
    from .training import checkpoint_identity, load_checkpoint, train
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("CPU smoke output must be a new directory")
    specs = tiny_models(root / "models")
    cfg = smoke_config(specs)
    counter = TokenCounter.from_spec(specs["causal"])
    data = fixture_data(root / "data")
    rows = load_examples(data / "train.jsonl", expected_split="train")
    reward = EngineeringReward()
    generate_supervision(rows, cfg, FixtureTeacher(), reward, counter, root / "supervision")
    registry, changes = {}, {}
    for method in ("rewrite", "delete"):
        seed_all(11)
        backend = CausalBackend(specs["causal"], trainable=True, device="cpu", lora=False) if method == "rewrite" else SelectorBackend(specs["encoder"], device="cpu")
        before = {n: p.detach().clone() for n,p in backend.model.named_parameters() if p.requires_grad}
        records, identity = load_supervision(root / "supervision", method)
        for stage in ("sft", "rl"):
            path = root / method / stage
            train(backend, records if stage == "sft" else rows, cfg, path, method, stage, 11,
                identity if stage == "sft" else digest([r.model_dump() for r in rows]), counter, reward,
                root / method / "sft" if stage == "rl" else None, checkpoint_every=1)
            registry[f"{method}_{stage}"] = str(path)
        changes[method] = any(not __import__("torch").equal(before[n], p) for n,p in backend.model.named_parameters() if n in before)
    def build(method, seed):
        if method in ("original", "truncate"):
            return BaselineCompressor(method, counter)
        if method == "rewrite":
            backend = CausalBackend(specs["causal"], trainable=True, device="cpu", lora=False)
            load_checkpoint(backend, registry["rewrite_rl"])
            return RewriteCompressor(backend, counter)
        backend = SelectorBackend(specs["encoder"], device="cpu")
        load_checkpoint(backend, registry["delete_rl"])
        return DeleteCompressor(backend, counter)
    targets = [FixtureTarget(alias, cfg.models[alias], counter) for alias in (*cfg.training_targets, cfg.transfer_target)]
    evaluate(load_examples(data / "test.jsonl"), cfg, counter, targets, build, root / "evaluation",
        {k: checkpoint_identity(v) for k,v in registry.items()}, ("original", "truncate", "rewrite", "delete"))
    report(root / "evaluation", root / "report", 100)
    result = {"parameter_updates": changes, "report": str(root / "report"),
        "scientific_result": False, "limitations": "Random tiny models and artificial reward/solver; engineering validation only."}
    atomic_json(root / "validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    print(__import__("json").dumps(run(p.parse_args().output), indent=2))
