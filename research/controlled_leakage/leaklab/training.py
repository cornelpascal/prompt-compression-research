from __future__ import annotations

import json
import math
from pathlib import Path
import random

from .backends import HFBackend
from .compressors import Compressor, bounded_text, rewrite_prompt
from .data import load_examples
from .io import atomic_json, digest, file_hash, read_jsonl, source_hash
from .metrics import answer_accuracy
from .observations import observation_prompt, observe, probability_text, ProbabilityView
from .runner import run_lock, seed_everything
from .schema import Observation
from .devices import cpu_tree, release_cache, resolve_device, step as accelerator_step


def require_train(rows):
    if not rows or any((r.split if hasattr(r, "split") else r["split"]) != "train" for r in rows):
        raise ValueError("Training accepts exclusively nonempty training splits")


def supervised_pairs(examples, tokenizer, ratios, no_query=False):
    require_train(examples)
    teacher = Compressor("redact_compress", tokenizer)
    pairs = []
    for e in examples:
        for ratio in ratios:
            budget = math.floor(len(tokenizer.encode(e.request.context))*ratio)
            query = "" if no_query else e.request.query
            response = teacher.compress(e.request.context, query, e.request.privacy_policy, budget).text
            pairs.append((rewrite_prompt(e.request.context, query, e.request.privacy_policy, budget, "lora"), response))
    return pairs


def sft(backend, pairs, directory, epochs=1, lr=2e-4, seed=11):
    """Response-only likelihood; atomic epoch checkpoints retain optimizer state.

    The caller constructs the backend from latest_adapter(directory) on resume.
    An interrupted epoch is replayed from its last complete checkpoint.
    """
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = digest({"pairs": pairs, "epochs": epochs, "lr": lr, "seed": seed,
        "model": backend.model_id, "revision": backend.revision, "code": source_hash()})
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"epoch": 0, "identity": identity}
    if state["identity"] != identity:
        raise ValueError("SFT resume inputs changed")
    optimizer = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad], lr=lr)
    if state["epoch"]:
        optimizer.load_state_dict(torch.load(directory / f"epoch-{state['epoch']:03d}" / "optimizer.pt", map_location="cpu", weights_only=True))
    for epoch in range(state["epoch"], epochs):
        seed_everything(seed + epoch)
        order = list(range(len(pairs)))
        random.Random(seed + epoch).shuffle(order)
        backend.model.train()
        total = 0.
        for index in order:
            prompt, answer = pairs[index]
            optimizer.zero_grad(set_to_none=True)
            loss = -backend.response_logprob(prompt, answer, normalize=True)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SFT loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in backend.model.parameters() if p.requires_grad], 1.)
            accelerator_step(optimizer, backend.device)
            total += float(loss.detach())
        checkpoint = directory / f"epoch-{epoch+1:03d}"
        backend.save(checkpoint)
        torch.save(cpu_tree(optimizer.state_dict()), checkpoint / "optimizer.pt")
        state = {"identity": identity, "epoch": epoch+1, "loss": total/max(1,len(pairs)), "adapter": str(checkpoint.resolve())}
        atomic_json(state_path, state)
    backend.model.eval()
    return state


def latest_adapter(directory):
    path = Path(directory) / "state.json"
    return json.loads(path.read_text()).get("adapter") if path.exists() else None


def reward_components(utility, attack_probability, raw_count, budget, privacy_weight=1., budget_weight=1.):
    overflow = max(0, raw_count-budget) / max(1, budget)
    return {"utility": float(utility), "attack_probability": float(attack_probability),
            "overflow": overflow, "reward": float(utility - privacy_weight*attack_probability-budget_weight*overflow)}


def train_compressor(config_path, data, output):
    import torch
    cfg = json.loads(Path(config_path).read_text())
    rows = load_examples(data)
    require_train(rows)
    required = {"compressor_model", "task_model", "attacker_model", "revisions", "seeds", "ratios", "rounds", "sft_epochs", "attacker_epochs", "learning_rate", "privacy_weight", "budget_weight", "no_query", "no_attacker_retraining"}
    if set(cfg) != required:
        raise ValueError(f"Training config requires exactly {sorted(required)}")
    if not cfg["seeds"] or cfg["rounds"] < 1 or cfg["sft_epochs"] < 1 or cfg["attacker_epochs"] < 1:
        raise ValueError("Positive seeds, rounds and epochs required")
    if any(not 0 < r <= 1 for r in cfg["ratios"]):
        raise ValueError("Invalid retained-length ratios")
    if resolve_device().type not in ("cuda", "xla"):
        raise RuntimeError("Full compressor training requires CUDA or XLA; CPU pilot is available via run")
    with run_lock(output) as root:
        identity = digest({"config": cfg, "data": file_hash(data), "code": source_hash()})
        identity_path = root / "manifest.json"
        if identity_path.exists() and json.loads(identity_path.read_text())["identity"] != identity:
            raise ValueError("Training inputs changed; choose a new output directory")
        atomic_json(identity_path, {"identity": identity, "config": cfg, "data_sha256": file_hash(data)})
        registry = {}
        for seed in cfg["seeds"]:
            seed_everything(seed)
            folder = root / f"seed-{seed}"
            folder.mkdir(exist_ok=True)
            state_path = folder / "round-state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"round": 0}
            warm = folder / "warmstart"
            adapter = state.get("compressor_adapter") or latest_adapter(warm)
            compressor = HFBackend(cfg["compressor_model"], cfg["revisions"][cfg["compressor_model"]], adapter=adapter, trainable=True)
            task = HFBackend(cfg["task_model"], cfg["revisions"][cfg["task_model"]])
            pairs = supervised_pairs(rows, task.counter, cfg["ratios"], cfg["no_query"])
            if not state["round"]:
                sft(compressor, pairs, warm, cfg["sft_epochs"], cfg["learning_rate"], seed)
            attacker_adapter = state.get("attacker_adapter")
            for round_index in range(state["round"], cfg["rounds"]):
                seed_everything(seed + round_index)
                round_dir = folder / f"round-{round_index+1:03d}"
                round_dir.mkdir(exist_ok=True)
                attacker_dir = round_dir / "attacker"
                partial_attacker = latest_adapter(attacker_dir)
                attacker = HFBackend(cfg["attacker_model"], cfg["revisions"][cfg["attacker_model"]],
                    adapter=partial_attacker or attacker_adapter, trainable=True)
                if round_index == 0 or not cfg["no_attacker_retraining"]:
                    attack_pairs = []
                    wrapper = Compressor("lora", task.counter, compressor)
                    for e in rows:
                        for ratio in cfg["ratios"]:
                            budget = math.floor(len(task.counter.encode(e.request.context))*ratio)
                            compressed = wrapper.compress(e.request.context, "" if cfg["no_query"] else e.request.query, e.request.privacy_policy, budget)
                            answer = task.answer(compressed.text, e.request.query).text
                            for access in ("context", "text", "public"):
                                obs = observe(e.request, access, compressed.text, answer)
                                attack_pairs.append((observation_prompt(obs), json.dumps(e.secrets, sort_keys=True)))
                    attack_state = sft(attacker, attack_pairs, attacker_dir, cfg["attacker_epochs"], cfg["learning_rate"], seed)
                    attacker_adapter = attack_state["adapter"]
                attacker.model.requires_grad_(False).eval()
                # Make replay independent of whether attacker SFT resumed or ran.
                seed_everything(seed + round_index + 1_000_000)
                # A new optimizer each alternating round is intentional and recorded.
                optimizer = torch.optim.AdamW([p for p in compressor.model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
                logs = []
                for e in rows:
                    for ratio in cfg["ratios"]:
                        budget = math.floor(len(task.counter.encode(e.request.context))*ratio)
                        query = "" if cfg["no_query"] else e.request.query
                        prompt = rewrite_prompt(e.request.context, query, e.request.privacy_policy, budget, "lora")
                        candidates = [compressor.generate(prompt, max_new_tokens=max(1,min(1024,budget*2)), sample=True) for _ in range(2)]
                        rewards = []
                        for generated in candidates:
                            raw = generated.text
                            z = bounded_text(raw, budget, task.counter)
                            answer = task.answer(z, e.request.query).text
                            risks = []
                            with torch.no_grad():
                                for access in ("context", "text"):
                                    obs = observe(e.request, access, z, answer)
                                    # Smooth surrogate for optimization; actual recovery is evaluated separately.
                                    risk = attacker.response_logprob(observation_prompt(obs), json.dumps(e.secrets, sort_keys=True), normalize=True).exp().item()
                                    risks.append(risk)
                            reward = reward_components(answer_accuracy(e,answer), max(risks), len(task.counter.encode(raw)), budget,
                                cfg["privacy_weight"], cfg["budget_weight"])
                            rewards.append(reward["reward"])
                            logs.append({"id":e.id,"ratio":ratio,**reward})
                        optimizer.zero_grad(set_to_none=True)
                        compressor.model.eval()  # Match the no-dropout sampling policy.
                        # Leave-one-out baseline: other sample is independent of this action.
                        for i, generated in enumerate(candidates):
                            advantage = rewards[i] - rewards[1-i]
                            loss = -advantage * compressor.response_ids_logprob(prompt, generated.token_ids) / 2
                            if not torch.isfinite(loss):
                                raise RuntimeError("Non-finite policy-gradient loss")
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_([p for p in compressor.model.parameters() if p.requires_grad], 1.)
                        accelerator_step(optimizer, compressor.device)
                checkpoint = round_dir / "compressor"
                compressor.save(checkpoint)
                atomic_json(round_dir / "rewards.json", logs)
                state = {"round":round_index+1,"compressor_adapter":str(checkpoint.resolve()),
                    "attacker_adapter":attacker_adapter,"seed":seed,"optimizer_reset_each_round":True}
                atomic_json(state_path, state)
                del attacker
                release_cache(compressor.device)
            registry[str(seed)] = state["compressor_adapter"]
            # The wrapper and optimizer retain model references across seeds.
            if "wrapper" in locals():
                del wrapper
            if "optimizer" in locals():
                del optimizer
            if "loss" in locals():
                del loss
            del compressor, task
            release_cache(resolve_device())
        atomic_json(root / "compressor-adapters.json", registry)
        return {"adapters": registry, "training_identity": identity}


def train_text_attacker(export, scope, output, model, revision, epochs=1, seed=11):
    from .runner import run_lock
    data = [r for r in read_jsonl(export) if r["scope"] == scope]
    require_train(data)
    if len({r["protocol_id"] for r in data}) != 1:
        raise ValueError("Training scope combines different compressor protocols")
    pairs = []
    for row in data:
        obs = Observation.model_validate(row["observation"])
        if obs.access not in ("context", "public", "text"):
            raise ValueError("Use train-inverter for probability observations")
        pairs.append((observation_prompt(obs), json.dumps(row["labels"]["secrets"], sort_keys=True)))
    with run_lock(output) as directory:
        seed_everything(seed)
        backend = HFBackend(model, revision, adapter=latest_adapter(directory), trainable=True)
        state = sft(backend, pairs, directory, epochs=epochs, seed=seed)
        atomic_json(directory / "attack.json", {"scope":scope,"model":model,"revision":revision,
            "protocol_id":data[0]["protocol_id"],
            "adapter":state["adapter"],"export_sha256":file_hash(export),"training_ids":[r["id"] for r in data],
            "scope_metadata":data[0]["scope_metadata"],"access":data[0]["observation"]["access"],"target":"secrets"})
        return state
