from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, lock_models
from .io import atomic_json, canonical, digest, file_hash, read_json


def parser():
    p = argparse.ArgumentParser(prog="promptcomp")
    commands = p.add_subparsers(dest="command", required=True)
    d = commands.add_parser("prepare")
    d.add_argument("--train")
    d.add_argument("--dev")
    d.add_argument("--download-dir")
    d.add_argument("--output", required=True)
    d.add_argument("--seed", type=int, default=11)
    l = commands.add_parser("lock-models")
    l.add_argument("--config", required=True)
    l.add_argument("--output", required=True)
    for name in ("preflight", "supervise", "train", "evaluate", "compress", "estimate"):
        c = commands.add_parser(name)
        c.add_argument("--config", required=True)
        if name == "preflight":
            required = c.add_mutually_exclusive_group()
            required.add_argument("--require-cuda", action="store_true")
            required.add_argument("--require-xla", action="store_true")
            continue
        if name == "estimate":
            c.add_argument("--data", required=True)
            c.add_argument("--pilot-dir", required=True)
            continue
        if name != "compress":
            c.add_argument("--output", required=True)
        if name in ("supervise", "train", "evaluate"):
            c.add_argument("--data", required=True)
            c.add_argument("--cache", default="runs/inference-cache")
        if name == "train":
            c.add_argument("--method", choices=("rewrite", "delete", "attention"), required=True)
            c.add_argument("--stage", choices=("sft", "rl"), required=True)
            c.add_argument("--seed", type=int, required=True)
            c.add_argument("--supervision")
            c.add_argument("--initial-checkpoint")
            c.add_argument("--checkpoint-every", type=int, default=50)
        if name in ("evaluate", "compress"):
            c.add_argument("--registry")
            c.add_argument("--selection")
            c.add_argument("--calibration")
            if name == "evaluate":
                c.add_argument("--split", choices=("validation", "test"), required=True)
                c.add_argument("--methods", nargs="+")
                c.add_argument("--variant", choices=("canonical", "paraphrase", "entity"), default="canonical")
            else:
                c.add_argument("--method", required=True)
                c.add_argument("--seed", type=int, default=11)
                c.add_argument("--input", required=True)
                c.add_argument("--budget", type=int)
                c.add_argument("--focus-start", type=int)
                c.add_argument("--focus-end", type=int)
    r = commands.add_parser("report")
    r.add_argument("--run", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--bootstrap-samples", type=int, default=2000)
    s = commands.add_parser("select")
    s.add_argument("--validation", required=True)
    s.add_argument("--registry", required=True)
    s.add_argument("--output", required=True)
    g = commands.add_parser("register")
    g.add_argument("--registry", required=True)
    g.add_argument("--name", required=True, help="e.g. rewrite_sft:11")
    g.add_argument("--checkpoint", required=True)
    a = commands.add_parser("calibrate-attention")
    a.add_argument("--validation", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--threshold", type=float, default=.98)
    a.add_argument("--bootstrap-samples", type=int, default=2000)
    b = commands.add_parser("benchmark-attention")
    for name in ("config", "data", "calibration", "method", "output"):
        b.add_argument("--" + name, required=True)
    b.add_argument("--registry")
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--samples", type=int, default=16)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--warmup", type=int, default=2)
    return p


def execute(args):
    if args.command == "prepare":
        from .data import download_squad, prepare_squad
        if args.download_dir:
            if args.train or args.dev:
                raise ValueError("Choose download-dir or local train/dev files")
            train, dev = download_squad(args.download_dir)
        elif args.train and args.dev:
            train, dev = args.train, args.dev
        else:
            raise ValueError("Supply train/dev files or download-dir")
        return prepare_squad(train, dev, args.output, args.seed)
    if args.command == "lock-models":
        return lock_models(args.config, args.output)
    if args.command == "report":
        from .reporting import report
        return report(args.run, args.output, args.bootstrap_samples)
    if args.command == "select":
        from .evaluation import select_checkpoints
        return select_checkpoints(args.validation, read_json(args.registry), args.output)
    if args.command == "calibrate-attention":
        from .calibration import calibrate
        return calibrate(args.validation, args.output, args.threshold, args.bootstrap_samples)
    if args.command == "register":
        from .training import checkpoint_identity
        identity = checkpoint_identity(args.checkpoint)
        method, seed = args.name.rsplit(":", 1)
        family, stage = (method[:-4], "sft") if method.endswith("_sft") else (method, "rl")
        inputs = read_json(Path(args.checkpoint) / "manifest.json")["inputs"]
        if family not in ("rewrite", "delete", "attention") or (family == "attention" and stage != "sft") or (inputs["method"], inputs["stage"], inputs["seed"]) != (family, stage, int(seed)):
            raise ValueError("Registry name does not match checkpoint method, stage, and seed")
        path = Path(args.registry)
        registry = read_json(path) if path.exists() else {}
        resolved = str(Path(args.checkpoint).resolve())
        if args.name in registry and registry[args.name] != resolved:
            raise ValueError("Registry entry exists with a different checkpoint")
        registry[args.name] = resolved
        atomic_json(path, registry)
        return {"registered": args.name, "identity": identity}
    cfg = load_config(args.config)
    from .backends import CausalBackend, Target, TokenCounter, generator
    from .runtime import checkpoint_registry, factory, solver_pool
    if args.command == "preflight":
        from .runtime import preflight
        result = preflight(cfg)
        if args.require_cuda and not result["cuda_available"]:
            raise RuntimeError("CUDA is required for cloud training; use promptcomp.smoke for CPU engineering tests")
        if args.require_xla and not result["xla_available"]:
            raise RuntimeError("Set PROMPTCOMP_DEVICE=xla and install matching torch_xla for TPU training")
        return result
    if args.command == "estimate":
        from .runtime import estimate
        return estimate(cfg, args.data, args.pilot_dir)
    counter = TokenCounter.from_spec(cfg.models[cfg.reference_tokenizer])
    if args.command == "benchmark-attention":
        from .attention_benchmark import benchmark
        from .data import load_examples
        if args.samples < 1:
            raise ValueError("Benchmark sample count must be positive")
        registry, identity = checkpoint_registry(args.registry)
        rows = load_examples(Path(args.data) / "validation.jsonl", args.samples, cfg.seed, "validation")
        targets = [Target(t, cfg.models[t], Path(args.output) / "unused-cache" / t, cfg.answer_max_tokens)
                   for t in cfg.training_targets]
        return benchmark(rows, cfg, counter, targets, factory(cfg, counter, registry), identity,
            read_json(args.calibration), args.method, args.seed, args.output, args.repeats, args.warmup)
    if args.command in ("compress", "evaluate"):
        registry, identity = checkpoint_registry(args.registry)
        selection = read_json(args.selection) if args.selection else None
        if selection:
            from .training import checkpoint_identity
            identity["selection"] = {"file": file_hash(args.selection), "checkpoints": {
                k: checkpoint_identity(v["path"]) for k,v in selection["selected"].items()}}
        build = factory(cfg, counter, registry, selection)
        calibration = read_json(args.calibration) if args.calibration else None
        if args.command == "compress":
            import math
            from .compressors import compress
            text = Path(args.input).read_text()
            if (args.budget is None) == (calibration is None):
                raise ValueError("Supply exactly one of --budget and --calibration")
            focus = (args.focus_start, args.focus_end)
            if (focus[0] is None) != (focus[1] is None):
                raise ValueError("Supply both --focus-start and --focus-end")
            budget = args.budget
            if calibration is not None:
                from .calibration import validate_calibration
                ratio = validate_calibration(calibration, cfg, args.method, args.seed, identity)
                if ratio is None:
                    raise ValueError("No qualifying compressed setting in calibration")
                budget = max(1, math.floor(counter.count(text) * ratio))
            return compress(text, budget, build(args.method, args.seed),
                            focus_span=None if focus[0] is None else focus).model_dump()
        from .data import load_examples
        from .evaluation import evaluate
        if args.split == "test" and cfg.profile in ("smoke", "pilot"):
            raise ValueError("Smoke and pilot profiles use validation; final test data and transfer model are reserved for the full profile")
        rows = load_examples(Path(args.data) / f"{args.split}.jsonl", cfg.eval_limit, cfg.seed, args.split)
        aliases = cfg.training_targets if args.split == "validation" else (*cfg.training_targets, cfg.transfer_target)
        targets = [Target(alias, cfg.models[alias], Path(args.cache) / alias, cfg.answer_max_tokens) for alias in aliases]
        return evaluate(rows, cfg, counter, targets, build, args.output, identity, args.methods,
                        variant=args.variant, calibration=calibration)
    from .data import load_examples
    rows = load_examples(Path(args.data) / "train.jsonl", cfg.train_limit, cfg.seed, "train")
    if args.command == "supervise":
        from .supervision import generate_supervision
        return generate_supervision(rows, cfg, generator("teacher", cfg.models["teacher"]),
            solver_pool(cfg, args.cache), counter, args.output)
    if args.command == "train":
        from .compressors import SelectorBackend
        from .supervision import load_supervision, seed_all
        from .training import Distributed, train
        dist = Distributed()
        seed_all(args.seed)
        if args.method == "attention":
            from .attention import AttentionBackend
            if args.stage != "sft" or args.initial_checkpoint:
                raise ValueError("Attention supports supervised training from pretrained Q/K weights only")
            backend = AttentionBackend(cfg.models["attention"], key_block=cfg.attention_key_block, query_block=cfg.attention_query_block)
        else:
            backend = CausalBackend(cfg.models["rewriter"], trainable=True) if args.method == "rewrite" else SelectorBackend(cfg.models["selector"])
        if args.stage == "sft":
            if not args.supervision:
                raise ValueError("SFT requires --supervision")
            records, identity = load_supervision(args.supervision, "delete" if args.method == "attention" else args.method)
            allowed_ids = {r.id for r in rows}
            records = [r for r in records if r["example_id"] in allowed_ids]
            if args.method == "attention":
                from .attention import attention_records
                records = attention_records(records, rows)
                identity = digest({"supervision": identity, "records": records})
            solvers = None
        else:
            records = rows
            identity = digest([r.model_dump() for r in rows])
            solvers = solver_pool(cfg, args.cache)
        result = train(backend, records, cfg, args.output, args.method, args.stage, args.seed, identity,
            counter, solvers, args.initial_checkpoint, args.checkpoint_every)
        return result if dist.rank == 0 else None
    raise ValueError("Unknown command")


def main():
    args = parser().parse_args()
    try:
        result = execute(args)
    except (ValueError, RuntimeError, KeyError, FileNotFoundError) as error:
        # No HTTP response bodies, environment values, or credential-bearing URLs.
        parser().exit(1, f"promptcomp: {error}\n")
    if result is not None:
        print(canonical(result))
