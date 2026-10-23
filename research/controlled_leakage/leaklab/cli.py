from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .io import atomic_json


def parser():
    p = argparse.ArgumentParser(prog="leaklab")
    commands = p.add_subparsers(dest="command", required=True)
    g = commands.add_parser("generate")
    g.add_argument("--output", required=True)
    g.add_argument("--size", type=int, default=120, help="Base records per split, each with two counterfactual variants")
    g.add_argument("--seed", type=int, default=11)
    r = commands.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--output", required=True)
    t = commands.add_parser("theory")
    t.add_argument("--output", required=True)
    c = commands.add_parser("compare")
    c.add_argument("--run", required=True)
    c.add_argument("--method", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--metric", default="secret_recovery", choices=["secret_recovery", "utility", "excess_recovery"])
    c.add_argument("--output", required=True)
    e = commands.add_parser("export-attacks")
    e.add_argument("--run", required=True)
    e.add_argument("--output", required=True)
    scopes = commands.add_parser("list-scopes")
    scopes.add_argument("--export",required=True)
    i = commands.add_parser("import-external")
    for name in ("source", "output", "mapping", "source-url"):
        i.add_argument("--"+name, required=True)
    tr = commands.add_parser("train-compressor")
    for name in ("config", "data", "output"):
        tr.add_argument("--"+name, required=True)
    adaptive = commands.add_parser("tune-anonymizer")
    for option in ("config","data","output"):
        adaptive.add_argument("--"+option,required=True)
    adaptive.add_argument("--rounds",type=int,default=2)
    for name in ("train-attacker", "train-inverter"):
        a = commands.add_parser(name)
        for option in ("export", "scope", "output", "model", "revision"):
            a.add_argument("--"+option, required=True)
        a.add_argument("--epochs", type=int, default=1)
        a.add_argument("--seed", type=int, default=11)
        if name == "train-inverter":
            a.add_argument("--run", required=True)
            a.add_argument("--target", choices=["secrets", "compressed_text"], default="secrets")
            a.add_argument("--public-only",action="store_true",help="Train a matched public-information-only prior")
    ev = commands.add_parser("evaluate-attacker")
    for option in ("run", "data", "checkpoint", "output"):
        ev.add_argument("--"+option, required=True)
    ev.add_argument("--public-checkpoint", help="Separately trained public-only attacker for the same scope")
    lock = commands.add_parser("lock-models")
    lock.add_argument("--config", required=True)
    lock.add_argument("--output", required=True)
    config = commands.add_parser("configure-evaluation")
    config.add_argument("--base",required=True)
    config.add_argument("--adapters",help="compressor-adapters.json emitted by train-compressor")
    config.add_argument("--policy",help="policy.json emitted by tune-anonymizer")
    config.add_argument("--split",choices=["test","shift","external"],default="test")
    config.add_argument("--evaluation-file")
    config.add_argument("--output",required=True)
    return p


def execute(args):
    if args.command == "generate":
        from .data import generate_dataset
        return generate_dataset(args.output, args.size, args.seed)
    if args.command == "run":
        from .runner import run
        return run(args.config, args.data, args.output)
    if args.command == "theory":
        from .theory import frontier
        result = frontier()
        atomic_json(args.output, result)
        return {"output":args.output,"points":len(result),"statuses":{s:sum(x["status"]==s for x in result) for s in sorted({x["status"] for x in result})}}
    if args.command == "compare":
        from .metrics import grouped_comparison
        result = grouped_comparison(json.loads((Path(args.run)/"results.json").read_text()), args.method,args.baseline,args.metric)
        atomic_json(args.output,result)
        return {"output":args.output,"comparisons":len(result)}
    if args.command == "export-attacks":
        from .runner import export_attack_data
        return export_attack_data(args.run,args.output)
    if args.command == "list-scopes":
        from .io import read_jsonl
        result = {}
        for row in read_jsonl(args.export):
            if row["split"] != "train":
                raise ValueError("Attack export contains a nontraining row")
            item = result.setdefault(row["scope"],{"scope":row["scope"],**row["scope_metadata"],"access":row["observation"]["access"],"rows":0})
            item["rows"] += 1
        return list(result.values())
    if args.command == "import-external":
        from .data import import_external
        return import_external(args.source,args.output,args.mapping,args.source_url)
    if args.command == "train-compressor":
        from .training import train_compressor
        return train_compressor(args.config,args.data,args.output)
    if args.command == "tune-anonymizer":
        from .adaptive import tune
        return tune(args.config,args.data,args.output,args.rounds)
    if args.command == "train-attacker":
        from .training import train_text_attacker
        return train_text_attacker(args.export,args.scope,args.output,args.model,args.revision,args.epochs,args.seed)
    if args.command == "train-inverter":
        from .inversion import train_inverter
        return train_inverter(args.export,args.run,args.scope,args.output,args.model,args.revision,args.target,args.epochs,args.seed,args.public_only)
    if args.command == "evaluate-attacker":
        from .evaluation import evaluate_attacker
        return evaluate_attacker(args.run,args.data,args.checkpoint,args.output,args.public_checkpoint)
    if args.command == "lock-models":
        from huggingface_hub import HfApi
        cfg = json.loads(Path(args.config).read_text())
        models = {cfg[k] for k in ("compressor_model","task_model","attacker_model") if k in cfg}
        models.update(cfg.get("task_models",[]))
        if "llmlingua2" in cfg.get("methods",[]):
            models.add("microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
        if any(x != "text" for x in cfg.get("probability_access",[])):
            models.add("google-t5/t5-base")
        cfg["revisions"] = {m:HfApi().model_info(m).sha for m in sorted(models) if m != "rules"}
        atomic_json(args.output,cfg)
        return {"output":args.output,"revisions":cfg["revisions"]}
    if args.command == "configure-evaluation":
        from .schema import ExperimentConfig
        cfg = json.loads(Path(args.base).read_text())
        cfg["split"] = args.split
        if args.adapters:
            cfg["compressor_adapters"] = json.loads(Path(args.adapters).read_text())
            cfg["methods"] = list(dict.fromkeys(cfg["methods"]+["lora"]))
        if args.policy:
            cfg["adaptive_policy"] = str(Path(args.policy).resolve())
            cfg["methods"] = list(dict.fromkeys(cfg["methods"]+["adaptive"]))
        if args.evaluation_file:
            cfg["evaluation_file"] = str(Path(args.evaluation_file).resolve())
        validated = ExperimentConfig.model_validate(cfg)
        atomic_json(args.output,validated.model_dump())
        return {"output":args.output,"methods":validated.methods,"split":validated.split}
    raise ValueError("Unknown command")


def main(argv=None):
    try:
        result = execute(parser().parse_args(argv))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(json.dumps({"error":type(exc).__name__,"message":str(exc)}),file=sys.stderr)
        return 1
