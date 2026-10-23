"""Invoked by the two-process integration check; no downloads or remote solvers."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from promptcomp.backends import CausalBackend
from promptcomp.io import atomic_json, read_json, digest
from promptcomp.schema import ModelSpec
from promptcomp.smoke import smoke_config
from promptcomp.supervision import seed_all
from promptcomp.training import Distributed, train

p = argparse.ArgumentParser()
p.add_argument("--fixture", required=True)
p.add_argument("--output", required=True)
p.add_argument("--stop-after", type=int)
p.add_argument("--method", choices=("rewrite", "attention"), default="rewrite")
args = p.parse_args()
torch.set_num_threads(1)
dist = Distributed()
specs = {k:ModelSpec.model_validate(v) for k,v in read_json(args.fixture).items()}
cfg = smoke_config(specs)
seed_all(11)
if args.method == "attention":
    from promptcomp.attention import AttentionBackend
    cfg = cfg.model_copy(update={"models": {**cfg.models, "attention": specs["attention"]},
        "attention_key_block": 2, "attention_query_block": 1})
    backend = AttentionBackend(specs["attention"], device="cpu", key_block=2, query_block=1)
    records = [{"split": "train", "request": {"prompt": "What color sky?", "budget_tokens": 2, "focus_span": [0, 15]},
        "attention_target": target} for target in ([0., 1., 0.], [.5, 0., .5], [1., 0., 0.], [0., 0., 1.])]
else:
    backend = CausalBackend(specs["causal"], trainable=True, device="cpu", lora=False)
    records = [{"split": "train", "request": {"prompt": "What color sky?", "budget_tokens": 2},
        "selected": {"text": answer}} for answer in ("blue", "red", "green", "blue")]
train(backend, records, cfg, args.output, args.method, "sft", 11, digest(records), checkpoint_every=1, stop_after=args.stop_after)
signature = digest({name:p.detach().cpu().tolist() for name,p in backend.model.named_parameters()})
atomic_json(Path(args.output) / f"rank-{dist.rank}.json", {"signature": signature})
dist.barrier()
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

