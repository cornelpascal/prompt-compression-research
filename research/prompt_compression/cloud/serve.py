"""Run inside vllm/vllm-openai:v0.10.2 (separate from the trainer environment)."""
import argparse
import json
import os
import re
import sys

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--alias", required=True)
p.add_argument("--port", type=int, default=8000)
p.add_argument("--tensor-parallel-size", type=int, default=1)
args = p.parse_args()
spec = json.load(open(args.config))["models"][args.alias]
if not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
    raise SystemExit("A locked model revision is required")
command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", spec["model"],
    "--revision", spec["revision"], "--tokenizer-revision", spec["revision"],
    "--served-model-name", args.alias + "@" + spec["revision"],
    "--port", str(args.port), "--tensor-parallel-size", str(args.tensor_parallel_size),
    "--max-model-len", "8192", "--generation-config", "vllm"]
os.execv(sys.executable, command)

