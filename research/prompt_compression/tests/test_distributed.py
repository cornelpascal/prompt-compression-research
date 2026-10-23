import os
from pathlib import Path
import subprocess
import sys
import signal
import socket

import pytest
import torch
from promptcomp.io import atomic_json, read_json


@pytest.mark.skipif(os.environ.get("PROMPTCOMP_DISTRIBUTED_TEST") != "1", reason="Explicit two-process loopback integration test")
@pytest.mark.parametrize("method", ["rewrite", "attention"])
def test_two_rank_training_and_resume(pc_models, tmp_path, method):
    fixture = tmp_path / "models.json"
    atomic_json(fixture, {k:v.model_dump() for k,v in pc_models.items()})
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
        "--master-addr=127.0.0.1", f"--master-port={port}",
        str(Path(__file__).parent / "distributed_worker.py"), "--fixture", str(fixture), "--method", method]
    interfaces = {name for _,name in socket.if_nameindex()}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo0" if "lo0" in interfaces else "lo"}
    for folder, extra in (("full", []), ("resume", ["--stop-after", "1"]), ("resume", [])):
        process = subprocess.Popen(command + ["--output", str(tmp_path / folder)] + extra,
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail("Distributed test timed out: " + stderr[-6000:])
        assert process.returncode == 0, stderr[-6000:]
        assert read_json(tmp_path / folder / "rank-0.json") == read_json(tmp_path / folder / "rank-1.json")
    assert read_json(tmp_path / "full" / "rank-0.json") == read_json(tmp_path / "resume" / "rank-0.json")
