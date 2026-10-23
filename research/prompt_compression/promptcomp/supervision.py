from __future__ import annotations

import math

from .compressors import align_extraction, render_mask, rewrite_instruction, word_spans
from .io import atomic_json, complete_run, digest, file_hash, read_json, run_directory, source_hash
from .schema import CompressionRequest


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_supervision(rows, cfg, teacher, solvers, counter, output):
    if not rows or any(r.split != "train" for r in rows):
        raise ValueError("Supervision generation accepts training examples only")
    identity = {"stage": "supervision", "config": cfg.model_dump(mode="json"),
        "data": digest([r.model_dump() for r in rows]), "counter": counter.name, "code": source_hash()}
    with run_directory(output, identity) as root:
        if read_json(root / "manifest.json")["complete"]:
            index = read_json(root / "index.json")
            if any(file_hash(root / "cases" / name) != index["checksums"][name] for name in index["files"]):
                raise ValueError("Supervision checksum mismatch")
            return {"accepted": index["accepted"], "cases": index["total"]}
        files, accepted = [], 0
        for row in rows:
            for ratio in cfg.ratios:
                budget = max(1, math.floor(counter.count(row.prompt) * ratio))
                request = CompressionRequest(prompt=row.prompt, budget_tokens=budget)
                for method in ("rewrite", "delete"):
                    name = digest([row.id, ratio, method]) + ".json"
                    path = root / "cases" / name
                    if path.exists():
                        record = read_json(path)
                    else:
                        candidates = []
                        for index in range(cfg.candidates):
                            seed_all(int(digest([cfg.seed, row.id, ratio, method, index])[:8], 16))
                            generated = teacher.generate(rewrite_instruction(request, method == "delete"),
                                max_new_tokens=max(16, budget * 2), sample=True)
                            text, mask = generated.text, None
                            status = "ok"
                            if method == "delete":
                                try:
                                    mask = align_extraction(row.prompt, text)
                                    text = render_mask(row.prompt, word_spans(row.prompt), mask)
                                except ValueError:
                                    status = "invalid_extraction"
                            count = counter.count(text)
                            if not text.strip():
                                status = "empty"
                            elif count > budget:
                                status = "overflow"
                            if generated.exhausted:
                                status = "generation_limit"
                            scores = solvers.score(row, text) if status == "ok" else []
                            candidates.append({"text": text, "mask": mask, "tokens": count, "status": status,
                                "scores": scores, "teacher_input_tokens": generated.input_tokens,
                                "teacher_output_tokens": generated.output_tokens, "seconds": generated.seconds})
                        valid = [c for c in candidates if c["status"] == "ok"]
                        chosen = max(valid, key=lambda c: (min(s["f1"] for s in c["scores"]), -c["tokens"])) if valid else None
                        record = {"example_id": row.id, "split": "train", "method": method,
                            "request": request.model_dump(), "ratio": ratio, "candidates": candidates, "selected": chosen}
                        atomic_json(path, record)
                    accepted += record["selected"] is not None
                    files.append(name)
        atomic_json(root / "index.json", {"files": files, "checksums": {name: file_hash(root / "cases" / name) for name in files},
            "accepted": accepted, "total": len(files)})
        complete_run(root, accepted=accepted, cases=len(files))
        return {"accepted": accepted, "cases": len(files)}


def load_supervision(directory, method):
    from pathlib import Path
    from .io import file_hash
    root = Path(directory)
    if not read_json(root / "manifest.json")["complete"]:
        raise ValueError("Supervision is incomplete")
    index = read_json(root / "index.json")
    files = index["files"]
    if any(file_hash(root / "cases" / name) != index["checksums"][name] for name in files):
        raise ValueError("Supervision checksum mismatch")
    records = [read_json(root / "cases" / name) for name in files]
    if any(r["split"] != "train" for r in records):
        raise ValueError("Non-training record in supervision")
    selected = [r for r in records if r["method"] == method and r["selected"] is not None]
    if not selected:
        raise ValueError(f"No accepted supervision for {method}; inspect teacher candidates")
    return selected, digest({name: file_hash(root / "cases" / name) for name in files})

