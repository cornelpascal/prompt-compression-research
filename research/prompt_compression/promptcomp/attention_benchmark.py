"""Uncached, warmed timing measurements, separate from quality evaluation."""
import gc
import math
from pathlib import Path
import time

import numpy as np

from .attention import question_span
from .calibration import validate_calibration
from .io import atomic_json, complete_run, digest, read_json, run_directory
from .schema import CompressionRequest


def benchmark(rows, cfg, counter, targets, factory, artifacts, calibration, method, seed,
              output, repeats=3, warmup=2):
    import torch
    if not rows or {r.split for r in rows} != {"validation"} or repeats < 1 or warmup < 1:
        raise ValueError("Benchmark requires validation examples and positive repeats/warmup")
    if {t.alias for t in targets} != set(cfg.training_targets):
        raise ValueError("Benchmark uses the two training receivers only")
    ratio = validate_calibration(calibration, cfg, method, seed, artifacts)
    if ratio is None:
        raise ValueError("No qualifying setting to benchmark")
    identity = {"stage": "attention_benchmark", "calibration": calibration["identity"],
        "method": method, "seed": seed, "data": digest([r.model_dump() for r in rows]),
        "repeats": repeats, "warmup": warmup, "config": cfg.model_dump(mode="json")}
    with run_directory(output, identity) as root:
        if read_json(root / "manifest.json")["complete"]:
            return read_json(root / "measurements.json")
        requests = [CompressionRequest(prompt=r.prompt, budget_tokens=max(1, math.floor(counter.count(r.prompt) * ratio)),
                                       focus_span=question_span(r)) for r in rows]
        compressor = factory(method, seed)
        for i in range(warmup):
            compressor.compress(requests[i % len(requests)])
        compressed = []
        for repeat in range(repeats):
            for row, request in zip(rows, requests):
                item = compressor.compress(request)
                compressed.append((row, repeat, item))
        del compressor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cases = []
        for target in targets:
            for i in range(warmup):
                target.answer(rows[i % len(rows)].prompt, use_cache=False)
            for index, (row, repeat, item) in enumerate(compressed):
                # Alternate order so one variant isn't always advantaged by warm state.
                variants = [("original", row.prompt, 0.), (method, item.text, item.seconds)]
                if index % 2:
                    variants.reverse()
                for variant, text, compression_seconds in variants:
                    started = time.perf_counter()
                    answer = target.answer(text, use_cache=False)
                    solver_seconds = time.perf_counter() - started
                    if answer["cache_hit"]:
                        raise RuntimeError("Benchmark unexpectedly received a cached answer")
                    cases.append({"id": row.id, "repeat": repeat, "target": target.alias, "method": variant,
                        "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"],
                        "compression_seconds": compression_seconds, "solver_seconds": solver_seconds,
                        "total_seconds": compression_seconds + solver_seconds,
                        "peak_compressor_memory_bytes": item.peak_memory_bytes if variant == method else 0})
            if not target.endpoint:
                target.unload()
        summary = []
        for alias in cfg.training_targets:
            for variant in ("original", method):
                group = [c for c in cases if (c["target"], c["method"]) == (alias, variant)]
                summary.append({"target": alias, "method": variant, "cases": len(group),
                    "total_seconds_median": float(np.median([c["total_seconds"] for c in group])),
                    "total_seconds_p95": float(np.quantile([c["total_seconds"] for c in group], .95)),
                    "compression_seconds_mean": float(np.mean([c["compression_seconds"] for c in group])),
                    "peak_compressor_memory_bytes": max(c["peak_compressor_memory_bytes"] for c in group)})
        result = {"summary": summary, "cases": cases, "profile": cfg.profile,
            "scientific_result": cfg.profile != "offline-engineering",
            "measurement": "Warmed compression plus uncached receiver request latency; model loading excluded.",
            "limitations": ["Disable server-side prefix caching for comparable prefill measurements.",
                "Remote request times include network and queueing; scorer and receiver are timed separately.",
                "Peak scorer allocation is CUDA-only; zero on CPU means unmeasured."]}
        atomic_json(root / "measurements.json", result)
        complete_run(root)
        return result
