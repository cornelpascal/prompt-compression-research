from collections import defaultdict
import csv
from pathlib import Path

from .evaluation import load_results
from .io import atomic_json, read_json
from .metrics import clustered_interval


def report(directory, output, bootstrap_samples=2000):
    rows = load_results(directory)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    baseline = {(r["id"], r["seed"], r["target"]): r for r in rows if r["method"] == "original"}
    for row in rows:
        groups[row["method"], row["target"], row["ratio"]].append(row)
    summary = []
    for (method, target, ratio), cases in sorted(groups.items()):
        article_ids = [r["group_id"] for r in cases]
        entry = {"method": method, "target": target, "requested_retained_fraction": ratio,
            "cases": len(cases), "seeds": sorted({r["seed"] for r in cases}),
            "f1": clustered_interval([r["score"]["f1"] for r in cases], article_ids, bootstrap_samples),
            "em": sum(r["score"]["em"] for r in cases) / len(cases),
            "budgeted_f1": sum(r["budgeted_score"]["f1"] for r in cases) / len(cases),
            "budget_compliance": sum(r["budget_valid"] for r in cases) / len(cases),
            "reference_retained_fraction": sum(r["compression"]["output_tokens"] for r in cases) / max(1, sum(r["compression"]["input_tokens"] for r in cases)),
            "request_retained_fraction": sum(r["answer"]["input_tokens"] for r in cases) / max(1, sum(r["original_request_tokens"] for r in cases)),
            "compression_seconds": sum(r["compression"]["seconds"] for r in cases) / len(cases),
            "peak_compressor_memory_bytes": max(r["compression"].get("peak_memory_bytes", 0) for r in cases),
            "total_seconds": sum(r["total_seconds"] for r in cases) / len(cases),
            "cache_hits": sum(r["answer"]["cache_hit"] for r in cases),
            "answer_only_outputs": sum(r["diagnostics"]["answer_only_match"] for r in cases),
            "per_seed_f1": {str(seed): sum(r["score"]["f1"] for r in cases if r["seed"] == seed) / sum(r["seed"] == seed for r in cases) for seed in sorted({r["seed"] for r in cases})}}
        paired = [(r, baseline[(r["id"], r["seed"], target)]) for r in cases if (r["id"], r["seed"], target) in baseline]
        if paired:
            entry["paired_f1_delta"] = clustered_interval([r["score"]["f1"]-b["score"]["f1"] for r,b in paired],
                [r["group_id"] for r,b in paired], bootstrap_samples)
            entry["correct_to_wrong"] = sum(b["score"]["em"] == 1 and r["score"]["em"] == 0 for r,b in paired)
        summary.append(entry)
    manifest = read_json(Path(directory) / "manifest.json")
    profile = manifest["inputs"]["config"]["profile"]
    training_compute = []
    for path in sorted(Path(directory).parent.rglob("state.json")):
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        training_run, state = read_json(manifest_path), read_json(path)
        inputs = training_run.get("inputs", {})
        if inputs.get("stage") not in ("sft", "rl"):
            continue
        seconds = state["elapsed_seconds"]
        training_compute.append({"method": inputs["method"], "stage": inputs["stage"], "seed": inputs["seed"],
            "world_size": inputs["world_size"], "device_type": inputs["device_type"], "wall_seconds": seconds,
            "allocated_training_gpu_hours": seconds * inputs["world_size"] / 3600 if inputs["device_type"] == "cuda" else 0.,
            "allocated_training_tpu_hours": seconds * inputs["world_size"] / 3600 if inputs["device_type"] == "xla" else 0.,
            "peak_memory_bytes": state["peak_memory_bytes"]})
    result = {"evaluation_identity": manifest["identity"], "profile": profile,
        "scientific_result": profile != "offline-engineering", "training_compute": training_compute, "summary": summary,
        "limitations": ["Lexical diagnostics do not prove semantic fidelity.",
            "Intervals resample articles; training seed results are also reported individually.",
            "Cached solver durations are original observed inference durations, not current wall time.",
            "SQuAD 1.1 has no canonical unanswerable cases and may be present in model pretraining.",
            "Different compressor sizes are not a parameter-matched architectural comparison."]}
    if "calibration" in manifest["inputs"]:
        from .calibration import threshold_report
        result["attention_threshold"] = threshold_report(directory, bootstrap_samples)
    atomic_json(root / "summary.json", result)
    with (root / "summary.csv").open("w", newline="") as f:
        names = ["method", "target", "requested_retained_fraction", "request_retained_fraction", "reference_retained_fraction", "f1", "em", "budgeted_f1", "budget_compliance", "compression_seconds", "total_seconds"]
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for entry in summary:
            writer.writerow({name: entry[name]["mean"] if name == "f1" else entry[name] for name in names})
    # Export complete audit candidates, including raw prompts via dataset ids.
    flagged = [r for r in rows if r["diagnostics"]["answer_only_match"] or r["diagnostics"]["new_numeric_strings"] or not r["budget_valid"]]
    atomic_json(root / "audit.json", {"cases": flagged, "flags_are_not_verified_errors": True})
    plot(summary, root, engineering=profile == "offline-engineering")
    return {"groups": len(summary), "output": str(root), "audit_cases": len(flagged)}


def plot(summary, root, engineering=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for target in sorted({e["target"] for e in summary if e["target"] != "compressor_direct"}):
        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
        for method in sorted({e["method"] for e in summary if e["target"] == target}):
            points = sorted([e for e in summary if e["target"] == target and e["method"] == method], key=lambda e: e["request_retained_fraction"])
            x = [e["request_retained_fraction"] for e in points]
            y = [e["f1"]["mean"] for e in points]
            # Percentile intervals can exclude the sample mean in small samples.
            axis.plot(x, y, "o-", label=method)
            axis.fill_between(x, [e["f1"]["low"] for e in points], [e["f1"]["high"] for e in points], alpha=.12)
        title = "ENGINEERING FIXTURE — " + target if engineering else target
        axis.set(xlabel="Achieved input-token fraction (target chat template included)", ylabel="SQuAD F1", title=title, ylim=(0, 1.02))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        figure.savefig(root / f"tradeoff-{target}.png", dpi=160)
        figure.savefig(root / f"tradeoff-{target}.svg")
        plt.close(figure)
