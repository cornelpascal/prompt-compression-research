"""Publish aggregate prompt-inversion observations without raw prompt claims."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def export(source: Path, destination: Path) -> dict:
    raw = source.read_bytes()
    run = json.loads(raw)
    if run.get("evidence") != "exploratory_black_box_prompt_inversion" or run.get("model") != "gpt-6-luna":
        raise ValueError("Unexpected prompt-inversion result")
    cases = run["cases"]
    if len(cases) != 48 or any(case["status"] != "completed" for case in cases):
        raise ValueError("Only a complete 48-request run may be published")
    families = []
    for family in dict.fromkeys(case["family"] for case in cases):
        control = [c for c in cases if c["family"] == family and c["target"] == "control"]
        live = [c for c in cases if c["family"] == family and c["target"] == "live"]
        if len(control) != 6 or len(live) != 6:
            raise ValueError(f"Incomplete family: {family}")
        families.append({"family": family, "pairs": 6,
                         "controlExactPrompt": sum(bool(c["control_exact_prompt"]) for c in control),
                         "controlMarker": sum(bool(c["control_exact_marker"]) for c in control),
                         "liveInputTokens": round(sum(c["input_tokens"] for c in live) / 6, 1),
                         "liveOutputTokens": round(sum(c["output_tokens"] for c in live) / 6, 1),
                         "liveLatencySeconds": round(sum(c["seconds"] for c in live) / 6, 2)})
    result = {"schemaVersion": 1, "evidence": "exploratory_black_box_prompt_inversion",
              "recordedAt": datetime.now(timezone.utc).isoformat(),
              "model": run["model"], "reasoningEffort": run["reasoning_effort"],
              "sourceSha256": hashlib.sha256(raw).hexdigest(), "requests": 48,
              "controlledPromptKnown": True, "livePromptGroundTruthAvailable": False,
              "families": families,
              "interpretation": "Control counts measure literal recovery of supplied synthetic text. Live Luna outputs cannot verify its hidden system prompt.",
              "limitations": run["limitations"]}
    destination.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("site-data/prompt-inversion.json"))
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.output)))
