from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

from .attacks import AttributeAttack,NeuralAttack,parse_prediction
from .backends import HFBackend
from .data import load_examples
from .io import atomic_json,digest,file_hash,source_hash,environment
from .metrics import exact,normalize,summarize
from .runner import result_row,scope_id,run_lock,read_case
from .schema import Observation


def text_f1(prediction,target):
    a,b = Counter(normalize(prediction).split()),Counter(normalize(target).split())
    overlap = sum((a&b).values())
    return 2*overlap/(sum(a.values())+sum(b.values())) if a or b else 1.


def checkpoint_identity(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)):file_hash(p) for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != ".lock"}


def evaluate_attacker(run_directory,data_path,checkpoint,output,public_checkpoint=None):
    run_directory,checkpoint = Path(run_directory),Path(checkpoint)
    metadata = json.loads((checkpoint/"attack.json").read_text())
    run_manifest = json.loads((run_directory/"manifest.json").read_text())
    if file_hash(data_path) != run_manifest["identity"]["evaluation_sha256"]:
        raise ValueError("Evaluation labels do not match the run's frozen dataset")
    rows = {e.id:e for e in load_examples(data_path)}
    if any(e.split == "train" for e in rows.values()) or set(rows)&set(metadata["training_ids"]):
        raise ValueError("Attacker evaluation overlaps its training set")
    cases = [read_case(p) for p in sorted((run_directory/"cases").glob("*.json"))]
    scope = metadata["scope_metadata"]
    cases = [c for c in cases if all(c[k]==v for k,v in scope.items())]
    if any(c["protocol_id"] != metadata["protocol_id"] for c in cases):
        raise ValueError("Attacker was trained against a different compressor/model protocol")
    if metadata.get("public_only"):
        raise ValueError("Pass a public-only inverter as --public-checkpoint, not as the main attack")
    access = metadata["access"]
    if scope_id(scope["seed"],scope["model"],scope["method"],scope["ratio"],access) != metadata["scope"]:
        raise ValueError("Attacker scope metadata is inconsistent")
    training = [c for c in cases if c["split"]=="train"]
    evaluation = [c for c in cases if c["id"] in rows and c["split"] != "train"]
    if not training or not evaluation:
        raise ValueError("No matching training/evaluation observations")
    if public_checkpoint:
        prior_meta = json.loads((Path(public_checkpoint)/"attack.json").read_text())
        prior_sequence = prior_meta.get("kind") == "pils_style_sequence_adapter"
        is_public = prior_meta.get("public_only",False) if prior_sequence else prior_meta["access"]=="public"
        if not is_public or prior_meta["scope_metadata"] != scope or prior_meta["protocol_id"] != metadata["protocol_id"] or set(rows)&set(prior_meta["training_ids"]):
            raise ValueError("Public-only attacker has mismatched scope or overlapping training")
        if any(prior_meta[k] != metadata[k] for k in ("model","revision","target")):
            raise ValueError("Public-only control must match the attack's model, revision and target")
        if prior_sequence:
            from .inversion import TrainedInverter
            prior = TrainedInverter(public_checkpoint)
        else:
            prior = NeuralAttack(HFBackend(prior_meta["model"],prior_meta["revision"],adapter=prior_meta["adapter"]))
        prior_name = "independent_neural_public"
    else:
        prior_sequence = False
        prior = AttributeAttack().fit((Observation.model_validate(c["observations"]["public"]),c["evaluation"]["secrets"]) for c in training)
        prior_name = "attribute_nb_public (weaker than a neural prior; descriptive comparison only)"
    probability = metadata.get("kind") == "pils_style_sequence_adapter"
    if probability:
        from .inversion import TrainedInverter,read_features
        attacker = TrainedInverter(checkpoint)
    else:
        attacker = NeuralAttack(HFBackend(metadata["model"],metadata["revision"],adapter=metadata["adapter"]))
    results = []
    reconstruction = []
    with run_lock(output) as directory:
        identity = {"run_manifest_sha256":file_hash(run_directory/"manifest.json"),
            "data_sha256":file_hash(data_path),"code_sha256":source_hash(),
            "checkpoint":checkpoint_identity(checkpoint),
            "public_checkpoint":checkpoint_identity(public_checkpoint) if public_checkpoint else None}
        manifest_path = directory/"manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
            raise ValueError("Attacker evaluation inputs changed; choose a new output directory")
        manifest = {"identity":identity,"environment":environment(),"attack":metadata,
            "public_prior":prior_name,"status":"running"}
        atomic_json(manifest_path,manifest)
        for case in evaluation:
            e = rows[case["id"]]
            saved = directory/"cases"/(digest([e.id,metadata["scope"]])+".json")
            if saved.exists():
                cached = read_case(saved)
                if cached["result"] is not None:
                    results.append(cached["result"])
                if cached["reconstruction"] is not None:
                    reconstruction.append(cached["reconstruction"])
                continue
            obs = Observation.model_validate(case["observations"][access])
            if probability:
                prediction = attacker.predict(obs,read_features(run_directory,case["probabilities"][access],access))
                if metadata["target"] == "compressed_text":
                    item = {"id":e.id,"pair_id":e.pair_id,"prediction":prediction,
                        "compressed_input_exact":exact(prediction,case["compression"]["text"]),
                        "compressed_input_token_f1":text_f1(prediction,case["compression"]["text"])}
                    reconstruction.append(item)
                    payload = {"result":None,"reconstruction":item}
                    atomic_json(saved,{**payload,"checkpoint_sha256":digest(payload)})
                    continue
                guesses = parse_prediction(prediction,obs.protected_attributes)
            else:
                guesses = attacker.predict(obs)
            if prior_sequence:
                public_guesses = parse_prediction(prior.predict(obs,None),obs.protected_attributes)
            else:
                public_guesses = prior.predict(Observation.model_validate(case["observations"]["public"]))
            result = result_row(case,e,access,"learned:"+metadata["model"],guesses,public_guesses)
            results.append(result)
            payload = {"result":result,"reconstruction":None}
            atomic_json(saved,{**payload,"checkpoint_sha256":digest(payload)})
        atomic_json(directory/"results.json",results)
        atomic_json(directory/"reconstruction.json",reconstruction)
        if results:
            summarize(results,directory)
        atomic_json(manifest_path,{**manifest,"status":"complete","rows":len(results),
            "reconstruction_rows":len(reconstruction),"results_sha256":file_hash(directory/"results.json"),
            "reconstruction_sha256":file_hash(directory/"reconstruction.json")})
    return {"rows":len(results),"reconstruction_rows":len(reconstruction),"output":str(output)}
