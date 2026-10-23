"""Validation-tuned reflective anonymization baseline; not a GEPA reproduction."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from .attacks import AttributeAttack,DirectCopyAttack
from .backends import HFBackend
from .compressors import Compressor
from .data import load_examples,validate_splits
from .io import atomic_json,file_hash,digest,source_hash
from .metrics import exact,answer_accuracy
from .observations import observe
from .runner import run_lock


SEED_INSTRUCTIONS = (
    "Remove personal identifiers and retain the minimum factual context needed to answer the query.",
    "Generalize identifying attributes and remove indirect identifying clues. Preserve exact task-relevant dates, quantities, and relationships.",
    "Rewrite the evidence as short factual statements. Omit protected values and correlated metadata whenever the query can still be answered.",
)


def tune(config_path,data_directory,output,rounds=2):
    cfg = json.loads(Path(config_path).read_text())
    train_path,valid_path = Path(data_directory)/"train.jsonl",Path(data_directory)/"validation.jsonl"
    train,valid = load_examples(train_path),load_examples(valid_path)
    if any(e.split!="train" for e in train) or any(e.split!="validation" for e in valid):
        raise ValueError("Policy optimization requires train and validation splits")
    validate_splits(train+valid)
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    model = cfg["compressor_model"]
    with run_lock(output) as directory:
        identity = digest({"config":cfg,"train":file_hash(train_path),"validation":file_hash(valid_path),"rounds":rounds,"code":source_hash()})
        identity_file = directory/"manifest.json"
        if identity_file.exists() and json.loads(identity_file.read_text())["identity"] != identity:
            raise ValueError("Adaptive tuning inputs changed")
        atomic_json(identity_file,{"identity":identity})
        rewriter = HFBackend(model,cfg["revisions"][model])
        policies = {}
        trials = []
        for task_model in cfg["task_models"]:
            task = HFBackend(task_model,cfg["revisions"][task_model])
            for ratio in cfg["ratios"]:
                candidates = list(SEED_INSTRUCTIONS)
                evaluated = []
                for index in range(len(SEED_INSTRUCTIONS)+rounds):
                    if index >= len(candidates):
                        best = max(evaluated,key=lambda x:x["objective"])
                        feedback = {k:best[k] for k in ("instruction","utility","leakage","retention")}
                        proposal = rewriter.generate("Improve this anonymization instruction using the aggregate validation feedback. Return only an instruction, without examples or private data.\n"+json.dumps(feedback),max_new_tokens=256).text
                        candidates.append(proposal)
                    instruction = candidates[index]
                    key = digest([task_model,ratio,index,instruction])
                    cached = directory/f"trial-{key}.json"
                    if cached.exists():
                        record = json.loads(cached.read_text())
                    else:
                        compressor = Compressor("adaptive",task.counter,rewriter,instruction=instruction)
                        training = []
                        for e in train:
                            budget = math.floor(len(task.counter.encode(e.request.context))*ratio)
                            z = compressor.compress(e.request.context,e.request.query,e.request.privacy_policy,budget).text
                            training.append((observe(e.request,"context",z),e.secrets))
                        attack = AttributeAttack().fit(training)
                        utility,leakage,retention = [],[],[]
                        for e in valid:
                            original = len(task.counter.encode(e.request.context))
                            z = compressor.compress(e.request.context,e.request.query,e.request.privacy_policy,math.floor(original*ratio))
                            answer = task.answer(z.text,e.request.query).text
                            obs = observe(e.request,"context",z.text)
                            guesses = [DirectCopyAttack().predict(obs),attack.predict(obs)]
                            utility.append(answer_accuracy(e,answer))
                            leakage.append(float(np.mean([max(exact(g.get(a),v) for g in guesses) for a,v in e.secrets.items()])))
                            retention.append(z.token_count/max(1,original))
                        record = {"instruction":instruction,"utility":float(np.mean(utility)),"leakage":float(np.mean(leakage)),
                            "retention":float(np.mean(retention)),"task_model":task_model,"ratio":ratio,"candidate":index}
                        record["objective"] = record["utility"]-record["leakage"]
                        atomic_json(cached,record)
                    evaluated.append(record)
                best = max(evaluated,key=lambda x:x["objective"])
                policies[f"{task_model}|{ratio}"] = best["instruction"]
                trials.extend(evaluated)
            del task
        artifact = {"kind":"validation_tuned_reflective_anonymization","identity":identity,"policies":policies,
            "training_ids":[e.id for e in train],"validation_ids":[e.id for e in valid],"config":cfg,
            "trials":trials,"limitations":["Independent adaptation, not the published GEPA implementation", "Selection attackers are direct-copy and attribute NB; final evaluation must use stronger retrained inverters"]}
        atomic_json(directory/"policy.json",artifact)
        return {"policy":str(directory/"policy.json"),"candidates_evaluated":len(trials)}
