"""PILS-style sequence inversion with explicitly separated observation access.

This is an adaptation, not a claim to reproduce the original PILS experiments.
Full access uses selected additive-log-ratio coordinates; top-k uses an embedding
of only returned (token id, unrenormalized log probability) pairs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import random
import os

import numpy as np
import torch
from torch import nn

from .io import atomic_json, digest, file_hash, read_jsonl, source_hash
from .observations import observation_prompt, ProbabilityView
from .schema import Observation


def read_features(run_directory, descriptor, access):
    path = (Path(run_directory) / descriptor["path"]).resolve()
    if not path.is_relative_to(Path(run_directory).resolve()):
        raise ValueError("Probability artifact must be inside its run directory")
    if file_hash(path) != descriptor["sha256"]:
        raise ValueError("Probability artifact checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        result = {k: arrays[k].copy() for k in arrays.files}
    if str(result["access"].item()) != access:
        raise ValueError("Probability artifact access mismatch")
    if access == "full":
        if result["features"].ndim != 2 or not np.isfinite(result["features"]).all():
            raise ValueError("Invalid ALR feature array")
    else:
        ProbabilityView(access,result["token_ids"],result["logprobs"],int(result["vocabulary_size"]))
    return result


class SequenceInverter(nn.Module):
    def __init__(self, decoder, access, vocabulary_size, feature_dim=0):
        super().__init__()
        self.decoder, self.access = decoder, access
        width = decoder.config.d_model
        if access == "full":
            if feature_dim < 1:
                raise ValueError("ALR feature dimension required")
            self.project = nn.Sequential(nn.Linear(feature_dim,width),nn.LayerNorm(width))
        elif access in ("top5","top20"):
            self.token_embedding = nn.Embedding(vocabulary_size,64)
            self.probability_embedding = nn.Sequential(nn.Linear(1,64),nn.Tanh())
            self.project = nn.Sequential(nn.Linear(64,width),nn.LayerNorm(width))
        else:
            raise ValueError("Sequence inverter requires probability access")

    def inputs(self, data, public_ids):
        device = next(self.parameters()).device
        if data is None:
            text = self.decoder.get_input_embeddings()(public_ids.to(device))
            return {"inputs_embeds":text,"attention_mask":torch.ones(text.shape[:2],dtype=torch.long,device=device)}
        if self.access == "full":
            probability = self.project(torch.as_tensor(data["features"],dtype=torch.float32,device=device))
        else:
            token_ids = torch.as_tensor(data["token_ids"],dtype=torch.long,device=device)
            logp = torch.as_tensor(data["logprobs"],dtype=torch.float32,device=device).unsqueeze(-1)
            probability = self.project((self.token_embedding(token_ids)+self.probability_embedding(logp)).mean(1))
        text = self.decoder.get_input_embeddings()(public_ids.to(device))[0]
        embeddings = torch.cat([probability,text],dim=0).unsqueeze(0)
        return {"inputs_embeds":embeddings,"attention_mask":torch.ones(embeddings.shape[:2],dtype=torch.long,device=device)}

    def forward(self,data,public_ids,labels):
        return self.decoder(**self.inputs(data,public_ids),labels=labels).loss

    def reconstruct(self,data,public_ids,max_new_tokens=512):
        with torch.no_grad():
            return self.decoder.generate(**self.inputs(data,public_ids),max_new_tokens=max_new_tokens,do_sample=False)


def build_inverter(model,revision,access,vocabulary_size,feature_dim,device=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if not re.fullmatch(r"[a-fA-F0-9]{40}",revision or ""):
        raise ValueError("Pin the inverter model's immutable revision")
    decoder = AutoModelForSeq2SeqLM.from_pretrained(model,revision=revision,trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model,revision=revision,trust_remote_code=False)
    net = SequenceInverter(decoder,access,vocabulary_size,feature_dim)
    return net.to(device or ("cuda" if torch.cuda.is_available() else "cpu")),tokenizer


def save_torch(path,value):
    path = Path(path)
    temporary = path.with_suffix(".writing")
    torch.save(value,temporary)
    os.replace(temporary,path)


def train_inverter(export,run_directory,scope,output,model,revision,target="secrets",epochs=1,seed=11,public_only=False):
    from .runner import run_lock,seed_everything
    from .training import require_train
    data = [r for r in read_jsonl(export) if r["scope"] == scope]
    require_train(data)
    if len({r["protocol_id"] for r in data}) != 1:
        raise ValueError("Training scope combines different compressor protocols")
    if epochs < 1 or target not in ("secrets","compressed_text"):
        raise ValueError("Invalid inverter training options")
    access = data[0]["observation"]["access"]
    if access not in ("top5","top20","full") or any(r["observation"]["access"] != access for r in data):
        raise ValueError("Train one probability access per inverter")
    first = read_features(run_directory,data[0]["probabilities"],access)
    feature_dim = first["features"].shape[1] if access == "full" else 0
    with run_lock(output) as directory:
        seed_everything(seed)
        identity = digest({"export":file_hash(export),"scope":scope,"model":model,"revision":revision,
            "epochs":epochs,"seed":seed,"target":target,"public_only":public_only,"code":source_hash()})
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"identity":identity,"epoch":0}
        if state["identity"] != identity:
            raise ValueError("Inverter training identity changed")
        net,tokenizer = build_inverter(model,revision,access,int(first["vocabulary_size"]),feature_dim)
        device = next(net.parameters()).device
        optimizer = torch.optim.AdamW(net.parameters(),lr=2e-4)
        if state["epoch"]:
            saved = torch.load(directory/f"epoch-{state['epoch']:03d}.pt",map_location=device,weights_only=True)
            net.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
        for epoch in range(state["epoch"],epochs):
            seed_everything(seed+epoch)
            order = list(range(len(data)))
            random.Random(seed+epoch).shuffle(order)
            net.train()
            losses = []
            for i in order:
                row = data[i]
                obs = Observation.model_validate(row["observation"])
                inputs = None if public_only else read_features(run_directory,row["probabilities"],access)
                if public_only:
                    obs = obs.model_copy(update={"access":"public","visible_text":""})
                if not public_only and access == "full" and not np.array_equal(first["selected_tokens"],inputs["selected_tokens"]):
                    raise ValueError("ALR coordinate system changed within training scope")
                text = observation_prompt(obs) if target == "secrets" else "Reconstruct the hidden compressed context.\n"+json.dumps(obs.model_dump())
                public_ids = tokenizer(text,return_tensors="pt").input_ids.to(device)
                label = json.dumps(row["labels"][target],sort_keys=True) if target == "secrets" else row["labels"][target]
                labels = tokenizer(label,return_tensors="pt").input_ids.to(device)
                steps = 0 if inputs is None else inputs[next(k for k in ("features","token_ids") if k in inputs)].shape[0]
                if public_ids.shape[1]+steps > 4096 or labels.shape[1] > 2048:
                    raise ValueError("Inverter example exceeds explicit training length limits")
                optimizer.zero_grad(set_to_none=True)
                loss = net(inputs,public_ids,labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite inverter loss")
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
                losses.append(loss.item())
            save_torch(directory/f"epoch-{epoch+1:03d}.pt",{"model":net.state_dict(),"optimizer":optimizer.state_dict()})
            state = {"identity":identity,"epoch":epoch+1,"loss":float(np.mean(losses))}
            atomic_json(state_path,state)
        metadata = {"kind":"pils_style_sequence_adapter","scope":scope,"scope_metadata":data[0]["scope_metadata"],
            "protocol_id":data[0]["protocol_id"],
            "access":access,"target":target,"public_only":public_only,"model":model,"revision":revision,"feature_dim":feature_dim,
            "vocabulary_size":int(first["vocabulary_size"]),"epoch":epochs,"seed":seed,
            "training_ids":[r["id"] for r in data],"export_sha256":file_hash(export),
            "selected_tokens":first["selected_tokens"].tolist() if access == "full" else None,
            "claims":"PILS-inspired adaptation; losslessness and original-paper reproduction not established"}
        atomic_json(directory/"attack.json",metadata)
        return {"output":str(directory),"scope":scope,"target":target,"epochs":epochs}


class TrainedInverter:
    def __init__(self,directory):
        directory = Path(directory)
        self.metadata = json.loads((directory/"attack.json").read_text())
        m = self.metadata
        self.net,self.tokenizer = build_inverter(m["model"],m["revision"],m["access"],m["vocabulary_size"],m["feature_dim"])
        weights = torch.load(directory/f"epoch-{m['epoch']:03d}.pt",map_location=next(self.net.parameters()).device,weights_only=True)
        self.net.load_state_dict(weights["model"])
        self.net.eval()

    def predict(self,observation,data):
        m = self.metadata
        if m.get("public_only"):
            observation = observation.model_copy(update={"access":"public","visible_text":""})
            data = None
        elif observation.access != m["access"] or int(data["vocabulary_size"]) != m["vocabulary_size"]:
            raise ValueError("Inverter observation schema mismatch")
        if data is not None and m["access"] == "full" and not np.array_equal(data["selected_tokens"],m["selected_tokens"]):
            raise ValueError("Inverter ALR coordinate mismatch")
        prompt = observation_prompt(observation) if m["target"] == "secrets" else "Reconstruct the hidden compressed context.\n"+json.dumps(observation.model_dump())
        ids = self.tokenizer(prompt,return_tensors="pt").input_ids
        result = self.net.reconstruct(data,ids,max_new_tokens=128 if m["target"]=="secrets" else 2048)
        return self.tokenizer.decode(result[0],skip_special_tokens=True)
