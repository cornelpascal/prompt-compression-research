from __future__ import annotations

import json
import re
from dataclasses import dataclass
import numpy as np

from .compressors import HFTokenizer
from .data import fields


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    logprobs: np.ndarray | None = None
    token_ids: list[int] | None = None


def task_prompt(context, query):
    return "Answer the query using only the supplied context. If unavailable, answer unknown. Return only the answer.\n" + json.dumps({"context": context, "query": query}, ensure_ascii=False)


class RuleTask:
    """Deterministic synthetic task checker, explicitly not an LLM."""
    model_id = "rules"
    revision = "structured-oracle-v1"

    def answer(self, context, query):
        f = fields(context)
        if "private_code" in query:
            return f.get("private_code", "unknown")
        try:
            if "refund" in query:
                return "eligible" if int(f["elapsed_days"]) <= int(f["refund_window_days"]) and f["item_opened"] == "no" else "ineligible"
            if "slot" in query:
                return "available" if f["requested_slot"] == f["available_slot"] else "unavailable"
        except (KeyError, ValueError):
            pass
        return "unknown"


class HFBackend:
    def __init__(self, model_id, revision, adapter=None, trainable=False, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError(f"Pin a 40-character Hugging Face commit for {model_id}")
        self.model_id, self.revision = model_id, revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
            dtype=dtype, trust_remote_code=False).to(self.device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=trainable)
        elif trainable:
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(r=16, lora_alpha=32,
                target_modules=["q_proj", "v_proj"], lora_dropout=.05, task_type="CAUSAL_LM"))
        if not trainable:
            self.model.requires_grad_(False)
        self.model.eval()
        self.counter = HFTokenizer(self.tokenizer, f"{model_id}@{revision}")

    def prompt_ids(self, prompt):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)

    def generate(self, prompt, max_new_tokens=32, trace=False, sample=False):
        import torch
        ids = torch.tensor([self.prompt_ids(prompt)], device=self.device)
        limit = getattr(self.model.config, "max_position_embeddings", 32768)
        if ids.shape[1] + max_new_tokens > limit:
            raise ValueError("Request exceeds model context limit; refusing silent truncation")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens, do_sample=sample, temperature=1.0 if sample else None,
                    top_k=0 if sample else None, top_p=1.0 if sample else None,
                    pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True,
                    output_logits=trace)
                new = output.sequences[0, ids.shape[1]:]
                lp = None
                if trace:
                    # Raw logits: no sampling penalties, top-k renormalization or teacher forcing.
                    lp = torch.stack(output.logits).squeeze(1).float().log_softmax(-1).cpu().numpy()
            return Generation(self.tokenizer.decode(new, skip_special_tokens=True), ids.shape[1], len(new), lp, new.tolist())
        finally:
            self.model.train(was_training)

    def answer(self, context, query, trace=False, max_new_tokens=32):
        return self.generate(task_prompt(context, query), max_new_tokens=max_new_tokens, trace=trace)

    def response_logprob(self, prompt, response, normalize=False):
        response_ids = self.tokenizer.encode(response, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        return self.response_ids_logprob(prompt, response_ids, normalize)

    def response_ids_logprob(self, prompt, response_ids, normalize=False):
        import torch
        if not response_ids:
            raise ValueError("Response token sequence must be nonempty")
        prompt_ids = self.prompt_ids(prompt)
        sequence = prompt_ids + response_ids
        if len(sequence) > getattr(self.model.config, "max_position_embeddings", 32768):
            raise ValueError("Training sequence exceeds context limit")
        ids = torch.tensor([sequence], device=self.device)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, len(prompt_ids)-1:-1].float()
        targets = torch.tensor(response_ids, device=self.device)
        logp = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
        return logp.mean() if normalize else logp.sum()

    def save(self, path):
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
