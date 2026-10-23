from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrivacyPolicy(Strict):
    protected_attributes: tuple[str, ...]


class Request(Strict):
    context: str
    query: str
    privacy_policy: PrivacyPolicy
    public_auxiliary: dict[str, str] = Field(default_factory=dict)


class Example(Strict):
    id: str
    split: Literal["train", "validation", "test", "shift", "external"]
    entity_id: str
    template_id: str
    pair_id: str
    domain: str
    regime: Literal["independent", "correlated", "essential", "unspecified"]
    request: Request
    # These are evaluator/training labels. Never pass Example to a model backend.
    answer: str
    secrets: dict[str, str]
    required_facts: tuple[str, ...]

    @model_validator(mode="after")
    def check_labels(self):
        if set(self.secrets) != set(self.request.privacy_policy.protected_attributes):
            raise ValueError("Secret labels must match the policy's attribute names")
        if not self.id or not self.secrets or any(not x for x in self.secrets.values()):
            raise ValueError("Empty ids or secret labels are invalid")
        if not self.answer.strip() or not self.required_facts or any(not x.strip() for x in self.required_facts):
            raise ValueError("Task answers and required facts must be explicitly nonempty")
        return self


class Compression(Strict):
    text: str
    token_count: int = Field(ge=0)
    requested_budget: int = Field(ge=0)
    tokenizer: str
    raw_token_count: int = Field(ge=0)
    compressor_input_tokens: int = Field(default=0, ge=0)
    compressor_output_tokens: int = Field(default=0, ge=0)


class Observation(Strict):
    """Whitelist boundary: deliberately has no source, labels, IDs or split."""
    access: Literal["public", "context", "text", "top5", "top20", "full"]
    query: str
    public_auxiliary: dict[str, str]
    protected_attributes: tuple[str, ...]
    visible_text: str = ""
    # Probability arrays are delivered separately, never as source-record paths.


class ExperimentConfig(Strict):
    backend: Literal["rules", "hf"] = "rules"
    seeds: tuple[int, ...] = (11, 23, 47)
    ratios: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    methods: tuple[str, ...] = ("original", "empty", "redact", "redact_compress")
    task_models: tuple[str, ...] = ("rules",)
    compressor_model: str = "Qwen/Qwen2.5-3B-Instruct"
    revisions: dict[str, str] = Field(default_factory=dict)
    compressor_adapters: dict[str, str] = Field(default_factory=dict)
    adaptive_policy: str | None = None
    attacker_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    bootstrap_samples: int = Field(default=2000, ge=100)
    split: Literal["validation", "test", "shift", "external"] = "test"
    evaluation_file: str | None = None
    no_query: bool = False
    probability_access: tuple[Literal["text", "top5", "top20", "full"], ...] = ("text",)

    @model_validator(mode="after")
    def validate_design(self):
        allowed = {"original", "empty", "redact", "redact_compress", "summarize", "adaptive", "llmlingua2", "lora"}
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be nonempty and unique")
        if any(s < 0 or s >= 2**32 for s in self.seeds):
            raise ValueError("Seeds must be unsigned 32-bit integers")
        if not self.probability_access or len(set(self.probability_access)) != len(self.probability_access):
            raise ValueError("Observation accesses must be nonempty and unique")
        if not self.ratios or any(not 0 < r <= 1 for r in self.ratios) or len(set(self.ratios)) != len(self.ratios):
            raise ValueError("Ratios must be distinct and in (0,1]")
        if not self.methods or len(set(self.methods)) != len(self.methods) or not set(self.methods) <= allowed:
            raise ValueError("Unknown, duplicate or empty methods")
        if not self.task_models or len(set(self.task_models)) != len(self.task_models):
            raise ValueError("Task models must be distinct and nonempty")
        if self.backend == "rules" and (self.task_models != ("rules",) or set(self.methods) - {"original", "empty", "redact", "redact_compress"} or self.probability_access != ("text",)):
            raise ValueError("CPU rules pilot cannot impersonate neural models or probability conditions")
        if self.backend == "hf" and "rules" in self.task_models:
            raise ValueError("HF experiments require real task models")
        if "lora" in self.methods and set(self.compressor_adapters) != {str(s) for s in self.seeds}:
            raise ValueError("LoRA evaluation needs one explicitly mapped adapter per training seed")
        if "adaptive" in self.methods and not self.adaptive_policy:
            raise ValueError("Adaptive evaluation requires a tuned policy artifact")
        if self.evaluation_file and self.split != "external":
            raise ValueError("An explicit evaluation_file is reserved for external evaluation")
        return self
