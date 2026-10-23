from __future__ import annotations

import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompressionRequest(Strict):
    """The entire inference boundary: deliberately contains no gold labels."""
    prompt: str = Field(min_length=1)
    budget_tokens: int = Field(ge=1, strict=True)
    focus_span: tuple[StrictInt, StrictInt] | None = None

    @model_validator(mode="after")
    def nonempty_prompt(self):
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be whitespace only")
        if self.focus_span is not None:
            a, b = self.focus_span
            if not 0 <= a < b <= len(self.prompt) or not self.prompt[a:b].strip():
                raise ValueError("Focus span must identify nonempty text inside the prompt")
        return self


class CompressionResult(Strict):
    text: str
    budget_tokens: int
    input_tokens: int
    output_tokens: int
    raw_output_tokens: int
    seconds: float = 0
    peak_memory_bytes: int = 0
    status: Literal["ok", "overflow", "empty", "generation_limit"]
    # Source character ranges exist only for extractive methods.
    spans: tuple[tuple[int, int], ...] = ()


class Answer(Strict):
    text: str = Field(min_length=1)
    start: int = Field(ge=0)


class Example(Strict):
    id: str
    dataset: str = "squad_v1"
    group_id: str
    split: Literal["train", "validation", "test"]
    prompt: str
    context: str
    question: str
    answers: tuple[Answer, ...]

    @model_validator(mode="after")
    def validate_answers(self):
        if not self.id or not self.group_id or not self.answers or not self.prompt.strip():
            raise ValueError("Nonempty identities, prompt, and answers required")
        if any(self.context[a.start:a.start + len(a.text)] != a.text for a in self.answers):
            raise ValueError("Answer offset does not match source context")
        return self


class ModelSpec(Strict):
    model: str
    revision: str = ""
    chat_options: dict = Field(default_factory=dict)
    endpoint_env: str | None = None


class StudyConfig(Strict):
    profile: str
    seed: int = 11
    seeds: tuple[int, ...] = (11, 23, 47)
    ratios: tuple[float, ...] = (.75, .5, .25, .125)
    train_limit: int | None = None
    eval_limit: int | None = None
    candidates: int = 4
    sft_epochs: int = 1
    rl_epochs: int = 1
    learning_rate: float = 2e-5
    selector_learning_rate: float = 2e-5
    kl_beta: float = .02
    clip_epsilon: float = .2
    answer_max_tokens: int = 64
    bootstrap_samples: int = 2000
    attention_key_block: int = Field(default=256, ge=1)
    attention_query_block: int = Field(default=32, ge=1)
    models: dict[str, ModelSpec]
    training_targets: tuple[str, ...] = ("qwen", "mistral")
    transfer_target: str = "phi"
    reference_tokenizer: str = "rewriter"
    methods: tuple[str, ...] = ("original", "truncate", "llmlingua2", "untrained_rewrite", "rewrite_sft", "rewrite", "delete_sft", "delete", "question_only", "direct_answer")

    @model_validator(mode="after")
    def validate_study(self):
        if len(self.training_targets) != 2 or len(set(self.training_targets)) != 2 or self.transfer_target in self.training_targets:
            raise ValueError("Two distinct training targets and one separate transfer target required")
        required = {"rewriter", "selector", "teacher", self.reference_tokenizer, *self.training_targets, self.transfer_target, "llmlingua2"}
        if any(m.startswith("attention_") for m in self.methods):
            required.add("attention")
        if not required <= self.models.keys():
            raise ValueError("Missing model configuration")
        ids = [self.models[x].model for x in (*self.training_targets, self.transfer_target)]
        if len(set(ids)) != 3:
            raise ValueError("Target models must be distinct")
        if self.candidates < 2 or self.sft_epochs < 1 or self.rl_epochs < 1 or self.answer_max_tokens < 1:
            raise ValueError("Positive epochs, output limit, and at least two candidates required")
        if not self.ratios or len(set(self.ratios)) != len(self.ratios) or any(not math.isfinite(r) or not 0 < r <= 1 for r in self.ratios):
            raise ValueError("Retained ratios must be distinct and in (0, 1]")
        if any(x is not None and x < 1 for x in (self.train_limit, self.eval_limit)):
            raise ValueError("Limits must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            raise ValueError("Distinct nonnegative seeds required")
        if self.learning_rate <= 0 or self.selector_learning_rate <= 0 or self.kl_beta < 0 or not 0 < self.clip_epsilon < 1:
            raise ValueError("Invalid optimizer parameters")
        return self
