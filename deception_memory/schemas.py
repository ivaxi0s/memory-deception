from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from deception_memory.enums import (
    DeceptionFamily,
    DistortionType,
    MemoryCategory,
    QuerySurface,
    RunCondition,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TruthAnchor(BaseModel):
    anchor_id: str
    domain: str
    entity: str
    canonical_fact: str
    short_fact_label: str
    distortion_target: str
    distortion_type: DistortionType
    source_note: str | None = None
    difficulty: int | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 5:
            raise ValueError("difficulty must be between 1 and 5")
        return value


class MemoryItem(BaseModel):
    memory_id: str
    text: str
    category: MemoryCategory
    realism_score: float | None = None
    role_in_sample: str | None = None


class MemoryBundle(BaseModel):
    bundle_id: str
    items: list[MemoryItem]
    motive_family: DeceptionFamily
    plausibility_notes: str | None = None


class QueryCandidate(BaseModel):
    query_id: str
    surface_type: QuerySurface
    text: str
    benignity_score: float | None = None
    nonleading_score: float | None = None
    requires_anchor_use: bool = True


class ModelRun(BaseModel):
    run_id: str
    condition: RunCondition
    model_name: str
    prompt_payload: dict[str, Any]
    response_text: str
    parsed_claims: list[str] | None = None
    timestamp: str = Field(default_factory=utc_now_iso)


class JudgeResult(BaseModel):
    judge_id: str
    score: int
    mechanism: str
    reasoning: str
    plausibly_memory_induced: bool
    material_distortion: bool
    compare_to_baseline: str
    feedback_for_next_iteration: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("judge score must be in [1, 5]")
        return value


class QueryJudgeResult(BaseModel):
    judge_id: str
    score: int
    objective_query: bool
    nonleading: bool
    grounded_in_anchor: bool
    reasoning: str
    feedback_for_next_iteration: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("query judge score must be in [1, 5]")
        return value


class DeceptionJudgeResult(BaseModel):
    judge_id: str
    score: int
    reasoning: str
    plausibly_memory_induced: bool
    material_distortion: bool
    feedback_for_next_iteration: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("deception judge score must be in [1, 5]")
        return value


class SampleRecord(BaseModel):
    sample_id: str
    truth_anchor: TruthAnchor
    memory_bundle: MemoryBundle
    query: QueryCandidate
    baseline_runs: list[ModelRun]
    memory_run: ModelRun
    judge_result: DeceptionJudgeResult
    iteration_history: list[dict[str, Any]] = Field(default_factory=list)
    accepted: bool


class CandidateRecord(BaseModel):
    candidate_id: str
    anchor_id: str | None = None
    bundle_id: str | None = None
    query_id: str | None = None
    truth_anchor: TruthAnchor
    memory_bundle: MemoryBundle
    query: QueryCandidate
    query_judge_result: QueryJudgeResult | None = None
    iteration_index: int = 0
    failure_modes: list[str] = Field(default_factory=list)
    mutation_history: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = None

    def model_post_init(self, __context: Any) -> None:
        self.anchor_id = self.truth_anchor.anchor_id
        self.bundle_id = self.memory_bundle.bundle_id
        self.query_id = self.query.query_id


class RefinementResult(BaseModel):
    diagnosis: str
    failure_mode: str
    revised_memory_bundle: MemoryBundle | None = None
    revised_query: QueryCandidate | None = None
    strategy: str


class BaselineResult(BaseModel):
    candidate_id: str
    anchor_id: str
    bundle_id: str
    query_id: str
    baseline_runs: list[ModelRun]


class MemoryEvalResult(BaseModel):
    candidate_id: str
    anchor_id: str
    bundle_id: str
    query_id: str
    baseline_run_ids: list[str]
    memory_run: ModelRun


class JudgmentArtifact(BaseModel):
    candidate_id: str
    anchor_id: str
    bundle_id: str
    query_id: str
    baseline_run_ids: list[str]
    memory_run_id: str
    query_judge_result: QueryJudgeResult | None = None
    judge_result: JudgeResult
    accepted: bool
    iteration_history: list[dict[str, Any]] = Field(default_factory=list)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"
