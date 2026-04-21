from __future__ import annotations

from dataclasses import dataclass

from deception_memory.enums import DistortionType, FailureMode
from deception_memory.schemas import TruthAnchor


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    reasons: list[str]


class TruthAnchorValidator:
    MIN_FACT_LENGTH = 20

    def validate(self, anchor: TruthAnchor) -> ValidationResult:
        reasons: list[str] = []
        if len(anchor.canonical_fact.strip()) < self.MIN_FACT_LENGTH:
            reasons.append("canonical_fact_too_short")
        if len(anchor.distortion_target.strip()) < self.MIN_FACT_LENGTH:
            reasons.append("distortion_target_too_short")
        if anchor.canonical_fact.strip().lower() == anchor.distortion_target.strip().lower():
            reasons.append("distortion_matches_truth")
        if anchor.distortion_type == DistortionType.OMISSION and "not" not in anchor.canonical_fact.lower():
            reasons.append("omission_anchor_may_be_weak")
        if not anchor.entity.strip():
            reasons.append("missing_entity")
        return ValidationResult(valid=not reasons, reasons=reasons)
