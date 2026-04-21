from __future__ import annotations

from collections import Counter

from deception_memory.schemas import SampleRecord


def compute_summary(samples: list[SampleRecord]) -> dict:
    total = len(samples)
    accepted = [sample for sample in samples if sample.accepted]
    by_family = Counter(sample.memory_bundle.motive_family.value for sample in samples)
    by_distortion = Counter(sample.truth_anchor.distortion_type.value for sample in samples)
    by_surface = Counter(sample.query.surface_type.value for sample in samples)
    failure_feedback = Counter(
        sample.judge_result.feedback_for_next_iteration for sample in samples if not sample.accepted
    )
    return {
        "generated_candidates": total,
        "accepted_samples": len(accepted),
        "acceptance_rate": (len(accepted) / total) if total else 0.0,
        "acceptance_by_deception_family": dict(by_family),
        "acceptance_by_distortion_type": dict(by_distortion),
        "acceptance_by_query_surface": dict(by_surface),
        "mean_judge_score": (sum(sample.judge_result.score for sample in samples) / total) if total else 0.0,
        "top_failure_reasons": dict(failure_feedback),
    }
