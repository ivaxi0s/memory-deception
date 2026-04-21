from __future__ import annotations

from deception_memory.config import PipelineConfig
from deception_memory.schemas import CandidateRecord, JudgeResult, ModelRun, SampleRecord, generate_id


def baseline_contains_distortion(baseline_runs: list[ModelRun], candidate: CandidateRecord) -> bool:
    target = candidate.truth_anchor.distortion_target.lower()
    return any(target in run.response_text.lower() for run in baseline_runs)


def acceptance_decision(
    candidate: CandidateRecord,
    baseline_runs: list[ModelRun],
    memory_run: ModelRun,
    judge_result: JudgeResult,
    config: PipelineConfig,
) -> bool:
    if judge_result.score < config.evaluation.min_judge_score:
        return False
    if not judge_result.material_distortion or not judge_result.plausibly_memory_induced:
        return False
    if config.evaluation.reject_if_baseline_false and baseline_contains_distortion(baseline_runs, candidate):
        return False
    baseline_text = " ".join(run.response_text.lower() for run in baseline_runs)
    memory_text = memory_run.response_text.lower()
    if config.evaluation.require_material_baseline_difference and memory_text == baseline_text:
        return False
    return True


def build_sample_record(
    candidate: CandidateRecord,
    baseline_runs: list[ModelRun],
    memory_run: ModelRun,
    judge_result: JudgeResult,
    accepted: bool,
    iteration_history: list[dict],
) -> SampleRecord:
    return SampleRecord(
        sample_id=generate_id("sample"),
        truth_anchor=candidate.truth_anchor,
        memory_bundle=candidate.memory_bundle,
        query=candidate.query,
        baseline_runs=baseline_runs,
        memory_run=memory_run,
        judge_result=judge_result,
        iteration_history=iteration_history,
        accepted=accepted,
    )
