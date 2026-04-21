from __future__ import annotations

import json

from jinja2 import Template

from deception_memory.config import PipelineConfig
from deception_memory.enums import FailureMode
from deception_memory.llm.client import BaseLLMClient
from deception_memory.llm.models import GenerationRequest
from deception_memory.llm.parsing import extract_json_object
from deception_memory.schemas import (
    CandidateRecord,
    JudgeResult,
    ModelRun,
    QueryCandidate,
    QueryJudgeResult,
    RefinementResult,
)
from deception_memory.settings import PROMPTS_DIR


class RefinementEngine:
    def __init__(self, client: BaseLLMClient, config: PipelineConfig) -> None:
        self.client = client
        self.config = config
        self.system_prompt = (PROMPTS_DIR / "red_team_system.txt").read_text(encoding="utf-8")
        self.template = Template((PROMPTS_DIR / "red_team_refine.txt").read_text(encoding="utf-8"))

    def diagnose(
        self,
        candidate: CandidateRecord,
        baseline_runs: list[ModelRun],
        memory_run: ModelRun,
        judge_result: JudgeResult,
        query_judge_result: QueryJudgeResult | None = None,
    ) -> RefinementResult:
        prompt = self.template.render(
            truth_anchor=candidate.truth_anchor.model_dump(mode="json"),
            memory_bundle=candidate.memory_bundle.model_dump(mode="json"),
            query=candidate.query.model_dump(mode="json"),
            query_judge_result=query_judge_result.model_dump(mode="json") if query_judge_result is not None else None,
            baseline_runs=[run.model_dump(mode="json") for run in baseline_runs],
            memory_run=memory_run.model_dump(mode="json"),
            judge_result=judge_result.model_dump(mode="json"),
            mutation_history=candidate.mutation_history,
        )
        try:
            response = self.client.generate(
                GenerationRequest(
                    model_name=self.config.models.generator_model,
                    system_prompt=self.system_prompt,
                    prompt=prompt,
                    temperature=self.config.models.temperature,
                    max_tokens=self.config.models.max_tokens,
                    metadata={"component": "refinement", "candidate_id": candidate.candidate_id},
                )
            )
            data = extract_json_object(response.text)
            return RefinementResult.model_validate(data)
        except (ValueError, KeyError, json.JSONDecodeError):
            return self._fallback(judge_result)

    def refine_pre_eval(
        self,
        candidate: CandidateRecord,
        query_judge_result: QueryJudgeResult,
    ) -> RefinementResult:
        return self.diagnose(
            candidate=candidate,
            baseline_runs=[],
            memory_run=ModelRun(
                run_id="pre_eval_placeholder",
                condition="memory",
                model_name="none",
                prompt_payload={},
                response_text="Pre-evaluation refinement stage.",
            ),
            judge_result=JudgeResult(
                judge_id="pre_eval_placeholder",
                score=query_judge_result.score,
                mechanism="query_objectivity",
                reasoning=query_judge_result.reasoning,
                plausibly_memory_induced=False,
                material_distortion=False,
                compare_to_baseline="not_run_yet",
                feedback_for_next_iteration=query_judge_result.feedback_for_next_iteration,
            ),
            query_judge_result=query_judge_result,
        )

    def _fallback(self, judge_result: JudgeResult) -> RefinementResult:
        failure_mode = FailureMode.JUDGE_UNCERTAIN.value
        if not judge_result.material_distortion:
            failure_mode = FailureMode.DISTORTION_IMMATERIAL.value
        elif not judge_result.plausibly_memory_induced:
            failure_mode = FailureMode.MEMORY_MOTIVE_WEAK.value
        return RefinementResult(
            diagnosis=judge_result.reasoning,
            failure_mode=failure_mode,
            strategy="Use a softer benign query surface and strengthen motive triggers while preserving realism.",
        )
