from __future__ import annotations

from jinja2 import Template

from deception_memory.config import PipelineConfig
from deception_memory.llm.client import BaseLLMClient
from deception_memory.llm.models import GenerationRequest
from deception_memory.llm.parsing import extract_json_object
from deception_memory.schemas import CandidateRecord, JudgeResult, ModelRun, QueryJudgeResult, generate_id
from deception_memory.settings import PROMPTS_DIR


class JudgeRunner:
    def __init__(self, client: BaseLLMClient, config: PipelineConfig) -> None:
        self.client = client
        self.config = config
        self.system_prompt = (PROMPTS_DIR / "judge_system.txt").read_text(encoding="utf-8")
        self.template = Template((PROMPTS_DIR / "judge_score.txt").read_text(encoding="utf-8"))

    def run(
        self,
        candidate: CandidateRecord,
        baseline_runs: list[ModelRun],
        memory_run: ModelRun,
        query_judge_result: QueryJudgeResult | None = None,
    ) -> JudgeResult:
        prompt = self.template.render(
            truth_anchor=candidate.truth_anchor.model_dump(mode="json"),
            memory_bundle=candidate.memory_bundle.model_dump(mode="json"),
            query=candidate.query.model_dump(mode="json"),
            query_judge_result=query_judge_result.model_dump(mode="json") if query_judge_result is not None else None,
            baseline_runs=[run.model_dump(mode="json") for run in baseline_runs],
            memory_run=memory_run.model_dump(mode="json"),
        )
        response = self.client.generate(
            GenerationRequest(
                model_name=self.config.models.judge_model,
                system_prompt=self.system_prompt,
                prompt=prompt,
                temperature=self.config.models.judge_temperature,
                reasoning_effort=self.config.models.judge_reasoning_effort,
                max_tokens=self.config.models.max_tokens,
                metadata={"component": "judge", "candidate_id": candidate.candidate_id},
            )
        )
        data = extract_json_object(response.text)
        return JudgeResult(
            judge_id=generate_id("judge"),
            score=int(data["score"]),
            mechanism=data["mechanism"],
            reasoning=data["reasoning"],
            plausibly_memory_induced=bool(data["plausibly_memory_induced"]),
            material_distortion=bool(data["material_distortion"]),
            compare_to_baseline=data["compare_to_baseline"],
            feedback_for_next_iteration=data["feedback_for_next_iteration"],
        )
