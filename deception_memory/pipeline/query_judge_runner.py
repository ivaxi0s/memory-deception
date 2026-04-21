from __future__ import annotations

from jinja2 import Template

from deception_memory.config import PipelineConfig
from deception_memory.llm.client import BaseLLMClient
from deception_memory.llm.models import GenerationRequest
from deception_memory.llm.parsing import extract_json_object
from deception_memory.schemas import CandidateRecord, QueryJudgeResult, generate_id
from deception_memory.settings import PROMPTS_DIR


class QueryJudgeRunner:
    def __init__(self, client: BaseLLMClient, config: PipelineConfig) -> None:
        self.client = client
        self.config = config
        self.system_prompt = (PROMPTS_DIR / "query_judge_system.txt").read_text(encoding="utf-8")
        self.template = Template((PROMPTS_DIR / "query_judge.txt").read_text(encoding="utf-8"))

    def run(self, candidate: CandidateRecord) -> QueryJudgeResult:
        prompt = self.template.render(
            truth_anchor=candidate.truth_anchor.model_dump(mode="json"),
            memory_bundle=candidate.memory_bundle.model_dump(mode="json"),
            query=candidate.query.model_dump(mode="json"),
        )
        response = self.client.generate(
            GenerationRequest(
                model_name=self.config.models.judge_model,
                system_prompt=self.system_prompt,
                prompt=prompt,
                temperature=self.config.models.judge_temperature,
                reasoning_effort=self.config.models.judge_reasoning_effort,
                max_tokens=self.config.models.max_tokens,
                metadata={"component": "query_judge", "candidate_id": candidate.candidate_id},
            )
        )
        data = extract_json_object(response.text)
        return QueryJudgeResult(
            judge_id=generate_id("qjudge"),
            score=int(data["score"]),
            objective_query=bool(data["objective_query"]),
            nonleading=bool(data["nonleading"]),
            grounded_in_anchor=bool(data["grounded_in_anchor"]),
            reasoning=data["reasoning"],
            feedback_for_next_iteration=data["feedback_for_next_iteration"],
        )
