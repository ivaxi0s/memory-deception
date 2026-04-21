from __future__ import annotations

from deception_memory.config import PipelineConfig
from deception_memory.enums import RunCondition
from deception_memory.llm.client import BaseLLMClient
from deception_memory.llm.models import GenerationRequest
from deception_memory.llm.parsing import simple_claim_parser
from deception_memory.schemas import ModelRun, QueryCandidate, TruthAnchor, generate_id


class BaselineRunner:
    def __init__(self, client: BaseLLMClient, config: PipelineConfig) -> None:
        self.client = client
        self.config = config

    def run(self, anchor: TruthAnchor, query: QueryCandidate) -> list[ModelRun]:
        runs: list[ModelRun] = []
        for index in range(self.config.evaluation.baseline_runs_per_candidate):
            request = GenerationRequest(
                model_name=self.config.models.target_model,
                prompt=query.text,
                temperature=self.config.models.temperature,
                max_tokens=self.config.models.max_tokens,
                metadata={"condition": "no_memory", "anchor_id": anchor.anchor_id, "run_index": index},
            )
            response = self.client.generate(request)
            runs.append(
                ModelRun(
                    run_id=generate_id("run"),
                    condition=RunCondition.NO_MEMORY,
                    model_name=response.model_name,
                    prompt_payload=request.model_dump(mode="json"),
                    response_text=response.text,
                    parsed_claims=simple_claim_parser(response.text),
                )
            )
        return runs

    def run_single(self, query: str, system_prompt: str | None = None) -> str:
        """Run a single query and return response text."""
        request = GenerationRequest(
            model_name=self.config.models.target_model,
            system_prompt=system_prompt,
            prompt=query,
            temperature=self.config.models.temperature,
            max_tokens=self.config.models.max_tokens,
            metadata={"component": "baseline"},
        )
        response = self.client.generate(request)
        return response.text
