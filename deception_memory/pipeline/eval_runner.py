from __future__ import annotations

from deception_memory.config import PipelineConfig
from deception_memory.enums import RunCondition
from deception_memory.llm.client import BaseLLMClient
from deception_memory.llm.models import GenerationRequest
from deception_memory.llm.parsing import simple_claim_parser
from deception_memory.schemas import MemoryBundle, ModelRun, QueryCandidate, TruthAnchor, generate_id


class MemoryEvalRunner:
    def __init__(self, client: BaseLLMClient, config: PipelineConfig) -> None:
        self.client = client
        self.config = config

    def run(self, anchor: TruthAnchor, bundle: MemoryBundle, query: QueryCandidate) -> ModelRun:
        memory_block = "\n".join(f"- {item.text}" for item in bundle.items)
        request = GenerationRequest(
            model_name=self.config.models.target_model,
            prompt=f"Persistent user memories:\n{memory_block}\n\nUser query:\n{query.text}",
            temperature=self.config.models.temperature,
            max_tokens=self.config.models.max_tokens,
            metadata={"condition": "memory", "anchor_id": anchor.anchor_id, "bundle_id": bundle.bundle_id},
        )
        response = self.client.generate(request)
        return ModelRun(
            run_id=generate_id("run"),
            condition=RunCondition.MEMORY,
            model_name=response.model_name,
            prompt_payload=request.model_dump(mode="json"),
            response_text=response.text,
            parsed_claims=simple_claim_parser(response.text),
        )
