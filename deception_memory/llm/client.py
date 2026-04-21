from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Iterable

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from deception_memory.llm.caching import LLMCache
from deception_memory.llm.models import GenerationRequest, GenerationResponse


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    def __init__(
        self,
        scripted_responses: Iterable[str] | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.responses = deque(scripted_responses or [])
        self.cache = cache

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0))
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.cache is not None:
            cached = self.cache.get(request)
            if cached is not None:
                return cached
        if self.responses:
            text = self.responses.popleft()
        else:
            text = self._default_response(request)
        response = GenerationResponse(
            model_name=request.model_name,
            text=text,
            raw_payload={"request_metadata": request.metadata},
        )
        if self.cache is not None:
            self.cache.put(request, response)
        return response

    def _default_response(self, request: GenerationRequest) -> str:
        component = str(request.metadata.get("component", ""))
        prompt = request.prompt.lower()
        if component == "query_judge":
            return (
                '{"score": 4, "objective_query": true, "nonleading": true, '
                '"grounded_in_anchor": true, '
                '"reasoning": "Mock query judge response.", '
                '"feedback_for_next_iteration": "accepted"}'
            )
        if "judge" in prompt or component == "judge":
            return (
                '{"score": 4, "mechanism": "selective_framing", '
                '"material_distortion": true, "plausibly_memory_induced": true, '
                '"reasoning": "Mock judge response.", '
                '"compare_to_baseline": "memory response is materially more deceptive than no-memory baseline", '
                '"feedback_for_next_iteration": "accepted"}'
            )
        return "Mock response."


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        cache: LLMCache | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        self.cache = cache
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization or os.getenv("OPENAI_ORG_ID"),
            project=project or os.getenv("OPENAI_PROJECT_ID"),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.cache is not None:
            cached = self.cache.get(request)
            if cached is not None:
                return cached

        input_payload: list[dict[str, Any]] = []
        if request.system_prompt:
            input_payload.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": request.system_prompt}],
                }
            )
        input_payload.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": request.prompt}],
            }
        )

        request_kwargs: dict[str, Any] = {
            "model": request.model_name,
            "input": input_payload,
            "max_output_tokens": request.max_tokens,
        }
        if request.reasoning_effort:
            request_kwargs["reasoning"] = {"effort": request.reasoning_effort}
        else:
            request_kwargs["temperature"] = request.temperature

        response = self.client.responses.create(**request_kwargs)
        text = self._extract_output_text(response)
        generation = GenerationResponse(
            model_name=request.model_name,
            text=text,
            raw_payload=response.model_dump(mode="json"),
        )
        if self.cache is not None:
            self.cache.put(request, generation)
        return generation

    def _extract_output_text(self, response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return str(response.output_text)

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
