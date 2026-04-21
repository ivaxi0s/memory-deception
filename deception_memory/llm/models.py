from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    model_name: str
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.2
    reasoning_effort: str | None = None
    max_tokens: int = 512
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    model_name: str
    text: str
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    cached: bool = False
