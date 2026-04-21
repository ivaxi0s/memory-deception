from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deception_memory.llm.models import GenerationRequest, GenerationResponse


class LLMCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cache.json"

    def _key(self, request: GenerationRequest) -> str:
        payload = request.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _read_store(self) -> dict[str, dict]:
        if not self.cache_file.exists():
            return {}
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_store(self, store: dict[str, dict]) -> None:
        self.cache_file.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, request: GenerationRequest) -> GenerationResponse | None:
        store = self._read_store()
        key = self._key(request)
        if key not in store:
            return None
        data = store[key]
        return GenerationResponse.model_validate({**data, "cached": True})

    def put(self, request: GenerationRequest, response: GenerationResponse) -> None:
        store = self._read_store()
        store[self._key(request)] = response.model_dump(mode="json")
        self._write_store(store)
