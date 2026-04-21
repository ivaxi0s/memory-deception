from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"runs": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def update_run(self, run_id: str, payload: dict[str, Any]) -> None:
        data = self.load()
        data.setdefault("runs", {})[run_id] = payload
        self.save(data)
